using System.Diagnostics;
using System.Globalization;
using System.Net;
using System.Net.Http.Headers;
using System.Text;
using System.Text.Json;
using System.Text.Json.Serialization;
using System.Text.Json.Serialization.Metadata;
using SignLanguageApp.Model;
using SignLanguageApp.Helpers;

namespace SignLanguageApp.Services;

public interface IApiService
{
    // Authentication
    Task<LoginResponse?> LoginAsync(string email, string password);
    Task<(bool Success, string Message)> RegisterAsync(string email, string password, string name);
    Task<bool> LogoutAsync(); 
    Task<bool> RefreshTokenAsync(string refreshToken);
    void SetAuthToken(string token);
    Task<string?> GetAuthTokenAsync();
    string EnsureAbsoluteUrl(string? url);

    // Learning - Stats & Progress
    Task<UserStatsDto?> GetUserStatsAsync();
    Task<ApiResponse<List<MediaDto>>?> GetMediaAsync();
    Task<ApiResponse<List<SignerCreditDto>>?> GetSignerCreditsAsync();
    Task<bool> SubmitFeedbackAsync(FeedbackRequest feedback);

    // Learning - Categories
    Task<ApiResponse<IEnumerable<LessonCategoryDto>>?> GetCategoriesAsync();
    Task<ApiResponse<LessonCategoryDto>?> GetCategoryAsync(int categoryId);

    // Learning - Lessons
    Task<ApiResponse<IEnumerable<LessonDetailDto>>?> GetLessonsByCategoryAsync(int categoryId);
    Task<ApiResponse<LessonDetailDto>?> GetLessonAsync(int lessonId);
    Task<ApiResponse<IEnumerable<LessonDto>>?> GetLessonsAsync();

    // Learning - Daily Review & Spaced Repetition
    Task<ApiResponse<DailyGoalDto>?> GetDailyGoalAsync();
    Task<ApiResponse<IEnumerable<SpacedRepetitionLessonDto>>?> GetDailyReviewLessonsAsync();
    Task<ApiResponse<UpcomingReviewsDto>?> GetUpcomingReviewsAsync();

    // Learning - Recommendations
    Task<ApiResponse<PersonalizedRecommendationDto>?> GetPersonalizedRecommendationAsync();
    Task<ApiResponse<InteractiveLessonDto>?> GetInteractiveLessonAsync(int lessonId);

    // Learning - Progress Tracking
    Task<ApiResponse<bool>?> MarkLessonCompleteAsync(int lessonId);
    Task<ApiResponse<bool>?> MarkReviewCompleteAsync(int reviewLessonId, int qualityRating = 4);

    // Videos - Listing & Details
    Task<ApiResponse<IEnumerable<VideoDto>>?> GetVideosAsync();
    Task<ApiResponse<VideoDto>?> GetVideoAsync(int videoId);
    Task<ApiResponse<IEnumerable<VideoDto>>?> GetVideosByCategory(string category);

    // Videos - Interactions
    Task<bool> LikeVideoAsync(int videoId, bool like);
    Task<bool> WatchVideoAsync(int videoId);

    // Camera - Real-Time Gesture Prediction
    Task<GesturePredictionResponseDto?> PredictGestureFromImageAsync(byte[] imageData, string? targetSign = null, CancellationToken cancellationToken = default);
    // User - Profile Management
    Task<ApiResponse<UserProfileDto>?> GetUserProfileAsync();
    Task<bool> UpdateNameAsync(string newName);
    Task<bool> UpdatePasswordAsync(string oldPassword, string newPassword);
    Task<bool> UpdateAvatarAsync(string avatarUrl);
    Task<bool> UpdateDescriptionAsync(string description);
    void UpdateBaseAddress(string newUrl);
    string CurrentBaseUrl { get; }
    Task<bool> CheckConnectionAsync();

    // Social & Gamification
    Task<LeaderboardDto?> GetLeaderboardAsync(int count = 10);
    Task<IEnumerable<AchievementBadgeDto>?> GetAchievementsAsync();
}
public class ApiService : IApiService
{
    private readonly HttpClient _httpClient;
    private readonly IConnectivityService _connectivityService;
    private readonly IApiConfigService _apiConfig;
    private readonly ILessonPayloadSecurityService _securityService;
    private readonly IDatabaseService _databaseService;
    private string? _authToken;
    
    public ApiService(HttpClient httpClient, IConnectivityService connectivityService, IApiConfigService apiConfig, ILessonPayloadSecurityService securityService, IDatabaseService databaseService)
    {
        _httpClient = httpClient;
        _connectivityService = connectivityService;
        _apiConfig = apiConfig;
        _securityService = securityService;
        _databaseService = databaseService;
    }


    private async Task EnsureConnectivityAsync()
    {
        if (!_connectivityService.IsConnected)
        {
            throw new NoInternetException();
        }

        // Only check server reachability if we have a real base URL configured
        var currentBaseUrl = _apiConfig.BaseUrl;
        if (!string.IsNullOrEmpty(currentBaseUrl) && 
            !currentBaseUrl.Contains("api.internal"))
        {
            // Try to hit the health endpoint which is more reliable than a HEAD on the root
            var healthUrl = currentBaseUrl.TrimEnd('/') + "/gesture/health";
            Debug.WriteLine($"EnsureConnectivityAsync: Checking reachability of {healthUrl}");
            if (!await _connectivityService.IsServerReachableAsync(healthUrl))
            {
                Debug.WriteLine($"EnsureConnectivityAsync: Server {currentBaseUrl} is NOT reachable.");
                throw new ServerUnreachableException(currentBaseUrl);
            }
            Debug.WriteLine($"EnsureConnectivityAsync: Server {currentBaseUrl} is reachable.");
        }
    }

    private async Task<T?> ExecuteSafeAsync<T>(Func<Task<HttpResponseMessage>> action, JsonTypeInfo<ApiResponse<T>> typeInfo, CancellationToken ct = default) where T : class
    {
        try
        {
            await EnsureConnectivityAsync();

            using var response = await action();
            
            if (response.StatusCode == HttpStatusCode.Unauthorized)
            {
                var errorBody = await response.Content.ReadAsStringAsync();
                Debug.WriteLine($">>> LOGIN UNAUTHORIZED. Body: {errorBody}");
                throw new UnauthorizedException();
            }

            if (!response.IsSuccessStatusCode)
            {
                var errorContent = await response.Content.ReadAsStringAsync(ct);
                throw new ApiException($"Server returned an error.", (int)response.StatusCode, errorContent);
            }

            var json = await response.Content.ReadAsStringAsync(ct);
            
            var wrapped = JsonSerializer.Deserialize(json, typeInfo);
            if (wrapped != null && (wrapped.Success || wrapped.Data != null))
            {
                return wrapped.Data;
            }

            return null;
        }
        catch (Exception ex) when (ex is not NoInternetException && ex is not ServerUnreachableException && ex is not UnauthorizedException && ex is not ApiException)
        {
            Debug.WriteLine($"Unhandled API error: {ex.GetType().Name} - {ex.Message}");
            Debug.WriteLine($"Stack Trace: {ex.StackTrace}");
            throw new ApiException($"An unexpected error occurred while connecting to the server: {ex.Message}", 0, ex.Message);
        }
    }

    private async Task<T?> ExecuteSafeDirectAsync<T>(Func<Task<HttpResponseMessage>> action, JsonTypeInfo<T> typeInfo, CancellationToken ct = default) where T : class
    {
        try
        {
            await EnsureConnectivityAsync();

            using var response = await action();
            
            if (response.StatusCode == HttpStatusCode.Unauthorized)
            {
                throw new UnauthorizedException();
            }

            if (!response.IsSuccessStatusCode)
            {
                var errorContent = await response.Content.ReadAsStringAsync(ct);
                throw new ApiException($"Server returned an error.", (int)response.StatusCode, errorContent);
            }

            var json = await response.Content.ReadAsStringAsync(ct);
            return JsonSerializer.Deserialize(json, typeInfo);
        }
        catch (Exception ex) when (ex is not NoInternetException && ex is not ServerUnreachableException && ex is not UnauthorizedException && ex is not ApiException)
        {
            throw new ApiException($"An unexpected error occurred while connecting to the server: {ex.Message}", 0, ex.Message);
        }
    }

    private async Task<T?> ExecuteSafeWithRetryAsync<T>(Func<Task<HttpResponseMessage>> action, JsonTypeInfo<ApiResponse<T>> typeInfo, int maxRetries = 2, CancellationToken ct = default) where T : class
    {
        int retryCount = 0;
        while (true)
        {
            try
            {
                return await ExecuteSafeAsync<T>(action, typeInfo, ct);
            }
            catch (Exception ex) when (retryCount < maxRetries && (ex is ServerUnreachableException || ex is NoInternetException))
            {
                retryCount++;
                Debug.WriteLine($"Retrying API call ({retryCount}/{maxRetries}) due to: {ex.Message}");
                await Task.Delay(1000 * retryCount, ct);
            }
        }
    }

    public string CurrentBaseUrl => _httpClient.BaseAddress?.ToString() ?? _apiConfig.BaseUrl;

    public void UpdateBaseAddress(string newUrl)
    {
        try
        {
            if (string.IsNullOrWhiteSpace(newUrl)) return;

            // Ensure trailing slash
            if (!newUrl.EndsWith("/")) newUrl += "/";

            if (Uri.TryCreate(newUrl, UriKind.Absolute, out var uri))
            {
                _httpClient.BaseAddress = uri;
                _apiConfig.BaseUrl = newUrl; // Sync back to config
                Debug.WriteLine($"ApiService BaseAddress updated to: {newUrl}");
            }
            else
            {
                Debug.WriteLine($"Invalid BaseAddress URL: {newUrl}");
            }
        }
        catch (Exception ex)
        {
            Debug.WriteLine($"Failed to update BaseAddress: {ex.Message}");
        }
    }

    public async Task<bool> CheckConnectionAsync()
    {
        try
        {
            // We use a very short timeout for the health check
            using var cts = new CancellationTokenSource(TimeSpan.FromSeconds(5));
            
            // Try to hit the gesture health endpoint which is lightweight
            var response = await _httpClient.GetAsync("gesture/health", cts.Token);
            return response.IsSuccessStatusCode;
        }
        catch (Exception ex)
        {
            Debug.WriteLine($"Connection check failed: {ex.Message}");
            return false;
        }
    }

    public async Task<LoginResponse?> LoginAsync(string email, string password)
    {
        try
        {
            await EnsureConnectivityAsync();

            var request = new LoginRequest { Email = email, Password = password };
            var json = JsonSerializer.Serialize(request, AppJsonContext.Default.LoginRequest);
            var content = new StringContent(json, Encoding.UTF8, "application/json");

            Debug.WriteLine($">>> ATTEMPTING LOGIN for: {email} at {(_httpClient.BaseAddress?.ToString() ?? "null")}");
            var response = await _httpClient.PostAsync("auth/login", content);
            Debug.WriteLine($">>> LOGIN RESPONSE: {response.StatusCode}");

            var jsonContent = await response.Content.ReadAsStringAsync();

            if (response.IsSuccessStatusCode)
            {
                Debug.WriteLine($">>> LOGIN SUCCESS. Raw JSON: {jsonContent}");
                
                var apiResult = JsonSerializer.Deserialize(jsonContent, AppJsonContext.Default.LoginApiResponse);
                if (apiResult != null && !string.IsNullOrEmpty(apiResult.Token))
                {
                    var result = new LoginResponse
                    {
                        AccessToken = apiResult.Token,
                        RefreshToken = apiResult.RefreshToken,
                        User = new UserDto
                        {
                            Id = apiResult.UserId,
                            Name = apiResult.Name
                        }
                    };
                    Debug.WriteLine($">>> LOGIN DECODED: User={result.User.Name}, TokenPrefix={result.AccessToken.Substring(0, Math.Min(10, result.AccessToken.Length))}...");
                    return result;
                }
                
                Debug.WriteLine(">>> LOGIN FAILED TO DECODE. Unexpected JSON structure.");
                return null;
            }
            
            if (response.StatusCode == HttpStatusCode.Unauthorized)
            {
                Debug.WriteLine($">>> LOGIN UNAUTHORIZED. Body: {jsonContent}");
                throw new UnauthorizedException("Invalid email or password.");
            }
            
            Debug.WriteLine($">>> LOGIN FAILED. Status: {response.StatusCode}. Body: {jsonContent}");
            throw new ApiException("Login failed.", (int)response.StatusCode, jsonContent);
        }
        catch (Exception ex) when (ex is not NoInternetException && ex is not ServerUnreachableException && ex is not UnauthorizedException && ex is not ApiException)
        {
            Debug.WriteLine($">>> LOGIN UNEXPECTED ERROR: {ex.GetType().Name} - {ex.Message}");
            throw new ApiException($"Can't connect to server: {ex.Message}", 0, ex.Message);
        }
    }

    public async Task<(bool Success, string Message)> RegisterAsync(string email, string password, string name)
    {
        try
        {
            await EnsureConnectivityAsync();

            var request = new RegisterRequest { Email = email, Password = password, Name = name };
            var content = new StringContent(
                JsonSerializer.Serialize(request, AppJsonContext.Default.RegisterRequest),
                Encoding.UTF8,
                "application/json");

            var response = await _httpClient.PostAsync("auth/register", content);
            var jsonContent = await response.Content.ReadAsStringAsync();

            if (response.IsSuccessStatusCode)
            {
                return (true, "Registration successful!");
            }

            // AOT safe deserialize for generic object-based error responses
            // Since we only need the message, we can use a string-string dictionary or a dedicated error type
            var errorInfo = JsonSerializer.Deserialize(jsonContent, AppJsonContext.Default.DictionaryStringString);
            string? message = null;
            if (errorInfo != null && errorInfo.TryGetValue("message", out var msg)) message = msg;
            
            return (false, message ?? "Registration failed.");
        }
        catch (NoInternetException) { return (false, "Can't connect to network."); }
        catch (ServerUnreachableException) { return (false, "Can't connect to server."); }
        catch (Exception ex)
        {
            return (false, $"Registration error: {ex.Message}");
        }
    }

    public async Task<ApiResponse<IEnumerable<LessonDto>>?> GetLessonsAsync()
    {
        try
        {
            var learnPayload = await GetLearnDataPayloadAsync();
            if (learnPayload?.Data != null)
            {
                var lessons = learnPayload.Data.Lessons;
                var mappedLessons = lessons.Select(MapLessonDetailToSummary).ToList();
                
                // --- INJECT GAMIFIED WEB LESSONS ---
                mappedLessons.Add(new LessonDto
                {
                    Id = 9001,
                    Title = "Speed Round: Alphabet",
                    Description = "Race against the clock to sign the alphabet. (Web Video)",
                    DurationSeconds = 120,
                    Difficulty = "Intermediate",
                    InstructorName = "Deaf Signer",
                    ThumbnailUrl = "https://images.unsplash.com/photo-1516979187457-637abb4f9353?auto=format&fit=crop&q=80&w=300&h=200",
                    IsDownloaded = false
                });

                mappedLessons.Add(new LessonDto
                {
                    Id = 9002,
                    Title = "Survival Signs Quiz",
                    Description = "Test your knowledge of emergency and survival signs. (Web Video)",
                    DurationSeconds = 180,
                    Difficulty = "Advanced",
                    InstructorName = "Deaf Signer",
                    ThumbnailUrl = "https://images.unsplash.com/photo-1620336655052-a549d414a1a5?auto=format&fit=crop&q=80&w=300&h=200",
                    IsDownloaded = false
                });
                
                var filteredLessons = mappedLessons.Where(l => !string.IsNullOrEmpty(l.InstructorName) && l.InstructorName.Contains("Signer", StringComparison.OrdinalIgnoreCase)).ToList();

                return new ApiResponse<IEnumerable<LessonDto>>
                {
                    Success = true,
                    Data = filteredLessons,
                    Message = learnPayload.Message
                };
            }

            return null;
        }
        catch (Exception ex)
        {
            Debug.WriteLine($"Get lessons error: {ex.Message}");
            return null;
        }
    }

    public async Task<ApiResponse<LessonDetailDto>?> GetLessonAsync(int lessonId)
    {
        if (lessonId == 9001 || lessonId == 9002)
        {
            return new ApiResponse<LessonDetailDto>
            {
                Success = true,
                Message = string.Empty,
                Data = new LessonDetailDto
                {
                    Id = lessonId,
                    Title = lessonId == 9001 ? "Speed Round: Alphabet" : "Survival Signs Quiz",
                    Description = lessonId == 9001 ? "Race against the clock to sign the alphabet. (Web Video)" : "Test your knowledge of emergency and survival signs. (Web Video)",
                    DurationSeconds = lessonId == 9001 ? 120 : 180,
                    InstructorName = "Deaf Signer",
                    ThumbnailUrl = lessonId == 9001 ? "https://images.unsplash.com/photo-1516979187457-637abb4f9353?auto=format&fit=crop&q=80&w=300&h=200" : "https://images.unsplash.com/photo-1620336655052-a549d414a1a5?auto=format&fit=crop&q=80&w=300&h=200",
                    VideoUrl = lessonId == 9001 ? "https://commondatastorage.googleapis.com/gtv-videos-bucket/sample/ForBiggerBlazes.mp4" : "https://commondatastorage.googleapis.com/gtv-videos-bucket/sample/ForBiggerEscapes.mp4",
                    Data = new LessonDetailDataDto
                    {
                        UiLayout = new LessonUiLayoutDto
                        {
                            FileName = "LessonView.xaml",
                            XamlContent = "<ContentView xmlns=\"http://schemas.microsoft.com/dotnet/2021/maui\"><Label Text=\"Gamified lesson placeholder!\" HorizontalOptions=\"Center\" VerticalOptions=\"Center\"/></ContentView>"
                        }
                    }
                }
            };
        }

        try
        {
            await AddAuthHeaderAsync();
            var response = await _httpClient.GetAsync($"learn/lessons/{lessonId}");
            var jsonContent = await response.Content.ReadAsStringAsync();

            if (!response.IsSuccessStatusCode)
            {
                Debug.WriteLine($"Get lesson error: {response.StatusCode} - {jsonContent}");
                return null;
            }

            var wrappedResponse = JsonSerializer.Deserialize(jsonContent, AppJsonContext.Default.ApiResponseLessonDetailDto);
            if (wrappedResponse?.Data != null && IsMeaningfulLesson(wrappedResponse.Data))
            {
                DecodeLessonLayoutPayload(wrappedResponse.Data);

                if (wrappedResponse.Data.Data?.UiLayout == null)
                {
                    var extractedWrappedLesson = TryExtractLessonDetail(jsonContent);
                    if (extractedWrappedLesson != null)
                    {
                        DecodeLessonLayoutPayload(extractedWrappedLesson);
                        wrappedResponse.Data = extractedWrappedLesson;
                    }
                }

                return wrappedResponse;
            }

            var lesson = JsonSerializer.Deserialize(jsonContent, AppJsonContext.Default.LessonDetailDto);
            if (lesson != null && IsMeaningfulLesson(lesson))
            {
                DecodeLessonLayoutPayload(lesson);

                if (lesson.Data?.UiLayout == null)
                {
                    var extractedDirectLesson = TryExtractLessonDetail(jsonContent);
                    if (extractedDirectLesson != null)
                    {
                        DecodeLessonLayoutPayload(extractedDirectLesson);
                        lesson = extractedDirectLesson;
                    }
                }

                return new ApiResponse<LessonDetailDto>
                {
                    Success = true,
                    Data = lesson,
                    Message = string.Empty
                };
            }

            var extractedLesson = TryExtractLessonDetail(jsonContent);
            if (extractedLesson != null)
            {
                DecodeLessonLayoutPayload(extractedLesson);
                
                // Fix URLs
                extractedLesson.ThumbnailUrl = EnsureAbsoluteUrl(extractedLesson.ThumbnailUrl);
                extractedLesson.VideoUrl = EnsureAbsoluteUrl(extractedLesson.VideoUrl);

                return new ApiResponse<LessonDetailDto>
                {
                    Success = true,
                    Data = extractedLesson,
                    Message = string.Empty
                };
            }

            return null;
        }
        catch (Exception ex)
        {
            Debug.WriteLine($"Get lesson error: {ex.Message}");
            return null;
        }
    }

    public async Task<bool> LogoutAsync() 
    {
        try
        {
            await AddAuthHeaderAsync();
            var response = await _httpClient.PostAsync("auth/logout", null);

            if (response.IsSuccessStatusCode)
            {
                await DeleteSecureTokenAsync();
                _httpClient.DefaultRequestHeaders.Authorization = null;
                return true;
            }
            return false;
        }
        catch (Exception ex)
        {
            Debug.WriteLine($"Logout error: {ex.Message}");
            return false;
        }
    }

    public async Task<bool> RefreshTokenAsync(string refreshToken)
    {
        try
        {
            var request = new RefreshTokenRequest { RefreshToken = refreshToken };
            var json = JsonSerializer.Serialize(request, AppJsonContext.Default.RefreshTokenRequest);
            var content = new StringContent(json, Encoding.UTF8, "application/json");

            var response = await _httpClient.PostAsync("auth/refresh", content);
            if (response.IsSuccessStatusCode)
            {
                var jsonResponse = await response.Content.ReadAsStringAsync();
                var apiResponse = JsonSerializer.Deserialize(jsonResponse, AppJsonContext.Default.LoginApiResponse);

                var token = apiResponse?.Token;
                var newRefreshToken = apiResponse?.RefreshToken;

                if (!string.IsNullOrEmpty(token))
                {
                    // CRITICAL FIX: Save the new access token
                    SetAuthToken(token);
                    await SetSecureTokenAsync(token);

                    // CRITICAL FIX: Save the new refresh token if provided by server
                    if (!string.IsNullOrEmpty(newRefreshToken))
                    {
                        Debug.WriteLine(">>> Saving new refresh token received during refresh cycle.");
                        await _databaseService.SaveRefreshTokenAsync(newRefreshToken);
                    }
                    
                    return true;
                }
            }
            return false;
        }
        catch (Exception ex)
        {
            Debug.WriteLine($"Refresh token error: {ex.Message}");
            return false;
        }
    }


    public async Task<string?> GetAuthTokenAsync() 
    {
        return await _databaseService.GetAccessTokenAsync();
    }

    private async Task<ApiResponse<T>?> HandleResponse<T>(HttpResponseMessage response, JsonTypeInfo<ApiResponse<T>> wrappedTypeInfo, JsonTypeInfo<T> directTypeInfo)
    {
        var jsonContent = await response.Content.ReadAsStringAsync();

        if (response.IsSuccessStatusCode)
        {
            ApiResponse<T>? wrapped = null;
            try
            {
                wrapped = JsonSerializer.Deserialize(jsonContent, wrappedTypeInfo);
            }
            catch
            {
            }

            if (wrapped != null && HasDataProperty(jsonContent))
            {
                return wrapped;
            }

            T? direct;
            try
            {
                direct = JsonSerializer.Deserialize(jsonContent, directTypeInfo);
            }
            catch (Exception ex)
            {
                Debug.WriteLine($"HandleResponse parse error: {ex.Message}");
                return null;
            }

            if (direct != null)
            {
                return new ApiResponse<T>
                {
                    Success = true,
                    Message = string.Empty,
                    Data = direct
                };
            }
        }

        Debug.WriteLine($"API Error: {response.StatusCode} - {jsonContent}");
        return null;
    }

    private Task AddAuthHeaderAsync()
    {
        // NO-OP: We now rely on SetAuthToken being called during initialization/login
        // and DefaultRequestHeaders being persisted on the HttpClient.
        return Task.CompletedTask;
    }

    public void SetAuthToken(string token)
    {
        _authToken = token;
        
        if (!string.IsNullOrEmpty(token))
        {
            _httpClient.DefaultRequestHeaders.Authorization =
                new AuthenticationHeaderValue("Bearer", token);
        }
        else
        {
            _httpClient.DefaultRequestHeaders.Authorization = null;
        }
    }

    private async Task SetSecureTokenAsync(string token)
    {
        SetAuthToken(token); // Update cache and header immediately
        await _databaseService.SaveAccessTokenAsync(token);
    }

    private async Task DeleteSecureTokenAsync()
    {
        SetAuthToken(string.Empty);
        await _databaseService.SaveAccessTokenAsync(string.Empty);
        await _databaseService.SaveRefreshTokenAsync(string.Empty);
    }

    private async Task<ApiResponse<LearnDataDto>?> GetLearnDataPayloadAsync()
    {
        await AddAuthHeaderAsync();
        var response = await _httpClient.GetAsync("learn/data");
        var jsonContent = await response.Content.ReadAsStringAsync();

        if (!response.IsSuccessStatusCode)
        {
            Debug.WriteLine($"Get learn data error: {response.StatusCode} - {jsonContent}");
            return null;
        }

        var wrapped = JsonSerializer.Deserialize(jsonContent, AppJsonContext.Default.ApiResponseLearnDataDto);
        if (wrapped?.Data != null)
        {
            foreach (var lesson in wrapped.Data.Lessons)
            {
                DecodeLessonLayoutPayload(lesson);
                lesson.ThumbnailUrl = EnsureAbsoluteUrl(lesson.ThumbnailUrl);
                lesson.VideoUrl = EnsureAbsoluteUrl(lesson.VideoUrl);
            }

            foreach (var category in wrapped.Data.Categories)
            {
                category.IconUrl = EnsureAbsoluteUrl(category.IconUrl);
            }

            return wrapped;
        }

        var direct = JsonSerializer.Deserialize(jsonContent, AppJsonContext.Default.LearnDataDto);
        if (direct != null)
        {
            foreach (var lesson in direct.Lessons)
            {
                DecodeLessonLayoutPayload(lesson);
                lesson.ThumbnailUrl = EnsureAbsoluteUrl(lesson.ThumbnailUrl);
                lesson.VideoUrl = EnsureAbsoluteUrl(lesson.VideoUrl);
            }

            foreach (var category in direct.Categories)
            {
                category.IconUrl = EnsureAbsoluteUrl(category.IconUrl);
            }

            return new ApiResponse<LearnDataDto>
            {
                Success = true,
                Data = direct,
                Message = string.Empty
            };
        }

        return null;
    }

    private static LessonCategoryDto NormalizeCategory(LessonCategoryDto category)
    {
        if (string.IsNullOrWhiteSpace(category.Icon) && !string.IsNullOrWhiteSpace(category.IconUrl))
        {
            category.Icon = category.IconUrl;
        }

        return category;
    }

    private static LessonDto MapLessonDetailToSummary(LessonDetailDto lesson)
    {
        return new LessonDto
        {
            Id = lesson.Id,
            Title = lesson.Title,
            Description = lesson.Description,
            ThumbnailUrl = lesson.ThumbnailUrl,
            DurationSeconds = lesson.DurationSeconds,
            Difficulty = lesson.Difficulty,
            InstructorName = lesson.InstructorName
        };
    }

    private static bool HasDataProperty(string jsonContent)
    {
        try
        {
            using var document = JsonDocument.Parse(jsonContent);
            if (document.RootElement.ValueKind != JsonValueKind.Object)
            {
                return false;
            }

            foreach (var property in document.RootElement.EnumerateObject())
            {
                if (string.Equals(property.Name, "data", StringComparison.OrdinalIgnoreCase))
                {
                    return true;
                }
            }
        }
        catch
        {
            return false;
        }

        return false;
    }

    // ============ Learning APIs ============


    public async Task<ApiResponse<List<MediaDto>>?> GetMediaAsync()
    {
        try
        {
            await AddAuthHeaderAsync();
            var response = await _httpClient.GetAsync("media");
            if (response.IsSuccessStatusCode)
            {
                var content = await response.Content.ReadAsStringAsync();
                return JsonSerializer.Deserialize(content, AppJsonContext.Default.ApiResponseListMediaDto);
            }
            return null;
        }
        catch { return null; }
    }

    public async Task<ApiResponse<List<SignerCreditDto>>?> GetSignerCreditsAsync()
    {
        try
        {
            await AddAuthHeaderAsync();
            var response = await _httpClient.GetAsync("signer-credits");
            if (response.IsSuccessStatusCode)
            {
                var content = await response.Content.ReadAsStringAsync();
                return JsonSerializer.Deserialize(content, AppJsonContext.Default.ApiResponseListSignerCreditDto);
            }
            return null;
        }
        catch { return null; }
    }

    public async Task<UserStatsDto?> GetUserStatsAsync()
    {
        try
        {
            await AddAuthHeaderAsync();
            var response = await _httpClient.GetAsync("statistics");
            var jsonContent = await response.Content.ReadAsStringAsync();

            if (response.IsSuccessStatusCode)
            {
                return JsonSerializer.Deserialize(jsonContent, AppJsonContext.Default.UserStatsDto);
            }
            
            // Fallback to legacy if possible
            var learnDataResponse = await GetLearnDataPayloadAsync();
            var learnData = learnDataResponse?.Data;
            if (learnData != null)
            {
                return new UserStatsDto
                {
                    TotalXP = learnData.TotalXp,
                    CurrentStreak = learnData.CurrentStreak,
                    TotalProgress = learnData.Lessons.Count(l => l.CompletionPercentage >= 1)
                };
            }

            return null;
        }
        catch (Exception ex)
        {
            Debug.WriteLine($"GetUserStats error: {ex.Message}");
            return null;
        }
    }

    public async Task<bool> SubmitFeedbackAsync(FeedbackRequest feedback)
    {
        try
        {
            await AddAuthHeaderAsync();
            var json = JsonSerializer.Serialize(feedback, AppJsonContext.Default.FeedbackRequest);
            var content = new StringContent(json, Encoding.UTF8, "application/json");
            var response = await _httpClient.PostAsync("feedback", content);
            return response.IsSuccessStatusCode;
        }
        catch (Exception ex)
        {
            Debug.WriteLine($"SubmitFeedback error: {ex.Message}");
            return false;
        }
    }

    public async Task<ApiResponse<IEnumerable<LessonCategoryDto>>?> GetCategoriesAsync()
    {
        try
        {
            await AddAuthHeaderAsync();
            var response = await _httpClient.GetAsync("learn/categories");
            var categoriesResponse = await HandleResponse(response, AppJsonContext.Default.ApiResponseIEnumerableLessonCategoryDto, AppJsonContext.Default.IEnumerableLessonCategoryDto);
            if (categoriesResponse?.Data?.Any() == true)
            {
                return new ApiResponse<IEnumerable<LessonCategoryDto>>
                {
                    Success = categoriesResponse.Success,
                    Message = categoriesResponse.Message,
                    Data = categoriesResponse.Data.Select(NormalizeCategory)
                };
            }

            var learnPayload = await GetLearnDataPayloadAsync();
            if (learnPayload?.Data?.Categories is { Count: > 0 } categoriesFromLearnData)
            {
                return new ApiResponse<IEnumerable<LessonCategoryDto>>
                {
                    Success = true,
                    Message = learnPayload.Message,
                    Data = categoriesFromLearnData.Select(NormalizeCategory)
                };
            }

            return categoriesResponse;
        }
        catch (Exception ex)
        {
            Debug.WriteLine($"GetCategories error: {ex.Message}");
            return null;
        }
    }

    public async Task<ApiResponse<LessonCategoryDto>?> GetCategoryAsync(int categoryId)
    {
        var data = await ExecuteSafeWithRetryAsync(async () => await _httpClient.GetAsync($"learn/categories/{categoryId}"), AppJsonContext.Default.ApiResponseLessonCategoryDto);
        return data != null ? new ApiResponse<LessonCategoryDto> { Success = true, Data = data } : null;
    }

    public async Task<ApiResponse<IEnumerable<LessonDetailDto>>?> GetLessonsByCategoryAsync(int categoryId)
    {
        try
        {
            await AddAuthHeaderAsync();
            var response = await _httpClient.GetAsync($"learn/categories/{categoryId}/lessons");
            var lessonsResponse = await HandleResponse(response, AppJsonContext.Default.ApiResponseIEnumerableLessonDetailDto, AppJsonContext.Default.IEnumerableLessonDetailDto);
            if (lessonsResponse?.Data?.Any() == true)
            {
                foreach (var lesson in lessonsResponse.Data)
                {
                    DecodeLessonLayoutPayload(lesson);
                }
                return lessonsResponse;
            }

            var learnPayload = await GetLearnDataPayloadAsync();
            if (learnPayload?.Data?.Lessons is { Count: > 0 } lessonsFromLearnData)
            {
                var filtered = lessonsFromLearnData
                    .Where(lesson => lesson.CategoryId == categoryId || lesson.Data?.CategoryId == categoryId)
                    .ToList();
                foreach (var lesson in filtered)
                {
                    DecodeLessonLayoutPayload(lesson);
                }

                return new ApiResponse<IEnumerable<LessonDetailDto>>
                {
                    Success = true,
                    Data = filtered,
                    Message = learnPayload.Message
                };
            }

            return lessonsResponse;
        }
        catch (Exception ex)
        {
            Debug.WriteLine($"GetLessonsByCategory error: {ex.Message}");
            return null;
        }
    }

    public async Task<ApiResponse<DailyGoalDto>?> GetDailyGoalAsync()
    {
        try
        {
            await AddAuthHeaderAsync();
            var response = await _httpClient.GetAsync("learn/daily-goal");
            var jsonContent = await response.Content.ReadAsStringAsync();

            if (!response.IsSuccessStatusCode)
            {
                Debug.WriteLine($"GetDailyGoal error: {response.StatusCode} - {jsonContent}");
                return null;
            }

            var wrappedGoal = JsonSerializer.Deserialize(jsonContent, AppJsonContext.Default.ApiResponseDailyGoalDto);
            if (wrappedGoal?.Data != null)
            {
                return wrappedGoal;
            }

            var directGoal = JsonSerializer.Deserialize(jsonContent, AppJsonContext.Default.DailyGoalDto);
            if (directGoal != null && (directGoal.TotalRequired != 0 || directGoal.CompletedToday != 0))
            {
                return new ApiResponse<DailyGoalDto> { Success = true, Data = directGoal };
            }

            var apiGoal = JsonSerializer.Deserialize(jsonContent, AppJsonContext.Default.DailyGoalApiDto);
            if (apiGoal != null)
            {
                return new ApiResponse<DailyGoalDto>
                {
                    Success = true,
                    Data = new DailyGoalDto
                    {
                        CompletedToday = apiGoal.CompletedToday,
                        TotalRequired = apiGoal.DailyGoal > 0 ? apiGoal.DailyGoal : apiGoal.TotalReviewsDue
                    }
                };
            }

            return null;
        }
        catch (Exception ex)
        {
            Debug.WriteLine($"GetDailyGoal error: {ex.Message}");
            return null;
        }
    }

    public async Task<ApiResponse<IEnumerable<SpacedRepetitionLessonDto>>?> GetDailyReviewLessonsAsync()
    {
        try
        {
            await AddAuthHeaderAsync();
            var response = await _httpClient.GetAsync("learn/daily-reviews");
            return await HandleResponse(response, AppJsonContext.Default.ApiResponseIEnumerableSpacedRepetitionLessonDto, AppJsonContext.Default.IEnumerableSpacedRepetitionLessonDto);
        }
        catch (Exception ex)
        {
            Debug.WriteLine($"GetDailyReviewLessons error: {ex.Message}");
            return null;
        }
    }

    public async Task<ApiResponse<UpcomingReviewsDto>?> GetUpcomingReviewsAsync()
    {
        try
        {
            await AddAuthHeaderAsync();
            // CRITICAL FIX: Correct endpoint is "learn/upcoming-reviews" not "learn/daily-reviews"
            var response = await _httpClient.GetAsync("learn/upcoming-reviews");
            var jsonContent = await response.Content.ReadAsStringAsync();

            if (!response.IsSuccessStatusCode)
            {
                Debug.WriteLine($"GetUpcomingReviews error: {response.StatusCode} - {jsonContent}");
                return null;
            }

            var wrappedUpcoming = JsonSerializer.Deserialize(jsonContent, AppJsonContext.Default.ApiResponseUpcomingReviewsDto);
            if (wrappedUpcoming?.Data != null)
            {
                return wrappedUpcoming;
            }

            var directUpcoming = JsonSerializer.Deserialize(jsonContent, AppJsonContext.Default.UpcomingReviewsDto);
            if (directUpcoming != null && (directUpcoming.TomorrowCount != 0 || directUpcoming.ThisWeekCount != 0 || directUpcoming.NextWeekCount != 0))
            {
                return new ApiResponse<UpcomingReviewsDto> { Success = true, Data = directUpcoming };
            }

            var apiUpcoming = JsonSerializer.Deserialize(jsonContent, AppJsonContext.Default.UpcomingReviewsApiDto);
            if (apiUpcoming != null)
            {
                return new ApiResponse<UpcomingReviewsDto>
                {
                    Success = true,
                    Data = new UpcomingReviewsDto
                    {
                        TomorrowCount = apiUpcoming.DueTomorrow,
                        ThisWeekCount = apiUpcoming.DueThisWeek,
                        NextWeekCount = apiUpcoming.DueToday + apiUpcoming.Overdue
                    }
                };
            }

            return null;
        }
        catch (Exception ex)
        {
            Debug.WriteLine($"GetUpcomingReviews error: {ex.Message}");
            return null;
        }
    }

    public async Task<ApiResponse<PersonalizedRecommendationDto>?> GetPersonalizedRecommendationAsync()
    {
        try
        {
            await AddAuthHeaderAsync();
            var response = await _httpClient.GetAsync("learn/recommendations");
            return await HandleResponse(response, AppJsonContext.Default.ApiResponsePersonalizedRecommendationDto, AppJsonContext.Default.PersonalizedRecommendationDto);
        }
        catch (Exception ex)
        {
            Debug.WriteLine($"GetPersonalizedRecommendation error: {ex.Message}");
            return null;
        }
    }

    public async Task<ApiResponse<InteractiveLessonDto>?> GetInteractiveLessonAsync(int lessonId)
    {
        try
        {
            await AddAuthHeaderAsync();
            var response = await _httpClient.GetAsync($"learn/lessons/{lessonId}/interactive");
            return await HandleResponse(response, AppJsonContext.Default.ApiResponseInteractiveLessonDto, AppJsonContext.Default.InteractiveLessonDto);
        }
        catch (Exception ex)
        {
            Debug.WriteLine($"GetInteractiveLesson error: {ex.Message}");
            return null;
        }
    }

    public async Task<ApiResponse<bool>?> MarkLessonCompleteAsync(int lessonId)
    {
        try
        {
            await AddAuthHeaderAsync();
            var content = new StringContent(
                "{}",
                System.Text.Encoding.UTF8,
                new MediaTypeHeaderValue("application/json"));

            var response = await _httpClient.PostAsync($"learn/lessons/{lessonId}/complete", content);
            var jsonContent = await response.Content.ReadAsStringAsync();
            if (!response.IsSuccessStatusCode)
            {
                Debug.WriteLine($"MarkLessonComplete error: {response.StatusCode} - {jsonContent}");
                return null;
            }

            var wrapped = JsonSerializer.Deserialize(jsonContent, AppJsonContext.Default.ApiResponseBoolean);
            if (wrapped != null)
            {
                wrapped.Data = wrapped.Data || response.IsSuccessStatusCode;
                wrapped.Success = true;
                return wrapped;
            }

            if (bool.TryParse(jsonContent, out var directBool))
            {
                return new ApiResponse<bool>
                {
                    Success = true,
                    Data = directBool
                };
            }

            return new ApiResponse<bool>
            {
                Success = true,
                Data = true,
                Message = ExtractErrorMessage(jsonContent, "Lesson marked complete.")
            };
        }
        catch (Exception ex)
        {
            Debug.WriteLine($"MarkLessonComplete error: {ex.Message}");
            return null;
        }
    }

    public async Task<ApiResponse<bool>?> MarkReviewCompleteAsync(int reviewLessonId, int qualityRating = 4)
    {
        try
        {
            // Validate quality rating (0-5 scale)
            if (qualityRating < 0 || qualityRating > 5)
            {
                Debug.WriteLine($"Invalid quality rating: {qualityRating}. Must be between 0-5.");
                return null;
            }

            await AddAuthHeaderAsync();
            var qualityValue = qualityRating.ToString(CultureInfo.InvariantCulture);
            var response = await _httpClient.PostAsync(
                $"learn/daily-reviews/{reviewLessonId}/review?qualityRating={Uri.EscapeDataString(qualityValue)}",
                content: null);

            var jsonContent = await response.Content.ReadAsStringAsync();
            if (!response.IsSuccessStatusCode)
            {
                Debug.WriteLine($"MarkReviewComplete error: {response.StatusCode} - {jsonContent}");
                return null;
            }

            var wrapped = JsonSerializer.Deserialize(jsonContent, AppJsonContext.Default.ApiResponseBoolean);
            if (wrapped != null)
            {
                wrapped.Data = wrapped.Data || response.IsSuccessStatusCode;
                wrapped.Success = true;
                return wrapped;
            }

            if (bool.TryParse(jsonContent, out var directBool))
            {
                return new ApiResponse<bool>
                {
                    Success = true,
                    Data = directBool
                };
            }

            return new ApiResponse<bool>
            {
                Success = true,
                Data = true,
                Message = ExtractErrorMessage(jsonContent, "Daily review submitted.")
            };
        }
        catch (Exception ex)
        {
            Debug.WriteLine($"MarkReviewComplete error: {ex.Message}");
            return null;
        }
    }

    // ============ Video APIs ============

    public async Task<ApiResponse<IEnumerable<VideoDto>>?> GetVideosAsync()
    {
        try
        {
            await AddAuthHeaderAsync();
            var response = await _httpClient.GetAsync("videos");
            var result = await HandleResponse(response, AppJsonContext.Default.ApiResponseIEnumerableVideoDto, AppJsonContext.Default.IEnumerableVideoDto);
            
            if (result?.Data != null)
            {
                var signersVideos = result.Data.Where(v => !string.IsNullOrEmpty(v.Instructor) && v.Instructor.Contains("Signer", StringComparison.OrdinalIgnoreCase)).ToList();
                foreach (var video in signersVideos)
                {
                    video.VideoUrl = EnsureAbsoluteUrl(video.VideoUrl);
                    video.ThumbnailUrl = EnsureAbsoluteUrl(video.ThumbnailUrl);
                }
                result.Data = signersVideos;
            }
            return result;
        }
        catch (Exception ex)
        {
            Debug.WriteLine($"GetVideos error: {ex.Message}");
            return null;
        }
    }

    public async Task<ApiResponse<VideoDto>?> GetVideoAsync(int videoId)
    {
        try
        {
            await AddAuthHeaderAsync();
            var response = await _httpClient.GetAsync($"videos/{videoId}");
            return await HandleResponse(response, AppJsonContext.Default.ApiResponseVideoDto, AppJsonContext.Default.VideoDto);
        }
        catch (Exception ex)
        {
            Debug.WriteLine($"GetVideo error: {ex.Message}");
            return null;
        }
    }

    public async Task<ApiResponse<IEnumerable<VideoDto>>?> GetVideosByCategory(string category)
    {
        try
        {
            await AddAuthHeaderAsync();
            var response = await _httpClient.GetAsync($"videos/category/{Uri.EscapeDataString(category)}");
            return await HandleResponse(response, AppJsonContext.Default.ApiResponseIEnumerableVideoDto, AppJsonContext.Default.IEnumerableVideoDto);
        }
        catch (Exception ex)
        {
            Debug.WriteLine($"GetVideosByCategory error: {ex.Message}");
            return null;
        }
    }

    public async Task<bool> LikeVideoAsync(int videoId, bool like)
    {
        try
        {
            await AddAuthHeaderAsync();
            var endpoint = like ? $"videos/{videoId}/like" : $"videos/{videoId}/unlike";
            var content = new StringContent(
                JsonSerializer.Serialize(new Dictionary<string, string>(), AppJsonContext.Default.DictionaryStringString),
                Encoding.UTF8,
                "application/json");

            var response = await _httpClient.PostAsync(endpoint, content);
            return response.IsSuccessStatusCode;
        }
        catch (Exception ex)
        {
            Debug.WriteLine($"LikeVideo error: {ex.Message}");
            return false;
        }
    }

    private static string ExtractErrorMessage(string jsonContent, string? fallback)
    {
        try
        {
            using var document = JsonDocument.Parse(jsonContent);
            if (document.RootElement.ValueKind == JsonValueKind.Object)
            {
                if (document.RootElement.TryGetProperty("message", out var messageProp))
                {
                    return messageProp.GetString() ?? fallback ?? "Request failed";
                }

                if (document.RootElement.TryGetProperty("Message", out var upperMessageProp))
                {
                    return upperMessageProp.GetString() ?? fallback ?? "Request failed";
                }
            }
        }
        catch
        {
        }

        return fallback ?? "Request failed";
    }

    private static bool IsMeaningfulLesson(LessonDetailDto lesson)
    {
        return lesson.Id > 0
               || !string.IsNullOrWhiteSpace(lesson.Title)
               || lesson.Data?.UiLayout != null;
    }

    private static LessonDetailDto? TryExtractLessonDetail(string jsonContent)
    {
        try
        {
            using var doc = JsonDocument.Parse(jsonContent);
            var root = doc.RootElement;

            var lessonNode = root;
            if (TryGetJsonProperty(root, out var dataNode, "data"))
            {
                lessonNode = dataNode;
            }

            if (TryGetJsonProperty(lessonNode, out var nestedLessonNode, "lesson"))
            {
                lessonNode = nestedLessonNode;
            }

            if (lessonNode.ValueKind != JsonValueKind.Object)
            {
                return null;
            }

            var lesson = new LessonDetailDto
            {
                Id = GetIntProperty(lessonNode, "id"),
                Title = GetStringProperty(lessonNode, "title"),
                Description = GetStringProperty(lessonNode, "description"),
                ThumbnailUrl = GetStringProperty(lessonNode, "thumbnail", "thumbnailUrl", "thumbnail_url"),
                DurationSeconds = GetIntProperty(lessonNode, "durationSeconds", "duration_seconds"),
                CompletionPercentage = GetDoubleProperty(lessonNode, "completionPercentage", "completion_percentage"),
                Difficulty = GetStringProperty(lessonNode, "difficulty"),
                InstructorName = GetStringProperty(lessonNode, "instructorName", "instructor_name"),
                CategoryId = GetIntProperty(lessonNode, "categoryId", "category_id")
            };

            JsonElement dataPayload;
            if (TryGetJsonProperty(lessonNode, out dataPayload, "data") && dataPayload.ValueKind == JsonValueKind.Object)
            {
                lesson.Data = new LessonDetailDataDto
                {
                    DurationSeconds = GetIntProperty(dataPayload, "durationSeconds", "duration_seconds"),
                    Difficulty = GetStringProperty(dataPayload, "difficulty"),
                    CompletionPercentage = GetDoubleProperty(dataPayload, "completionPercentage", "completion_percentage"),
                    InstructorName = GetStringProperty(dataPayload, "instructorName", "instructor_name"),
                    CategoryId = GetIntProperty(dataPayload, "categoryId", "category_id")
                };

                JsonElement uiLayoutNode;
                if (TryGetJsonProperty(dataPayload, out uiLayoutNode, "uiLayout", "ui_layout") && uiLayoutNode.ValueKind == JsonValueKind.Object)
                {
                    lesson.Data.UiLayout = new LessonUiLayoutDto
                    {
                        FileName = GetStringProperty(uiLayoutNode, "fileName", "file_name"),
                        XamlContent = GetStringProperty(uiLayoutNode, "xamlContent", "xaml_content"),
                        CodeBehindContent = GetStringProperty(uiLayoutNode, "codeBehindContent", "code_behind_content")
                    };
                }
            }

            if (lesson.Id <= 0 && string.IsNullOrWhiteSpace(lesson.Title) && lesson.Data?.UiLayout == null)
            {
                return null;
            }

            return lesson;
        }
        catch
        {
            return null;
        }
    }

    private static void DecodeLessonLayoutPayload(LessonDetailDto lesson)
    {
        var layout = lesson.Data?.UiLayout;
        if (layout == null)
        {
            return;
        }

        layout.XamlContent = DecodePayloadText(layout.XamlContent);
        layout.CodeBehindContent = DecodePayloadText(layout.CodeBehindContent);
    }

    private static string DecodePayloadText(string? value)
    {
        if (string.IsNullOrWhiteSpace(value))
        {
            return string.Empty;
        }

        var text = value.Trim();

        // URL encoded payload
        if (text.Contains('%'))
        {
            try
            {
                text = Uri.UnescapeDataString(text);
            }
            catch
            {
            }
        }

        // HTML encoded payload
        if (text.Contains("&lt;", StringComparison.OrdinalIgnoreCase) ||
            text.Contains("&gt;", StringComparison.OrdinalIgnoreCase))
        {
            text = WebUtility.HtmlDecode(text);
        }

        // Base64 encoded payload
        if (!text.Contains('<') && LooksLikeBase64(text))
        {
            try
            {
                var base64 = text.Replace('-', '+').Replace('_', '/');
                var padLength = 4 - (base64.Length % 4);
                if (padLength < 4)
                {
                    base64 = base64.PadRight(base64.Length + padLength, '=');
                }
                
                var bytes = Convert.FromBase64String(base64);
                var decoded = Encoding.UTF8.GetString(bytes);
                if (!string.IsNullOrWhiteSpace(decoded))
                {
                    text = decoded;
                }
            }
            catch
            {
            }
        }

        return text;
    }

    private static bool LooksLikeBase64(string input)
    {
        if (string.IsNullOrWhiteSpace(input) || input.Length < 16)
        {
            return false;
        }

        foreach (var c in input)
        {
            if (!(char.IsLetterOrDigit(c) || c == '+' || c == '/' || c == '=' || c == '-' || c == '_'))
            {
                return false;
            }
        }

        return true;
    }

    private static bool TryGetJsonProperty(JsonElement element, out JsonElement value, params string[] names)
    {
        if (element.ValueKind == JsonValueKind.Object)
        {
            foreach (var name in names)
            {
                if (element.TryGetProperty(name, out value))
                {
                    return true;
                }

                var fallback = ToCamelCase(name);
                if (!string.Equals(name, fallback, StringComparison.Ordinal) && element.TryGetProperty(fallback, out value))
                {
                    return true;
                }
            }

            foreach (var property in element.EnumerateObject())
            {
                foreach (var name in names)
                {
                    if (string.Equals(property.Name, name, StringComparison.OrdinalIgnoreCase) ||
                        string.Equals(property.Name, ToCamelCase(name), StringComparison.OrdinalIgnoreCase))
                    {
                        value = property.Value;
                        return true;
                    }
                }
            }
        }

        value = default;
        return false;
    }

    private static int GetIntProperty(JsonElement element, params string[] names)
    {
        if (TryGetJsonProperty(element, out var value, names))
        {
            if (value.ValueKind == JsonValueKind.Number && value.TryGetInt32(out var number))
            {
                return number;
            }

            if (value.ValueKind == JsonValueKind.String && int.TryParse(value.GetString(), out number))
            {
                return number;
            }
        }

        return 0;
    }

    private static double GetDoubleProperty(JsonElement element, params string[] names)
    {
        if (TryGetJsonProperty(element, out var value, names))
        {
            if (value.ValueKind == JsonValueKind.Number && value.TryGetDouble(out var number))
            {
                return number;
            }

            if (value.ValueKind == JsonValueKind.String && double.TryParse(value.GetString(), out number))
            {
                return number;
            }
        }

        return 0d;
    }

    private static string GetStringProperty(JsonElement element, params string[] names)
    {
        if (TryGetJsonProperty(element, out var value, names))
        {
            if (value.ValueKind == JsonValueKind.String)
            {
                return value.GetString() ?? string.Empty;
            }

            if (value.ValueKind != JsonValueKind.Null && value.ValueKind != JsonValueKind.Undefined)
            {
                return value.ToString();
            }
        }

        return string.Empty;
    }

    private static string ToCamelCase(string value)
    {
        if (string.IsNullOrWhiteSpace(value) || value.Length == 1)
        {
            return value;
        }

        return char.ToLowerInvariant(value[0]) + value[1..];
    }

    public string EnsureAbsoluteUrl(string? url)
    {
        if (string.IsNullOrWhiteSpace(url)) return string.Empty;
        if (url.StartsWith("http", StringComparison.OrdinalIgnoreCase)) return url;

        var baseUrl = _apiConfig.BaseUrl.TrimEnd('/');
        return $"{baseUrl}/{url.TrimStart('/')}";
    }

    public async Task<bool> WatchVideoAsync(int videoId)
    {
        try
        {
            await AddAuthHeaderAsync();
            var content = new StringContent(
                "{}",
                System.Text.Encoding.UTF8,
                new MediaTypeHeaderValue("application/json"));

            var response = await _httpClient.PostAsync($"videos/{videoId}/watch", content);
            return response.IsSuccessStatusCode;
        }
        catch (Exception ex)
        {
            Debug.WriteLine($"WatchVideo error: {ex.Message}");
            return false;
        }
    }

    // ============ Camera - Real-Time Gesture Prediction ============

    public async Task<GesturePredictionResponseDto?> PredictGestureFromImageAsync(byte[] imageData, string? targetSign = null, CancellationToken cancellationToken = default)
    {
        int retryCount = 0;
        const int maxRetries = 1;

        while (retryCount <= maxRetries)
        {
            try
            {
                var url = "gesture/predict";
                if (!string.IsNullOrEmpty(targetSign))
                {
                    url += $"?targetSign={Uri.EscapeDataString(targetSign)}";
                }

                // Ensure we have a valid absolute URI for the routing handler to process
                var baseAddr = _httpClient.BaseAddress?.ToString() ?? "http://api.internal/";
                var finalUrl = new Uri(new Uri(baseAddr), url).ToString();

                using var request = new HttpRequestMessage(HttpMethod.Post, finalUrl);
                
                // Ensure we have the latest token from storage if memory cache is empty
                if (string.IsNullOrEmpty(_authToken))
                {
                    _authToken = await _databaseService.GetAccessTokenAsync();
                }

                if (!string.IsNullOrEmpty(_authToken))
                {
                    request.Headers.Authorization = new AuthenticationHeaderValue("Bearer", _authToken);
                }
                
                using var content = new MultipartFormDataContent();
                var imageContent = new ByteArrayContent(imageData);
                imageContent.Headers.ContentType = new MediaTypeHeaderValue("image/jpeg");
                content.Add(imageContent, "image", "frame.jpg");
                request.Content = content;

                Debug.WriteLine($">>> GESTURE PREDICT: Sending request to {(_httpClient.BaseAddress?.ToString() ?? "null")}{url} (Size: {imageData.Length})");
                var response = await _httpClient.SendAsync(request, cancellationToken);
                Debug.WriteLine($">>> GESTURE PREDICT RESPONSE: {response.StatusCode} ({(int)response.StatusCode})");
                
                if (response.StatusCode == HttpStatusCode.Unauthorized && retryCount < maxRetries)
                {
                    Debug.WriteLine(">>> Gesture prediction 401 Unauthorized. Attempting token refresh...");
                    var refreshToken = await _databaseService.GetRefreshTokenAsync();
                    if (!string.IsNullOrEmpty(refreshToken))
                    {
                        var refreshed = await RefreshTokenAsync(refreshToken);
                        if (refreshed)
                        {
                            retryCount++;
                            continue; 
                        }
                    }
                }

                if (!response.IsSuccessStatusCode)
                {
                    var error = await response.Content.ReadAsStringAsync(cancellationToken);
                    Debug.WriteLine($">>> Gesture prediction FAILED: {response.StatusCode}. Body: {error}");
                    
                    if (retryCount == 0)
                    {
                        retryCount++;
                        continue;
                    }
                    return null;
                }

                var json = await response.Content.ReadAsStringAsync(cancellationToken);
                return JsonSerializer.Deserialize(json, AppJsonContext.Default.GesturePredictionResponseDto);
            }
            catch (OperationCanceledException)
            {
                Debug.WriteLine(">>> Gesture prediction CANCELLED (timeout or user action).");
                return null;
            }
            catch (Exception ex)
            {
                Debug.WriteLine($">>> Gesture prediction EXCEPTION: {ex.GetType().Name} - {ex.Message}");
                
                if (retryCount == 0)
                {
                    retryCount++;
                    continue;
                }
                
                return null;
            }
        }
        return null;
    }


    // ============ User - Profile Management ============

    public async Task<ApiResponse<UserProfileDto>?> GetUserProfileAsync()
    {
        try
        {
            await AddAuthHeaderAsync();
            var response = await _httpClient.GetAsync("user/profile");
            return await HandleResponse(response, AppJsonContext.Default.ApiResponseUserProfileDto, AppJsonContext.Default.UserProfileDto);
        }
        catch (Exception ex)
        {
            Debug.WriteLine($"GetUserProfile error: {ex.Message}");
            return null;
        }
    }

    public async Task<bool> UpdateNameAsync(string newName)
    {
        try
        {
            await AddAuthHeaderAsync();
            var request = new UpdateNameRequest { NewName = newName };
            var content = new StringContent(
                JsonSerializer.Serialize(request, AppJsonContext.Default.UpdateNameRequest),
                Encoding.UTF8,
                "application/json");

            var response = await _httpClient.PutAsync("user/name", content);
            return response.IsSuccessStatusCode;
        }
        catch (Exception ex)
        {
            Debug.WriteLine($"UpdateName error: {ex.Message}");
            return false;
        }
    }

    public async Task<bool> UpdatePasswordAsync(string oldPassword, string newPassword)
    {
        try
        {
            await AddAuthHeaderAsync();
            var request = new UpdatePasswordRequest { OldPassword = oldPassword, NewPassword = newPassword };
            var content = new StringContent(
                JsonSerializer.Serialize(request, AppJsonContext.Default.UpdatePasswordRequest),
                Encoding.UTF8,
                "application/json");

            var response = await _httpClient.PutAsync("user/password", content);
            return response.IsSuccessStatusCode;
        }
        catch (Exception ex)
        {
            Debug.WriteLine($"UpdatePassword error: {ex.Message}");
            return false;
        }
    }

    public async Task<bool> UpdateAvatarAsync(string avatarUrl)
    {
        try
        {
            await AddAuthHeaderAsync();
            var request = new UpdateAvatarRequest { AvatarUrl = avatarUrl };
            var content = new StringContent(
                JsonSerializer.Serialize(request, AppJsonContext.Default.UpdateAvatarRequest),
                Encoding.UTF8,
                "application/json");

            var response = await _httpClient.PutAsync("user/avatar", content);
            return response.IsSuccessStatusCode;
        }
        catch (Exception ex)
        {
            Debug.WriteLine($"UpdateAvatar error: {ex.Message}");
            return false;
        }
    }

    public async Task<bool> UpdateDescriptionAsync(string description)
    {
        try
        {
            await AddAuthHeaderAsync();
            var request = new UpdateDescriptionRequest { Description = description };
            var content = new StringContent(
                JsonSerializer.Serialize(request, AppJsonContext.Default.UpdateDescriptionRequest),
                Encoding.UTF8,
                "application/json");

            var response = await _httpClient.PutAsync("user/description", content);
            return response.IsSuccessStatusCode;
        }
        catch (Exception ex)
        {
            Debug.WriteLine($"UpdateDescription error: {ex.Message}");
            return false;
        }
    }

    public async Task<LeaderboardDto?> GetLeaderboardAsync(int count = 10)
    {
        try
        {
            await AddAuthHeaderAsync();
            var response = await _httpClient.GetAsync($"leaderboard?count={count}");
            if (response.IsSuccessStatusCode)
            {
                var json = await response.Content.ReadAsStringAsync();
                return JsonSerializer.Deserialize(json, AppJsonContext.Default.LeaderboardDto);
            }
            return null;
        }
        catch (Exception ex)
        {
            Debug.WriteLine($"GetLeaderboard error: {ex.Message}");
            return null;
        }
    }

    public async Task<IEnumerable<AchievementBadgeDto>?> GetAchievementsAsync()
    {
        try
        {
            await AddAuthHeaderAsync();
            var response = await _httpClient.GetAsync("user/achievements");
            if (response.IsSuccessStatusCode)
            {
                var json = await response.Content.ReadAsStringAsync();
                return JsonSerializer.Deserialize(json, AppJsonContext.Default.IEnumerableAchievementBadgeDto);
            }
            return null;
        }
        catch (Exception ex)
        {
            Debug.WriteLine($"GetAchievements error: {ex.Message}");
            return null;
        }
    }
}
