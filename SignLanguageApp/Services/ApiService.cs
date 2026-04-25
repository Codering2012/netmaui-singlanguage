using System.Diagnostics;
using System.Globalization;
using System.Net;
using System.Net.Http.Headers;
using System.Text;
using System.Text.Json;
using SignLanguageApp.Model;

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

    // Learning - Stats & Progress
    Task<ApiResponse<UserStatsDto>?> GetUserStatsAsync();

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
    Task<GesturePredictionResponseDto?> PredictGestureFromImageAsync(byte[] imageData, CancellationToken cancellationToken = default);
}
public class ApiService : IApiService
{
    private readonly HttpClient _httpClient;
    private const string TokenKey = "access_token";
    private const string LegacyTokenKey = "jwt_token";
    private static readonly JsonSerializerOptions SerializerOptions = new()
    {
        PropertyNameCaseInsensitive = true
    };

    public ApiService(HttpClient httpClient)
    {
        _httpClient = httpClient;
    }

    public async Task<LoginResponse?> LoginAsync(string email, string password)
    {
        try
        {
            var request = new LoginRequest { Email = email, Password = password };
            var content = new StringContent(
                JsonSerializer.Serialize(request),
                System.Text.Encoding.UTF8,
                new MediaTypeHeaderValue("application/json"));

            var response = await _httpClient.PostAsync("auth/login", content);

            if (response.IsSuccessStatusCode)
            {
                var jsonContent = await response.Content.ReadAsStringAsync();
                var wrappedResponse = JsonSerializer.Deserialize<ApiResponse<LoginApiResponse>>(jsonContent, SerializerOptions);
                var apiResponse = wrappedResponse?.Data ?? JsonSerializer.Deserialize<LoginApiResponse>(jsonContent, SerializerOptions);

                var token = apiResponse?.Token;
                if (string.IsNullOrEmpty(token))
                {
                    token = apiResponse?.AccessToken;
                }

                if (!string.IsNullOrEmpty(token))
                {
                    var loginResponse = new LoginResponse
                    {
                        AccessToken = token,
                        RefreshToken = apiResponse?.RefreshToken ?? string.Empty,
                        User = new UserDto
                        {
                            Id = apiResponse?.UserId ?? string.Empty,
                            Name = apiResponse?.Name ?? string.Empty,
                            Email = email
                        }
                    };

                    SetAuthToken(token);
                    await SetSecureTokenAsync(token);

                    return loginResponse;
                }
            }

            Debug.WriteLine($"Login error: {response.StatusCode}");
            return null;
        }
        catch (Exception ex)
        {
            Debug.WriteLine($"Login error: {ex.Message}");
            return null;
        }
    }

    public async Task<(bool Success, string Message)> RegisterAsync(string email, string password, string name)
    {
        try
        {
            var request = new RegisterRequest { Email = email, Password = password, Name = name };
            var content = new StringContent(
                JsonSerializer.Serialize(request),
                System.Text.Encoding.UTF8,
                new MediaTypeHeaderValue("application/json"));

            var response = await _httpClient.PostAsync("auth/register", content);

            if (response.IsSuccessStatusCode)
            {
                return (true, "Registration successful");
            }
            else
            {
                var jsonContent = await response.Content.ReadAsStringAsync();
                var errorResponse = JsonSerializer.Deserialize<dynamic>(jsonContent);
                return (false, errorResponse?.ToString() ?? "Registration failed");
            }
        }
        catch (Exception ex)
        {
            Debug.WriteLine($"Register error: {ex.Message}");
            return (false, ex.Message);
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
                return new ApiResponse<IEnumerable<LessonDto>>
                {
                    Success = true,
                    Data = mappedLessons,
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

            var wrappedResponse = JsonSerializer.Deserialize<ApiResponse<LessonDetailDto>>(jsonContent, SerializerOptions);
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

            var lesson = JsonSerializer.Deserialize<LessonDetailDto>(jsonContent, SerializerOptions);
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
            var request = new { refreshToken };
            var content = new StringContent(
                JsonSerializer.Serialize(request),
                System.Text.Encoding.UTF8,
                new MediaTypeHeaderValue("application/json"));

            var response = await _httpClient.PostAsync("auth/refresh", content);
            if (response.IsSuccessStatusCode)
            {
                var jsonResponse = await response.Content.ReadAsStringAsync();
                var wrappedResponse = JsonSerializer.Deserialize<ApiResponse<LoginApiResponse>>(jsonResponse, SerializerOptions);
                var apiResponse = wrappedResponse?.Data ?? JsonSerializer.Deserialize<LoginApiResponse>(jsonResponse, SerializerOptions);

                var token = apiResponse?.Token;
                if (string.IsNullOrEmpty(token))
                {
                    token = apiResponse?.AccessToken;
                }

                if (!string.IsNullOrEmpty(token))
                {
                    // CRITICAL FIX: Save the new token to secure storage
                    SetAuthToken(token);
                    await SetSecureTokenAsync(token);
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

    public void SetAuthToken(string token)
    {
        if (!string.IsNullOrEmpty(token))
        {
            _httpClient.DefaultRequestHeaders.Authorization =
                new AuthenticationHeaderValue("Bearer", token);
        }
    }

    public async Task<string?> GetAuthTokenAsync() 
    {
        try
        {
            var token = await SecureStorage.Default.GetAsync(TokenKey);
            if (!string.IsNullOrEmpty(token))
            {
                return token;
            }

            return await SecureStorage.Default.GetAsync(LegacyTokenKey);
        }
        catch
        {
            return null;
        }
    }

    private async Task<ApiResponse<T>?> HandleResponse<T>(HttpResponseMessage response)
    {
        var jsonContent = await response.Content.ReadAsStringAsync();

        if (response.IsSuccessStatusCode)
        {
            ApiResponse<T>? wrapped = null;
            try
            {
                wrapped = JsonSerializer.Deserialize<ApiResponse<T>>(jsonContent, SerializerOptions);
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
                direct = JsonSerializer.Deserialize<T>(jsonContent, SerializerOptions);
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

    private async Task AddAuthHeaderAsync()
    {
        try
        {
            var token = await SecureStorage.Default.GetAsync(TokenKey);
            if (string.IsNullOrEmpty(token))
            {
                token = await SecureStorage.Default.GetAsync(LegacyTokenKey);
            }

            if (!string.IsNullOrEmpty(token))
            {
                _httpClient.DefaultRequestHeaders.Authorization =
                    new AuthenticationHeaderValue("Bearer", token);
            }
        }
        catch (Exception ex)
        {
            Debug.WriteLine($"Add auth header warning: {ex.Message}");
        }
    }

    private async Task SetSecureTokenAsync(string token)
    {
        try
        {
            if (MainThread.IsMainThread)
            {
                await SecureStorage.Default.SetAsync(TokenKey, token);
                await SecureStorage.Default.SetAsync(LegacyTokenKey, token);
            }
            else
            {
                await MainThread.InvokeOnMainThreadAsync(async () =>
                {
                    await SecureStorage.Default.SetAsync(TokenKey, token);
                    await SecureStorage.Default.SetAsync(LegacyTokenKey, token);
                });
            }
        }
        catch (Exception ex)
        {
            Debug.WriteLine($"Secure storage error: {ex.Message}");
        }
    }

    private async Task DeleteSecureTokenAsync()
    {
        try
        {
            SecureStorage.Default.Remove(TokenKey);
            SecureStorage.Default.Remove(LegacyTokenKey);
        }
        catch (Exception ex)
        {
            Debug.WriteLine($"Secure storage delete error: {ex.Message}");
        }
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

        var wrapped = JsonSerializer.Deserialize<ApiResponse<LearnDataDto>>(jsonContent, SerializerOptions);
        if (wrapped?.Data != null)
        {
            foreach (var lesson in wrapped.Data.Lessons)
            {
                DecodeLessonLayoutPayload(lesson);
            }

            return wrapped;
        }

        var direct = JsonSerializer.Deserialize<LearnDataDto>(jsonContent, SerializerOptions);
        if (direct != null)
        {
            foreach (var lesson in direct.Lessons)
            {
                DecodeLessonLayoutPayload(lesson);
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

    public async Task<ApiResponse<UserStatsDto>?> GetUserStatsAsync()
    {
        try
        {
            var learnDataResponse = await GetLearnDataPayloadAsync();
            var learnData = learnDataResponse?.Data;
            if (learnData != null)
            {
                var totalXp = learnData.TotalXp != 0 ? learnData.TotalXp : learnData.TotalXP;
                var progress = learnData.ProgressPercentage;
                if (progress <= 0 && learnData.Lessons.Count > 0)
                {
                    progress = learnData.Lessons.Average(l => l.CompletionPercentage) * 100d;
                }

                return new ApiResponse<UserStatsDto>
                {
                    Success = true,
                    Data = new UserStatsDto
                    {
                        TotalProgress = (int)Math.Round(progress),
                        CurrentStreak = learnData.CurrentStreak,
                        TotalXP = totalXp
                    },
                    Message = learnDataResponse?.Message ?? string.Empty
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

    public async Task<ApiResponse<IEnumerable<LessonCategoryDto>>?> GetCategoriesAsync()
    {
        try
        {
            await AddAuthHeaderAsync();
            var response = await _httpClient.GetAsync("learn/categories");
            var categoriesResponse = await HandleResponse<IEnumerable<LessonCategoryDto>>(response);
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
                    Data = categoriesFromLearnData.Select(NormalizeCategory).ToList(),
                    Message = learnPayload.Message
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
        try
        {
            await AddAuthHeaderAsync();
            var response = await _httpClient.GetAsync($"learn/categories/{categoryId}");
            return await HandleResponse<LessonCategoryDto>(response);
        }
        catch (Exception ex)
        {
            Debug.WriteLine($"GetCategory error: {ex.Message}");
            return null;
        }
    }

    public async Task<ApiResponse<IEnumerable<LessonDetailDto>>?> GetLessonsByCategoryAsync(int categoryId)
    {
        try
        {
            await AddAuthHeaderAsync();
            var response = await _httpClient.GetAsync($"learn/categories/{categoryId}/lessons");
            var lessonsResponse = await HandleResponse<IEnumerable<LessonDetailDto>>(response);
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

            var wrappedGoal = JsonSerializer.Deserialize<ApiResponse<DailyGoalDto>>(jsonContent, SerializerOptions);
            if (wrappedGoal?.Data != null)
            {
                return wrappedGoal;
            }

            var directGoal = JsonSerializer.Deserialize<DailyGoalDto>(jsonContent, SerializerOptions);
            if (directGoal != null && (directGoal.TotalRequired != 0 || directGoal.CompletedToday != 0))
            {
                return new ApiResponse<DailyGoalDto> { Success = true, Data = directGoal };
            }

            var apiGoal = JsonSerializer.Deserialize<DailyGoalApiDto>(jsonContent, SerializerOptions);
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
            return await HandleResponse<IEnumerable<SpacedRepetitionLessonDto>>(response);
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

            var wrappedUpcoming = JsonSerializer.Deserialize<ApiResponse<UpcomingReviewsDto>>(jsonContent, SerializerOptions);
            if (wrappedUpcoming?.Data != null)
            {
                return wrappedUpcoming;
            }

            var directUpcoming = JsonSerializer.Deserialize<UpcomingReviewsDto>(jsonContent, SerializerOptions);
            if (directUpcoming != null && (directUpcoming.TomorrowCount != 0 || directUpcoming.ThisWeekCount != 0 || directUpcoming.NextWeekCount != 0))
            {
                return new ApiResponse<UpcomingReviewsDto> { Success = true, Data = directUpcoming };
            }

            var apiUpcoming = JsonSerializer.Deserialize<UpcomingReviewsApiDto>(jsonContent, SerializerOptions);
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
            return await HandleResponse<PersonalizedRecommendationDto>(response);
        }
        catch (Exception ex)
        {
            Debug.WriteLine($"GetPersonalizedRecommendation error: {ex.Message}");
            return null;
        }
    }

    public async Task<ApiResponse<bool>?> MarkLessonCompleteAsync(int lessonId)
    {
        try
        {
            await AddAuthHeaderAsync();
            var content = new StringContent(
                JsonSerializer.Serialize(new { }),
                System.Text.Encoding.UTF8,
                new MediaTypeHeaderValue("application/json"));

            var response = await _httpClient.PostAsync($"learn/lessons/{lessonId}/complete", content);
            var jsonContent = await response.Content.ReadAsStringAsync();
            if (!response.IsSuccessStatusCode)
            {
                Debug.WriteLine($"MarkLessonComplete error: {response.StatusCode} - {jsonContent}");
                return null;
            }

            var wrapped = JsonSerializer.Deserialize<ApiResponse<bool>>(jsonContent, SerializerOptions);
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

            var wrapped = JsonSerializer.Deserialize<ApiResponse<bool>>(jsonContent, SerializerOptions);
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
            return await HandleResponse<IEnumerable<VideoDto>>(response);
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
            return await HandleResponse<VideoDto>(response);
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
            return await HandleResponse<IEnumerable<VideoDto>>(response);
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
                JsonSerializer.Serialize(new { }),
                System.Text.Encoding.UTF8,
                new MediaTypeHeaderValue("application/json"));

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
                var bytes = Convert.FromBase64String(text);
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
        if (string.IsNullOrWhiteSpace(input) || input.Length < 16 || input.Length % 4 != 0)
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

    public async Task<bool> WatchVideoAsync(int videoId)
    {
        try
        {
            await AddAuthHeaderAsync();
            var content = new StringContent(
                JsonSerializer.Serialize(new { }),
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

    public async Task<GesturePredictionResponseDto?> PredictGestureFromImageAsync(byte[] imageData, CancellationToken cancellationToken = default)
    {
        try
        {
            if (imageData == null || imageData.Length == 0)
            {
                Debug.WriteLine("Gesture prediction warning: image payload was empty.");
                return null;
            }

            try
            {
                await AddAuthHeaderAsync();
            }
            catch (Exception ex)
            {
                Debug.WriteLine($"Gesture auth header warning: {ex.Message}");
            }

            Debug.WriteLine($"Sending gesture frame: {imageData.Length} bytes");

            using (var content = new MultipartFormDataContent())
            {
                // Add image as multipart form data
                var imageContent = new ByteArrayContent(imageData);
                imageContent.Headers.ContentType = new MediaTypeHeaderValue("image/jpeg");
                content.Add(imageContent, "image", "frame.jpg");

                var response = await _httpClient.PostAsync("gesture/predict", content, cancellationToken);

                if (response.IsSuccessStatusCode)
                {
                    var jsonContent = await response.Content.ReadAsStringAsync(cancellationToken);
                    var result = JsonSerializer.Deserialize<GesturePredictionResponseDto>(jsonContent, SerializerOptions);
                    if (result?.Data != null)
                    {
                        return result;
                    }

                    var wrapped = JsonSerializer.Deserialize<ApiResponse<GesturePredictionDataDto>>(jsonContent, SerializerOptions);
                    if (wrapped?.Data != null)
                    {
                        return new GesturePredictionResponseDto
                        {
                            Status = wrapped.Success ? "success" : "failed",
                            Message = wrapped.Message,
                            Data = wrapped.Data
                        };
                    }

                    var direct = JsonSerializer.Deserialize<GesturePredictionDataDto>(jsonContent, SerializerOptions);
                    if (direct != null)
                    {
                        return new GesturePredictionResponseDto
                        {
                            Status = "success",
                            Message = string.Empty,
                            Data = direct
                        };
                    }

                    Debug.WriteLine($"Gesture prediction parse error: unexpected payload format - {jsonContent}");
                    return null;
                }
                else
                {
                    var errorContent = await response.Content.ReadAsStringAsync(cancellationToken);
                    Debug.WriteLine($"Gesture prediction error: {response.StatusCode} - {errorContent}");
                    return null;
                }
            }
        }
        catch (OperationCanceledException)
        {
            Debug.WriteLine("Gesture prediction request was canceled.");
            return null;
        }
        catch (Exception ex)
        {
            Debug.WriteLine($"Gesture prediction error: {ex.Message}");
            return null;
        }
    }
}
