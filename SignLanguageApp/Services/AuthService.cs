using System.Diagnostics;
using SignLanguageApp.Model;

namespace SignLanguageApp.Services;

/// <summary>
/// Authentication service interface
/// </summary>
public interface IAuthService
{
    Task<LoginResponse?> LoginAsync(string email, string password);
    Task<UserDto?> RegisterAsync(string email, string password, string name);
    Task<bool> LogoutAsync();
    Task<User?> GetCurrentUserAsync();
    Task<bool> IsAuthenticatedAsync();
    Task<bool> RefreshTokenAsync();
}

/// <summary>
/// Authentication service for MAUI
/// </summary>
public class AuthService : IAuthService
{
    private readonly IApiService _apiService;
    private readonly IDatabaseService _databaseService;

    public AuthService(IApiService apiService, IDatabaseService databaseService)
    {
        _apiService = apiService;
        _databaseService = databaseService;
    }

    public async Task<LoginResponse?> LoginAsync(string email, string password)
    {
        try
        {
            var response = await _apiService.LoginAsync(email, password);

            if (response?.AccessToken != null)
            {
                // Save tokens
                await _databaseService.SaveAccessTokenAsync(response.AccessToken);
                await _databaseService.SaveRefreshTokenAsync(response.RefreshToken);

                // Save user
                if (response.User != null)
                {
                    var user = new User
                    {
                        Id = response.User.Id,
                        Email = response.User.Email,
                        Name = response.User.Name,
                        AvatarUrl = response.User.AvatarUrl ?? string.Empty,
                        LearningStreak = response.User.LearningStreak
                    };
                    await _databaseService.SaveUserAsync(user);
                }

                // Set auth token in API service
                _apiService.SetAuthToken(response.AccessToken);

                return response;
            }

            return null;
        }
        catch (Exception ex)
        {
            Debug.WriteLine($"Login error: {ex.Message}");
            return null;
        }
    }

    public async Task<UserDto?> RegisterAsync(string email, string password, string name)
    {
        try
        {
            var (success, message) = await _apiService.RegisterAsync(email, password, name);

            if (success)
            {
                // Return a basic UserDto for successful registration
                return new UserDto
                {
                    Email = email,
                    Name = name
                };
            }

            return null;
        }
        catch (Exception ex)
        {
            Debug.WriteLine($"Register error: {ex.Message}");
            return null;
        }
    }

    public async Task<bool> LogoutAsync()
    {
        try
        {
            await _databaseService.ClearAllAsync();
            _apiService.SetAuthToken(string.Empty);
            return true;
        }
        catch (Exception ex)
        {
            Debug.WriteLine($"Logout error: {ex.Message}");
            return false;
        }
    }

    public async Task<User?> GetCurrentUserAsync()
    {
        return await _databaseService.GetUserAsync();
    }

    public async Task<bool> IsAuthenticatedAsync()
    {
        var token = await _databaseService.GetAccessTokenAsync();
        return !string.IsNullOrEmpty(token);
    }

    public async Task<bool> RefreshTokenAsync()
    {
        try
        {
            var refreshToken = await _databaseService.GetRefreshTokenAsync();
            if (string.IsNullOrEmpty(refreshToken))
            {
                await _databaseService.ClearAllAsync();
                _apiService.SetAuthToken(string.Empty);
                return false;
            }

            var refreshed = await _apiService.RefreshTokenAsync(refreshToken);
            if (!refreshed)
            {
                await _databaseService.ClearAllAsync();
                _apiService.SetAuthToken(string.Empty);
            }

            return refreshed;
        }
        catch (Exception ex)
        {
            Debug.WriteLine($"Token refresh error: {ex.Message}");
            await _databaseService.ClearAllAsync();
            _apiService.SetAuthToken(string.Empty);
            return false;
        }
    }
}