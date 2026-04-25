using System.Diagnostics;
using System.Text.Json;
using SignLanguageApp.Model;

namespace SignLanguageApp.Services;

/// <summary>
/// Interface for local database operations
/// </summary>
public interface IDatabaseService
{
    Task<User?> GetUserAsync();
    Task SaveUserAsync(User user);
    Task<bool> DeleteUserAsync();
    Task<string?> GetAccessTokenAsync();
    Task SaveAccessTokenAsync(string token);
    Task<string?> GetRefreshTokenAsync();
    Task SaveRefreshTokenAsync(string token);
    Task ClearAllAsync();
}

/// <summary>
/// Database service for local storage using preferences and secure storage
/// </summary>
public class DatabaseService : IDatabaseService
{
    private const string UserKey = "current_user";
    private const string AccessTokenKey = "access_token";
    private const string RefreshTokenKey = "refresh_token";

    public async Task<User?> GetUserAsync()
    {
        try
        {
            var json = Preferences.Get(UserKey, string.Empty);
            if (string.IsNullOrEmpty(json))
                return null;

            return JsonSerializer.Deserialize<User>(json);
        }
        catch (Exception ex)
        {
            Debug.WriteLine($"GetUser error: {ex.Message}");
            return null;
        }
    }

    public async Task SaveUserAsync(User user)
    {
        try
        {
            var json = JsonSerializer.Serialize(user);
            Preferences.Set(UserKey, json);
            await Task.CompletedTask;
        }
        catch (Exception ex)
        {
            Debug.WriteLine($"SaveUser error: {ex.Message}");
        }
    }

    public async Task<bool> DeleteUserAsync()
    {
        try
        {
            Preferences.Remove(UserKey);
            Preferences.Remove(AccessTokenKey);
            Preferences.Remove(RefreshTokenKey);
            await Task.CompletedTask;
            return true;
        }
        catch (Exception ex)
        {
            Debug.WriteLine($"DeleteUser error: {ex.Message}");
            return false;
        }
    }

    public async Task<string?> GetAccessTokenAsync()
    {
        try
        {
            return await SecureStorage.Default.GetAsync(AccessTokenKey);
        }
        catch (Exception ex)
        {
            Debug.WriteLine($"GetAccessToken error: {ex.Message}");
            return null;
        }
    }

    public async Task SaveAccessTokenAsync(string token)
    {
        try
        {
            await SecureStorage.Default.SetAsync(AccessTokenKey, token);
        }
        catch (Exception ex)
        {
            Debug.WriteLine($"SaveAccessToken error: {ex.Message}");
        }
    }

    public async Task<string?> GetRefreshTokenAsync()
    {
        try
        {
            return await SecureStorage.Default.GetAsync(RefreshTokenKey);
        }
        catch (Exception ex)
        {
            Debug.WriteLine($"GetRefreshToken error: {ex.Message}");
            return null;
        }
    }

    public async Task SaveRefreshTokenAsync(string token)
    {
        try
        {
            await SecureStorage.Default.SetAsync(RefreshTokenKey, token);
        }
        catch (Exception ex)
        {
            Debug.WriteLine($"SaveRefreshToken error: {ex.Message}");
        }
    }

    public async Task ClearAllAsync()
    {
        try
        {
            await DeleteUserAsync();
        }
        catch (Exception ex)
        {
            Debug.WriteLine($"ClearAll error: {ex.Message}");
        }
    }
}
