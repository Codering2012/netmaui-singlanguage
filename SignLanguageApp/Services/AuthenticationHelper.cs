using System.Diagnostics;
using System.Text.Json;

namespace SignLanguageApp.Services;

/// <summary>
/// Helper service for authentication utilities
/// </summary>
public static class AuthenticationHelper
{
    /// <summary>
    /// Initializes authentication from stored credentials
    /// </summary>
    public static async Task InitializeAuthenticationAsync(
        IAuthService authService,
        IDatabaseService databaseService,
        IApiService apiService)
    {
        try
        {
            var accessToken = await databaseService.GetAccessTokenAsync();
            if (!string.IsNullOrEmpty(accessToken))
            {
                apiService.SetAuthToken(accessToken);
            }
        }
        catch (Exception ex)
        {
            Debug.WriteLine($"Initialize authentication error: {ex.Message}");
        }
    }

    /// <summary>
    /// Validates if token is still valid (basic check)
    /// </summary>
    public static bool IsTokenValid(string? token)
    {
        if (string.IsNullOrEmpty(token))
            return false;

        try
        {
            var parts = token.Split('.');
            if (parts.Length != 3)
                return false;

            // Decode the payload
            var payload = parts[1];
            var padded = payload.PadRight(payload.Length + (4 - payload.Length % 4) % 4, '=');
            var decoded = System.Convert.FromBase64String(padded);
            var json = System.Text.Encoding.UTF8.GetString(decoded);

            var jObject = JsonDocument.Parse(json);
            var expProperty = jObject.RootElement.GetProperty("exp");
            var expirationUnix = expProperty.GetInt64();
            var expirationDateTime = UnixTimeStampToDateTime(expirationUnix);

            return expirationDateTime > DateTime.UtcNow;
        }
        catch (Exception ex)
        {
            Debug.WriteLine($"Token validation error: {ex.Message}");
            return false;
        }
    }

    private static DateTime UnixTimeStampToDateTime(long unixTimeStamp)
    {
        var dateTime = new DateTime(1970, 1, 1, 0, 0, 0, 0, DateTimeKind.Utc);
        dateTime = dateTime.AddSeconds(unixTimeStamp).ToUniversalTime();
        return dateTime;
    }
}
