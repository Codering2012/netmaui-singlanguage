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

            // Decode the payload (Base64Url to Base64)
            var payload = parts[1];
            payload = payload.Replace('-', '+').Replace('_', '/');
            var padLength = 4 - (payload.Length % 4);
            if (padLength < 4)
            {
                payload = payload.PadRight(payload.Length + padLength, '=');
            }
            
            var decoded = System.Convert.FromBase64String(payload);
            var json = System.Text.Encoding.UTF8.GetString(decoded);

            var jObject = System.Text.Json.JsonDocument.Parse(json);
            if (jObject.RootElement.TryGetProperty("exp", out var expProperty))
            {
                long expirationUnix = 0;
                if (expProperty.ValueKind == System.Text.Json.JsonValueKind.Number)
                {
                    expirationUnix = expProperty.GetInt64();
                }
                
                var expirationDateTime = UnixTimeStampToDateTime(expirationUnix);
                return expirationDateTime > DateTime.UtcNow;
            }
            
            return true; // No expiration means valid
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
