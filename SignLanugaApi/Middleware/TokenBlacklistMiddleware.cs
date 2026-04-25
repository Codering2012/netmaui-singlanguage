using SignLanguageApi.Services;
using System.Text.Json;

namespace SignLanguageApi.Middleware;

public class TokenBlacklistMiddleware
{
    private readonly RequestDelegate _next;
    private readonly ILogger<TokenBlacklistMiddleware> _logger;

    public TokenBlacklistMiddleware(RequestDelegate next, ILogger<TokenBlacklistMiddleware> logger)
    {
        _next = next;
        _logger = logger;
    }

    public async Task InvokeAsync(HttpContext context, ITokenBlacklistService tokenBlacklistService)
    {
        // Check if the request has an Authorization header
        var authHeader = context.Request.Headers["Authorization"].ToString();
        if (!string.IsNullOrEmpty(authHeader) && authHeader.StartsWith("Bearer "))
        {
            var token = authHeader.Substring("Bearer ".Length).Trim();

            // Check if token is blacklisted
            var isBlacklisted = await tokenBlacklistService.IsTokenBlacklistedAsync(token);
            if (isBlacklisted)
            {
                _logger.LogWarning("Attempt to use blacklisted token. IP: {RemoteIP}", 
                    context.Connection.RemoteIpAddress?.ToString() ?? "Unknown");

                context.Response.StatusCode = StatusCodes.Status401Unauthorized;
                context.Response.ContentType = "application/json";

                await context.Response.WriteAsync(JsonSerializer.Serialize(new
                {
                    message = "Token is no longer valid. Please log in again."
                }));
                return;
            }
        }

        await _next(context);
    }
}

public static class TokenBlacklistMiddlewareExtensions
{
    public static IApplicationBuilder UseTokenBlacklistMiddleware(this IApplicationBuilder builder)
    {
        return builder.UseMiddleware<TokenBlacklistMiddleware>();
    }
}
