using System.Collections.Concurrent;
using System.Net;

namespace SignLanguageApi.Middleware;

public class RateLimitMiddleware
{
    private readonly RequestDelegate _next;
    private readonly ILogger<RateLimitMiddleware> _logger;
    private static readonly ConcurrentDictionary<string, (int Count, DateTime ResetTime)> _requestTracker =
        new ConcurrentDictionary<string, (int, DateTime)>();

    private const int MaxRequests = 5;
    private const int WindowMinutes = 15;

    public RateLimitMiddleware(RequestDelegate next, ILogger<RateLimitMiddleware> logger)
    {
        _next = next;
        _logger = logger;
    }

    public async Task InvokeAsync(HttpContext context)
    {
        // Only apply rate limiting to login and register endpoints
        var path = context.Request.Path.Value?.ToLower();
        if (path == null || (!path.Contains("auth/login") && !path.Contains("auth/register")))
        {
            await _next(context);
            return;
        }

        var clientIp = GetClientIpAddress(context);
        var now = DateTime.UtcNow;

        // Get or initialize the rate limit data for this IP
        var (count, resetTime) = _requestTracker.GetOrAdd(clientIp, _ => (0, now.AddMinutes(WindowMinutes)));

        // Check if the window has expired
        if (now > resetTime)
        {
            // Reset the counter
            _requestTracker.AddOrUpdate(clientIp, (1, now.AddMinutes(WindowMinutes)), (_, _) => (1, now.AddMinutes(WindowMinutes)));
            await _next(context);
            return;
        }

        // Check if we've exceeded the limit
        if (count >= MaxRequests)
        {
            _logger.LogWarning("Rate limit exceeded for IP {ClientIp}. Requests: {Count}, Window: {Window} minutes",
                clientIp, count, WindowMinutes);

            context.Response.StatusCode = StatusCodes.Status429TooManyRequests;
            context.Response.ContentType = "application/json";

            var timeRemaining = resetTime - now;
            var response = new SignLanguageApi.Dtos.RateLimitResponseDto
            {
                message = "Too many requests. Please try again later.",
                retryAfterSeconds = (int)timeRemaining.TotalSeconds
            };

            await context.Response.WriteAsJsonAsync(response, SignLanguageApi.Dtos.ApiJsonContext.Default.RateLimitResponseDto);
            return;
        }

        // Increment the counter
        _requestTracker.AddOrUpdate(clientIp, (count + 1, resetTime), (_, _) => (count + 1, resetTime));

        await _next(context);
    }

    private static string GetClientIpAddress(HttpContext context)
    {
        // Check for X-Forwarded-For header (used by proxies)
        if (context.Request.Headers.TryGetValue("X-Forwarded-For", out var forwardedFor))
        {
            var ips = forwardedFor.ToString().Split(',');
            if (ips.Length > 0 && IPAddress.TryParse(ips[0].Trim(), out _))
            {
                return ips[0].Trim();
            }
        }

        // Check for X-Real-IP header
        if (context.Request.Headers.TryGetValue("X-Real-IP", out var realIp))
        {
            if (IPAddress.TryParse(realIp.ToString(), out _))
            {
                return realIp.ToString();
            }
        }

        // Fall back to RemoteIpAddress
        return context.Connection.RemoteIpAddress?.ToString() ?? "Unknown";
    }
}

public static class RateLimitMiddlewareExtensions
{
    public static IApplicationBuilder UseRateLimitMiddleware(this IApplicationBuilder builder)
    {
        return builder.UseMiddleware<RateLimitMiddleware>();
    }
}
