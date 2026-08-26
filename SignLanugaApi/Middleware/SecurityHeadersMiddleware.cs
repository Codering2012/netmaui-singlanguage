using Microsoft.AspNetCore.Http;
using System.Threading.Tasks;

namespace SignLanguageApi.Middleware
{
    public class SecurityHeadersMiddleware
    {
        private readonly RequestDelegate _next;

        public SecurityHeadersMiddleware(RequestDelegate next)
        {
            _next = next;
        }

        public async Task InvokeAsync(HttpContext context)
        {
            context.Response.Headers["X-Content-Type-Options"] = "nosniff";
            context.Response.Headers["X-Frame-Options"] = "DENY";
            context.Response.Headers["X-XSS-Protection"] = "1; mode=block";
            context.Response.Headers["Referrer-Policy"] = "no-referrer";
            context.Response.Headers["Content-Security-Policy"] = "default-src 'self'; frame-ancestors 'none';";
            
            // HSTS for 1 year
            context.Response.Headers["Strict-Transport-Security"] = "max-age=31536000; includeSubDomains";

            await _next(context);
        }
    }
}
