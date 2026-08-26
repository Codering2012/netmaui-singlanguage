using Microsoft.AspNetCore.Http;
using System.Threading.Tasks;

namespace SignLanguageApi.Middleware
{
    public class ApiKeyMiddleware
    {
        private readonly RequestDelegate _next;
        private const string APIKEYNAME = "X-API-KEY";
        private const string APIKEY = "SignLang_Secure_v1_2026"; // In prod, this should be in Environment/Secrets

        public ApiKeyMiddleware(RequestDelegate next)
        {
            _next = next;
        }

        public async Task InvokeAsync(HttpContext context)
        {
            // Allow health check and swagger without API key if needed, 
            // but for maximum robustness we require it everywhere except maybe health.
            if (context.Request.Path.StartsWithSegments("/api/gesture/health"))
            {
                await _next(context);
                return;
            }

            if (!context.Request.Headers.TryGetValue(APIKEYNAME, out var extractedApiKey))
            {
                context.Response.StatusCode = 401;
                await context.Response.WriteAsync("API Key was not provided.");
                return;
            }

            if (!APIKEY.Equals(extractedApiKey))
            {
                context.Response.StatusCode = 401;
                await context.Response.WriteAsync("Unauthorized client.");
                return;
            }

            await _next(context);
        }
    }
}
