using System.Diagnostics;

namespace SignLanguageApp.Services
{
    public class DiscoveryRoutingHandler : DelegatingHandler
    {
        private readonly IApiConfigService _configService;

    public DiscoveryRoutingHandler(IApiConfigService configService)
    {
        _configService = configService;
    }

        protected override Task<HttpResponseMessage> SendAsync(HttpRequestMessage request, CancellationToken cancellationToken)
        {
            bool isInternal = true;
            try
            {
                var currentBase = _configService.BaseUrl;
                if (!string.IsNullOrWhiteSpace(currentBase) && Uri.TryCreate(currentBase, UriKind.Absolute, out var baseUri))
                {
                    var requestUri = request.RequestUri;
                    if (requestUri != null)
                    {
                        bool isRelative = !requestUri.IsAbsoluteUri;
                        
                        // If relative, resolve against a temporary base so we can use UriBuilder
                        if (isRelative)
                        {
                            requestUri = new Uri(new Uri("http://api.internal/"), requestUri);
                        }

                        isInternal = isRelative || 
                                     requestUri.Host == baseUri.Host ||
                                     requestUri.Host == "api.internal" || 
                                     requestUri.Host == "localhost" || 
                                     requestUri.Host == "127.0.0.1" ||
                                     requestUri.Host == "10.0.2.2";

                        // Replace placeholder or default hosts with the discovered one
                        if (isRelative || 
                            requestUri.Host == "api.internal" || 
                            requestUri.Host == "localhost" || 
                            requestUri.Host == "127.0.0.1" ||
                            requestUri.Host == "10.0.2.2")
                        {
                            var builder = new UriBuilder(requestUri);
                            builder.Scheme = baseUri.Scheme;
                            builder.Host = baseUri.Host;
                            builder.Port = baseUri.Port;

                            // Ensure path logic is sound
                            var basePath = baseUri.AbsolutePath.TrimEnd('/');
                            var currentPath = builder.Path.StartsWith("/") ? builder.Path : "/" + builder.Path;
                            
                            // If the base path contains /api and the request doesn't, prepend it
                            if (basePath.EndsWith("/api") && !currentPath.StartsWith("/api"))
                            {
                                builder.Path = basePath.TrimEnd('/') + currentPath;
                            }
                            else if (basePath != "/" && !currentPath.StartsWith(basePath))
                            {
                                builder.Path = basePath.TrimEnd('/') + currentPath;
                            }

                            Debug.WriteLine($">>> Routing request from {request.RequestUri} to {builder.Uri}");
                            request.RequestUri = builder.Uri;
                        }
                    }
                }
            }
            catch (Exception ex)
            {
                Debug.WriteLine($"Routing Error: {ex.GetType().Name} - {ex.Message}");
            }

            if (isInternal)
            {
                // Security: Add required API Key for internal API requests
                request.Headers?.TryAddWithoutValidation("X-API-KEY", "SignLang_Secure_v1_2026");
            }
            else
            {
                // Security: Prevent leaking Bearer JWT access tokens and API keys to third-party domains (e.g. CDNs or external image hosts)
                request.Headers?.Remove("Authorization");
                request.Headers?.Remove("X-API-KEY");
                Debug.WriteLine($">>> Security Warning: Stripped Authorization & X-API-KEY header from external request to {request.RequestUri}");
            }

            return base.SendAsync(request, cancellationToken);
        }
    }
}
