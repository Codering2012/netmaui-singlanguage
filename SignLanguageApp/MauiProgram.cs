using Microsoft.Extensions.Logging;
using SignLanguageApp.ViewModels;
using SignLanguageApp.Pages;
using SignLanguageApp.Services;
using SignLanguageApp.Model;
using CommunityToolkit.Maui;
using CommunityToolkit.Maui.Core;

namespace SignLanguageApp
{
    public static class MauiProgram
    {
        private const int DefaultApiPort = 5179;

        public static MauiApp CreateMauiApp()
        {
            var builder = MauiApp.CreateBuilder();
            builder
                .UseMauiApp<App>()
                .UseMauiCommunityToolkit()
                .UseMauiCommunityToolkitCamera()
                .ConfigureFonts(fonts =>
                {
                    fonts.AddFont("OpenSans-Regular.ttf", "OpenSansRegular");
                    fonts.AddFont("OpenSans-Semibold.ttf", "OpenSansSemibold");
                });

            // Register services
            builder.Services.AddSingleton(SecureStorage.Default);
            builder.Services.AddSingleton<IDatabaseService, DatabaseService>();
            builder.Services.AddSingleton<IAuthService, AuthService>();
            builder.Services.AddSingleton<IOnnxInferenceService, OnnxInferenceService>();
            builder.Services.AddSingleton<ILessonPayloadSecurityService, LessonPayloadSecurityService>();

            // Register and initialize cache service
            var cacheService = new CacheService();
            _ = cacheService.InitializeAsync().ConfigureAwait(false);
            builder.Services.AddSingleton<ICacheService>(cacheService);

            // Configure HttpClient for API communication
#if DEBUG
            var httpHandler = new HttpClientHandler
            {
                ServerCertificateCustomValidationCallback = HttpClientHandler.DangerousAcceptAnyServerCertificateValidator
            };
            var httpClient = new HttpClient(httpHandler)
            {
                BaseAddress = BuildApiBaseUri(),
                Timeout = TimeSpan.FromSeconds(30)
            };
#else
            var httpClient = new HttpClient
            {
                BaseAddress = BuildApiBaseUri(),
                Timeout = TimeSpan.FromSeconds(30)
            };
#endif
            builder.Services.AddSingleton(httpClient);
            builder.Services.AddSingleton<IApiService>(sp => new ApiService(httpClient));

            // Register ViewModels
            builder.Services.AddSingleton<LoginViewModel>();
            builder.Services.AddSingleton<RegisterViewModel>();
            builder.Services.AddSingleton<HomeViewModel>();
            builder.Services.AddSingleton<AccountViewModel>();
            builder.Services.AddSingleton<CameraTranslationViewModel>(sp =>
                new CameraTranslationViewModel(sp.GetRequiredService<IApiService>()));
            builder.Services.AddSingleton<LearnViewModel>();
            builder.Services.AddSingleton<VideoViewModel>();

            // Register Pages
            builder.Services.AddSingleton<LoginPage>();
            builder.Services.AddSingleton<RegisterPage>();
            builder.Services.AddSingleton<HomePage>();
            builder.Services.AddSingleton<AccountPage>();
            builder.Services.AddSingleton<CameraTranslationPage>();
            builder.Services.AddSingleton<LearnPage>();
            builder.Services.AddSingleton<VideosPage>();
            builder.Services.AddSingleton<ProfilePage>();
            builder.Services.AddSingleton<MainPage>();

#if DEBUG
            builder.Logging.AddDebug();
#endif

            return builder.Build();
        }

        private static Uri BuildApiBaseUri()
        {
            var configuredUrl = Environment.GetEnvironmentVariable("SIGNLANGUAGE_API_BASE_URL");
            var baseUrl = string.IsNullOrWhiteSpace(configuredUrl)
                ? BuildDefaultLanApiBaseUrl()
                : configuredUrl.Trim();

            if (!baseUrl.EndsWith("/", StringComparison.Ordinal))
            {
                baseUrl += "/";
            }

            return new Uri(baseUrl, UriKind.Absolute);
        }

        private static string BuildDefaultLanApiBaseUrl()
        {
            var host = "localhost";
#if ANDROID
            // Android emulators use 10.0.2.2 to reach the host machine.
            host = "10.0.2.2";
#endif
            return $"http://{host}:{DefaultApiPort}/api/";
        }
    }
}
