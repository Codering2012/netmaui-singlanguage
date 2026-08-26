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

            try
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
                        fonts.AddFont("Inter-Regular.ttf", "InterRegular");
                        fonts.AddFont("Inter-SemiBold.ttf", "InterSemiBold");
                        fonts.AddFont("Inter-Bold.ttf", "InterBold");
                    });

                // Register services
                builder.Services.AddSingleton(SecureStorage.Default);
                builder.Services.AddSingleton<IDatabaseService, DatabaseService>();
                builder.Services.AddSingleton<IAuthService, AuthService>();
                builder.Services.AddSingleton<IThemeService, ThemeService>();
                builder.Services.AddSingleton<ILessonPayloadSecurityService, LessonPayloadSecurityService>();

                // Register and initialize cache service
                var cacheService = new CacheService();
                _ = cacheService.InitializeAsync().ConfigureAwait(false);
                builder.Services.AddSingleton<ICacheService>(cacheService);

                // Configure HttpClient for API communication (Global Interface Ready)
                var httpHandler = new SocketsHttpHandler
                {
                    PooledConnectionLifetime = TimeSpan.FromMinutes(5), // Connection pooling
                    AutomaticDecompression = System.Net.DecompressionMethods.All // Brotli, GZip, Deflate
#if DEBUG
                    , SslOptions = new System.Net.Security.SslClientAuthenticationOptions
                    {
                        RemoteCertificateValidationCallback = (sender, cert, chain, sslPolicyErrors) => true
                    }
#endif
                };

                var hmacHandler = new HMACDelegatingHandler
                {
                    InnerHandler = httpHandler
                };

                var httpClient = new HttpClient(hmacHandler)
                {
                    BaseAddress = BuildApiBaseUri(),
                    Timeout = TimeSpan.FromSeconds(30),
                    DefaultRequestVersion = new Version(2, 0) // Force HTTP/2
                };
                builder.Services.AddSingleton(httpClient);
                
                // Register networking and caching
                builder.Services.AddSingleton<IConnectivityService, ConnectivityService>();
                builder.Services.AddSingleton<IApiConfigService, ApiConfigService>();
                builder.Services.AddSingleton<IApiService, ApiService>();
                builder.Services.AddSingleton<IMediaDownloadAndCacheService, MediaDownloadAndCacheService>();

                // Register ViewModels
                builder.Services.AddSingleton<LoginViewModel>();
                builder.Services.AddSingleton<RegisterViewModel>();
                builder.Services.AddSingleton<HomeViewModel>();
                builder.Services.AddSingleton<AccountViewModel>();
                builder.Services.AddSingleton<CameraTranslationViewModel>(sp =>
                    new CameraTranslationViewModel(sp.GetRequiredService<IApiService>()));
                builder.Services.AddSingleton<LearnViewModel>();
                builder.Services.AddSingleton<VideoViewModel>();
                builder.Services.AddSingleton<DictionaryViewModel>();
                
                // New ViewModels
                builder.Services.AddTransient<AchievementsViewModel>();
                builder.Services.AddTransient<DifficultyCalibrationViewModel>();
                builder.Services.AddTransient<DynamicLessonViewModel>();
                builder.Services.AddTransient<EditProfileViewModel>();
                builder.Services.AddTransient<FeedbackViewModel>();
                builder.Services.AddTransient<InteractiveLessonViewModel>();
                builder.Services.AddTransient<LeaderboardViewModel>();
                builder.Services.AddTransient<MistakeReplayViewModel>();
                builder.Services.AddTransient<ProgressReportsViewModel>();
                builder.Services.AddTransient<SentenceBuilderViewModel>();
                builder.Services.AddTransient<StatisticsViewModel>();

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

                // New Pages
                builder.Services.AddTransient<AchievementsPage>();
                builder.Services.AddTransient<CommunityHubPage>();
                builder.Services.AddTransient<DictionaryPage>();
                builder.Services.AddTransient<DifficultyCalibrationPage>();
                builder.Services.AddTransient<EditProfilePage>();
                builder.Services.AddTransient<FeedbackPage>();
                builder.Services.AddTransient<InteractiveLessonPage>();
                builder.Services.AddTransient<LeaderboardPage>();
                builder.Services.AddTransient<MistakeReplayPage>();
                builder.Services.AddTransient<ProgressReportsPage>();
                builder.Services.AddTransient<SentenceBuilderPage>();
                builder.Services.AddTransient<StatisticsPage>();
                builder.Services.AddTransient<TimeAttackPage>();

#if DEBUG
                builder.Logging.AddDebug();
#endif

                return builder.Build();
            }
            catch (Exception ex)
            {
                // This is absolutely critical. If DI fails, we need to know.
                System.Diagnostics.Debug.WriteLine($"CRITICAL DI FAILURE: {ex}");
                throw; // Rethrow because if MauiProgram fails, the app literally cannot start.
            }
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
