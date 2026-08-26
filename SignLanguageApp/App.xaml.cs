using SignLanguageApp.Services;
using Microsoft.Extensions.DependencyInjection;

namespace SignLanguageApp
{
    public partial class App : Application
    {
        private readonly IServiceProvider _serviceProvider;
        public static IServiceProvider Services { get; private set; }

        public App() : this(new ServiceCollection().BuildServiceProvider())
        {
        }

        public App(IServiceProvider serviceProvider)
        {
            try
            {
                _serviceProvider = serviceProvider;
                Services = serviceProvider;
                
                // Global Exception Hooks
                AppDomain.CurrentDomain.UnhandledException += (s, e) =>
                {
                    if (e.ExceptionObject is Exception ex)
                        SignLanguageApp.Helpers.GlobalExceptionHandler.HandleException(ex);
                };
                TaskScheduler.UnobservedTaskException += (s, e) =>
                {
                    SignLanguageApp.Helpers.GlobalExceptionHandler.HandleException(e.Exception);
                    e.SetObserved();
                };

                InitializeComponent();
                
                var themeService = _serviceProvider.GetService<IThemeService>();
                themeService?.InitializeTheme();
            }
            catch (Exception ex)
            {
                SignLanguageApp.Helpers.GlobalExceptionHandler.HandleException(ex);
            }
        }

        protected override Window CreateWindow(IActivationState? activationState)
        {
            try
            {
                return new Window(new Pages.StartupLoadingPage(_serviceProvider));
            }
            catch (Exception ex)
            {
                SignLanguageApp.Helpers.GlobalExceptionHandler.HandleException(ex);
                return new Window(new Pages.ErrorDebugPage(ex));
            }
        }
    }
}