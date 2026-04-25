using SignLanguageApp.Services;
using Microsoft.Extensions.DependencyInjection;

namespace SignLanguageApp
{
    public partial class App : Application
    {
        private readonly IServiceProvider _serviceProvider;

        public App() : this(new ServiceCollection().BuildServiceProvider())
        {
        }

        public App(IServiceProvider serviceProvider)
        {
            _serviceProvider = serviceProvider;
            InitializeComponent();
        }

        protected override Window CreateWindow(IActivationState? activationState)
        {
            return new Window(new Pages.StartupLoadingPage(_serviceProvider));
        }
    }
}