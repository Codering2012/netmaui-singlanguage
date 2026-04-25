namespace SignLanguageApp
{
    public partial class AppShell : Shell
    {
        public AppShell()
        {
            InitializeComponent();
            Routing.RegisterRoute("home_page", typeof(Pages.HomePage));
            Routing.RegisterRoute("account_page", typeof(Pages.AccountPage));
            Routing.RegisterRoute("learn", typeof(Pages.LearnPage));
            Routing.RegisterRoute("camera", typeof(Pages.CameraTranslationPage));
            Routing.RegisterRoute("camera-translation", typeof(Pages.CameraTranslationPage));
            Routing.RegisterRoute("translation", typeof(Pages.CameraTranslationPage));
        }
    }
}
