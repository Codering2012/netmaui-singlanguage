using Android.App;
using Android.Runtime;

namespace SignLanguageApp
{
    [Application]
    public class MainApplication : MauiApplication
    {
        public MainApplication(IntPtr handle, JniHandleOwnership ownership)
            : base(handle, ownership)
        {
            AndroidEnvironment.UnhandledExceptionRaiser += (sender, args) =>
            {
                args.Handled = true;
                SignLanguageApp.Helpers.GlobalExceptionHandler.HandleException(args.Exception);
            };
        }

        protected override MauiApp CreateMauiApp() => MauiProgram.CreateMauiApp();
    }
}
