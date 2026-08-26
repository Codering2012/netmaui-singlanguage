using SignLanguageApp.ViewModels;

namespace SignLanguageApp.Pages;

public partial class ProfilePage : ContentPage
{
    public ProfilePage(AccountViewModel viewModel)
    {
        InitializeComponent();
        BindingContext = viewModel;
    }

    protected override async void OnAppearing()
    {
        base.OnAppearing();
            this.Opacity = 0;
            this.TranslationY = 20;
            MainThread.BeginInvokeOnMainThread(async () => {
            await Task.Delay(100);
            await Task.WhenAll(
                this.FadeToAsync(1, 400, Easing.CubicOut),
                this.TranslateToAsync(0, 0, 400, Easing.CubicOut)
            );
        });
        if (BindingContext is AccountViewModel vm)
        {
            await vm.LoadDataCommand.ExecuteAsync(null);
        }
    }

    protected override void OnDisappearing()
    {
        base.OnDisappearing();
        if (BindingContext is AccountViewModel vm)
        {
            vm.OnDisappearing();
        }
    }
}
