using SignLanguageApp.ViewModels;

namespace SignLanguageApp.Pages;

public partial class EditProfilePage : ContentPage
{
    private readonly EditProfileViewModel _viewModel;

    public EditProfilePage(EditProfileViewModel viewModel)
    {
        InitializeComponent();
        BindingContext = _viewModel = viewModel;
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
        await _viewModel.InitializeAsync();
    }

        protected override async void OnDisappearing()
        {
            base.OnDisappearing();
            await this.FadeToAsync(0, 200, Easing.CubicIn);
        }
}
