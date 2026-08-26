using Microsoft.Maui.Controls;
using SignLanguageApp.ViewModels;

namespace SignLanguageApp.Pages;

public partial class RegisterPage : ContentPage
{
    public RegisterPage()
    {
        try
        {
            InitializeComponent();
            BindingContext = App.Services.GetService<RegisterViewModel>();
        }
        catch (System.Exception ex)
        {
            SignLanguageApp.Helpers.GlobalExceptionHandler.HandleException(ex);
        }
    }

    public RegisterPage(RegisterViewModel viewModel)
    {
        try
        {
            InitializeComponent();
            BindingContext = viewModel;
        }
        catch (System.Exception ex)
        {
            SignLanguageApp.Helpers.GlobalExceptionHandler.HandleException(ex);
        }
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
    }

    protected override async void OnDisappearing()
    {
        base.OnDisappearing();
        await this.FadeToAsync(0, 200, Easing.CubicIn);
    }
}
