using SignLanguageApp.ViewModels;
using SignLanguageApp.Helpers;

namespace SignLanguageApp.Pages;

public partial class ProgressReportsPage : ContentPage
{
    private readonly ProgressReportsViewModel _viewModel;

    public ProgressReportsPage(ProgressReportsViewModel viewModel)
    {
        InitializeComponent();
        _viewModel = viewModel;
        BindingContext = _viewModel;
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

        // Staggered entrance for main content cards
        if (MainLayout != null && MainLayout.Children.Count > 0)
        {
            _ = MainLayout.AnimateStaggeredChildren(100, 500, 50);
        }

        // Staggered pop-in for the chart bars after data binds
        MainThread.BeginInvokeOnMainThread(async () =>
        {
            await Task.Delay(150); // Short delay to let BindableLayout populate the chart bars
            if (ChartLayout != null && ChartLayout.Children.Count > 0)
            {
                await ChartLayout.AnimateStaggeredChildren(50, 400, 50);
            }
        });
    }

    protected override async void OnDisappearing()
    {
        base.OnDisappearing();
        await this.FadeToAsync(0, 200, Easing.CubicIn);
    }
}
