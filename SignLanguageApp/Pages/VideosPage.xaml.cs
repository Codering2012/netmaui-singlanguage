using SignLanguageApp.ViewModels;

namespace SignLanguageApp.Pages;

public partial class VideosPage : ContentPage
{
    public VideosPage()
    {
        InitializeComponent();
        BindingContext = App.Services.GetService<VideoViewModel>();
    }

    public VideosPage(VideoViewModel viewModel)
    {
        InitializeComponent();
        BindingContext = viewModel;
    }

    protected override async void OnAppearing()
    {
        try
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
            if (BindingContext is VideoViewModel vm)
            {
                await vm.LoadVideosCommand.ExecuteAsync(null);
            }
        }
        catch (Exception ex)
        {
            SignLanguageApp.Helpers.GlobalExceptionHandler.HandleException(ex);
        }
    }

    private async void OnVideoTapped(object sender, TappedEventArgs e)
    {
        if (sender is Border border)
        {
            var originalStroke = border.Stroke;
            var originalMargin = border.Margin;

            // Brighter border and enlarge using Dynamic Resource
            var highlightColor = Application.Current?.Resources.TryGetValue("PrimaryColor", out var hc) == true ? (Color)hc : Color.FromArgb("#56A8F0");
            border.Stroke = highlightColor;
            
            // Push other elements by increasing bottom margin while scaling
            border.Margin = new Thickness(originalMargin.Left, originalMargin.Top, originalMargin.Right, originalMargin.Bottom + 15);
            
            await border.ScaleToAsync(1.1, 150, Easing.CubicOut);
            
            await Task.Delay(150);

            await border.ScaleToAsync(1.0, 150, Easing.CubicIn);
            
            border.Margin = originalMargin;
            border.Stroke = originalStroke;
        }
    }

    protected override async void OnDisappearing()
    {
        base.OnDisappearing();
        await this.FadeToAsync(0, 200, Easing.CubicIn);
    }
}
