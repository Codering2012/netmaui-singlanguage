using SignLanguageApp.ViewModels;

namespace SignLanguageApp.Pages;

public partial class VideosPage : ContentPage
{
    public VideosPage(VideoViewModel viewModel)
    {
        InitializeComponent();
        BindingContext = viewModel;
    }

    protected override async void OnAppearing()
    {
        base.OnAppearing();
        if (BindingContext is VideoViewModel vm)
        {
            await vm.LoadVideosCommand.ExecuteAsync(null);
        }
    }
}
