using SignLanguageApp.ViewModels;

namespace SignLanguageApp.Pages
{
    public partial class LeaderboardPage : ContentPage
    {
        private readonly LeaderboardViewModel _viewModel;

        public LeaderboardPage(LeaderboardViewModel viewModel)
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
            await _viewModel.LoadLeaderboardCommand.ExecuteAsync(null);
        }
        protected override async void OnDisappearing()
        {
            base.OnDisappearing();
            await this.FadeToAsync(0, 200, Easing.CubicIn);
        }

    }
}
