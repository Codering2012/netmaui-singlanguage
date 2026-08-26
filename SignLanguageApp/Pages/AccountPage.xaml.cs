using Microsoft.Maui.Controls;
using SignLanguageApp.ViewModels;
using SignLanguageApp.Services;
using System.Threading.Tasks;
using System;

namespace SignLanguageApp.Pages
{
    public partial class AccountPage : ContentPage
    {
        private readonly AccountViewModel _viewModel;

        public AccountPage()
        {
            InitializeComponent();
            _viewModel = App.Services.GetService<AccountViewModel>();
            BindingContext = _viewModel;
            SyncThemeSwitchState();
        }

        public AccountPage(AccountViewModel viewModel)
        {
            InitializeComponent();
            _viewModel = viewModel;
            BindingContext = _viewModel;
            SyncThemeSwitchState();
        }

        private void SyncThemeSwitchState()
        {
            try
            {
                var themeService = App.Services?.GetService<IThemeService>();
                if (themeService != null && ThemeSwitch != null)
                {
                    ThemeSwitch.Toggled -= OnThemeSwitchToggled;

                    int currentTheme = themeService.CurrentTheme;
                    if (currentTheme == 0)
                    {
                        var requestedTheme = Application.Current?.RequestedTheme;
                        ThemeSwitch.IsToggled = (requestedTheme == AppTheme.Dark);
                    }
                    else
                    {
                        ThemeSwitch.IsToggled = (currentTheme == 2);
                    }

                    ThemeSwitch.Toggled += OnThemeSwitchToggled;
                }
            }
            catch { }
        }

        private void OnThemeSwitchToggled(object? sender, ToggledEventArgs e)
        {
            var themeService = App.Services?.GetService<IThemeService>() ?? new ThemeService();
            int newThemeMode = e.Value ? 2 : 1;
            if (themeService.CurrentTheme != newThemeMode)
            {
                themeService.CurrentTheme = newThemeMode;
            }
        }

        protected override async void OnAppearing()
        {
            try
            {
                base.OnAppearing();
                SyncThemeSwitchState();
                
                this.Opacity = 0;
                this.TranslationY = 20;
                MainThread.BeginInvokeOnMainThread(async () => {
                    await Task.Delay(100);
                    await Task.WhenAll(
                        this.FadeToAsync(1, 400, Easing.CubicOut),
                        this.TranslateToAsync(0, 0, 400, Easing.CubicOut)
                    );
                });

                if (_viewModel.LoadDataCommand.CanExecute(null))
                {
                    await _viewModel.LoadDataCommand.ExecuteAsync(null);
                }
            }
            catch (Exception ex)
            {
                SignLanguageApp.Helpers.GlobalExceptionHandler.HandleException(ex);
            }
        }

        protected override void OnDisappearing()
        {
            base.OnDisappearing();
            _viewModel.OnDisappearing();
        }
    }
}
