using Microsoft.Extensions.DependencyInjection;
using Microsoft.Maui.Controls.Shapes;
using SignLanguageApp.Services;
using SignLanguageApp.ViewModels;
using System.Linq;

namespace SignLanguageApp.Pages;

public sealed class StartupLoadingPage : ContentPage
{
    private readonly IServiceProvider _serviceProvider;
    private readonly Label _statusLabel;
    private readonly Label _stageLabel;
    private bool _initialized;

    public StartupLoadingPage(IServiceProvider serviceProvider)
    {
        _serviceProvider = serviceProvider;

        Content = new Grid
        {
            Background = new LinearGradientBrush(
                new GradientStopCollection
                {
                    new GradientStop(Color.FromArgb("#06111C"), 0.0f),
                    new GradientStop(Color.FromArgb("#10283D"), 0.5f),
                    new GradientStop(Color.FromArgb("#113752"), 1.0f)
                },
                new Point(0, 0),
                new Point(1, 1)),
            Padding = 24,
            Children =
            {
                new Border
                {
                    StrokeThickness = 0,
                    Background = Color.FromArgb("#1B2E42CC"),
                    StrokeShape = new RoundRectangle { CornerRadius = 24 },
                    Padding = 24,
                    VerticalOptions = LayoutOptions.Center,
                    HorizontalOptions = LayoutOptions.Center,
                    Content = new VerticalStackLayout
                    {
                        Spacing = 14,
                        WidthRequest = 280,
                        Children =
                        {
                            (_stageLabel = new Label
                            {
                                Text = "SignLanguage",
                                TextColor = Colors.White,
                                FontSize = 28,
                                FontAttributes = FontAttributes.Bold,
                                HorizontalTextAlignment = TextAlignment.Center
                            }),
                            new Label
                            {
                                Text = "Preparing your personalized learning space",
                                TextColor = Color.FromArgb("#BFD7E8"),
                                FontSize = 13,
                                HorizontalTextAlignment = TextAlignment.Center
                            },
                            new ActivityIndicator
                            {
                                IsRunning = true,
                                Color = Colors.White,
                                WidthRequest = 38,
                                HeightRequest = 38,
                                HorizontalOptions = LayoutOptions.Center
                            },
                            (_statusLabel = new Label
                            {
                                Text = "Preparing resources",
                                TextColor = Color.FromArgb("#A0BCCF"),
                                FontSize = 12,
                                HorizontalTextAlignment = TextAlignment.Center
                            })
                        }
                    }
                }
            }
        };
    }

    protected override async void OnAppearing()
    {
        base.OnAppearing();

        if (_initialized)
        {
            return;
        }

        _initialized = true;

        var nextRoot = await PrepareAndResolveRootAsync();
        var window = Application.Current?.Windows?.FirstOrDefault();
        if (window != null)
        {
            window.Page = nextRoot;
        }
    }

    private async Task<Page> PrepareAndResolveRootAsync()
    {
        try
        {
            await MainThread.InvokeOnMainThreadAsync(() =>
            {
                _stageLabel.Text = "SignLanguage";
                _statusLabel.Text = "Warming up services";
            });
            _ = _serviceProvider.GetService<ICacheService>();
            _ = _serviceProvider.GetService<IDatabaseService>();
            _ = _serviceProvider.GetService<IApiService>();

            await MainThread.InvokeOnMainThreadAsync(() => _statusLabel.Text = "Checking session");
            var authService = _serviceProvider.GetService<IAuthService>();
            var isAuthenticated = authService != null && await authService.IsAuthenticatedAsync();

            if (isAuthenticated)
            {
                var refreshSucceeded = await authService!.RefreshTokenAsync();
                if (refreshSucceeded)
                {
                    return new AppShell();
                }

                await authService.LogoutAsync();
                return new LoginShell();
            }

            return new LoginShell();
        }
        catch
        {
            return new LoginShell();
        }
    }
}
