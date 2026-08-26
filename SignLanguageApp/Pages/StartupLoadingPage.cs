using Microsoft.Extensions.DependencyInjection;
using Microsoft.Maui.Controls.Shapes;
using SignLanguageApp.Services;
using SignLanguageApp.ViewModels;
using System.Linq;
using SignLanguageApp.Helpers;

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
        try
        {
            base.OnAppearing();

            if (_initialized)
            {
                return;
            }

            _initialized = true;

            FileLogger.Log("[STARTUP] PrepareAndResolveRootAsync completed!");
            var nextRoot = await PrepareAndResolveRootAsync();
            FileLogger.Log("[STARTUP] nextRoot resolved!");
            
            if (Application.Current != null)
            {
                FileLogger.Log("[STARTUP] Changing MainPage...");
                MainThread.BeginInvokeOnMainThread(async () => 
                {
                    try
                    {
                        FileLogger.Log("[STARTUP] Inside BeginInvokeOnMainThread...");
                        await Task.Delay(50);
                        FileLogger.Log("[STARTUP] Task.Delay finished...");
                        
                        var window = this.Window ?? Application.Current.Windows.FirstOrDefault();
                        FileLogger.Log($"[STARTUP] Window resolved: {window != null}");
                        
                        FileLogger.Log("[STARTUP] ABOUT TO ASSIGN MAINPAGE!");
#pragma warning disable CS0618
                        Application.Current.MainPage = nextRoot;
#pragma warning restore CS0618
                        FileLogger.Log("[STARTUP] MainPage changed successfully!");
                    }
                    catch (Exception ex)
                    {
                        FileLogger.Log($"[FATAL ERROR IN LAMBDA]: {ex}");
                        SignLanguageApp.Helpers.GlobalExceptionHandler.HandleException(ex);
                    }
                });
            }
        }
        catch (Exception ex)
        {
            SignLanguageApp.Helpers.GlobalExceptionHandler.HandleException(ex);
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
                System.Diagnostics.Debug.WriteLine("[STARTUP] Calling RefreshTokenAsync...");
                var refreshSucceeded = await authService!.RefreshTokenAsync();
                System.Diagnostics.Debug.WriteLine($"[STARTUP] RefreshTokenAsync returned: {refreshSucceeded}");
                if (refreshSucceeded)
                {
                    FileLogger.Log("[STARTUP] Instantiating AppShell...");
                    var shell = await MainThread.InvokeOnMainThreadAsync(() => new AppShell())
                                                .WithTimeout("AppShell Constructor", 5000);
                    FileLogger.Log("[STARTUP] AppShell instantiated successfully!");
                    return shell;
                }

                await authService.LogoutAsync();
                return await MainThread.InvokeOnMainThreadAsync(() => new LoginShell());
            }

            return await MainThread.InvokeOnMainThreadAsync(() => new LoginShell());
        }
        catch (Exception ex)
        {
            throw; // Let OnAppearing catch this and handle it properly
        }
    }
}
