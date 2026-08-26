using Microsoft.Maui.Controls;
using SignLanguageApp.ViewModels;
using System.Threading;
using System.Threading.Tasks;
using System;

namespace SignLanguageApp.Pages;

public partial class LoginPage : ContentPage
{
    private System.Threading.Timer? _slideshowTimer;
    private int _currentSlideIndex = 0;
    private bool _isAnimating = false;

    public LoginPage()
    {
        try
        {
            InitializeComponent();
            BindingContext = App.Services.GetService<LoginViewModel>();
        }
        catch (System.Exception ex)
        {
            SignLanguageApp.Helpers.GlobalExceptionHandler.HandleException(ex);
        }
    }

    public LoginPage(LoginViewModel viewModel)
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

    private void OnThemeToggleTapped(object sender, EventArgs e)
    {
        var themeService = App.Services?.GetService<SignLanguageApp.Services.IThemeService>();
        if (themeService != null)
        {
            var current = themeService.CurrentTheme;
            if (current == 0) current = Application.Current?.RequestedTheme == AppTheme.Dark ? 2 : 1;
            
            themeService.CurrentTheme = current == 2 ? 1 : 2;
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

        StartSlideshowTimer();
    }

    protected override async void OnDisappearing()
    {
        base.OnDisappearing();
        StopSlideshowTimer();
        await this.FadeToAsync(0, 200, Easing.CubicIn);
    }

    private void StartSlideshowTimer()
    {
        StopSlideshowTimer();
        _slideshowTimer = new System.Threading.Timer(_ =>
        {
            MainThread.BeginInvokeOnMainThread(() =>
            {
                int nextIndex = (_currentSlideIndex + 1) % 3;
                SwitchToSlide(nextIndex);
            });
        }, null, 4000, 4000);
    }

    private void StopSlideshowTimer()
    {
        _slideshowTimer?.Dispose();
        _slideshowTimer = null;
    }

    private void OnDot1Tapped(object sender, EventArgs e) => SwitchToSlide(0);
    private void OnDot2Tapped(object sender, EventArgs e) => SwitchToSlide(1);
    private void OnDot3Tapped(object sender, EventArgs e) => SwitchToSlide(2);

    private async void SwitchToSlide(int targetIndex)
    {
        if (_currentSlideIndex == targetIndex || _isAnimating) return;
        _isAnimating = true;

        try
        {
            Grid[] slides = new[] { Slide1, Slide2, Slide3 };
            Border[] dots = new[] { Dot1, Dot2, Dot3 };

            if (targetIndex < 0 || targetIndex >= slides.Length) return;

            var currentSlide = slides[_currentSlideIndex];
            var nextSlide = slides[targetIndex];

            _currentSlideIndex = targetIndex;

            // Update dots
            for (int i = 0; i < dots.Length; i++)
            {
                if (dots[i] != null)
                {
                    bool isActive = i == targetIndex;
                    dots[i].WidthRequest = isActive ? 24 : 8;
                    dots[i].Opacity = isActive ? 1.0 : 0.4;
                    if (Application.Current?.Resources != null)
                    {
                        if (isActive && Application.Current.Resources.TryGetValue("PrimaryColor", out var primary))
                            dots[i].BackgroundColor = (Color)primary;
                        else if (!isActive && Application.Current.Resources.TryGetValue("SecondaryTextColor", out var secondary))
                            dots[i].BackgroundColor = (Color)secondary;
                    }
                }
            }

            // Animate transition
            if (currentSlide != null && nextSlide != null)
            {
                nextSlide.Opacity = 0;
                nextSlide.TranslationX = 15;
                nextSlide.IsVisible = true;

                await Task.WhenAll(
                    currentSlide.FadeToAsync(0, 250, Easing.CubicIn),
                    currentSlide.TranslateToAsync(-15, 0, 250, Easing.CubicIn),
                    nextSlide.FadeToAsync(1, 300, Easing.CubicOut),
                    nextSlide.TranslateToAsync(0, 0, 300, Easing.CubicOut)
                );

                currentSlide.IsVisible = false;
                currentSlide.TranslationX = 0;
            }
        }
        catch { }
        finally
        {
            _isAnimating = false;
        }
    }

    protected override void OnSizeAllocated(double width, double height)
    {
        base.OnSizeAllocated(width, height);
        if (width < 800)
        {
            if (RightColumn != null) RightColumn.IsVisible = false;
            if (MainGrid != null && MainGrid.ColumnDefinitions.Count >= 2)
            {
                MainGrid.ColumnDefinitions[0].Width = new GridLength(1, GridUnitType.Star);
                MainGrid.ColumnDefinitions[1].Width = new GridLength(0);
            }
        }
        else
        {
            if (RightColumn != null) RightColumn.IsVisible = true;
            if (MainGrid != null && MainGrid.ColumnDefinitions.Count >= 2)
            {
                MainGrid.ColumnDefinitions[0].Width = new GridLength(450);
                MainGrid.ColumnDefinitions[1].Width = new GridLength(1, GridUnitType.Star);
            }
        }
    }
}