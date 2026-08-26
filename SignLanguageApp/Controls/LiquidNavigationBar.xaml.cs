using Microsoft.Maui.Controls;
using Microsoft.Maui.Graphics;
using System;
using System.Threading.Tasks;
using MauiPath = Microsoft.Maui.Controls.Shapes.Path;

namespace SignLanguageApp.Controls;

public partial class LiquidNavigationBar : ContentView
{
    public LiquidNavigationBar()
    {
        InitializeComponent();
        Loaded += OnLiquidNavLoaded;
        Unloaded += OnLiquidNavUnloaded;
    }

    private void OnLiquidNavLoaded(object? sender, EventArgs e)
    {
        if (Shell.Current != null)
        {
            Shell.Current.Navigated += OnShellNavigated;
            UpdateActiveTabState(Shell.Current.CurrentState?.Location?.OriginalString);
        }
    }

    private void OnLiquidNavUnloaded(object? sender, EventArgs e)
    {
        if (Shell.Current != null)
        {
            Shell.Current.Navigated -= OnShellNavigated;
        }
    }

    private void OnShellNavigated(object? sender, ShellNavigatedEventArgs e)
    {
        UpdateActiveTabState(e.Current?.Location?.OriginalString);
    }

    private async void OnTabTapped(object sender, TappedEventArgs e)
    {
        if (e.Parameter is string route)
        {
            if (sender is VisualElement container)
            {
                await container.ScaleToAsync(1.15, 80, Easing.CubicOut);
                await container.ScaleToAsync(1.0, 100, Easing.CubicIn);
            }

            if (Shell.Current != null)
            {
                await Shell.Current.GoToAsync(route);
            }
        }
    }

    private void UpdateActiveTabState(string? location)
    {
        location ??= string.Empty;
        location = location.ToLowerInvariant();

        string activeTab = "home";
        if (location.Contains("learn"))
            activeTab = "learn";
        else if (location.Contains("translation") || location.Contains("camera"))
            activeTab = "camera";
        else if (location.Contains("dictionary"))
            activeTab = "dict";
        else if (location.Contains("profile") || location.Contains("account"))
            activeTab = "profile";
        else if (location.Contains("home"))
            activeTab = "home";

        SetTabState(PillHome, TextHome, IconHome, activeTab == "home");
        SetTabState(PillLearn, TextLearn, IconLearn, activeTab == "learn");
        SetTabState(PillCamera, TextCamera, IconCamera, activeTab == "camera");
        SetTabState(PillDict, TextDict, IconDict, activeTab == "dict");
        SetTabState(PillProfile, TextProfile, IconProfile, activeTab == "profile");
    }

    private void SetTabState(Border pill, Label text, MauiPath icon, bool isActive)
    {
        if (Application.Current?.Resources == null) return;

        var resources = Application.Current.Resources;

        Color activeColor = resources.TryGetValue("NavBarActiveColor", out var ac) && ac is Color c1 ? c1 : Color.FromArgb("#0284C7");
        Color inactiveColor = resources.TryGetValue("NavBarInactiveColor", out var ic) && ic is Color c2 ? c2 : Color.FromArgb("#334155");
        Color pillColor = resources.TryGetValue("NavBarActivePillColor", out var pc) && pc is Color c3 ? c3 : Color.FromArgb("#250284C7");
        Color pillStroke = resources.TryGetValue("NavBarActivePillStroke", out var ps) && ps is Color c4 ? c4 : Color.FromArgb("#500284C7");

        if (isActive)
        {
            pill.BackgroundColor = pillColor;
            pill.Stroke = pillStroke;
            text.TextColor = activeColor;
            text.FontAttributes = FontAttributes.Bold;
            icon.Fill = new SolidColorBrush(activeColor);
        }
        else
        {
            pill.BackgroundColor = Colors.Transparent;
            pill.Stroke = Colors.Transparent;
            text.TextColor = inactiveColor;
            text.FontAttributes = FontAttributes.None;
            icon.Fill = new SolidColorBrush(inactiveColor);
        }
    }
}
