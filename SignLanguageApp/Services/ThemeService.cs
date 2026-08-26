using Microsoft.Maui.ApplicationModel;
using SignLanguageApp.Resources.Styles;

namespace SignLanguageApp.Services;

public class ThemeService : IThemeService
{
    private const string ThemeKey = "AppThemePref";

    // 0 = System, 1 = Light, 2 = Dark
    public int CurrentTheme
    {
        get
        {
            try
            {
                return Preferences.Default.Get(ThemeKey, 0);
            }
            catch
            {
                return 0;
            }
        }
        set
        {
            try
            {
                Preferences.Default.Set(ThemeKey, value);
            }
            catch
            {
                // Fallback for headless environments
            }
            SetTheme(value);
        }
    }

    public void InitializeTheme()
    {
        SetTheme(CurrentTheme);
        
        // Listen to system theme changes if using System mode
        if (Application.Current != null)
        {
            Application.Current.RequestedThemeChanged += (s, e) =>
            {
                if (CurrentTheme == 0)
                {
                    ApplyTheme(e.RequestedTheme);
                }
            };
        }
    }

    public async void SetTheme(int themeMode)
    {
        var app = Application.Current;
        if (app == null) return;

        // Cinematic Transition: Fade out before swapping
        if (app.MainPage != null && !Microsoft.Maui.Controls.Application.Current.MainPage.IsSet(Microsoft.Maui.Controls.VisualElement.OpacityProperty))
        {
            await app.MainPage.FadeTo(0, 250, Easing.CubicIn);
        }
        else if (app.MainPage != null)
        {
            await app.MainPage.FadeTo(0, 250, Easing.CubicIn);
        }

        if (themeMode == 1)
            ApplyTheme(AppTheme.Light);
        else if (themeMode == 2)
            ApplyTheme(AppTheme.Dark);
        else
            ApplyTheme(app.PlatformAppTheme);
            
        app.UserAppTheme = themeMode switch
        {
            1 => AppTheme.Light,
            2 => AppTheme.Dark,
            _ => AppTheme.Unspecified
        };

        // Cinematic Transition: Fade back in
        if (app.MainPage != null)
        {
            await app.MainPage.FadeTo(1, 350, Easing.CubicOut);
        }
    }

    private void ApplyTheme(AppTheme theme)
    {
        var dictionaries = Application.Current?.Resources?.MergedDictionaries;
        if (dictionaries == null) return;

        // Remove existing theme dictionaries
        var existingThemes = dictionaries
            .Where(d => d is LightTheme or DarkTheme)
            .ToList();

        foreach (var existing in existingThemes)
        {
            dictionaries.Remove(existing);
        }

        // Add the appropriate theme dictionary by instantiating the typed class
        if (theme == AppTheme.Dark)
        {
            dictionaries.Add(new DarkTheme());
        }
        else
        {
            dictionaries.Add(new LightTheme());
        }
    }
}

