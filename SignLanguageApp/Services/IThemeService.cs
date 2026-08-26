namespace SignLanguageApp.Services;

public interface IThemeService
{
    int CurrentTheme { get; set; }
    void InitializeTheme();
    void SetTheme(int themeMode);
}
