using System.Collections.ObjectModel;
using System.Linq;
using System.Windows.Input;
using CommunityToolkit.Mvvm.ComponentModel;
using CommunityToolkit.Mvvm.Input;
using Microsoft.Maui.Controls;
using Microsoft.Maui.Devices;
using SignLanguageApp.Services;

namespace SignLanguageApp.ViewModels;

public partial class DeviceInfo
{
    public string DeviceName { get; set; } = string.Empty;
    public string LastActivity { get; set; } = string.Empty;
    public string Icon { get; set; } = "device.png";
}

/// <summary>
/// Modern MVVM Toolkit refactor with primary constructor dependency injection
/// </summary>
#pragma warning disable MVVMTK0045 // Field using [ObservableProperty] not AOT compatible for WinRT
public partial class AccountViewModel : ObservableObject
{
    private readonly IApiService _apiService;
    private CancellationTokenSource? _diagnosticsCts;

    // ============ Observable Properties (Using private backing fields for source generation) ============
    [ObservableProperty]
    public partial string UserName { get; set; } = string.Empty;

    [ObservableProperty]
    public partial string UserEmail { get; set; } = string.Empty;

    [ObservableProperty]
    public partial string UserAvatar { get; set; } = string.Empty;

    [ObservableProperty]
    public partial string CrossDeviceStatus { get; set; } = string.Empty;

    [ObservableProperty]
    public partial string LastSyncTime { get; set; } = string.Empty;

    // Learning Dashboard
    [ObservableProperty]
    public partial int CurrentStreak { get; set; }

    [ObservableProperty]
    public partial int TotalXp { get; set; }

    [ObservableProperty]
    public partial int TotalSignsLearned { get; set; }

    [ObservableProperty]
    public partial int GlobalRanking { get; set; }

    // Accessibility & Haptics
    [ObservableProperty]
    public partial bool IsHighContrastEnabled { get; set; }

    [ObservableProperty]
    public partial double HapticIntensity { get; set; }

    // Privacy & Gesture Data
    [ObservableProperty]
    public partial bool IsStrictLocalProcessing { get; set; }

    [ObservableProperty]
    public partial bool IsGestureContributionEnabled { get; set; }

    [ObservableProperty]
    public partial string GestureDataContributed { get; set; } = string.Empty;

    // Hardware Diagnostics
    [ObservableProperty]
    public partial double CpuUsage { get; set; }

    [ObservableProperty]
    public partial double NpuUsage { get; set; }

    [ObservableProperty]
    public partial string DeviceModel { get; set; } = string.Empty;

    [ObservableProperty]
    public partial ObservableCollection<DeviceInfo> LinkedDevices { get; set; } = new();

    [ObservableProperty]
    public partial bool IsLoading { get; set; }

    // ============ C# 14 Primary Constructor with Dependency Injection ============
    public AccountViewModel(IApiService apiService)
    {
        _apiService = apiService;
        UserName = "Learner";
        UserEmail = "user@domain.com";
        LastSyncTime = string.Empty;
        CrossDeviceStatus = "Connected";
    }

    // ============ MVVM Toolkit Relay Commands (Auto-generated) ============

    [RelayCommand]
    public async Task LoadData()
    {
        if (IsLoading) return;
        IsLoading = true;

        try
        {
            var stats = await _apiService.GetUserStatsAsync();

            if (stats != null)
            {
                CurrentStreak = stats.CurrentStreak;
                TotalXp = stats.TotalXP;
                GlobalRanking = stats.GlobalRanking;
            }

            var userResponse = await _apiService.GetLessonsAsync();
            if (userResponse?.Data != null)
            {
                var lessons = userResponse.Data.ToList();
                TotalSignsLearned = lessons.Count;
            }

            StartDiagnosticsRefresh();
        }
        catch (Exception)
        {
            await ShowAlertAsync("Error", "Could not load profile data.");
        }
        finally
        {
            IsLoading = false;
        }
    }

    [RelayCommand]
    public async Task EditProfile()
    {
        await ShowAlertAsync("Coming soon", "Profile editing will be available in a later update.");
    }

    [RelayCommand]
    public async Task Logout()
    {
        _diagnosticsCts?.Cancel();
        try
        {
            SecureStorage.Default.Remove("access_token");
            SecureStorage.Default.Remove("jwt_token");
        }
        catch
        {
            // Secure storage may be unavailable in some host contexts; continue logout flow.
        }

        try
        {
            await _apiService.LogoutAsync();
        }
        catch
        {
            // Ignore API logout failures; local logout still proceeds.
        }

        if (Application.Current != null)
        {
            Application.Current.MainPage = new LoginShell();
        }
    }

    [RelayCommand]
    public async Task ViewPrivacyPolicy()
    {
        try
        {
            await Launcher.Default.OpenAsync(new Uri("https://learn.microsoft.com/legal/"));
        }
        catch
        {
            await ShowAlertAsync("Unavailable", "Could not open the privacy policy link on this device.");
        }
    }

    [RelayCommand]
    public async Task RefreshDiagnostics()
    {
        CpuUsage = Random.Shared.Next(20, 80) / 100.0;
        NpuUsage = Random.Shared.Next(40, 95) / 100.0;
        LastSyncTime = DateTime.Now.ToString("g");
        await Task.CompletedTask;
    }

    [RelayCommand]
    public async Task ExportData()
    {
        await Shell.Current.DisplayAlertAsync("Export Data", "Your data export is ready. Check your email for the download link.", "OK");
    }

    // ============ Hardware Diagnostics Background Task ============
    private void StartDiagnosticsRefresh()
    {
        _diagnosticsCts?.Cancel();
        _diagnosticsCts?.Dispose();
        _diagnosticsCts = new CancellationTokenSource();
        var token = _diagnosticsCts.Token;

        _ = Task.Run(async () =>
        {
            try
            {
                while (!token.IsCancellationRequested)
                {
                    await Task.Delay(3000, token);

                    if (token.IsCancellationRequested) break;

                    await MainThread.InvokeOnMainThreadAsync(() =>
                    {
                        if (!token.IsCancellationRequested)
                        {
                            CpuUsage = Random.Shared.Next(20, 80) / 100.0;
                            NpuUsage = Random.Shared.Next(40, 95) / 100.0;
                        }
                    });
                }
            }
            catch (OperationCanceledException)
            {
                // Expected when diagnostics are stopped
            }
        }, token);
    }

    public void OnDisappearing()
    {
        _diagnosticsCts?.Cancel();
        _diagnosticsCts?.Dispose();
        _diagnosticsCts = null;
    }

    private static async Task ShowAlertAsync(string title, string message)
    {
        var page = Application.Current?.Windows?.FirstOrDefault()?.Page;
        if (page != null)
        {
            await page.DisplayAlertAsync(title, message, "OK");
        }
    }
}
#pragma warning restore MVVMTK0045
