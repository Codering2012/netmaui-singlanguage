using CommunityToolkit.Mvvm.ComponentModel;
using CommunityToolkit.Mvvm.Input;
using SignLanguageApp.Services;
using System.Collections.ObjectModel;

namespace SignLanguageApp.ViewModels;

public partial class ProgressReportsViewModel : ObservableObject
{
    private readonly IAnalyticsService _analyticsService;

    [ObservableProperty]
    public partial WeeklyReport? Report { get; set; }

    [ObservableProperty]
    public partial bool IsBusy { get; set; }

    public ProgressReportsViewModel(IAnalyticsService analyticsService)
    {
        _analyticsService = analyticsService;
    }

    public async Task InitializeAsync()
    {
        IsBusy = true;
        try
        {
            Report = await _analyticsService.GetWeeklyReportAsync();
        }
        finally
        {
            IsBusy = false;
        }
    }

    [RelayCommand]
    private async Task Back()
    {
        await Helpers.NavigationHelper.SafeNavigateAsync("..");
    }
}

