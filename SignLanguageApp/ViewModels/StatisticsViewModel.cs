using CommunityToolkit.Mvvm.ComponentModel;
using CommunityToolkit.Mvvm.Input;
using SignLanguageApp.Model;
using SignLanguageApp.Services;
using System.Collections.ObjectModel;

namespace SignLanguageApp.ViewModels
{
    public partial class StatisticsViewModel : BaseViewModel
    {
        private readonly IApiService _apiService;

        [ObservableProperty]
        public partial int TotalXp { get; set; }

        [ObservableProperty]
        public partial int LearningStreak { get; set; }

        [ObservableProperty]
        public partial int LessonsCompleted { get; set; }

        public ObservableCollection<DailyXpDto> WeeklyXp { get; } = new();
        public ObservableCollection<CategoryProgressDto> CategoryProgress { get; } = new();

        public StatisticsViewModel(IApiService apiService)
        {
            _apiService = apiService;
            Title = "Statistics";
        }

        [RelayCommand]
        private async Task LoadStatsAsync()
        {
            await RunSafeAsync(async () =>
            {
                var stats = await _apiService.GetUserStatsAsync();
                if (stats != null)
                {
                    TotalXp = stats.TotalXP;
                    LearningStreak = stats.CurrentStreak;
                    LessonsCompleted = stats.TotalProgress;

                    WeeklyXp.Clear();
                    // foreach (var day in stats.WeeklyXp)
                    //     WeeklyXp.Add(day);

                    CategoryProgress.Clear();
                    // foreach (var category in stats.CategoryProgress)
                    //     CategoryProgress.Add(category);
                }
            });
        }
    }
}



