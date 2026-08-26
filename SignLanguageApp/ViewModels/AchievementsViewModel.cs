using CommunityToolkit.Mvvm.ComponentModel;
using CommunityToolkit.Mvvm.Input;
using SignLanguageApp.Model;
using SignLanguageApp.Services;
using System.Collections.ObjectModel;

namespace SignLanguageApp.ViewModels
{
    public partial class AchievementsViewModel : BaseViewModel
    {
        private readonly IApiService _apiService;

        public ObservableCollection<AchievementBadgeDto> Achievements { get; } = new();

        [ObservableProperty]
        public partial int UnlockedCount { get; set; }

        [ObservableProperty]
        public partial double Progress { get; set; }

        public AchievementsViewModel(IApiService apiService)
        {
            _apiService = apiService;
            Title = "My Achievements";
            _ = LoadAchievementsAsync();
        }

        [RelayCommand]
        public async Task GoBack()
        {
            await Helpers.NavigationHelper.SafeNavigateAsync("..");
        }

        [RelayCommand]
        public async Task LoadAchievementsAsync()
        {
            IsBusy = true;
            try
            {
                var list = await _apiService.GetAchievementsAsync();
                if (list != null)
                {
                    Achievements.Clear();
                    int unlocked = 0;
                    foreach (var item in list)
                    {
                        Achievements.Add(item);
                        if (item.IsUnlocked) unlocked++;
                    }
                    UnlockedCount = unlocked;
                    Progress = list.Any() ? (double)unlocked / list.Count() : 0;
                }
            }
            catch (Exception ex)
            {
                System.Diagnostics.Debug.WriteLine($"Error loading achievements: {ex.Message}");
            }
            finally
            {
                IsBusy = false;
            }
        }
    }
}


