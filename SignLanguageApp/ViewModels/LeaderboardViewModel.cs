using CommunityToolkit.Mvvm.ComponentModel;
using CommunityToolkit.Mvvm.Input;
using SignLanguageApp.Model;
using SignLanguageApp.Services;
using System.Collections.ObjectModel;

namespace SignLanguageApp.ViewModels
{
    public partial class LeaderboardViewModel : BaseViewModel
    {
        private readonly IApiService _apiService;

        public ObservableCollection<LeaderboardEntryDto> TopEntries { get; } = new();

        [ObservableProperty]
        public partial LeaderboardEntryDto? CurrentUserEntry { get; set; }

        public LeaderboardViewModel(IApiService apiService)
        {
            _apiService = apiService;
            Title = "Leaderboard";
        }

        [RelayCommand]
        public async Task LoadLeaderboardAsync()
        {
            IsBusy = true;
            try
            {
                var leaderboard = await _apiService.GetLeaderboardAsync();
                if (leaderboard != null)
                {
                    TopEntries.Clear();
                    foreach (var entry in leaderboard.TopEntries)
                    {
                        entry.AvatarUrl = _apiService.EnsureAbsoluteUrl(entry.AvatarUrl);
                        TopEntries.Add(entry);
                    }
                    if (leaderboard.CurrentUserEntry != null)
                    {
                        leaderboard.CurrentUserEntry.AvatarUrl = _apiService.EnsureAbsoluteUrl(leaderboard.CurrentUserEntry.AvatarUrl);
                        CurrentUserEntry = leaderboard.CurrentUserEntry;
                    }
                }
            }
            catch (Exception ex)
            {
                System.Diagnostics.Debug.WriteLine($"Error loading leaderboard: {ex.Message}");
            }
            finally
            {
                IsBusy = false;
            }
        }
    }
}

