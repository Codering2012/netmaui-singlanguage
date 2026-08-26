namespace SignLanguageApp.Services;

public interface IRewardService
{
    Task<List<string>> GetUnlockedBadgesAsync();
    Task<int> GetCurrentStreakAsync();
    Task TrackActivityAsync();
}

public class RewardService : IRewardService
{
    private const string StreakKey = "user_streak";
    private const string LastActivityKey = "last_activity";
    private const string BadgesKey = "user_badges";

    public async Task<List<string>> GetUnlockedBadgesAsync()
    {
        var badges = Preferences.Get(BadgesKey, "Newbie");
        return badges.Split(',').ToList();
    }

    public async Task<int> GetCurrentStreakAsync()
    {
        return Preferences.Get(StreakKey, 0);
    }

    public async Task TrackActivityAsync()
    {
        var lastActivity = Preferences.Get(LastActivityKey, DateTime.MinValue.Ticks);
        var lastDate = new DateTime(lastActivity).Date;
        var today = DateTime.Today;

        if (lastDate < today)
        {
            if (lastDate == today.AddDays(-1))
            {
                var streak = Preferences.Get(StreakKey, 0);
                Preferences.Set(StreakKey, streak + 1);
            }
            else if (lastDate < today.AddDays(-1))
            {
                Preferences.Set(StreakKey, 1);
            }
            Preferences.Set(LastActivityKey, today.Ticks);
        }
        await Task.CompletedTask;
    }
}
