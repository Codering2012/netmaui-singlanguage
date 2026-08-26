using SignLanguageApp.Model;
using System.Text.Json;

namespace SignLanguageApp.Services;

public interface IAnalyticsService
{
    Task<WeeklyReport> GetWeeklyReportAsync();
    Task TrackSessionAsync(int durationSeconds, int correctCount, int totalCount);
}

public class WeeklyReport
{
    public List<DayActivity> DailyActivity { get; set; } = new();
    public double AverageAccuracy { get; set; }
    public int TotalTimeSpentMinutes { get; set; }
    public string TopMissedSign { get; set; } = string.Empty;
}

public class DayActivity
{
    public string DayName { get; set; } = string.Empty;
    public int MinutesSpent { get; set; }
    public double Accuracy { get; set; }
}

public class AnalyticsService : IAnalyticsService
{
    private readonly IDatabaseService _databaseService;

    public AnalyticsService(IDatabaseService databaseService)
    {
        _databaseService = databaseService;
    }

    public async Task<WeeklyReport> GetWeeklyReportAsync()
    {
        var performances = await _databaseService.GetAllSignPerformancesAsync();
        
        var report = new WeeklyReport
        {
            AverageAccuracy = performances.Any() ? performances.Average(p => (double)p.CorrectCount / (p.CorrectCount + p.IncorrectCount + 1)) : 0,
            TotalTimeSpentMinutes = 120, // Mocked for now
            TopMissedSign = performances.OrderByDescending(p => p.IncorrectCount).FirstOrDefault()?.SignId ?? "None"
        };

        // Mocking daily activity for visualization
        var days = new[] { "Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun" };
        var random = new Random();
        foreach (var day in days)
        {
            report.DailyActivity.Add(new DayActivity
            {
                DayName = day,
                MinutesSpent = random.Next(10, 45),
                Accuracy = 0.7 + (random.NextDouble() * 0.25)
            });
        }

        return report;
    }

    public async Task TrackSessionAsync(int durationSeconds, int correctCount, int totalCount)
    {
        // Log session to local storage for future report generation
        await Task.CompletedTask;
    }
}
