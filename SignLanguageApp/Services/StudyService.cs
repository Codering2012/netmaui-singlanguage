using SignLanguageApp.Model;

namespace SignLanguageApp.Services;

public interface IStudyService
{
    Task UpdateSignPerformanceAsync(string signId, bool wasCorrect, double accuracy = 1.0);
    Task<List<string>> GetReviewQueueAsync(int limit = 10);
    Task<List<string>> GetForgottenSignsAsync();
}

public class StudyService : IStudyService
{
    private readonly IDatabaseService _databaseService;

    public StudyService(IDatabaseService databaseService)
    {
        _databaseService = databaseService;
    }

    public async Task UpdateSignPerformanceAsync(string signId, bool wasCorrect, double accuracy = 1.0)
    {
        var performance = await _databaseService.GetSignPerformanceAsync(signId) 
                          ?? new SignPerformance { SignId = signId };

        // Simple SM-2 implementation
        int grade = CalculateGrade(wasCorrect, accuracy);
        
        if (grade >= 3)
        {
            if (performance.Repetitions == 0)
                performance.Interval = 1;
            else if (performance.Repetitions == 1)
                performance.Interval = 6;
            else
                performance.Interval = (int)Math.Round(performance.Interval * performance.EaseFactor);

            performance.Repetitions++;
        }
        else
        {
            performance.Repetitions = 0;
            performance.Interval = 1;
        }

        performance.EaseFactor = performance.EaseFactor + (0.1 - (5 - grade) * (0.08 + (5 - grade) * 0.02));
        if (performance.EaseFactor < 1.3) performance.EaseFactor = 1.3;

        performance.LastReviewed = DateTime.UtcNow;
        performance.NextReviewDate = DateTime.UtcNow.AddDays(performance.Interval);
        
        if (wasCorrect) performance.CorrectCount++;
        else performance.IncorrectCount++;

        await _databaseService.SaveSignPerformanceAsync(performance);
    }

    public async Task<List<string>> GetReviewQueueAsync(int limit = 10)
    {
        var performances = await _databaseService.GetAllSignPerformancesAsync();
        return performances
            .Where(p => p.NextReviewDate <= DateTime.UtcNow)
            .OrderBy(p => p.NextReviewDate)
            .Take(limit)
            .Select(p => p.SignId)
            .ToList();
    }

    public async Task<List<string>> GetForgottenSignsAsync()
    {
        var performances = await _databaseService.GetAllSignPerformancesAsync();
        // Definition of "forgotten": has performance record but accuracy is dropping or not reviewed in a long time
        return performances
            .Where(p => p.IncorrectCount > p.CorrectCount * 0.5 || (DateTime.UtcNow - p.LastReviewed).TotalDays > 30)
            .OrderByDescending(p => p.IncorrectCount)
            .Select(p => p.SignId)
            .ToList();
    }

    private int CalculateGrade(bool wasCorrect, double accuracy)
    {
        if (!wasCorrect) return 1;
        if (accuracy < 0.6) return 2;
        if (accuracy < 0.8) return 3;
        if (accuracy < 0.95) return 4;
        return 5;
    }
}
