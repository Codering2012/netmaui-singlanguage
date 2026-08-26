using System;
using SQLite;

namespace SignLanguageApp.Model;

public class SignPerformance
{
    [PrimaryKey]
    public string SignId { get; set; } = string.Empty;
    public DateTime LastReviewed { get; set; }
    public DateTime NextReviewDate { get; set; }
    
    // SRS Parameters (SuperMemo-2 style)
    public int Interval { get; set; } = 0; // days
    public int Repetitions { get; set; } = 0;
    public double EaseFactor { get; set; } = 2.5;
    
    // Stats
    public int CorrectCount { get; set; }
    public int IncorrectCount { get; set; }
    public double AverageAccuracy { get; set; }
}
