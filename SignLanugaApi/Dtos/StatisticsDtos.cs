namespace SignLanguageApi.Dtos
{
    public class UserStatsDto
    {
        public int TotalXp { get; set; }
        public int LearningStreak { get; set; }
        public int LessonsCompleted { get; set; }
        public List<DailyXpDto> WeeklyXp { get; set; } = new();
        public List<CategoryProgressDto> CategoryProgress { get; set; } = new();
    }

    public class DailyXpDto
    {
        public string Date { get; set; } = string.Empty;
        public int Xp { get; set; }
    }

    public class CategoryProgressDto
    {
        public string CategoryName { get; set; } = string.Empty;
        public double Progress { get; set; }
    }
}
