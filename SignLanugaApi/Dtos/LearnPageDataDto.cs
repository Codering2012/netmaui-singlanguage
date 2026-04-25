namespace SignLanguageApi.Dtos
{
    public class LearnPageDataDto
    {
        public List<LessonCategoryDto> Categories { get; set; } = new();

        public List<LessonDto> Lessons { get; set; } = new();

        public List<AchievementBadgeDto> Achievements { get; set; } = new();

        public List<SpacedRepetitionLessonDto> DailyReviewLessons { get; set; } = new();

        public int TotalXp { get; set; } = 0;

        public int CurrentStreak { get; set; } = 0;

        public double ProgressPercentage { get; set; } = 0;

        public string RecommendationReason { get; set; } = "Continue learning sign language at your own pace!";

        public int DailyGoalCompleted { get; set; } = 0;

        public int DailyGoalTotal { get; set; } = 5;

        public double DailyGoalProgress => DailyGoalTotal > 0 ? (DailyGoalCompleted / (double)DailyGoalTotal) * 100 : 0;

        public int TomorrowReviewCount { get; set; } = 0;

        public int ThisWeekReviewCount { get; set; } = 0;

        public double UpcomingReviewProgress { get; set; } = 0;
    }
}
