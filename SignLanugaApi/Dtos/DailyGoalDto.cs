namespace SignLanguageApi.Dtos
{
    public class DailyGoalDto
    {
        public int TotalReviewsDue { get; set; }
        public int CompletedToday { get; set; }
        public int DailyGoal { get; set; } = 5;
        public double ProgressPercentage { get; set; }
    }

    public class UpcomingReviewsDto
    {
        public int DueToday { get; set; }
        public int DueTomorrow { get; set; }
        public int DueThisWeek { get; set; }
        public int Overdue { get; set; }
    }
}
