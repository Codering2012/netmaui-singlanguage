namespace SignLanguageApi.Data
{
    public class UserActivity
    {
        public int Id { get; set; }
        public string UserId { get; set; } = string.Empty;
        public string ActivityType { get; set; } = string.Empty; // "Lesson", "Quiz", "Practice"
        public int PointsGained { get; set; }
        public DateTime Timestamp { get; set; }
        public string Description { get; set; } = string.Empty;
    }
}
