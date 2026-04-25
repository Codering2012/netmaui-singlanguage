namespace SignLanguageApi.Data
{
    public class SpacedRepetitionLesson
    {
        public int Id { get; set; }

        public string UserId { get; set; } = string.Empty;

        public int LessonId { get; set; }

        public User? User { get; set; }

        public Lesson? Lesson { get; set; }

        public DateTime DueDate { get; set; } = DateTime.UtcNow;

        public int RepetitionCount { get; set; } = 0;

        public double RetentionPercentage { get; set; } = 0;

        public int Interval { get; set; } = 1;

        public double EaseFactor { get; set; } = 2.5;

        public DateTime LastReviewedAt { get; set; } = DateTime.UtcNow;

        public bool IsReviewDue => DateTime.UtcNow >= DueDate;
    }
}
