namespace SignLanguageApi.Data
{
    public class UserLesson
    {
        public int Id { get; set; }

        public string UserId { get; set; } = string.Empty;

        public int LessonId { get; set; }

        public User? User { get; set; }

        public Lesson? Lesson { get; set; }

        public int CompletionPercentage { get; set; } = 0;

        public bool IsCompleted { get; set; } = false;

        public DateTime? CompletedAt { get; set; }

        public DateTime StartedAt { get; set; } = DateTime.UtcNow;

        public int TotalAttempts { get; set; } = 0;
    }
}
