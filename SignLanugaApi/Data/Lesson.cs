namespace SignLanguageApi.Data
{
    public class Lesson
    {
        public int Id { get; set; }

        public string Title { get; set; } = string.Empty;

        public string Description { get; set; } = string.Empty;

        public string? ThumbnailUrl { get; set; }

        public int DurationSeconds { get; set; }

        public string InstructorName { get; set; } = string.Empty;

        public string Difficulty { get; set; } = "Beginner";

        public int ViewCount { get; set; } = 0;

        public int CategoryId { get; set; }

        public LessonCategory? Category { get; set; }

        public ICollection<UserLesson> UserLessons { get; set; } = new List<UserLesson>();
    }
}
