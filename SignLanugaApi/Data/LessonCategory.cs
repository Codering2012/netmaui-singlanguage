namespace SignLanguageApi.Data
{
    public class LessonCategory
    {
        public int Id { get; set; }

        public string Title { get; set; } = string.Empty;

        public string Description { get; set; } = string.Empty;

        public string Difficulty { get; set; } = "Beginner";

        public string? IconUrl { get; set; }

        public ICollection<Lesson> Lessons { get; set; } = new List<Lesson>();

        public DateTime CreatedAt { get; set; } = DateTime.UtcNow;
    }
}
