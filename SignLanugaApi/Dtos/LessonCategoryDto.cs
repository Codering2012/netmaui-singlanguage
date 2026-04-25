namespace SignLanguageApi.Dtos
{
    public class LessonCategoryDto
    {
        public int Id { get; set; }

        public string Title { get; set; } = string.Empty;

        public string Description { get; set; } = string.Empty;

        public string Difficulty { get; set; } = "Beginner";

        public string? IconUrl { get; set; }

        public double Progress { get; set; } = 0;
    }
}
