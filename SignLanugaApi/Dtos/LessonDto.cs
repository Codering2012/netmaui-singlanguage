namespace SignLanguageApi.Dtos
{
    public class LessonUiLayoutDto
    {
        public string FileName { get; set; } = "LessonView.xaml";

        public string XamlContent { get; set; } = string.Empty;

        public string CodeBehindContent { get; set; } = string.Empty;
    }

    public class LessonDataDto
    {
        public int DurationSeconds { get; set; }

        public string Difficulty { get; set; } = "Beginner";

        public double CompletionPercentage { get; set; } = 0;

        public string InstructorName { get; set; } = string.Empty;

        public int CategoryId { get; set; }

        public LessonUiLayoutDto UiLayout { get; set; } = new();
    }

    public class LessonDto
    {
        public int Id { get; set; }

        public string Title { get; set; } = string.Empty;

        public string Description { get; set; } = string.Empty;

        public string? Thumbnail { get; set; }

        public int DurationSeconds { get; set; }

        public string Difficulty { get; set; } = "Beginner";

        public double CompletionPercentage { get; set; } = 0;

        public string InstructorName { get; set; } = string.Empty;

        public int CategoryId { get; set; }

        public LessonDataDto Data { get; set; } = new();
    }
}
