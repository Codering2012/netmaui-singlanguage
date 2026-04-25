namespace SignLanguageApi.Dtos
{
    public class SpacedRepetitionLessonDto
    {
        public int Id { get; set; }

        public string Title { get; set; } = string.Empty;

        public string DueDate { get; set; } = string.Empty;

        public int RepetitionCount { get; set; } = 0;

        public double RetentionPercentage { get; set; } = 0;

        public bool IsReviewDue { get; set; } = false;

        public int LessonId { get; set; }
    }
}
