namespace SignLanguageApi.Dtos
{
    public class PersonalizedRecommendationDto
    {
        public int RecommendedLessonId { get; set; }
        public string LessonTitle { get; set; } = string.Empty;
        public string LessonDescription { get; set; } = string.Empty;
        public int CategoryId { get; set; }
        public string CategoryName { get; set; } = string.Empty;
        public string Reason { get; set; } = string.Empty;
        public double CurrentProgress { get; set; }
        public string Difficulty { get; set; } = string.Empty;
    }
}
