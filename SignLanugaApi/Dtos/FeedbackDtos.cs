namespace SignLanguageApi.Dtos
{
    public class FeedbackRequestDto
    {
        public string Subject { get; set; } = string.Empty;
        public string Message { get; set; } = string.Empty;
        public int Rating { get; set; }
    }
}
