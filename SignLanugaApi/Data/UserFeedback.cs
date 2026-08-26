namespace SignLanguageApi.Data
{
    public class UserFeedback
    {
        public int Id { get; set; }
        public string UserId { get; set; } = string.Empty;
        public string Subject { get; set; } = string.Empty;
        public string Message { get; set; } = string.Empty;
        public int Rating { get; set; } // 1-5
        public DateTime SubmittedAt { get; set; }
    }
}
