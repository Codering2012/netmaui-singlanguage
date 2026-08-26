namespace SignLanguageApi.Dtos
{
    public class MediaDto
    {
        public string FileName { get; set; } = string.Empty;
        public string Url { get; set; } = string.Empty;
        public string Type { get; set; } = "image"; // image or video
        public DateTime UploadedAt { get; set; } = DateTime.UtcNow;
    }
}
