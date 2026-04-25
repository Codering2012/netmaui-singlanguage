namespace SignLanguageApi.Dtos
{
    public class VideoDto
    {
        public int Id { get; set; }
        public string FileName { get; set; } = string.Empty;
        public string Title { get; set; } = string.Empty;
        public string Category { get; set; } = string.Empty;
        public string Path { get; set; } = string.Empty;
        public int Likes { get; set; }
        public int Views { get; set; }
    }
}
