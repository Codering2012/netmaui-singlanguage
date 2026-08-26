namespace SignLanguageApi.Dtos
{
    public class SignerCreditDto
    {
        public string SignerName { get; set; } = string.Empty;
        public string AvatarUrl { get; set; } = string.Empty;
        public string SocialLinks { get; set; } = string.Empty;
        public string LicenseType { get; set; } = string.Empty;
        public int ContributedVideosCount { get; set; }
        public string Bio { get; set; } = string.Empty;
        public string SourceUrl { get; set; } = string.Empty;
    }
}
