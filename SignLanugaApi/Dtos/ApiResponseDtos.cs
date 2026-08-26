namespace SignLanguageApi.Dtos
{
    public class ApiMessageDto
    {
        public string message { get; set; } = string.Empty;
    }

    public class AuthTokenResponseDto
    {
        public string token { get; set; } = string.Empty;
        public string refreshToken { get; set; } = string.Empty;
        public string userId { get; set; } = string.Empty;
        public string name { get; set; } = string.Empty;
    }

    public class RateLimitResponseDto
    {
        public string message { get; set; } = string.Empty;
        public int retryAfterSeconds { get; set; }
    }
}
