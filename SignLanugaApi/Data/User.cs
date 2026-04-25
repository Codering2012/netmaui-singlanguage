namespace SignLanguageApi.Data
{
    public class User
    {
        public string Id { get; set; } = Guid.NewGuid().ToString();

        public string Email { get; set; } = string.Empty;

        public string Name { get; set; } = string.Empty;

        public string PasswordHash { get; set; } = string.Empty;

        public string? AvatarUrl { get; set; }

        public int LearningStreak { get; set; } = 0;

        public int TotalXp { get; set; } = 0;

        public DateTime CreatedAt { get; set; } = DateTime.UtcNow;

        public DateTime? LastLoginAt { get; set; }

        // Token refresh support
        public string? RefreshToken { get; set; }

        public DateTime? RefreshTokenExpiryTime { get; set; }
    }
}
