namespace SignLanguageApi.Data
{
    public class UserAchievement
    {
        public int Id { get; set; }

        public string UserId { get; set; } = string.Empty;

        public int AchievementId { get; set; }

        public User? User { get; set; }

        public Achievement? Achievement { get; set; }

        public DateTime UnlockedAt { get; set; } = DateTime.UtcNow;
    }
}
