namespace SignLanguageApi.Dtos
{
    public class AchievementBadgeDto
    {
        public int Id { get; set; }

        public string Title { get; set; } = string.Empty;

        public string Color { get; set; } = "#10B981";

        public string IconChar { get; set; } = "⭐";

        public bool Unlocked { get; set; } = false;

        public DateTime? UnlockedAt { get; set; }
    }
}
