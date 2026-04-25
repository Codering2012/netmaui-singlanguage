namespace SignLanguageApi.Data
{
    public class Achievement
    {
        public int Id { get; set; }

        public string Title { get; set; } = string.Empty;

        public string Description { get; set; } = string.Empty;

        public string BadgeColor { get; set; } = "#10B981";

        public int RequiredPoints { get; set; } = 0;

        public string IconChar { get; set; } = "⭐";

        public ICollection<UserAchievement> UserAchievements { get; set; } = new List<UserAchievement>();
    }
}
