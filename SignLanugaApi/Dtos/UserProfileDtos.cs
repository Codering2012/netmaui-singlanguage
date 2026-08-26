namespace SignLanguageApi.Dtos
{
    public class UpdateNameRequest
    {
        public string NewName { get; set; } = string.Empty;
    }

    public class UpdatePasswordRequest
    {
        public string OldPassword { get; set; } = string.Empty;
        public string NewPassword { get; set; } = string.Empty;
    }

    public class UpdateAvatarRequest
    {
        public string AvatarUrl { get; set; } = string.Empty;
    }

    public class UpdateDescriptionRequest
    {
        public string Description { get; set; } = string.Empty;
    }

    public class UserProfileDto
    {
        public string Id { get; set; } = string.Empty;
        public string Email { get; set; } = string.Empty;
        public string Name { get; set; } = string.Empty;
        public string? AvatarUrl { get; set; }
        public string? ProfileDescription { get; set; }
        public int LearningStreak { get; set; }
        public int TotalXp { get; set; }
    }
}
