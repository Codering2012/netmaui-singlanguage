using System;

namespace SignLanguageApi.Dtos
{
    public class UserAccountRecordDto
    {
        public string Id { get; set; } = string.Empty;
        public string Email { get; set; } = string.Empty;
        public string Name { get; set; } = string.Empty;
        public DateTime CreatedAt { get; set; }
        public int LearningStreak { get; set; }
    }
}
