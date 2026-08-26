using System.Text.Json.Serialization;

namespace SignLanguageApi.Dtos
{
    public class LeaderboardEntryDto
    {
        [JsonPropertyName("userId")]
        public string UserId { get; set; } = string.Empty;

        [JsonPropertyName("name")]
        public string Name { get; set; } = string.Empty;

        [JsonPropertyName("avatarUrl")]
        public string? AvatarUrl { get; set; }

        [JsonPropertyName("totalXp")]
        public int TotalXp { get; set; }

        [JsonPropertyName("rank")]
        public int Rank { get; set; }

        [JsonPropertyName("isCurrentUser")]
        public bool IsCurrentUser { get; set; }
    }

    public class LeaderboardDto
    {
        [JsonPropertyName("topEntries")]
        public List<LeaderboardEntryDto> TopEntries { get; set; } = new();

        [JsonPropertyName("currentUserEntry")]
        public LeaderboardEntryDto? CurrentUserEntry { get; set; }
    }
}
