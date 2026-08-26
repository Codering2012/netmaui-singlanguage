using System.Text.Json.Serialization;
using SignLanguageApi.Dtos;
using SignLanguageApi.Data;
using SignLanguageApi.Services;

namespace SignLanguageApi.Dtos
{
    [JsonSourceGenerationOptions(PropertyNamingPolicy = JsonKnownNamingPolicy.CamelCase, GenerationMode = JsonSourceGenerationMode.Default)]
    [JsonSerializable(typeof(RegisterRequest))]
    [JsonSerializable(typeof(LoginRequest))]
    [JsonSerializable(typeof(ApiMessageDto))]
    [JsonSerializable(typeof(AuthTokenResponseDto))]
    [JsonSerializable(typeof(GestureLandmarksRequestDto))]
    [JsonSerializable(typeof(GesturePredictionResponseDto))]
    [JsonSerializable(typeof(Landmark3DDto))]
    [JsonSerializable(typeof(CoordinateDto))]
    [JsonSerializable(typeof(GesturePredictionDataDto))]
    [JsonSerializable(typeof(LandmarkRequestDto))]
    [JsonSerializable(typeof(LearnPageDataDto))]
    [JsonSerializable(typeof(LessonCategoryDto))]
    [JsonSerializable(typeof(LessonDto))]
    [JsonSerializable(typeof(PersonalizedRecommendationDto))]
    [JsonSerializable(typeof(RefreshTokenRequest))]
    [JsonSerializable(typeof(SpacedRepetitionLessonDto))]
    [JsonSerializable(typeof(UserProfileDto))]
    [JsonSerializable(typeof(VideoDto))]
    [JsonSerializable(typeof(MediaDto))]
    [JsonSerializable(typeof(AchievementBadgeDto))]
    [JsonSerializable(typeof(User))]
    [JsonSerializable(typeof(Lesson))]
    [JsonSerializable(typeof(LessonCategory))]
    [JsonSerializable(typeof(UserLesson))]
    [JsonSerializable(typeof(Achievement))]
    [JsonSerializable(typeof(UserAchievement))]
    [JsonSerializable(typeof(SpacedRepetitionLesson))]
    [JsonSerializable(typeof(List<LessonCategoryDto>))]
    [JsonSerializable(typeof(List<LessonDto>))]
    [JsonSerializable(typeof(List<AchievementBadgeDto>))]
    [JsonSerializable(typeof(List<SpacedRepetitionLessonDto>))]
    [JsonSerializable(typeof(List<MediaDto>))]
    [JsonSerializable(typeof(UserProgressData))]
    [JsonSerializable(typeof(List<UserLessonProgressData>))]
    [JsonSerializable(typeof(UserAccountRecordDto))]
    [JsonSerializable(typeof(AuditLogEntryDto))]
    [JsonSerializable(typeof(List<UserAccountRecordDto>))]
    [JsonSerializable(typeof(ErrorResponseDto))]
    [JsonSerializable(typeof(RateLimitResponseDto))]
    [JsonSerializable(typeof(UserStatsDto))]
    [JsonSerializable(typeof(DailyXpDto))]
    [JsonSerializable(typeof(CategoryProgressDto))]
    [JsonSerializable(typeof(FeedbackRequestDto))]
    [JsonSerializable(typeof(List<DailyXpDto>))]
    [JsonSerializable(typeof(List<CategoryProgressDto>))]
    public partial class ApiJsonContext : JsonSerializerContext
    {
    }

    public class ErrorResponseDto
    {
        public string message { get; set; } = string.Empty;
        public int statusCode { get; set; }
    }
}
