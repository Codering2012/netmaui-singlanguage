namespace SignLanguageApp.Model;

using System.Text.Json.Serialization;

public class LoginRequest
{
    public string Email { get; set; } = string.Empty;
    public string Password { get; set; } = string.Empty;
}

public class RegisterRequest
{
    public string Email { get; set; } = string.Empty;
    public string Password { get; set; } = string.Empty;
    public string Name { get; set; } = string.Empty;
}

public class RefreshTokenRequest
{
    public string RefreshToken { get; set; } = string.Empty;
}

public class LoginApiResponse
{
    [JsonPropertyName("token")]
    public string Token { get; set; } = string.Empty;

    [JsonPropertyName("accessToken")]
    public string AccessToken { get; set; } = string.Empty;

    [JsonPropertyName("refreshToken")]
    public string RefreshToken { get; set; } = string.Empty;

    [JsonPropertyName("userId")]
    public string UserId { get; set; } = string.Empty;

    [JsonPropertyName("name")]
    public string Name { get; set; } = string.Empty;
}

public class ApiResponse<T>
{
    public bool Success { get; set; }
    public string Message { get; set; } = string.Empty;
    public T? Data { get; set; }
}

public class UserDto
{
    public string Id { get; set; } = string.Empty;
    public string Email { get; set; } = string.Empty;
    public string Name { get; set; } = string.Empty;
    public string? AvatarUrl { get; set; }
    public int LearningStreak { get; set; }
    public int TotalXP { get; set; }
}

public class LessonDto
{
    public int Id { get; set; }
    public string Title { get; set; } = string.Empty;
    public string Description { get; set; } = string.Empty;
    public string? ThumbnailUrl { get; set; }
    public int DurationSeconds { get; set; }
    public string InstructorName { get; set; } = string.Empty;
    public string Difficulty { get; set; } = "Beginner";
    public int ViewCount { get; set; }
    [System.Text.Json.Serialization.JsonIgnore] public bool IsDownloaded { get; set; }
}

public class LoginResponse
{
    [JsonPropertyName("token")]
    public string AccessToken { get; set; } = string.Empty;

    [JsonPropertyName("refreshToken")]
    public string RefreshToken { get; set; } = string.Empty;

    [JsonPropertyName("userId")]
    public string UserId { get; set; } = string.Empty;

    [JsonPropertyName("name")]
    public string Name { get; set; } = string.Empty;

    // Computed property for backward compatibility
    [JsonIgnore]
    public UserDto User { get; set; } = new UserDto();
}



/// <summary>
/// Learning Progress and Stats DTOs
/// </summary>
/// <summary>
/// Learning Progress and Stats DTOs
/// </summary>
public class UserStatsDto
{
    [JsonPropertyName("totalXp")]
    public int TotalXp { get; set; }

    [JsonPropertyName("learningStreak")]
    public int LearningStreak { get; set; }

    [JsonPropertyName("lessonsCompleted")]
    public int LessonsCompleted { get; set; }
    [JsonIgnore]
    public int TotalXP { get => TotalXp; set => TotalXp = value; }

    [JsonIgnore]
    public int CurrentStreak { get => LearningStreak; set => LearningStreak = value; }

    [JsonIgnore]
    public int TotalProgress { get => LessonsCompleted; set => LessonsCompleted = value; }
    public int GlobalRanking { get; set; }

    [JsonPropertyName("weeklyXp")]
    public List<DailyXpDto> WeeklyXp { get; set; } = new();

    [JsonPropertyName("categoryProgress")]
    public List<CategoryProgressDto> CategoryProgress { get; set; } = new();
}

public class DailyXpDto
{
    [JsonPropertyName("date")]
    public string Date { get; set; } = string.Empty;

    [JsonPropertyName("xp")]
    public int Xp { get; set; }
}

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

public class AchievementBadgeDto
{
    [JsonPropertyName("id")]
    public int Id { get; set; }

    [JsonPropertyName("title")]
    public string Title { get; set; } = string.Empty;

    [JsonPropertyName("description")]
    public string Description { get; set; } = string.Empty;

    [JsonPropertyName("icon")]
    public string Icon { get; set; } = string.Empty;

    [JsonPropertyName("color")]
    public string Color { get; set; } = "#10B981";

    [JsonPropertyName("isUnlocked")]
    public bool IsUnlocked { get; set; }
}

public class LeaderboardDto
{
    [JsonPropertyName("topEntries")]
    public List<LeaderboardEntryDto> TopEntries { get; set; } = new();

    [JsonPropertyName("currentUserEntry")]
    public LeaderboardEntryDto? CurrentUserEntry { get; set; }
}

public class CategoryProgressDto
{
    [JsonPropertyName("categoryName")]
    public string CategoryName { get; set; } = string.Empty;

    [JsonPropertyName("progress")]
    public double Progress { get; set; }
}

public class FeedbackRequest
{
    [JsonPropertyName("subject")]
    public string Subject { get; set; } = string.Empty;

    [JsonPropertyName("message")]
    public string Message { get; set; } = string.Empty;

    [JsonPropertyName("rating")]
    public int Rating { get; set; }
}

public class LessonCategoryDto
{
    [JsonPropertyName("id")]
    public int Id { get; set; }

    [JsonPropertyName("title")]
    public string Title { get; set; } = string.Empty;

    [JsonPropertyName("description")]
    public string Description { get; set; } = string.Empty;

    [JsonPropertyName("icon")]
    public string? Icon { get; set; }

    [JsonPropertyName("iconUrl")]
    public string? IconUrl { get; set; }

    [JsonPropertyName("progress")]
    public double Progress { get; set; }

    [JsonPropertyName("totalLessons")]
    public int TotalLessons { get; set; }

    [JsonPropertyName("completedLessons")]
    public int CompletedLessons { get; set; }

    [JsonPropertyName("difficulty")]
    public string Difficulty { get; set; } = "Beginner";
}

public class LessonDetailDto
{
    [JsonPropertyName("id")]
    public int Id { get; set; }

    [JsonPropertyName("title")]
    public string Title { get; set; } = string.Empty;

    [JsonPropertyName("description")]
    public string Description { get; set; } = string.Empty;

    [JsonPropertyName("thumbnail")]
    public string? ThumbnailUrl { get; set; }

    [JsonPropertyName("durationSeconds")]
    public int DurationSeconds { get; set; }

    [JsonPropertyName("completionPercentage")]
    public double CompletionPercentage { get; set; }

    [JsonPropertyName("difficulty")]
    public string Difficulty { get; set; } = "Beginner";

    [JsonPropertyName("instructorName")]
    public string InstructorName { get; set; } = string.Empty;

    [JsonPropertyName("categoryId")]
    public int CategoryId { get; set; }

    [JsonPropertyName("data")]
    public LessonDetailDataDto? Data { get; set; }

    [JsonPropertyName("videoUrl")]
    public string? VideoUrl { get; set; }
    [System.Text.Json.Serialization.JsonIgnore] public bool IsDownloaded { get; set; }
}

public class LessonStepDto
{
    [JsonPropertyName("type")]
    public LessonStepType Type { get; set; }

    [JsonPropertyName("title")]
    public string Title { get; set; } = string.Empty;

    [JsonPropertyName("description")]
    public string Description { get; set; } = string.Empty;

    [JsonPropertyName("imageUrl")]
    public string? ImageUrl { get; set; }

    [JsonPropertyName("options")]
    public List<string>? Options { get; set; }

    [JsonPropertyName("correctOption")]
    public string? CorrectOption { get; set; }

    [JsonPropertyName("targetGesture")]
    public string? TargetGesture { get; set; }
    public string? TargetSentence { get; set; }
    public List<MatchingPairDto>? MatchingPairs { get; set; }
    public List<string>? SequenceTokens { get; set; }
}

public class InteractiveLessonDto
{
    [JsonPropertyName("id")]
    public int Id { get; set; }

    [JsonPropertyName("title")]
    public string Title { get; set; } = string.Empty;

    [JsonPropertyName("steps")]
    public List<LessonStepDto> Steps { get; set; } = new();
}

public class LessonDetailDataDto
{
    [JsonPropertyName("durationSeconds")]
    public int DurationSeconds { get; set; }

    [JsonPropertyName("difficulty")]
    public string Difficulty { get; set; } = string.Empty;

    [JsonPropertyName("completionPercentage")]
    public double CompletionPercentage { get; set; }

    [JsonPropertyName("instructorName")]
    public string InstructorName { get; set; } = string.Empty;

    [JsonPropertyName("categoryId")]
    public int CategoryId { get; set; }

    [JsonPropertyName("uiLayout")]
    public LessonUiLayoutDto? UiLayout { get; set; }

    [JsonPropertyName("interactiveLesson")]
    public InteractiveLessonDto? InteractiveLesson { get; set; }
}

public class LessonUiLayoutDto
{
    [JsonPropertyName("fileName")]
    public string FileName { get; set; } = string.Empty;

    [JsonPropertyName("xamlContent")]
    public string XamlContent { get; set; } = string.Empty;

    [JsonPropertyName("codeBehindContent")]
    public string CodeBehindContent { get; set; } = string.Empty;
}

public class DailyGoalDto
{
    [JsonPropertyName("totalRequired")]
    public int TotalRequired { get; set; }

    [JsonPropertyName("completedToday")]
    public int CompletedToday { get; set; }
}

public class LearnDataDto
{
    [JsonPropertyName("progressPercentage")]
    public double ProgressPercentage { get; set; }

    [JsonPropertyName("currentStreak")]
    public int CurrentStreak { get; set; }

    [JsonPropertyName("totalXp")]
    public int TotalXp { get; set; }

    [JsonIgnore]
    public int TotalXP
    {
        get => TotalXp;
        set => TotalXp = value;
    }

    [JsonPropertyName("categories")]
    public List<LessonCategoryDto> Categories { get; set; } = [];

    [JsonPropertyName("lessons")]
    public List<LessonDetailDto> Lessons { get; set; } = [];

    [JsonPropertyName("dailyReviewLessons")]
    public List<SpacedRepetitionLessonDto> DailyReviewLessons { get; set; } = [];

    [JsonPropertyName("recommendationReason")]
    public string RecommendationReason { get; set; } = string.Empty;

    [JsonPropertyName("dailyGoalCompleted")]
    public int DailyGoalCompleted { get; set; }

    [JsonPropertyName("dailyGoalTotal")]
    public int DailyGoalTotal { get; set; }

    [JsonPropertyName("tomorrowReviewCount")]
    public int TomorrowReviewCount { get; set; }

    [JsonPropertyName("thisWeekReviewCount")]
    public int ThisWeekReviewCount { get; set; }
}

public class DailyGoalApiDto
{
    [JsonPropertyName("totalReviewsDue")]
    public int TotalReviewsDue { get; set; }

    [JsonPropertyName("completedToday")]
    public int CompletedToday { get; set; }

    [JsonPropertyName("dailyGoal")]
    public int DailyGoal { get; set; }

    [JsonPropertyName("progressPercentage")]
    public double ProgressPercentage { get; set; }
}

public class SpacedRepetitionLessonDto
{
    [JsonPropertyName("id")]
    public int Id { get; set; }

    [JsonPropertyName("title")]
    public string Title { get; set; } = string.Empty;

    [JsonPropertyName("dueDate")]
    public DateTime DueDate { get; set; }

    [JsonPropertyName("repetitionCount")]
    public int RepetitionCount { get; set; }

    [JsonPropertyName("retentionPercentage")]
    public double RetentionPercentage { get; set; }

    [JsonPropertyName("completionPercentage")]
    public double CompletionPercentage { get; set; }

    [JsonPropertyName("difficulty")]
    public string Difficulty { get; set; } = "Beginner";

    [JsonPropertyName("instructorName")]
    public string InstructorName { get; set; } = string.Empty;
}

public class PersonalizedRecommendationDto
{
    [JsonPropertyName("recommendedLessonId")]
    public int RecommendedLessonId { get; set; }

    [JsonPropertyName("recommendedCategoryId")]
    public int RecommendedCategoryId { get; set; }

    [JsonPropertyName("recommendedCategoryTitle")]
    public string RecommendedCategoryTitle { get; set; } = string.Empty;

    [JsonPropertyName("categoryId")]
    public int CategoryId { get; set; }

    [JsonPropertyName("categoryName")]
    public string CategoryName { get; set; } = string.Empty;

    [JsonPropertyName("lessonTitle")]
    public string LessonTitle { get; set; } = string.Empty;

    [JsonPropertyName("lessonDescription")]
    public string LessonDescription { get; set; } = string.Empty;

    [JsonPropertyName("reason")]
    public string Reason { get; set; } = string.Empty;

    [JsonPropertyName("thumbnailUrl")]
    public string ThumbnailUrl { get; set; } = string.Empty;
}

public class UpcomingReviewsDto
{
    [JsonPropertyName("tomorrow")]
    public int TomorrowCount { get; set; }

    [JsonPropertyName("thisWeek")]
    public int ThisWeekCount { get; set; }

    [JsonPropertyName("nextWeek")]
    public int NextWeekCount { get; set; }
}

public class UserProfileDto
{
    [JsonPropertyName("email")]
    public string Email { get; set; } = string.Empty;

    [JsonPropertyName("name")]
    public string Name { get; set; } = string.Empty;

    [JsonPropertyName("avatarUrl")]
    public string? AvatarUrl { get; set; }

    [JsonPropertyName("profileDescription")]
    public string? ProfileDescription { get; set; }

    [JsonPropertyName("learningStreak")]
    public int LearningStreak { get; set; }

    [JsonPropertyName("totalXp")]
    public int TotalXp { get; set; }
}

public class UpdateNameRequest
{
    [JsonPropertyName("newName")]
    public string NewName { get; set; } = string.Empty;
}

public class UpdatePasswordRequest
{
    [JsonPropertyName("oldPassword")]
    public string OldPassword { get; set; } = string.Empty;

    [JsonPropertyName("newPassword")]
    public string NewPassword { get; set; } = string.Empty;
}

public class UpdateAvatarRequest
{
    [JsonPropertyName("avatarUrl")]
    public string AvatarUrl { get; set; } = string.Empty;
}

public class UpdateDescriptionRequest
{
    [JsonPropertyName("description")]
    public string Description { get; set; } = string.Empty;
}

public class UpcomingReviewsApiDto
{
    [JsonPropertyName("dueToday")]
    public int DueToday { get; set; }

    [JsonPropertyName("dueTomorrow")]
    public int DueTomorrow { get; set; }

    [JsonPropertyName("dueThisWeek")]
    public int DueThisWeek { get; set; }

    [JsonPropertyName("overdue")]
    public int Overdue { get; set; }
}

/// <summary>
/// Video DTOs for YouTube/TikTok-style video library
/// </summary>
public class VideoDto
{
    [JsonPropertyName("id")]
    public int Id { get; set; }

    [JsonPropertyName("title")]
    public string Title { get; set; } = string.Empty;

    [JsonPropertyName("description")]
    public string Description { get; set; } = string.Empty;

    [JsonPropertyName("thumbnailUrl")]
    public string ThumbnailUrl { get; set; } = string.Empty;

    [JsonPropertyName("videoUrl")]
    public string VideoUrl { get; set; } = string.Empty;

    [JsonPropertyName("uploadDate")]
    public DateTime UploadDate { get; set; }

    [JsonPropertyName("viewCount")]
    public int ViewCount { get; set; }
    [System.Text.Json.Serialization.JsonIgnore] public bool IsDownloaded { get; set; }

    [JsonPropertyName("likeCount")]
    public int LikeCount { get; set; }

    [JsonPropertyName("durationSeconds")]
    public double DurationSeconds { get; set; }

    [JsonPropertyName("category")]
    public string Category { get; set; } = string.Empty;

    [JsonPropertyName("instructor")]
    public string Instructor { get; set; } = string.Empty;

    [JsonPropertyName("isLiked")]
    public bool IsLiked { get; set; }
}

/// <summary>
/// Hand Gesture Prediction DTOs for Real-Time Camera Processing
/// </summary>
public class CoordinateDto
{
    [JsonPropertyName("x")]
    public double X { get; set; }

    [JsonPropertyName("y")]
    public double Y { get; set; }
}

public class GesturePredictionDataDto
{
    [JsonPropertyName("count")]
    public int Count { get; set; }

    [JsonPropertyName("coordinates")]
    public List<CoordinateDto> Coordinates { get; set; } = [];

    [JsonPropertyName("letter")]
    public string? Letter { get; set; }

    [JsonPropertyName("confidence")]
    public float Confidence { get; set; }

    [JsonPropertyName("processingTimeMs")]
    public double ProcessingTimeMs { get; set; }

    [JsonPropertyName("sentence")]
    public string? Sentence { get; set; }

    [JsonPropertyName("isDrawing")]
    public bool IsDrawing { get; set; }
}

public class GesturePredictionResponseDto
{
    [JsonPropertyName("status")]
    public string Status { get; set; } = string.Empty;

    [JsonPropertyName("message")]
    public string Message { get; set; } = string.Empty;

    [JsonPropertyName("data")]
    public GesturePredictionDataDto? Data { get; set; }
}


public class MediaDto
{
    public int Id { get; set; }
    public string Url { get; set; } = string.Empty;
}

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

public class MatchingPairDto
{
    public int Id { get; set; }
    public string SignTitle { get; set; } = string.Empty;
    public string TextTranslation { get; set; } = string.Empty;
    public string? ImageUrl { get; set; }
    public string? VideoUrl { get; set; }
}
