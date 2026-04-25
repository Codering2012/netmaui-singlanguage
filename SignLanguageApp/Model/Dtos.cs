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
}

public class LoginResponse
{
    public string AccessToken { get; set; } = string.Empty;
    public string RefreshToken { get; set; } = string.Empty;
    public UserDto? User { get; set; }
}

public class User
{
    public string Id { get; set; } = string.Empty;
    public string Email { get; set; } = string.Empty;
    public string Name { get; set; } = string.Empty;
    public string? AvatarUrl { get; set; }
    public int LearningStreak { get; set; }
    public int TotalXP { get; set; }
}

/// <summary>
/// Learning Progress and Stats DTOs
/// </summary>
public class UserStatsDto
{
    [JsonPropertyName("totalProgress")]
    public int TotalProgress { get; set; }

    [JsonPropertyName("currentStreak")]
    public int CurrentStreak { get; set; }

    [JsonPropertyName("totalXP")]
    public int TotalXP { get; set; }

    [JsonPropertyName("globalRanking")]
    public int GlobalRanking { get; set; }
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
