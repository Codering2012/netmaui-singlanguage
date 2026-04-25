namespace SignLanguageApi.Services
{
    /// <summary>
    /// Interface for saving and loading user progress data to/from persistent storage
    /// </summary>
    public interface IUserProgressService
    {
        /// <summary>
        /// Save user progress data (TotalXp, Streak, etc.) to a JSON file
        /// </summary>
        Task SaveUserProgressAsync(string userId, int totalXp, int learningStreak, DateTime lastProgressUpdate);

        /// <summary>
        /// Load user progress data from a JSON file
        /// </summary>
        Task<UserProgressData?> LoadUserProgressAsync(string userId);

        /// <summary>
        /// Save all lessons progress for a user
        /// </summary>
        Task SaveUserLessonsProgressAsync(string userId, List<UserLessonProgressData> lessonsProgress);

        /// <summary>
        /// Load all lessons progress for a user
        /// </summary>
        Task<List<UserLessonProgressData>> LoadUserLessonsProgressAsync(string userId);

        /// <summary>
        /// Delete user data files when account is deleted
        /// </summary>
        Task DeleteUserDataAsync(string userId);
    }

    /// <summary>
    /// Represents user progress data that can be persisted
    /// </summary>
    public class UserProgressData
    {
        public string UserId { get; set; } = string.Empty;
        public int TotalXp { get; set; } = 0;
        public int LearningStreak { get; set; } = 0;
        public DateTime LastProgressUpdate { get; set; } = DateTime.UtcNow;
        public DateTime SavedAt { get; set; } = DateTime.UtcNow;
    }

    /// <summary>
    /// Represents individual lesson progress that can be persisted
    /// </summary>
    public class UserLessonProgressData
    {
        public int LessonId { get; set; }
        public string LessonTitle { get; set; } = string.Empty;
        public int CompletionPercentage { get; set; } = 0;
        public bool IsCompleted { get; set; } = false;
        public DateTime? CompletedAt { get; set; }
        public DateTime StartedAt { get; set; } = DateTime.UtcNow;
        public int TotalAttempts { get; set; } = 0;
        public int XpEarned { get; set; } = 0;
    }
}
