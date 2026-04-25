using System.Text.Json;

namespace SignLanguageApi.Services
{
    /// <summary>
    /// Service for saving and loading user progress data to/from JSON files in AppData
    /// </summary>
    public class UserProgressService : IUserProgressService
    {
        private readonly ILogger<UserProgressService> _logger;
        private readonly string _progressDataPath;
        private readonly JsonSerializerOptions _jsonOptions;

        public UserProgressService(ILogger<UserProgressService> logger)
        {
            _logger = logger;
            
            // Set up directory path: AppData\SignLanguageApp\UserProgress
            string appDataPath = Environment.GetFolderPath(Environment.SpecialFolder.ApplicationData);
            _progressDataPath = Path.Combine(appDataPath, "SignLanguageApp", "UserProgress");
            
            // Create directory if it doesn't exist
            if (!Directory.Exists(_progressDataPath))
            {
                Directory.CreateDirectory(_progressDataPath);
                _logger.LogInformation("Created user progress data directory: {Path}", _progressDataPath);
            }

            // Configure JSON serialization options
            _jsonOptions = new JsonSerializerOptions
            {
                WriteIndented = true,
                PropertyNameCaseInsensitive = true
            };
        }

        /// <summary>
        /// Save user progress data (TotalXp, Streak, etc.) to a JSON file
        /// </summary>
        public async Task SaveUserProgressAsync(string userId, int totalXp, int learningStreak, DateTime lastProgressUpdate)
        {
            try
            {
                var progressData = new UserProgressData
                {
                    UserId = userId,
                    TotalXp = totalXp,
                    LearningStreak = learningStreak,
                    LastProgressUpdate = lastProgressUpdate,
                    SavedAt = DateTime.UtcNow
                };

                string filePath = GetProgressFilePath(userId);
                string json = JsonSerializer.Serialize(progressData, _jsonOptions);
                
                await System.IO.File.WriteAllTextAsync(filePath, json);
                _logger.LogInformation("User progress saved: UserId={UserId}, TotalXp={TotalXp}, Streak={Streak}", 
                    userId, totalXp, learningStreak);
            }
            catch (Exception ex)
            {
                _logger.LogError(ex, "Error saving user progress for UserId={UserId}", userId);
                throw;
            }
        }

        /// <summary>
        /// Load user progress data from a JSON file
        /// </summary>
        public async Task<UserProgressData?> LoadUserProgressAsync(string userId)
        {
            try
            {
                string filePath = GetProgressFilePath(userId);
                
                if (!System.IO.File.Exists(filePath))
                {
                    _logger.LogDebug("No progress file found for UserId={UserId}", userId);
                    return null;
                }

                string json = await System.IO.File.ReadAllTextAsync(filePath);
                var progressData = JsonSerializer.Deserialize<UserProgressData>(json, _jsonOptions);
                
                _logger.LogInformation("User progress loaded: UserId={UserId}, TotalXp={TotalXp}, Streak={Streak}", 
                    userId, progressData?.TotalXp ?? 0, progressData?.LearningStreak ?? 0);
                
                return progressData;
            }
            catch (Exception ex)
            {
                _logger.LogError(ex, "Error loading user progress for UserId={UserId}", userId);
                return null;
            }
        }

        /// <summary>
        /// Save all lessons progress for a user
        /// </summary>
        public async Task SaveUserLessonsProgressAsync(string userId, List<UserLessonProgressData> lessonsProgress)
        {
            try
            {
                string filePath = GetLessonsProgressFilePath(userId);
                string json = JsonSerializer.Serialize(lessonsProgress, _jsonOptions);
                
                await System.IO.File.WriteAllTextAsync(filePath, json);
                _logger.LogInformation("User lessons progress saved: UserId={UserId}, LessonCount={Count}", 
                    userId, lessonsProgress.Count);
            }
            catch (Exception ex)
            {
                _logger.LogError(ex, "Error saving lessons progress for UserId={UserId}", userId);
                throw;
            }
        }

        /// <summary>
        /// Load all lessons progress for a user
        /// </summary>
        public async Task<List<UserLessonProgressData>> LoadUserLessonsProgressAsync(string userId)
        {
            try
            {
                string filePath = GetLessonsProgressFilePath(userId);
                
                if (!System.IO.File.Exists(filePath))
                {
                    _logger.LogDebug("No lessons progress file found for UserId={UserId}", userId);
                    return new List<UserLessonProgressData>();
                }

                string json = await System.IO.File.ReadAllTextAsync(filePath);
                var lessonsProgress = JsonSerializer.Deserialize<List<UserLessonProgressData>>(json, _jsonOptions) 
                    ?? new List<UserLessonProgressData>();
                
                _logger.LogInformation("User lessons progress loaded: UserId={UserId}, LessonCount={Count}", 
                    userId, lessonsProgress.Count);
                
                return lessonsProgress;
            }
            catch (Exception ex)
            {
                _logger.LogError(ex, "Error loading lessons progress for UserId={UserId}", userId);
                return new List<UserLessonProgressData>();
            }
        }

        /// <summary>
        /// Delete all user data files when account is deleted
        /// </summary>
        public async Task DeleteUserDataAsync(string userId)
        {
            try
            {
                string progressFilePath = GetProgressFilePath(userId);
                string lessonsFilePath = GetLessonsProgressFilePath(userId);

                if (System.IO.File.Exists(progressFilePath))
                {
                    System.IO.File.Delete(progressFilePath);
                    _logger.LogInformation("Deleted progress file for UserId={UserId}", userId);
                }

                if (System.IO.File.Exists(lessonsFilePath))
                {
                    System.IO.File.Delete(lessonsFilePath);
                    _logger.LogInformation("Deleted lessons progress file for UserId={UserId}", userId);
                }

                await Task.CompletedTask;
            }
            catch (Exception ex)
            {
                _logger.LogError(ex, "Error deleting user data for UserId={UserId}", userId);
                throw;
            }
        }

        // Helper methods
        private string GetProgressFilePath(string userId)
        {
            return Path.Combine(_progressDataPath, $"progress_{userId}.json");
        }

        private string GetLessonsProgressFilePath(string userId)
        {
            return Path.Combine(_progressDataPath, $"lessons_progress_{userId}.json");
        }
    }
}
