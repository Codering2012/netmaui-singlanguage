using System.Diagnostics;
using System.Text.Json;
using SQLite;
using SignLanguageApp.Model;

namespace SignLanguageApp.Services
{
    public class CacheService : ICacheService
    {
        private SQLiteAsyncConnection? _database;
        private readonly string _dbPath;
        private bool _isInitialized;

        public CacheService()
        {
            _dbPath = Path.Combine(FileSystem.AppDataDirectory, "cache.db");
        }

        public async Task InitializeAsync()
        {
            if (_isInitialized)
                return;

            try
            {
                _database = new SQLiteAsyncConnection(_dbPath);

                // Create tables
                await _database.CreateTableAsync<HandPoseCache>();
                await _database.CreateTableAsync<TranslationCache>();
                await _database.CreateTableAsync<LessonCache>();
                await _database.CreateTableAsync<UserProgressCache>();
                await _database.CreateTableAsync<VocabularyCache>();

                _isInitialized = true;
                Debug.WriteLine($"Cache database initialized at {_dbPath}");
            }
            catch (Exception ex)
            {
                Debug.WriteLine($"Failed to initialize cache database: {ex.Message}");
                throw;
            }
        }

        #region Hand Pose Cache Operations

        public async Task<HandPoseCache?> GetHandPoseCacheAsync(string imageHash)
        {
            try
            {
                var cache = await _database!.Table<HandPoseCache>()
                    .Where(c => c.ImageHash == imageHash && c.ExpiresAt > DateTime.UtcNow)
                    .FirstOrDefaultAsync();

                return cache;
            }
            catch (Exception ex)
            {
                Debug.WriteLine($"Error getting hand pose cache: {ex.Message}");
                return null;
            }
        }

        public async Task SaveHandPoseCacheAsync(HandPoseCache cache)
        {
            try
            {
                if (cache.ExpiresAt == default)
                    cache.ExpiresAt = DateTime.UtcNow.AddDays(7); // Default 7 days

                var existing = await _database!.Table<HandPoseCache>()
                    .Where(c => c.ImageHash == cache.ImageHash)
                    .FirstOrDefaultAsync();

                if (existing != null)
                    await _database.UpdateAsync(cache);
                else
                    await _database.InsertAsync(cache);
            }
            catch (Exception ex)
            {
                Debug.WriteLine($"Error saving hand pose cache: {ex.Message}");
            }
        }

        public async Task<bool> DeleteHandPoseCacheAsync(string imageHash)
        {
            try
            {
                await _database!.DeleteAsync<HandPoseCache>(imageHash);
                return true;
            }
            catch (Exception ex)
            {
                Debug.WriteLine($"Error deleting hand pose cache: {ex.Message}");
                return false;
            }
        }

        public async Task ClearExpiredHandPoseCacheAsync()
        {
            try
            {
                await _database!.ExecuteAsync(
                    $"DELETE FROM hand_pose_cache WHERE ExpiresAt <= datetime('now')");
            }
            catch (Exception ex)
            {
                Debug.WriteLine($"Error clearing expired hand pose cache: {ex.Message}");
            }
        }

        #endregion

        #region Translation Cache Operations

        public async Task<TranslationCache?> GetTranslationCacheAsync(string sourceText, string targetLanguage)
        {
            try
            {
                var cache = await _database!.Table<TranslationCache>()
                    .Where(c => c.SourceText == sourceText && c.TargetLanguage == targetLanguage && c.ExpiresAt > DateTime.UtcNow)
                    .FirstOrDefaultAsync();

                return cache;
            }
            catch (Exception ex)
            {
                Debug.WriteLine($"Error getting translation cache: {ex.Message}");
                return null;
            }
        }

        public async Task SaveTranslationCacheAsync(TranslationCache cache)
        {
            try
            {
                if (cache.ExpiresAt == default)
                    cache.ExpiresAt = DateTime.UtcNow.AddDays(30); // Default 30 days

                var existing = await _database!.Table<TranslationCache>()
                    .Where(c => c.SourceText == cache.SourceText && c.TargetLanguage == cache.TargetLanguage)
                    .FirstOrDefaultAsync();

                if (existing != null)
                    await _database.UpdateAsync(cache);
                else
                    await _database.InsertAsync(cache);
            }
            catch (Exception ex)
            {
                Debug.WriteLine($"Error saving translation cache: {ex.Message}");
            }
        }

        public async Task<bool> DeleteTranslationCacheAsync(string sourceText, string targetLanguage)
        {
            try
            {
                await _database!.ExecuteAsync(
                    $"DELETE FROM translation_cache WHERE SourceText = ? AND TargetLanguage = ?",
                    sourceText, targetLanguage);
                return true;
            }
            catch (Exception ex)
            {
                Debug.WriteLine($"Error deleting translation cache: {ex.Message}");
                return false;
            }
        }

        public async Task ClearExpiredTranslationCacheAsync()
        {
            try
            {
                await _database!.ExecuteAsync(
                    $"DELETE FROM translation_cache WHERE ExpiresAt <= datetime('now')");
            }
            catch (Exception ex)
            {
                Debug.WriteLine($"Error clearing expired translation cache: {ex.Message}");
            }
        }

        #endregion

        #region Lesson Cache Operations

        public async Task<LessonCache?> GetLessonCacheAsync(int lessonId)
        {
            try
            {
                var cache = await _database!.Table<LessonCache>()
                    .Where(c => c.LessonId == lessonId && c.ExpiresAt > DateTime.UtcNow)
                    .FirstOrDefaultAsync();

                return cache;
            }
            catch (Exception ex)
            {
                Debug.WriteLine($"Error getting lesson cache: {ex.Message}");
                return null;
            }
        }

        public async Task SaveLessonCacheAsync(LessonCache cache)
        {
            try
            {
                if (cache.ExpiresAt == default)
                    cache.ExpiresAt = DateTime.UtcNow.AddDays(14); // Default 14 days

                var existing = await _database!.Table<LessonCache>()
                    .Where(c => c.LessonId == cache.LessonId)
                    .FirstOrDefaultAsync();

                if (existing != null)
                    await _database.UpdateAsync(cache);
                else
                    await _database.InsertAsync(cache);
            }
            catch (Exception ex)
            {
                Debug.WriteLine($"Error saving lesson cache: {ex.Message}");
            }
        }

        public async Task<List<LessonCache>> GetAllLessonCacheAsync()
        {
            try
            {
                return await _database!.Table<LessonCache>()
                    .Where(c => c.ExpiresAt > DateTime.UtcNow)
                    .ToListAsync();
            }
            catch (Exception ex)
            {
                Debug.WriteLine($"Error getting all lesson cache: {ex.Message}");
                return new List<LessonCache>();
            }
        }

        public async Task<bool> DeleteLessonCacheAsync(int lessonId)
        {
            try
            {
                await _database!.DeleteAsync<LessonCache>(lessonId);
                return true;
            }
            catch (Exception ex)
            {
                Debug.WriteLine($"Error deleting lesson cache: {ex.Message}");
                return false;
            }
        }

        public async Task ClearExpiredLessonCacheAsync()
        {
            try
            {
                await _database!.ExecuteAsync(
                    $"DELETE FROM lesson_cache WHERE ExpiresAt <= datetime('now')");
            }
            catch (Exception ex)
            {
                Debug.WriteLine($"Error clearing expired lesson cache: {ex.Message}");
            }
        }

        #endregion

        #region User Progress Cache Operations

        public async Task<UserProgressCache?> GetUserProgressCacheAsync(string userId, int lessonId)
        {
            try
            {
                var cache = await _database!.Table<UserProgressCache>()
                    .Where(c => c.UserId == userId && c.LessonId == lessonId && c.ExpiresAt > DateTime.UtcNow)
                    .FirstOrDefaultAsync();

                return cache;
            }
            catch (Exception ex)
            {
                Debug.WriteLine($"Error getting user progress cache: {ex.Message}");
                return null;
            }
        }

        public async Task SaveUserProgressCacheAsync(UserProgressCache cache)
        {
            try
            {
                if (cache.ExpiresAt == default)
                    cache.ExpiresAt = DateTime.UtcNow.AddDays(90); // Default 90 days

                var existing = await _database!.Table<UserProgressCache>()
                    .Where(c => c.UserId == cache.UserId && c.LessonId == cache.LessonId)
                    .FirstOrDefaultAsync();

                if (existing != null)
                    await _database.UpdateAsync(cache);
                else
                    await _database.InsertAsync(cache);
            }
            catch (Exception ex)
            {
                Debug.WriteLine($"Error saving user progress cache: {ex.Message}");
            }
        }

        public async Task<List<UserProgressCache>> GetUserProgressCacheByUserIdAsync(string userId)
        {
            try
            {
                return await _database!.Table<UserProgressCache>()
                    .Where(c => c.UserId == userId && c.ExpiresAt > DateTime.UtcNow)
                    .ToListAsync();
            }
            catch (Exception ex)
            {
                Debug.WriteLine($"Error getting user progress cache: {ex.Message}");
                return new List<UserProgressCache>();
            }
        }

        public async Task<bool> DeleteUserProgressCacheAsync(string userId, int lessonId)
        {
            try
            {
                await _database!.ExecuteAsync(
                    $"DELETE FROM user_progress_cache WHERE UserId = ? AND LessonId = ?",
                    userId, lessonId);
                return true;
            }
            catch (Exception ex)
            {
                Debug.WriteLine($"Error deleting user progress cache: {ex.Message}");
                return false;
            }
        }

        public async Task ClearExpiredUserProgressCacheAsync()
        {
            try
            {
                await _database!.ExecuteAsync(
                    $"DELETE FROM user_progress_cache WHERE ExpiresAt <= datetime('now')");
            }
            catch (Exception ex)
            {
                Debug.WriteLine($"Error clearing expired user progress cache: {ex.Message}");
            }
        }

        #endregion

        #region Vocabulary Cache Operations

        public async Task<VocabularyCache?> GetVocabularyCacheAsync(string word)
        {
            try
            {
                var cache = await _database!.Table<VocabularyCache>()
                    .Where(c => c.Word == word && c.ExpiresAt > DateTime.UtcNow)
                    .FirstOrDefaultAsync();

                return cache;
            }
            catch (Exception ex)
            {
                Debug.WriteLine($"Error getting vocabulary cache: {ex.Message}");
                return null;
            }
        }

        public async Task SaveVocabularyCacheAsync(VocabularyCache cache)
        {
            try
            {
                if (cache.ExpiresAt == default)
                    cache.ExpiresAt = DateTime.UtcNow.AddDays(60); // Default 60 days

                var existing = await _database!.Table<VocabularyCache>()
                    .Where(c => c.Word == cache.Word)
                    .FirstOrDefaultAsync();

                if (existing != null)
                    await _database.UpdateAsync(cache);
                else
                    await _database.InsertAsync(cache);
            }
            catch (Exception ex)
            {
                Debug.WriteLine($"Error saving vocabulary cache: {ex.Message}");
            }
        }

        public async Task<List<VocabularyCache>> GetVocabularyCacheByCategoryAsync(string category)
        {
            try
            {
                return await _database!.Table<VocabularyCache>()
                    .Where(c => c.Category == category && c.ExpiresAt > DateTime.UtcNow)
                    .ToListAsync();
            }
            catch (Exception ex)
            {
                Debug.WriteLine($"Error getting vocabulary cache by category: {ex.Message}");
                return new List<VocabularyCache>();
            }
        }

        public async Task<List<VocabularyCache>> GetAllVocabularyCacheAsync()
        {
            try
            {
                return await _database!.Table<VocabularyCache>()
                    .Where(c => c.ExpiresAt > DateTime.UtcNow)
                    .ToListAsync();
            }
            catch (Exception ex)
            {
                Debug.WriteLine($"Error getting all vocabulary cache: {ex.Message}");
                return new List<VocabularyCache>();
            }
        }

        public async Task<bool> DeleteVocabularyCacheAsync(string word)
        {
            try
            {
                await _database!.ExecuteAsync(
                    $"DELETE FROM vocabulary_cache WHERE Word = ?", word);
                return true;
            }
            catch (Exception ex)
            {
                Debug.WriteLine($"Error deleting vocabulary cache: {ex.Message}");
                return false;
            }
        }

        public async Task ClearExpiredVocabularyCacheAsync()
        {
            try
            {
                await _database!.ExecuteAsync(
                    $"DELETE FROM vocabulary_cache WHERE ExpiresAt <= datetime('now')");
            }
            catch (Exception ex)
            {
                Debug.WriteLine($"Error clearing expired vocabulary cache: {ex.Message}");
            }
        }

        #endregion

        #region General Operations

        public async Task ClearAllCacheAsync()
        {
            try
            {
                await _database!.DeleteAllAsync<HandPoseCache>();
                await _database.DeleteAllAsync<TranslationCache>();
                await _database.DeleteAllAsync<LessonCache>();
                await _database.DeleteAllAsync<UserProgressCache>();
                await _database.DeleteAllAsync<VocabularyCache>();

                Debug.WriteLine("All cache cleared");
            }
            catch (Exception ex)
            {
                Debug.WriteLine($"Error clearing all cache: {ex.Message}");
            }
        }

        #endregion
    }
}
