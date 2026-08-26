using SQLite;
using SignLanguageApp.Model;
using System.Diagnostics;

namespace SignLanguageApp.Services
{
    public class CacheService : ICacheService
    {
        private SQLiteAsyncConnection? _database;
        private readonly string _dbPath;

        public CacheService()
        {
            _dbPath = Path.Combine(FileSystem.AppDataDirectory, "app_cache.db3");
        }

        public async Task InitializeAsync()
        {
            if (_database != null) return;

            try
            {
                _database = new SQLiteAsyncConnection(_dbPath);
                await _database.CreateTablesAsync<HandPoseCache, TranslationCache, LessonCache, UserProgressCache, VocabularyCache>();
                Debug.WriteLine("Cache database initialized.");
            }
            catch (Exception ex)
            {
                Debug.WriteLine($"Failed to initialize cache database: {ex.Message}");
            }
        }

        public async Task CacheHandPoseAsync(string imageHash, float[] poseData, float confidence)
        {
            if (_database == null) await InitializeAsync();
            
            var entry = new HandPoseCache
            {
                ImageHash = imageHash,
                PoseData = poseData,
                Confidence = (long)(confidence * 100),
                CreatedAt = DateTime.UtcNow,
                ExpiresAt = DateTime.UtcNow.AddDays(7)
            };

            await _database!.InsertOrReplaceAsync(entry);
        }

        public async Task<float[]?> GetCachedHandPoseAsync(string imageHash)
        {
            if (_database == null) await InitializeAsync();
            
            var entry = await _database!.Table<HandPoseCache>()
                .Where(x => x.ImageHash == imageHash && x.ExpiresAt > DateTime.UtcNow)
                .FirstOrDefaultAsync();

            return entry?.PoseData;
        }

        public async Task CacheLessonAsync(LessonDetailDto lesson)
        {
            if (_database == null) await InitializeAsync();

            var entry = new LessonCache
            {
                LessonId = lesson.Id,
                Title = lesson.Title ?? string.Empty,
                Description = lesson.Description ?? string.Empty,
                ThumbnailUrl = lesson.ThumbnailUrl ?? string.Empty,
                DurationSeconds = lesson.DurationSeconds,
                InstructorName = lesson.InstructorName ?? string.Empty,
                Difficulty = lesson.Difficulty ?? "Beginner",
                CachedAt = DateTime.UtcNow,
                ExpiresAt = DateTime.UtcNow.AddHours(24)
            };

            await _database!.InsertOrReplaceAsync(entry);
        }

        public async Task<LessonDetailDto?> GetCachedLessonAsync(int lessonId)
        {
            if (_database == null) await InitializeAsync();

            var entry = await _database!.Table<LessonCache>()
                .Where(x => x.LessonId == lessonId && x.ExpiresAt > DateTime.UtcNow)
                .FirstOrDefaultAsync();

            if (entry == null) return null;

            return new LessonDetailDto
            {
                Id = entry.LessonId,
                Title = entry.Title,
                Description = entry.Description,
                ThumbnailUrl = entry.ThumbnailUrl,
                DurationSeconds = entry.DurationSeconds,
                InstructorName = entry.InstructorName,
                Difficulty = entry.Difficulty
            };
        }

        public async Task ClearExpiredCacheAsync()
        {
            if (_database == null) await InitializeAsync();

            var now = DateTime.UtcNow;
            await _database!.Table<HandPoseCache>().Where(x => x.ExpiresAt < now).DeleteAsync();
            await _database!.Table<TranslationCache>().Where(x => x.ExpiresAt < now).DeleteAsync();
            await _database!.Table<LessonCache>().Where(x => x.ExpiresAt < now).DeleteAsync();
        }
    }
}
