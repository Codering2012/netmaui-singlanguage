using SignLanguageApp.Model;

namespace SignLanguageApp.Services
{
    public interface ICacheService
    {
        Task InitializeAsync();

        // Hand Pose Cache Operations
        Task<HandPoseCache?> GetHandPoseCacheAsync(string imageHash);
        Task SaveHandPoseCacheAsync(HandPoseCache cache);
        Task<bool> DeleteHandPoseCacheAsync(string imageHash);
        Task ClearExpiredHandPoseCacheAsync();

        // Translation Cache Operations
        Task<TranslationCache?> GetTranslationCacheAsync(string sourceText, string targetLanguage);
        Task SaveTranslationCacheAsync(TranslationCache cache);
        Task<bool> DeleteTranslationCacheAsync(string sourceText, string targetLanguage);
        Task ClearExpiredTranslationCacheAsync();

        // Lesson Cache Operations
        Task<LessonCache?> GetLessonCacheAsync(int lessonId);
        Task SaveLessonCacheAsync(LessonCache cache);
        Task<List<LessonCache>> GetAllLessonCacheAsync();
        Task<bool> DeleteLessonCacheAsync(int lessonId);
        Task ClearExpiredLessonCacheAsync();

        // User Progress Cache Operations
        Task<UserProgressCache?> GetUserProgressCacheAsync(string userId, int lessonId);
        Task SaveUserProgressCacheAsync(UserProgressCache cache);
        Task<List<UserProgressCache>> GetUserProgressCacheByUserIdAsync(string userId);
        Task<bool> DeleteUserProgressCacheAsync(string userId, int lessonId);
        Task ClearExpiredUserProgressCacheAsync();

        // Vocabulary Cache Operations
        Task<VocabularyCache?> GetVocabularyCacheAsync(string word);
        Task SaveVocabularyCacheAsync(VocabularyCache cache);
        Task<List<VocabularyCache>> GetVocabularyCacheByCategoryAsync(string category);
        Task<List<VocabularyCache>> GetAllVocabularyCacheAsync();
        Task<bool> DeleteVocabularyCacheAsync(string word);
        Task ClearExpiredVocabularyCacheAsync();

        // General Operations
        Task ClearAllCacheAsync();
    }
}
