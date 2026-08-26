using SignLanguageApp.Model;

namespace SignLanguageApp.Services
{
    public interface ICacheService
    {
        Task InitializeAsync();
        Task CacheHandPoseAsync(string imageHash, float[] poseData, float confidence);
        Task<float[]?> GetCachedHandPoseAsync(string imageHash);
        Task CacheLessonAsync(LessonDetailDto lesson);
        Task<LessonDetailDto?> GetCachedLessonAsync(int lessonId);
        Task ClearExpiredCacheAsync();
    }
}
