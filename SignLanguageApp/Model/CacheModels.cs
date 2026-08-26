using System;
using System.Linq;
using SQLite;

namespace SignLanguageApp.Model
{
    /// <summary>
    /// Model for caching hand pose inference results
    /// </summary>
    [Table("hand_pose_cache")]
    public class HandPoseCache
    {
        [PrimaryKey, AutoIncrement]
        public int Id { get; set; }

        [Indexed]
        public string ImageHash { get; set; } = string.Empty;

        [Ignore]
        public float[] PoseData
        {
            get
            {
                if (string.IsNullOrEmpty(SerializedPoseData)) return Array.Empty<float>();
                try
                {
                    return SerializedPoseData.Split(',', StringSplitOptions.RemoveEmptyEntries)
                                             .Select(x => float.Parse(x, System.Globalization.CultureInfo.InvariantCulture))
                                             .ToArray();
                }
                catch
                {
                    return Array.Empty<float>();
                }
            }
            set
            {
                if (value == null || value.Length == 0)
                {
                    SerializedPoseData = string.Empty;
                }
                else
                {
                    SerializedPoseData = string.Join(",", value.Select(x => x.ToString(System.Globalization.CultureInfo.InvariantCulture)));
                }
            }
        }

        public string SerializedPoseData { get; set; } = string.Empty;

        public long Confidence { get; set; }

        public DateTime CreatedAt { get; set; } = DateTime.UtcNow;

        public DateTime ExpiresAt { get; set; }
    }

    /// <summary>
    /// Model for caching translation results
    /// </summary>
    [Table("translation_cache")]
    public class TranslationCache
    {
        [PrimaryKey, AutoIncrement]
        public int Id { get; set; }

        [Indexed]
        public string SourceText { get; set; } = string.Empty;

        public string TargetLanguage { get; set; } = string.Empty;

        public string TranslatedText { get; set; } = string.Empty;

        public DateTime CreatedAt { get; set; } = DateTime.UtcNow;

        public DateTime ExpiresAt { get; set; }
    }

    /// <summary>
    /// Model for caching lesson data
    /// </summary>
    [Table("lesson_cache")]
    public class LessonCache
    {
        [PrimaryKey]
        public int LessonId { get; set; }

        public string Title { get; set; } = string.Empty;

        public string Description { get; set; } = string.Empty;

        public string ThumbnailUrl { get; set; } = string.Empty;

        public int DurationSeconds { get; set; }

        public string InstructorName { get; set; } = string.Empty;

        public string Difficulty { get; set; } = "Beginner";

        public int ViewCount { get; set; }

        public DateTime CachedAt { get; set; } = DateTime.UtcNow;

        public DateTime ExpiresAt { get; set; }
    }

    /// <summary>
    /// Model for caching user progress/learning data
    /// </summary>
    [Table("user_progress_cache")]
    public class UserProgressCache
    {
        [PrimaryKey, AutoIncrement]
        public int Id { get; set; }

        [Indexed]
        public string UserId { get; set; } = string.Empty;

        public int LessonId { get; set; }

        public bool IsCompleted { get; set; }

        public int ProgressPercentage { get; set; }

        public DateTime LastAccessedAt { get; set; } = DateTime.UtcNow;

        public DateTime CachedAt { get; set; } = DateTime.UtcNow;

        public DateTime ExpiresAt { get; set; }
    }

    /// <summary>
    /// Model for caching sign language vocabulary
    /// </summary>
    [Table("vocabulary_cache")]
    public class VocabularyCache
    {
        [PrimaryKey, AutoIncrement]
        public int Id { get; set; }

        [Indexed]
        public string Word { get; set; } = string.Empty;

        public string SignDescription { get; set; } = string.Empty;

        public string VideoUrl { get; set; } = string.Empty;

        public string Category { get; set; } = string.Empty;

        public DateTime CachedAt { get; set; } = DateTime.UtcNow;

        public DateTime ExpiresAt { get; set; }
    }
}
