using SignLanguageApi.Data;
using SignLanguageApi.Dtos;
using Microsoft.EntityFrameworkCore;

namespace SignLanguageApi.Services
{
    public interface ILearnService
    {
        Task<LearnPageDataDto> GetLearnPageDataAsync(string userId);
        Task<List<LessonDto>> GetLessonsByCategoryAsync(int categoryId, string userId);
        Task<LessonDto?> GetLessonAsync(int lessonId, string userId);
        Task UpdateLessonProgressAsync(string userId, int lessonId, int completionPercentage);
        Task CompleteLessonAsync(string userId, int lessonId);
        Task<List<SpacedRepetitionLessonDto>> GetDailyReviewLessonsAsync(string userId);
        Task ReviewLessonAsync(string userId, int spacedRepetitionId, double qualityRating);
        Task<List<LessonCategoryDto>> GetAllCategoriesAsync(string userId);
        Task<LessonCategoryDto?> GetCategoryAsync(int categoryId, string userId);
        Task<DailyGoalDto> GetDailyGoalAsync(string userId);
        Task<UpcomingReviewsDto> GetUpcomingReviewsAsync(string userId);
        Task<PersonalizedRecommendationDto> GetPersonalizedRecommendationAsync(string userId);
    }

    public class LearnService : ILearnService
    {
        private readonly AppDbContext _context;
        private readonly IUserProgressService _progressService;
        private readonly ILogger<LearnService> _logger;

        public LearnService(AppDbContext context, IUserProgressService progressService, ILogger<LearnService> logger)
        {
            _context = context;
            _progressService = progressService;
            _logger = logger;
        }

        public async Task<LearnPageDataDto> GetLearnPageDataAsync(string userId)
        {
            var utcNow = DateTime.UtcNow;
            var today = utcNow.Date;

            var user = await _context.Users.FindAsync(userId);
            if (user == null)
                throw new InvalidOperationException("User not found");

            var categories = await _context.LessonCategories
                .Include(c => c.Lessons)
                .ToListAsync();

            var allLessons = await _context.Lessons.ToListAsync();

            var userLessons = await _context.UserLessons
                .Where(ul => ul.UserId == userId)
                .Include(ul => ul.Lesson)
                .ToListAsync();

            var userAchievements = await _context.UserAchievements
                .Where(ua => ua.UserId == userId)
                .Include(ua => ua.Achievement)
                .ToListAsync();

            var allAchievements = await _context.Achievements.ToListAsync();

            var dailyReviews = await GetDailyReviewLessonsAsync(userId);

            var categoriesDto = categories.Select(c =>
            {
                var categoryLessons = userLessons.Where(ul => ul.Lesson?.CategoryId == c.Id).ToList();
                var progress = categoryLessons.Count > 0
                    ? categoryLessons.Average(ul => ul.CompletionPercentage) / 100
                    : 0;

                return new LessonCategoryDto
                {
                    Id = c.Id,
                    Title = c.Title,
                    Description = c.Description,
                    Difficulty = c.Difficulty,
                    IconUrl = c.IconUrl,
                    Progress = progress
                };
            }).ToList();

            var userLessonByLessonId = userLessons.ToDictionary(ul => ul.LessonId);

            var lessonsDto = allLessons.Select(lesson =>
            {
                userLessonByLessonId.TryGetValue(lesson.Id, out var userLesson);
                var completion = (userLesson?.CompletionPercentage ?? 0) / 100.0;

                return new LessonDto
                {
                    Id = lesson.Id,
                    Title = lesson.Title,
                    Description = lesson.Description,
                    Thumbnail = lesson.ThumbnailUrl,
                    DurationSeconds = lesson.DurationSeconds,
                    Difficulty = lesson.Difficulty,
                    CompletionPercentage = completion,
                    InstructorName = lesson.InstructorName,
                    CategoryId = lesson.CategoryId,
                    Data = BuildLessonData(
                        lesson.Id,
                        lesson.DurationSeconds,
                        lesson.Difficulty,
                        completion,
                        lesson.InstructorName,
                        lesson.CategoryId)
                };
            }).ToList();

            var achievementsDto = allAchievements.Select(a =>
            {
                var userUnlocked = userAchievements.FirstOrDefault(ua => ua.AchievementId == a.Id);
                return new AchievementBadgeDto
                {
                    Id = a.Id,
                    Title = a.Title,
                    Color = a.BadgeColor,
                    IconChar = a.IconChar,
                    Unlocked = userUnlocked != null,
                    UnlockedAt = userUnlocked?.UnlockedAt
                };
            }).ToList();

            var totalXp = user.TotalXp;
            var completedToday = userLessons.Count(ul => ul.IsCompleted && ul.CompletedAt?.Date == today);
            var tomorrowReviews = await _context.SpacedRepetitionLessons
                .Where(sr => sr.UserId == userId && sr.DueDate.Date == today.AddDays(1))
                .CountAsync();

            var weekReviews = await _context.SpacedRepetitionLessons
                .Where(sr => sr.UserId == userId && sr.DueDate.Date > today && sr.DueDate.Date <= today.AddDays(7))
                .CountAsync();

            return new LearnPageDataDto
            {
                Categories = categoriesDto,
                Lessons = lessonsDto,
                Achievements = achievementsDto,
                DailyReviewLessons = dailyReviews,
                TotalXp = totalXp,
                CurrentStreak = user.LearningStreak,
                ProgressPercentage = lessonsDto.Count > 0 ? lessonsDto.Average(l => l.CompletionPercentage) : 0,
                RecommendationReason = GetRecommendation(lessonsDto),
                DailyGoalCompleted = completedToday,
                DailyGoalTotal = 5,
                TomorrowReviewCount = tomorrowReviews,
                ThisWeekReviewCount = weekReviews,
                UpcomingReviewProgress = (dailyReviews.Count + tomorrowReviews + weekReviews) > 0
                    ? dailyReviews.Count / (double)(dailyReviews.Count + tomorrowReviews + weekReviews)
                    : 0
            };
        }

        public async Task<List<LessonDto>> GetLessonsByCategoryAsync(int categoryId, string userId)
        {
            var lessons = await _context.Lessons
                .Where(l => l.CategoryId == categoryId)
                .ToListAsync();

            var userLessons = await _context.UserLessons
                .Where(ul => ul.UserId == userId && ul.Lesson!.CategoryId == categoryId)
                .ToListAsync();

            return lessons.Select(l =>
            {
                var userLesson = userLessons.FirstOrDefault(ul => ul.LessonId == l.Id);
                return new LessonDto
                {
                    Id = l.Id,
                    Title = l.Title,
                    Description = l.Description,
                    Thumbnail = l.ThumbnailUrl,
                    DurationSeconds = l.DurationSeconds,
                    Difficulty = l.Difficulty,
                    CompletionPercentage = userLesson?.CompletionPercentage / 100.0 ?? 0,
                    InstructorName = l.InstructorName,
                    CategoryId = l.CategoryId,
                    Data = BuildLessonData(
                        l.Id,
                        l.DurationSeconds,
                        l.Difficulty,
                        userLesson?.CompletionPercentage / 100.0 ?? 0,
                        l.InstructorName,
                        l.CategoryId)
                };
            }).ToList();
        }

        public async Task<LessonDto?> GetLessonAsync(int lessonId, string userId)
        {
            var lesson = await _context.Lessons.FindAsync(lessonId);
            if (lesson == null)
                return null;

            var userLesson = await _context.UserLessons
                .FirstOrDefaultAsync(ul => ul.UserId == userId && ul.LessonId == lessonId);

            return new LessonDto
            {
                Id = lesson.Id,
                Title = lesson.Title,
                Description = lesson.Description,
                Thumbnail = lesson.ThumbnailUrl,
                DurationSeconds = lesson.DurationSeconds,
                Difficulty = lesson.Difficulty,
                CompletionPercentage = userLesson?.CompletionPercentage / 100.0 ?? 0,
                InstructorName = lesson.InstructorName,
                CategoryId = lesson.CategoryId,
                Data = BuildLessonData(
                    lesson.Id,
                    lesson.DurationSeconds,
                    lesson.Difficulty,
                    userLesson?.CompletionPercentage / 100.0 ?? 0,
                    lesson.InstructorName,
                    lesson.CategoryId)
            };
        }

        public async Task UpdateLessonProgressAsync(string userId, int lessonId, int completionPercentage)
        {
            var lessonExists = await _context.Lessons.AnyAsync(l => l.Id == lessonId);
            if (!lessonExists)
                throw new KeyNotFoundException("Lesson not found");

            var userLesson = await _context.UserLessons
                .FirstOrDefaultAsync(ul => ul.UserId == userId && ul.LessonId == lessonId);

            if (userLesson == null)
            {
                userLesson = new UserLesson
                {
                    UserId = userId,
                    LessonId = lessonId,
                    CompletionPercentage = completionPercentage,
                    IsCompleted = completionPercentage >= 100,
                    CompletedAt = completionPercentage >= 100 ? DateTime.UtcNow : null,
                    TotalAttempts = 1
                };
                _context.UserLessons.Add(userLesson);
            }
            else
            {
                userLesson.CompletionPercentage = Math.Max(userLesson.CompletionPercentage, completionPercentage);
                if (userLesson.CompletionPercentage >= 100)
                {
                    userLesson.IsCompleted = true;
                    userLesson.CompletedAt ??= DateTime.UtcNow;
                }
                userLesson.TotalAttempts++;
            }

            await _context.SaveChangesAsync();
        }

        public async Task CompleteLessonAsync(string userId, int lessonId)
        {
            var user = await _context.Users.FindAsync(userId);
            if (user == null)
                throw new InvalidOperationException("User not found");

            var lessonExists = await _context.Lessons.AnyAsync(l => l.Id == lessonId);
            if (!lessonExists)
                throw new KeyNotFoundException("Lesson not found");

            var userLesson = await _context.UserLessons
                .FirstOrDefaultAsync(ul => ul.UserId == userId && ul.LessonId == lessonId);

            const int xpReward = 10; // XP earned per lesson completion
            if (userLesson == null)
            {
                userLesson = new UserLesson
                {
                    UserId = userId,
                    LessonId = lessonId,
                    CompletionPercentage = 100,
                    IsCompleted = true,
                    CompletedAt = DateTime.UtcNow,
                    TotalAttempts = 1
                };
                _context.UserLessons.Add(userLesson);

                // Award XP to user
                user.TotalXp += xpReward;
                _logger.LogInformation("User {UserId} completed lesson {LessonId}, earned {XP} XP. Total: {TotalXP}", 
                    userId, lessonId, xpReward, user.TotalXp);
            }
            else if (!userLesson.IsCompleted)
            {
                userLesson.CompletionPercentage = 100;
                userLesson.IsCompleted = true;
                userLesson.CompletedAt = DateTime.UtcNow;
                userLesson.TotalAttempts++;

                // Award XP to user
                user.TotalXp += xpReward;
                _logger.LogInformation("User {UserId} completed lesson {LessonId}, earned {XP} XP. Total: {TotalXP}", 
                    userId, lessonId, xpReward, user.TotalXp);
            }
            else
            {
                userLesson.TotalAttempts++;
                _logger.LogInformation("User {UserId} reviewed lesson {LessonId} (already completed)", userId, lessonId);
            }

            // Add to spaced repetition
            var existingRep = await _context.SpacedRepetitionLessons
                .FirstOrDefaultAsync(sr => sr.UserId == userId && sr.LessonId == lessonId);

            if (existingRep == null)
            {
                var newRep = new SpacedRepetitionLesson
                {
                    UserId = userId,
                    LessonId = lessonId,
                    DueDate = DateTime.UtcNow.AddDays(1),
                    RepetitionCount = 0,
                    RetentionPercentage = 100,
                    Interval = 1,
                    EaseFactor = 2.5
                };
                _context.SpacedRepetitionLessons.Add(newRep);
            }

            // Save to database
            await _context.SaveChangesAsync();

            // Save progress to file
            try
            {
                await _progressService.SaveUserProgressAsync(user.Id, user.TotalXp, user.LearningStreak, DateTime.UtcNow);
            }
            catch (Exception ex)
            {
                _logger.LogWarning(ex, "Failed to save user progress to file for UserId={UserId}", userId);
                // Don't fail the operation if file save fails
            }
        }

        public async Task<List<SpacedRepetitionLessonDto>> GetDailyReviewLessonsAsync(string userId)
        {
            var utcNow = DateTime.UtcNow;

            var reviews = await _context.SpacedRepetitionLessons
                .Where(sr => sr.UserId == userId && sr.DueDate <= utcNow)
                .Include(sr => sr.Lesson)
                .OrderBy(sr => sr.DueDate)
                .ToListAsync();

            return reviews.Select(r => new SpacedRepetitionLessonDto
            {
                Id = r.Id,
                Title = r.Lesson?.Title ?? string.Empty,
                DueDate = r.DueDate.ToString("O"),
                RepetitionCount = r.RepetitionCount,
                RetentionPercentage = r.RetentionPercentage,
                IsReviewDue = r.DueDate <= utcNow,
                LessonId = r.LessonId
            }).ToList();
        }

        public async Task ReviewLessonAsync(string userId, int spacedRepetitionId, double qualityRating)
        {
            var user = await _context.Users.FindAsync(userId);
            if (user == null)
                throw new InvalidOperationException("User not found");

            var spacedRep = await _context.SpacedRepetitionLessons.FindAsync(spacedRepetitionId);
            if (spacedRep == null || spacedRep.UserId != userId)
                throw new KeyNotFoundException("Spaced repetition lesson not found");

            // SM-2 Algorithm
            spacedRep.RepetitionCount++;
            spacedRep.LastReviewedAt = DateTime.UtcNow;

            if (qualityRating >= 3)
            {
                spacedRep.EaseFactor = Math.Max(1.3, spacedRep.EaseFactor + 0.1 - (5 - qualityRating) * (0.08 + (5 - qualityRating) * 0.02));
                if (spacedRep.RepetitionCount == 1)
                    spacedRep.Interval = 1;
                else if (spacedRep.RepetitionCount == 2)
                    spacedRep.Interval = 3;
                else
                    spacedRep.Interval = (int)Math.Ceiling(spacedRep.Interval * spacedRep.EaseFactor);
            }
            else
            {
                spacedRep.RepetitionCount = 0;
                spacedRep.Interval = 1;
                spacedRep.EaseFactor = 2.5;
            }

            spacedRep.DueDate = DateTime.UtcNow.AddDays(spacedRep.Interval);
            spacedRep.RetentionPercentage = Math.Min(100, spacedRep.RetentionPercentage + (qualityRating * 10));

            // Award XP for successful review (quality rating >= 3)
            if (qualityRating >= 3)
            {
                const int reviewXpReward = 5;
                user.TotalXp += reviewXpReward;
                _logger.LogInformation("User {UserId} completed review for lesson {LessonId}, earned {XP} XP. Total: {TotalXP}", 
                    userId, spacedRep.LessonId, reviewXpReward, user.TotalXp);
            }

            await _context.SaveChangesAsync();

            // Save progress to file
            try
            {
                await _progressService.SaveUserProgressAsync(user.Id, user.TotalXp, user.LearningStreak, DateTime.UtcNow);
            }
            catch (Exception ex)
            {
                _logger.LogWarning(ex, "Failed to save user progress to file for UserId={UserId}", userId);
                // Don't fail the operation if file save fails
            }
        }

        private string GetRecommendation(List<LessonDto> lessons)
        {
            if (lessons.Count == 0)
                return "Start learning sign language with our beginner-friendly lessons!";

            var avgCompletion = lessons.Average(l => l.CompletionPercentage);
            if (avgCompletion < 0.25)
                return "You're just getting started! Keep up the momentum.";
            if (avgCompletion < 0.5)
                return "Great progress! Continue with the next category.";
            if (avgCompletion < 0.75)
                return "You're doing amazing! Time to tackle more advanced lessons.";

            return "You've come a long way! Challenge yourself with expert-level content.";
        }

        private LessonDataDto BuildLessonData(
            int lessonId,
            int durationSeconds,
            string difficulty,
            double completionPercentage,
            string instructorName,
            int categoryId)
        {
            return new LessonDataDto
            {
                DurationSeconds = durationSeconds,
                Difficulty = difficulty,
                CompletionPercentage = completionPercentage,
                InstructorName = instructorName,
                CategoryId = categoryId,
                UiLayout = GetUiLayoutForLesson(lessonId)
            };
        }

        private LessonUiLayoutDto GetUiLayoutForLesson(int lessonId)
        {
            if (lessonId == 7)
                return BuildRealtimeLayout("RealtimeHandSignalPracticeSet1View");

            if (lessonId == 8)
                return BuildRealtimeLayout("RealtimeHandSignalPracticeSet2View");

            if (lessonId == 9)
                return BuildRealtimeLayout("RealtimeHandSignalPracticeSet3View");

            if (lessonId == 10)
                return BuildLessonLayout(
                    "FamilySignsView",
                    "Family Signs",
                    "Practice signs for mother, father, sister, brother, and family.");

            if (lessonId == 11)
                return BuildLessonLayout(
                    "DaysAndTimeView",
                    "Days and Time",
                    "Practice signs for weekdays, today, tomorrow, and common time phrases.");

            if (lessonId == 12)
                return BuildLessonLayout(
                    "ShoppingDialoguesView",
                    "Shopping Dialogues",
                    "Practice transactional phrases: price, payment, receipt, and help.");

            if (lessonId == 13)
                return BuildLessonLayout(
                    "SchoolWorkPhrasesView",
                    "School & Work Phrases",
                    "Practice communication used in class and workplace settings.");

            if (lessonId == 14)
                return BuildLessonLayout(
                    "NarrativeClassifiersView",
                    "Narrative Classifiers",
                    "Apply classifier storytelling with movement and spatial setup.");

            return new LessonUiLayoutDto
            {
                FileName = "LessonView.xaml",
                XamlContent = "<ContentPage xmlns=\"http://schemas.microsoft.com/dotnet/2021/maui\" xmlns:x=\"http://schemas.microsoft.com/winfx/2009/xaml\" x:Class=\"App.LessonView\"><VerticalStackLayout Padding=\"16\" Spacing=\"12\"><Label Text=\"Lesson Content\" FontAttributes=\"Bold\" FontSize=\"20\"/><Label Text=\"Follow the instructions, then complete practice.\"/></VerticalStackLayout></ContentPage>",
                CodeBehindContent = "namespace App;\\n\\npublic partial class LessonView : ContentPage\\n{\\n    public LessonView()\\n    {\\n        InitializeComponent();\\n    }\\n}"
            };
        }

        private LessonUiLayoutDto BuildLessonLayout(string className, string title, string instructions)
        {
            return new LessonUiLayoutDto
            {
                FileName = $"{className}.xaml",
                XamlContent = $"<ContentPage xmlns=\"http://schemas.microsoft.com/dotnet/2021/maui\" xmlns:x=\"http://schemas.microsoft.com/winfx/2009/xaml\" x:Class=\"App.{className}\"><VerticalStackLayout Padding=\"16\" Spacing=\"12\"><Label Text=\"{title}\" FontAttributes=\"Bold\" FontSize=\"20\"/><Label Text=\"{instructions}\"/><Button Text=\"Start Practice\"/><Button Text=\"Mark Complete\"/></VerticalStackLayout></ContentPage>",
                CodeBehindContent = $"namespace App;\\n\\npublic partial class {className} : ContentPage\\n{{\\n    public {className}()\\n    {{\\n        InitializeComponent();\\n    }}\\n}}"
            };
        }

        private LessonUiLayoutDto BuildRealtimeLayout(string className)
        {
            return new LessonUiLayoutDto
            {
                FileName = $"{className}.xaml",
                XamlContent = $"<ContentPage xmlns=\"http://schemas.microsoft.com/dotnet/2021/maui\" xmlns:x=\"http://schemas.microsoft.com/winfx/2009/xaml\" x:Class=\"App.{className}\"><Grid RowDefinitions=\"Auto,*,Auto\"><Label Text=\"Real-time Hand Signal Practice\" FontAttributes=\"Bold\" FontSize=\"20\" Padding=\"16\"/><Border Grid.Row=\"1\" StrokeThickness=\"1\" Stroke=\"#BDBDBD\"><Label Text=\"[Camera Preview Placeholder]\" HorizontalOptions=\"Center\" VerticalOptions=\"Center\"/></Border><HorizontalStackLayout Grid.Row=\"2\" Padding=\"16\" Spacing=\"12\"><Button Text=\"Start Camera\"/><Button Text=\"Capture & Predict\"/><Label Text=\"Prediction: -\" VerticalOptions=\"Center\"/></HorizontalStackLayout></Grid></ContentPage>",
                CodeBehindContent = $"namespace App;\\n\\npublic partial class {className} : ContentPage\\n{{\\n    public {className}()\\n    {{\\n        InitializeComponent();\\n    }}\\n\\n    // Capture frame then POST to /api/gesture/predict with JWT token\\n}}"
            };
        }

        public async Task<List<LessonCategoryDto>> GetAllCategoriesAsync(string userId)
        {
            var user = await _context.Users.FindAsync(userId);
            if (user == null)
                throw new InvalidOperationException("User not found");

            var categories = await _context.LessonCategories
                .Include(c => c.Lessons)
                .ToListAsync();

            var userLessons = await _context.UserLessons
                .Where(ul => ul.UserId == userId)
                .Include(ul => ul.Lesson)
                .ToListAsync();

            return categories.Select(c =>
            {
                var categoryLessons = userLessons.Where(ul => ul.Lesson?.CategoryId == c.Id).ToList();
                var progress = categoryLessons.Count > 0
                    ? categoryLessons.Average(ul => ul.CompletionPercentage) / 100
                    : 0;

                return new LessonCategoryDto
                {
                    Id = c.Id,
                    Title = c.Title,
                    Description = c.Description,
                    Difficulty = c.Difficulty,
                    IconUrl = c.IconUrl,
                    Progress = progress
                };
            }).ToList();
        }

        public async Task<LessonCategoryDto?> GetCategoryAsync(int categoryId, string userId)
        {
            var user = await _context.Users.FindAsync(userId);
            if (user == null)
                throw new InvalidOperationException("User not found");

            var category = await _context.LessonCategories
                .Include(c => c.Lessons)
                .FirstOrDefaultAsync(c => c.Id == categoryId);

            if (category == null)
                return null;

            var userLessons = await _context.UserLessons
                .Where(ul => ul.UserId == userId && ul.Lesson!.CategoryId == categoryId)
                .Include(ul => ul.Lesson)
                .ToListAsync();

            var progress = userLessons.Count > 0
                ? userLessons.Average(ul => ul.CompletionPercentage) / 100
                : 0;

            return new LessonCategoryDto
            {
                Id = category.Id,
                Title = category.Title,
                Description = category.Description,
                Difficulty = category.Difficulty,
                IconUrl = category.IconUrl,
                Progress = progress
            };
        }

        public async Task<DailyGoalDto> GetDailyGoalAsync(string userId)
        {
            var utcNow = DateTime.UtcNow;

            var user = await _context.Users.FindAsync(userId);
            if (user == null)
                throw new InvalidOperationException("User not found");

            var reviewsDue = await _context.SpacedRepetitionLessons
                .Where(sr => sr.UserId == userId && sr.DueDate <= utcNow)
                .CountAsync();

            var completedToday = await _context.UserLessons
                .Where(ul => ul.UserId == userId && ul.IsCompleted && ul.CompletedAt!.Value.Date == utcNow.Date)
                .CountAsync();

            const int dailyGoal = 5;
            var progressPercentage = dailyGoal > 0 ? (completedToday / (double)dailyGoal) * 100 : 0;

            return new DailyGoalDto
            {
                TotalReviewsDue = reviewsDue,
                CompletedToday = completedToday,
                DailyGoal = dailyGoal,
                ProgressPercentage = Math.Min(100, progressPercentage)
            };
        }

        public async Task<UpcomingReviewsDto> GetUpcomingReviewsAsync(string userId)
        {
            var user = await _context.Users.FindAsync(userId);
            if (user == null)
                throw new InvalidOperationException("User not found");

            var today = DateTime.UtcNow.Date;
            var tomorrow = today.AddDays(1);
            var weekEnd = today.AddDays(7);

            var dueToday = await _context.SpacedRepetitionLessons
                .Where(sr => sr.UserId == userId && sr.DueDate.Date == today)
                .CountAsync();

            var dueTomorrow = await _context.SpacedRepetitionLessons
                .Where(sr => sr.UserId == userId && sr.DueDate.Date == tomorrow)
                .CountAsync();

            var dueThisWeek = await _context.SpacedRepetitionLessons
                .Where(sr => sr.UserId == userId && sr.DueDate.Date > tomorrow && sr.DueDate.Date <= weekEnd)
                .CountAsync();

            var overdue = await _context.SpacedRepetitionLessons
                .Where(sr => sr.UserId == userId && sr.DueDate.Date < today)
                .CountAsync();

            return new UpcomingReviewsDto
            {
                DueToday = dueToday,
                DueTomorrow = dueTomorrow,
                DueThisWeek = dueThisWeek,
                Overdue = overdue
            };
        }

        public async Task<PersonalizedRecommendationDto> GetPersonalizedRecommendationAsync(string userId)
        {
            var user = await _context.Users.FindAsync(userId);
            if (user == null)
                throw new InvalidOperationException("User not found");

            var userLessons = await _context.UserLessons
                .Where(ul => ul.UserId == userId)
                .Include(ul => ul.Lesson)
                .ToListAsync();

            var allLessons = await _context.Lessons
                .Include(l => l.Category)
                .ToListAsync();

            // Find the category with the lowest average progress
            var categoryProgress = allLessons
                .GroupBy(l => l.CategoryId)
                .Select(g => new
                {
                    CategoryId = g.Key,
                    Category = g.First().Category,
                    AverageProgress = userLessons
                        .Where(ul => ul.Lesson?.CategoryId == g.Key)
                        .Average(ul => (double?)ul.CompletionPercentage) ?? 0
                })
                .OrderBy(x => x.AverageProgress)
                .FirstOrDefault();

            if (categoryProgress == null)
            {
                // No categories, recommend the first lesson
                var firstLesson = allLessons.FirstOrDefault();
                if (firstLesson == null)
                    throw new InvalidOperationException("No lessons available");

                return new PersonalizedRecommendationDto
                {
                    RecommendedLessonId = firstLesson.Id,
                    LessonTitle = firstLesson.Title,
                    LessonDescription = firstLesson.Description,
                    CategoryId = firstLesson.CategoryId,
                    CategoryName = firstLesson.Category?.Title ?? "Unknown",
                    Reason = "Start with our beginner-friendly lessons!",
                    CurrentProgress = 0,
                    Difficulty = firstLesson.Difficulty
                };
            }

            // Find an incomplete lesson in the recommended category
            var recommendedLesson = allLessons
                .Where(l => l.CategoryId == categoryProgress.CategoryId)
                .FirstOrDefault(l => !userLessons.Any(ul => ul.LessonId == l.Id && ul.IsCompleted));

            // If all lessons in the category are complete, get the first incomplete lesson overall
            if (recommendedLesson == null)
            {
                recommendedLesson = allLessons
                    .FirstOrDefault(l => !userLessons.Any(ul => ul.LessonId == l.Id && ul.IsCompleted));
            }

            // If still no lesson found, recommend the least completed one
            if (recommendedLesson == null)
            {
                var lessonProgress = userLessons
                    .OrderBy(ul => ul.CompletionPercentage)
                    .FirstOrDefault();

                recommendedLesson = lessonProgress?.Lesson;
            }

            if (recommendedLesson == null)
                throw new InvalidOperationException("No lessons available");

            var reason = categoryProgress.AverageProgress < 25
                ? "You need to focus on this category!"
                : categoryProgress.AverageProgress < 50
                ? "Keep improving in this category."
                : "Challenge yourself with the next category!";

            return new PersonalizedRecommendationDto
            {
                RecommendedLessonId = recommendedLesson.Id,
                LessonTitle = recommendedLesson.Title,
                LessonDescription = recommendedLesson.Description,
                CategoryId = recommendedLesson.CategoryId,
                CategoryName = recommendedLesson.Category?.Title ?? "Unknown",
                Reason = reason,
                CurrentProgress = categoryProgress.AverageProgress / 100.0,
                Difficulty = recommendedLesson.Difficulty
            };
        }
    }
}
