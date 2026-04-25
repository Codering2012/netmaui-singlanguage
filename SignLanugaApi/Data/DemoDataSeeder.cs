using Microsoft.EntityFrameworkCore;

namespace SignLanguageApi.Data
{
    public static class DemoDataSeeder
    {
        public static async Task SeedAsync(AppDbContext context, ILogger logger)
        {
            var utcNow = DateTime.UtcNow;

            if (!await context.LessonCategories.AnyAsync())
            {
                context.LessonCategories.AddRange(
                    new LessonCategory
                    {
                        Id = 1,
                        Title = "Basics",
                        Description = "Foundational signs for daily communication.",
                        Difficulty = "Beginner",
                        IconUrl = "https://images.unsplash.com/photo-1455390582262-044cdead277a?w=300"
                    },
                    new LessonCategory
                    {
                        Id = 2,
                        Title = "Conversations",
                        Description = "Practical conversational phrases and responses.",
                        Difficulty = "Intermediate",
                        IconUrl = "https://images.unsplash.com/photo-1521737604893-d14cc237f11d?w=300"
                    },
                    new LessonCategory
                    {
                        Id = 3,
                        Title = "Advanced Grammar",
                        Description = "Sentence structure and complex sign patterns.",
                        Difficulty = "Advanced",
                        IconUrl = "https://images.unsplash.com/photo-1450101499163-c8848c66ca85?w=300"
                    });
            }

            if (!await context.Lessons.AnyAsync())
            {
                context.Lessons.AddRange(
                    new Lesson { Id = 1, CategoryId = 1, Title = "Alphabet A-M", Description = "Learn letters A through M.", Difficulty = "Beginner", DurationSeconds = 420, InstructorName = "Sarah Kim", ThumbnailUrl = "https://images.unsplash.com/photo-1513258496099-48168024aec0?w=640" },
                    new Lesson { Id = 2, CategoryId = 1, Title = "Alphabet N-Z", Description = "Learn letters N through Z.", Difficulty = "Beginner", DurationSeconds = 420, InstructorName = "Sarah Kim", ThumbnailUrl = "https://images.unsplash.com/photo-1523240795612-9a054b0db644?w=640" },
                    new Lesson { Id = 3, CategoryId = 1, Title = "Numbers 1-20", Description = "Sign common numbers for everyday use.", Difficulty = "Beginner", DurationSeconds = 360, InstructorName = "Daniel Chen", ThumbnailUrl = "https://images.unsplash.com/photo-1588072432836-e10032774350?w=640" },
                    new Lesson { Id = 4, CategoryId = 2, Title = "Greetings", Description = "Say hello, goodbye, and introduce yourself.", Difficulty = "Intermediate", DurationSeconds = 480, InstructorName = "Ava Brooks", ThumbnailUrl = "https://images.unsplash.com/photo-1529156069898-49953e39b3ac?w=640" },
                    new Lesson { Id = 5, CategoryId = 2, Title = "Questions", Description = "Ask common questions clearly and confidently.", Difficulty = "Intermediate", DurationSeconds = 540, InstructorName = "Ava Brooks", ThumbnailUrl = "https://images.unsplash.com/photo-1522202176988-66273c2fd55f?w=640" },
                    new Lesson { Id = 6, CategoryId = 3, Title = "Classifier Basics", Description = "Use classifiers for spatial and movement concepts.", Difficulty = "Advanced", DurationSeconds = 720, InstructorName = "Luis Ortega", ThumbnailUrl = "https://images.unsplash.com/photo-1434030216411-0b793f4b4173?w=640" },
                    new Lesson { Id = 7, CategoryId = 1, Title = "Camera Practice Set 1: Alphabet", Description = "Real-time camera practice for basic alphabet hand signs.", Difficulty = "Beginner", DurationSeconds = 600, InstructorName = "Realtime Coach", ThumbnailUrl = "https://images.unsplash.com/photo-1516321497487-e288fb19713f?w=640" },
                    new Lesson { Id = 8, CategoryId = 2, Title = "Camera Practice Set 2: Greetings", Description = "Use camera mode to practice greeting signs with instant prediction.", Difficulty = "Intermediate", DurationSeconds = 720, InstructorName = "Realtime Coach", ThumbnailUrl = "https://images.unsplash.com/photo-1522075469751-3a6694fb2f61?w=640" },
                    new Lesson { Id = 9, CategoryId = 3, Title = "Camera Practice Set 3: Advanced Phrases", Description = "Advanced real-time drills using gesture recognition feedback.", Difficulty = "Advanced", DurationSeconds = 900, InstructorName = "Realtime Coach", ThumbnailUrl = "https://images.unsplash.com/photo-1509062522246-3755977927d7?w=640" },
                    new Lesson { Id = 10, CategoryId = 1, Title = "Family Signs", Description = "Learn common signs for family members and relationships.", Difficulty = "Beginner", DurationSeconds = 510, InstructorName = "Sarah Kim", ThumbnailUrl = "https://images.unsplash.com/photo-1511895426328-dc8714191300?w=640" },
                    new Lesson { Id = 11, CategoryId = 1, Title = "Days and Time", Description = "Practice signs for days of the week and daily time expressions.", Difficulty = "Beginner", DurationSeconds = 540, InstructorName = "Daniel Chen", ThumbnailUrl = "https://images.unsplash.com/photo-1506784983877-45594efa4cbe?w=640" },
                    new Lesson { Id = 12, CategoryId = 2, Title = "Shopping Dialogues", Description = "Use practical signs for buying, asking prices, and checkout.", Difficulty = "Intermediate", DurationSeconds = 660, InstructorName = "Ava Brooks", ThumbnailUrl = "https://images.unsplash.com/photo-1472851294608-062f824d29cc?w=640" },
                    new Lesson { Id = 13, CategoryId = 2, Title = "School & Work Phrases", Description = "Build confidence with signs used in classroom and workplace conversations.", Difficulty = "Intermediate", DurationSeconds = 690, InstructorName = "Ava Brooks", ThumbnailUrl = "https://images.unsplash.com/photo-1454165804606-c3d57bc86b40?w=640" },
                    new Lesson { Id = 14, CategoryId = 3, Title = "Narrative Classifiers", Description = "Apply classifiers in storytelling with movement and spatial context.", Difficulty = "Advanced", DurationSeconds = 840, InstructorName = "Luis Ortega", ThumbnailUrl = "https://images.unsplash.com/photo-1456324504439-367cee3b3c32?w=640" });
            }

            if (!await context.Achievements.AnyAsync())
            {
                context.Achievements.AddRange(
                    new Achievement { Id = 1, Title = "First Steps", Description = "Complete your first lesson.", RequiredPoints = 10, BadgeColor = "#10B981", IconChar = "🌟" },
                    new Achievement { Id = 2, Title = "Consistency", Description = "Keep a 3-day learning streak.", RequiredPoints = 30, BadgeColor = "#3B82F6", IconChar = "🔥" },
                    new Achievement { Id = 3, Title = "Dedicated Learner", Description = "Reach 100 XP.", RequiredPoints = 100, BadgeColor = "#F59E0B", IconChar = "🏆" });
            }

            var demoUser = await context.Users.FirstOrDefaultAsync(u => u.Email == "demo@signlanguage.app");
            if (demoUser == null)
            {
                demoUser = new User
                {
                    Id = "demo-user-001",
                    Email = "demo@signlanguage.app",
                    Name = "Demo Learner",
                    PasswordHash = BCrypt.Net.BCrypt.HashPassword("DemoPass123!"),
                    LearningStreak = 2,
                    TotalXp = 45,
                    CreatedAt = utcNow.AddDays(-10),
                    LastLoginAt = utcNow.AddHours(-2)
                };

                context.Users.Add(demoUser);
            }

            await context.SaveChangesAsync();

            if (!await context.UserLessons.AnyAsync(ul => ul.UserId == demoUser.Id))
            {
                context.UserLessons.AddRange(
                    new UserLesson { UserId = demoUser.Id, LessonId = 1, CompletionPercentage = 100, IsCompleted = true, CompletedAt = utcNow.AddDays(-3), StartedAt = utcNow.AddDays(-3), TotalAttempts = 2 },
                    new UserLesson { UserId = demoUser.Id, LessonId = 2, CompletionPercentage = 70, IsCompleted = false, StartedAt = utcNow.AddDays(-2), TotalAttempts = 3 },
                    new UserLesson { UserId = demoUser.Id, LessonId = 4, CompletionPercentage = 30, IsCompleted = false, StartedAt = utcNow.AddDays(-1), TotalAttempts = 1 },
                    new UserLesson { UserId = demoUser.Id, LessonId = 7, CompletionPercentage = 40, IsCompleted = false, StartedAt = utcNow.AddHours(-12), TotalAttempts = 2 },
                    new UserLesson { UserId = demoUser.Id, LessonId = 8, CompletionPercentage = 15, IsCompleted = false, StartedAt = utcNow.AddHours(-6), TotalAttempts = 1 },
                    new UserLesson { UserId = demoUser.Id, LessonId = 9, CompletionPercentage = 0, IsCompleted = false, StartedAt = utcNow.AddHours(-2), TotalAttempts = 1 });
            }

            if (!await context.SpacedRepetitionLessons.AnyAsync(sr => sr.UserId == demoUser.Id))
            {
                context.SpacedRepetitionLessons.AddRange(
                    new SpacedRepetitionLesson { UserId = demoUser.Id, LessonId = 1, DueDate = utcNow.AddHours(-3), RepetitionCount = 2, RetentionPercentage = 82, Interval = 3, EaseFactor = 2.4, LastReviewedAt = utcNow.AddDays(-2) },
                    new SpacedRepetitionLesson { UserId = demoUser.Id, LessonId = 2, DueDate = utcNow.AddDays(1), RepetitionCount = 1, RetentionPercentage = 65, Interval = 1, EaseFactor = 2.5, LastReviewedAt = utcNow.AddDays(-1) },
                    new SpacedRepetitionLesson { UserId = demoUser.Id, LessonId = 4, DueDate = utcNow.AddDays(4), RepetitionCount = 0, RetentionPercentage = 50, Interval = 1, EaseFactor = 2.5, LastReviewedAt = utcNow.AddDays(-1) });
            }

            if (!await context.UserAchievements.AnyAsync(ua => ua.UserId == demoUser.Id))
            {
                context.UserAchievements.Add(new UserAchievement
                {
                    UserId = demoUser.Id,
                    AchievementId = 1,
                    UnlockedAt = utcNow.AddDays(-3)
                });
            }

            await context.SaveChangesAsync();

            logger.LogInformation("Demo data ready. Demo account: demo@signlanguage.app / DemoPass123!");
        }
    }
}
