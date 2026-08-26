using Microsoft.EntityFrameworkCore;

namespace SignLanguageApi.Data
{
    public static class DemoDataSeeder
    {
        public static async Task SeedAsync(AppDbContext context, ILogger logger)
        {
            var utcNow = DateTime.UtcNow;

            // 1. Categories
            if (!await context.LessonCategories.AnyAsync())
            {
                context.LessonCategories.AddRange(
                    new LessonCategory
                    {
                        Id = 1,
                        Title = "ASL Alphabet & Letters",
                        Description = "Master all 26 manual alphabet handshapes from A to Z.",
                        Difficulty = "Beginner",
                        IconUrl = "/api/media/image/alphabet.png"
                    },
                    new LessonCategory
                    {
                        Id = 2,
                        Title = "Conversations & Greetings",
                        Description = "Practical conversational phrases and everyday greetings.",
                        Difficulty = "Intermediate",
                        IconUrl = "/api/media/image/phrases.png"
                    },
                    new LessonCategory
                    {
                        Id = 3,
                        Title = "Advanced Grammar & Classifiers",
                        Description = "Spatial placement, classifiers, and complex sign structures.",
                        Difficulty = "Advanced",
                        IconUrl = "/api/media/image/dynamic.png"
                    },
                    new LessonCategory
                    {
                        Id = 4,
                        Title = "Numbers & Everyday Vocabulary",
                        Description = "Count numbers 1-20, tell time, and sign family members.",
                        Difficulty = "Beginner",
                        IconUrl = "/api/media/image/basics.png"
                    },
                    new LessonCategory
                    {
                        Id = 5,
                        Title = "Real-Time Camera Drills",
                        Description = "Interactive camera-based sign language practice.",
                        Difficulty = "All Levels",
                        IconUrl = "/api/media/image/spelling_practice.png"
                    });

                await context.SaveChangesAsync();
            }

            // 2. Lessons
            if (!await context.Lessons.AnyAsync())
            {
                var lessons = new List<Lesson>();
                int lessonId = 1;

                // Category 1: ASL Alphabet A-Z (26 Individual Letter Lessons)
                string[] letters = { "A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K", "L", "M", "N", "O", "P", "Q", "R", "S", "T", "U", "V", "W", "X", "Y", "Z" };
                foreach (var letter in letters)
                {
                    lessons.Add(new Lesson
                    {
                        Id = lessonId++,
                        CategoryId = 1,
                        Title = $"Letter {letter}",
                        Description = $"Learn the official ASL handshape for the letter '{letter}' with high-definition video demonstration.",
                        Difficulty = "Beginner",
                        DurationSeconds = 120,
                        InstructorName = "Sarah Kim (ASL Specialist)",
                        ThumbnailUrl = $"/api/media/image/letters_{(letter.ToLower() == "a" ? "a" : letter.ToLower() == "b" ? "b" : "a")}.png",
                        VideoUrl = $"/api/videos/letter/{letter}"
                    });
                }

                // Group Alphabet Lessons
                lessons.Add(new Lesson
                {
                    Id = lessonId++,
                    CategoryId = 1,
                    Title = "Alphabet Mastery: A through M",
                    Description = "Comprehensive video drill for letters A to M.",
                    Difficulty = "Beginner",
                    DurationSeconds = 420,
                    InstructorName = "Sarah Kim",
                    ThumbnailUrl = "/api/media/image/letters_a_m.png",
                    VideoUrl = "/api/videos/letter/A"
                });

                lessons.Add(new Lesson
                {
                    Id = lessonId++,
                    CategoryId = 1,
                    Title = "Alphabet Mastery: N through Z",
                    Description = "Comprehensive video drill for letters N to Z.",
                    Difficulty = "Beginner",
                    DurationSeconds = 420,
                    InstructorName = "Sarah Kim",
                    ThumbnailUrl = "/api/media/image/letters_n_z.png",
                    VideoUrl = "/api/videos/letter/N"
                });

                // Category 2: Conversations & Greetings
                lessons.Add(new Lesson { Id = lessonId++, CategoryId = 2, Title = "Greetings & Introductions", Description = "Say hello, goodbye, and introduce yourself in ASL.", Difficulty = "Intermediate", DurationSeconds = 480, InstructorName = "Ava Brooks", ThumbnailUrl = "/api/media/image/phrases.png", VideoUrl = "/api/videos/letter/H" });
                lessons.Add(new Lesson { Id = lessonId++, CategoryId = 2, Title = "Common Questions", Description = "Ask WHO, WHAT, WHERE, WHEN, and WHY in ASL.", Difficulty = "Intermediate", DurationSeconds = 540, InstructorName = "Ava Brooks", ThumbnailUrl = "/api/media/image/phrases.png", VideoUrl = "/api/videos/letter/W" });
                lessons.Add(new Lesson { Id = lessonId++, CategoryId = 2, Title = "Shopping & Prices", Description = "Practical signs for buying, asking prices, and checkout.", Difficulty = "Intermediate", DurationSeconds = 660, InstructorName = "Ava Brooks", ThumbnailUrl = "/api/media/image/spelling_practice.png" });
                lessons.Add(new Lesson { Id = lessonId++, CategoryId = 2, Title = "Workplace & School", Description = "Signs used in classroom and office environments.", Difficulty = "Intermediate", DurationSeconds = 690, InstructorName = "Ava Brooks", ThumbnailUrl = "/api/media/image/basics.png" });

                // Category 3: Advanced Grammar & Classifiers
                lessons.Add(new Lesson { Id = lessonId++, CategoryId = 3, Title = "Classifier Basics (CL:1, CL:3)", Description = "Use classifiers for spatial movement and object descriptions.", Difficulty = "Advanced", DurationSeconds = 720, InstructorName = "Luis Ortega", ThumbnailUrl = "/api/media/image/dynamic.png" });
                lessons.Add(new Lesson { Id = lessonId++, CategoryId = 3, Title = "Narrative Classifiers & Storytelling", Description = "Apply spatial agreement and classifiers in fluid ASL narratives.", Difficulty = "Advanced", DurationSeconds = 840, InstructorName = "Luis Ortega", ThumbnailUrl = "/api/media/image/dynamic.png" });

                // Category 4: Numbers & Everyday Vocabulary
                lessons.Add(new Lesson { Id = lessonId++, CategoryId = 4, Title = "Numbers 1-20", Description = "Sign numbers clearly for everyday counting.", Difficulty = "Beginner", DurationSeconds = 360, InstructorName = "Daniel Chen", ThumbnailUrl = "/api/media/image/basics.png" });
                lessons.Add(new Lesson { Id = lessonId++, CategoryId = 4, Title = "Family & Relationships", Description = "Learn signs for Mother, Father, Friend, and Siblings.", Difficulty = "Beginner", DurationSeconds = 510, InstructorName = "Sarah Kim", ThumbnailUrl = "/api/media/image/basics.png" });
                lessons.Add(new Lesson { Id = lessonId++, CategoryId = 4, Title = "Days, Months & Time", Description = "Signs for days of the week, months, and time of day.", Difficulty = "Beginner", DurationSeconds = 540, InstructorName = "Daniel Chen", ThumbnailUrl = "/api/media/image/basics.png" });

                // Category 5: Real-Time Camera Practice Drills
                lessons.Add(new Lesson { Id = lessonId++, CategoryId = 5, Title = "Camera Practice: Alphabet Drill", Description = "Real-time camera practice with immediate AI hand landmarker feedback.", Difficulty = "Beginner", DurationSeconds = 600, InstructorName = "Realtime AI Coach", ThumbnailUrl = "/api/media/image/spelling_practice.png" });
                lessons.Add(new Lesson { Id = lessonId++, CategoryId = 5, Title = "Camera Practice: Dynamic Gestures", Description = "Practice dynamic motion signs with camera recognition.", Difficulty = "Intermediate", DurationSeconds = 720, InstructorName = "Realtime AI Coach", ThumbnailUrl = "/api/media/image/speed_test.png" });

                context.Lessons.AddRange(lessons);
                await context.SaveChangesAsync();
            }

            // 3. Achievements
            if (!await context.Achievements.AnyAsync())
            {
                context.Achievements.AddRange(
                    new Achievement { Id = 1, Title = "First Steps", Description = "Complete your first ASL lesson.", RequiredPoints = 10, BadgeColor = "#10B981", IconChar = "🌟" },
                    new Achievement { Id = 2, Title = "Streak Master", Description = "Maintain a 3-day learning streak.", RequiredPoints = 30, BadgeColor = "#3B82F6", IconChar = "🔥" },
                    new Achievement { Id = 3, Title = "Alphabet Champion", Description = "Master all 26 ASL alphabet letters.", RequiredPoints = 100, BadgeColor = "#F59E0B", IconChar = "🏆" },
                    new Achievement { Id = 4, Title = "Camera Prodigy", Description = "Complete 5 real-time camera drills.", RequiredPoints = 150, BadgeColor = "#8B5CF6", IconChar = "📷" });

                await context.SaveChangesAsync();
            }

            // 4. Demo User & Seed Progress
            var demoUser = await context.Users.FirstOrDefaultAsync(u => u.Email == "demo@signlanguage.app");
            if (demoUser == null)
            {
                demoUser = new User
                {
                    Id = "demo-user-001",
                    Email = "demo@signlanguage.app",
                    Name = "Demo Learner",
                    PasswordHash = BCrypt.Net.BCrypt.HashPassword("DemoPass123!"),
                    LearningStreak = 3,
                    TotalXp = 120,
                    CreatedAt = utcNow.AddDays(-10),
                    LastLoginAt = utcNow.AddHours(-1)
                };

                context.Users.Add(demoUser);
                await context.SaveChangesAsync();
            }

            // User Lessons
            if (!await context.UserLessons.AnyAsync(ul => ul.UserId == demoUser.Id))
            {
                context.UserLessons.AddRange(
                    new UserLesson { UserId = demoUser.Id, LessonId = 1, CompletionPercentage = 100, IsCompleted = true, CompletedAt = utcNow.AddDays(-3), StartedAt = utcNow.AddDays(-3), TotalAttempts = 2 },
                    new UserLesson { UserId = demoUser.Id, LessonId = 2, CompletionPercentage = 100, IsCompleted = true, CompletedAt = utcNow.AddDays(-2), StartedAt = utcNow.AddDays(-2), TotalAttempts = 1 },
                    new UserLesson { UserId = demoUser.Id, LessonId = 3, CompletionPercentage = 80, IsCompleted = false, StartedAt = utcNow.AddDays(-1), TotalAttempts = 3 },
                    new UserLesson { UserId = demoUser.Id, LessonId = 27, CompletionPercentage = 100, IsCompleted = true, CompletedAt = utcNow.AddHours(-12), StartedAt = utcNow.AddHours(-12), TotalAttempts = 1 },
                    new UserLesson { UserId = demoUser.Id, LessonId = 29, CompletionPercentage = 40, IsCompleted = false, StartedAt = utcNow.AddHours(-2), TotalAttempts = 1 });

                await context.SaveChangesAsync();
            }

            // Spaced Repetition Cards
            if (!await context.SpacedRepetitionLessons.AnyAsync(sr => sr.UserId == demoUser.Id))
            {
                context.SpacedRepetitionLessons.AddRange(
                    new SpacedRepetitionLesson { UserId = demoUser.Id, LessonId = 1, DueDate = utcNow.AddHours(-1), RepetitionCount = 3, RetentionPercentage = 90, Interval = 4, EaseFactor = 2.5, LastReviewedAt = utcNow.AddDays(-2) },
                    new SpacedRepetitionLesson { UserId = demoUser.Id, LessonId = 2, DueDate = utcNow.AddDays(1), RepetitionCount = 2, RetentionPercentage = 80, Interval = 2, EaseFactor = 2.4, LastReviewedAt = utcNow.AddDays(-1) },
                    new SpacedRepetitionLesson { UserId = demoUser.Id, LessonId = 3, DueDate = utcNow.AddDays(2), RepetitionCount = 1, RetentionPercentage = 70, Interval = 1, EaseFactor = 2.5, LastReviewedAt = utcNow.AddHours(-6) });

                await context.SaveChangesAsync();
            }

            // Achievements Unlocked
            if (!await context.UserAchievements.AnyAsync(ua => ua.UserId == demoUser.Id))
            {
                context.UserAchievements.AddRange(
                    new UserAchievement { UserId = demoUser.Id, AchievementId = 1, UnlockedAt = utcNow.AddDays(-3) },
                    new UserAchievement { UserId = demoUser.Id, AchievementId = 2, UnlockedAt = utcNow.AddDays(-1) });

                await context.SaveChangesAsync();
            }

            logger.LogInformation("Demo dataset fully seeded with 26 ASL Letter videos, categories, achievements, and user progress!");
        }
    }
}
