using Microsoft.EntityFrameworkCore;
using SignLanguageApi.Data;

namespace SignLanguageApi.Helpers
{
    public static class DbSeeder
    {
        public static async Task SeedAsync(AppDbContext context)
        {
            await context.Database.EnsureCreatedAsync();

            if (!await context.LessonCategories.AnyAsync())
            {
                var categories = new List<LessonCategory>
                {
                    new LessonCategory { Id = 1, Title = "Alphabet", Description = "Master the basics of ASL alphabet", Difficulty = "Beginner", IconUrl = "https://cdn-icons-png.flaticon.com/512/3843/3843940.png" },
                    new LessonCategory { Id = 2, Title = "Numbers", Description = "Learn to sign numbers 1-100", Difficulty = "Beginner", IconUrl = "https://cdn-icons-png.flaticon.com/512/3743/3743126.png" },
                    new LessonCategory { Id = 3, Title = "Phrases", Description = "Common daily phrases and greetings", Difficulty = "Intermediate", IconUrl = "https://cdn-icons-png.flaticon.com/512/3063/3063822.png" }
                };

                await context.LessonCategories.AddRangeAsync(categories);
                await context.SaveChangesAsync();
            }

            if (!await context.Lessons.AnyAsync())
            {
                var lessons = new List<Lesson>
                {
                    new Lesson { Id = 1, Title = "Alphabet Part 1", CategoryId = 1, Difficulty = "Beginner", Description = "Learn letters A through M with clear demonstrations", ViewCount = 1250, VideoUrl = "Alphabet_Part1.mp4", ThumbnailUrl = "Alphabet_Part1.png", DurationSeconds = 45, InstructorName = "Sarah Miller" },
                    new Lesson { Id = 2, Title = "Alphabet Part 2", CategoryId = 1, Difficulty = "Beginner", Description = "Continue your journey with letters N through Z", ViewCount = 980, VideoUrl = "Alphabet_Part2.mp4", ThumbnailUrl = "Alphabet_Part2.png", DurationSeconds = 52, InstructorName = "Sarah Miller" },
                    new Lesson { Id = 3, Title = "A-M Masterclass", CategoryId = 1, Difficulty = "Intermediate", Description = "Mastering the first half of the alphabet with precision", ViewCount = 850, VideoUrl = "Letters_Alphabet A-M.mp4", ThumbnailUrl = "Letters_Alphabet A-M.png", DurationSeconds = 120, InstructorName = "John Doe" },
                    new Lesson { Id = 4, Title = "Common Greetings", CategoryId = 3, Difficulty = "Intermediate", Description = "Learn how to say Hello, Goodbye, and more", ViewCount = 2100, VideoUrl = "Spelling_HELLO.mp4", ThumbnailUrl = "Spelling_HELLO.png", DurationSeconds = 12, InstructorName = "Emily Chen" },
                    new Lesson { Id = 5, Title = "3-Letter Words", CategoryId = 1, Difficulty = "Intermediate", Description = "Practice signing common three-letter words", ViewCount = 3200, VideoUrl = "Alphabet_Part1.mp4", ThumbnailUrl = "Alphabet_Part1.png", DurationSeconds = 46, InstructorName = "Sarah Miller" }
                };

                await context.Lessons.AddRangeAsync(lessons);
                await context.SaveChangesAsync();
            }

            if (!await context.Achievements.AnyAsync())
            {
                var achievements = new List<Achievement>
                {
                    new Achievement { Id = 1, Title = "First Steps", Description = "Completed your first lesson", RequiredPoints = 50 },
                    new Achievement { Id = 2, Title = "Streak Master", Description = "Learned for 3 days in a row", RequiredPoints = 150 },
                    new Achievement { Id = 3, Title = "Expert", Description = "Mastered 50 signs", RequiredPoints = 500 }
                };

                await context.Achievements.AddRangeAsync(achievements);
                await context.SaveChangesAsync();
            }
        }
    }
}
