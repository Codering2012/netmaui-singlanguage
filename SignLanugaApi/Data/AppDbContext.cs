using Microsoft.EntityFrameworkCore;

namespace SignLanguageApi.Data
{
    public class AppDbContext : DbContext
    {
        protected AppDbContext()
        {
        }

        public AppDbContext(DbContextOptions<AppDbContext> options) : base(options)
        {
        }

        public virtual DbSet<User> Users { get; set; }

        public virtual DbSet<Lesson> Lessons { get; set; }

        public virtual DbSet<LessonCategory> LessonCategories { get; set; }

        public virtual DbSet<UserLesson> UserLessons { get; set; }

        public virtual DbSet<Achievement> Achievements { get; set; }

        public virtual DbSet<UserAchievement> UserAchievements { get; set; }

        public virtual DbSet<SpacedRepetitionLesson> SpacedRepetitionLessons { get; set; }

        protected override void OnModelCreating(ModelBuilder modelBuilder)
        {
            base.OnModelCreating(modelBuilder);

            // Configure User entity
            modelBuilder.Entity<User>()
                .HasIndex(u => u.Email)
                .IsUnique();

            modelBuilder.Entity<User>()
                .Property(u => u.Id)
                .ValueGeneratedNever();

            // Configure Lesson entity
            modelBuilder.Entity<Lesson>()
                .Property(l => l.Difficulty)
                .HasDefaultValueSql("'Beginner'");

            modelBuilder.Entity<Lesson>()
                .Property(l => l.ViewCount)
                .HasDefaultValueSql("0");
        }
    }
}
