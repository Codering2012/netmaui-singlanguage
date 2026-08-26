using Microsoft.AspNetCore.Authorization;
using Microsoft.AspNetCore.Mvc;
using Microsoft.EntityFrameworkCore;
using SignLanguageApi.Data;
using SignLanguageApi.Dtos;
using System.Security.Claims;

namespace SignLanguageApi.Controllers
{
    [ApiController]
    [Route("api/[controller]")]
    [Authorize]
    public class StatisticsController : ControllerBase
    {
        private readonly AppDbContext _context;
        private readonly ILogger<StatisticsController> _logger;

        public StatisticsController(AppDbContext context, ILogger<StatisticsController> logger)
        {
            _context = context;
            _logger = logger;
        }

        private string GetUserId()
        {
            var userId = User.FindFirst(ClaimTypes.NameIdentifier)?.Value;
            if (string.IsNullOrWhiteSpace(userId))
                throw new InvalidOperationException("User ID not found in token");
            return userId;
        }

        [HttpGet]
        public async Task<ActionResult<UserStatsDto>> GetStats()
        {
            try
            {
                var userId = GetUserId();
                var user = await _context.Users.FindAsync(userId);
                if (user == null) return NotFound();

                var lessonsCompleted = await _context.UserLessons.CountAsync(ul => ul.UserId == userId && ul.IsCompleted);
                
                var weekAgo = DateTime.UtcNow.Date.AddDays(-6);
                var activities = await _context.UserActivities
                    .Where(ua => ua.UserId == userId && ua.Timestamp >= weekAgo)
                    .ToListAsync();

                var weeklyXp = new List<DailyXpDto>();
                for (int i = 0; i < 7; i++)
                {
                    var date = weekAgo.AddDays(i);
                    var xp = activities.Where(a => a.Timestamp.Date == date).Sum(a => a.PointsGained);
                    weeklyXp.Add(new DailyXpDto { Date = date.ToString("yyyy-MM-dd"), Xp = xp });
                }

                var categories = await _context.LessonCategories.ToListAsync();
                var allLessons = await _context.Lessons.ToListAsync();
                var userLessons = await _context.UserLessons.Where(ul => ul.UserId == userId).ToListAsync();
                
                var categoryProgress = categories.Select(c => {
                    var categoryLessons = allLessons.Where(l => l.CategoryId == c.Id).ToList();
                    var completedInCategory = userLessons.Count(ul => categoryLessons.Any(l => l.Id == ul.LessonId) && ul.IsCompleted);
                    return new CategoryProgressDto {
                        CategoryName = c.Title,
                        Progress = categoryLessons.Count > 0 ? (double)completedInCategory / categoryLessons.Count : 0
                    };
                }).ToList();

                return Ok(new UserStatsDto
                {
                    TotalXp = user.TotalXp,
                    LearningStreak = user.LearningStreak,
                    LessonsCompleted = lessonsCompleted,
                    WeeklyXp = weeklyXp,
                    CategoryProgress = categoryProgress
                });
            }
            catch (Exception ex)
            {
                _logger.LogError(ex, "Error fetching statistics");
                return StatusCode(500, new { message = "Error fetching statistics" });
            }
        }
    }
}
