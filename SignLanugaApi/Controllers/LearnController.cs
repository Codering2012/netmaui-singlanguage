using Microsoft.AspNetCore.Authorization;
using Microsoft.AspNetCore.Mvc;
using SignLanguageApi.Dtos;
using SignLanguageApi.Services;
using System.Security.Claims;

namespace SignLanguageApi.Controllers
{
    [ApiController]
    [Route("api/[controller]")]
    [Authorize]
    public class LearnController : ControllerBase
    {
        private readonly ILearnService _learnService;
        private readonly ILogger<LearnController> _logger;

        public LearnController(ILearnService learnService, ILogger<LearnController> logger)
        {
            _learnService = learnService;
            _logger = logger;
        }

        // Helper to get the authenticated user's ID from the JWT token
        private string GetUserId()
        {
            var userId = User.FindFirst(ClaimTypes.NameIdentifier)?.Value;
            if (string.IsNullOrWhiteSpace(userId))
                throw new InvalidOperationException("User ID not found in token");
            return userId;
        }

        /// <summary>
        /// Get all learn page data for the authenticated user ONLY.
        /// Requires a valid JWT token. User ID is taken from the token, not from a query parameter.
        /// </summary>
        [HttpGet("data")]
        public async Task<ActionResult<LearnPageDataDto>> GetLearnPageData()
        {
            string userId;
            try
            {
                // SECURITY: Only return data for the authenticated user (from JWT token)
                userId = GetUserId();
            }
            catch (InvalidOperationException ex)
            {
                _logger.LogError(ex, "User ID not found in token");
                return Unauthorized(new { message = "Unauthorized" });
            }

            try
            {
                var data = await _learnService.GetLearnPageDataAsync(userId);
                return Ok(data);
            }
            catch (Exception ex)
            {
                _logger.LogError(ex, "Error fetching learn page data");
                return StatusCode(500, new { message = "Error fetching learn page data" });
            }
        }

        /// <summary>
        /// Get lessons for a specific category for the authenticated user.
        /// </summary>
        [HttpGet("categories/{categoryId}/lessons")]
        public async Task<ActionResult<List<LessonDto>>> GetLessonsByCategory(int categoryId)
        {
            string userId;
            try
            {
                userId = GetUserId();
            }
            catch (InvalidOperationException ex)
            {
                _logger.LogError(ex, "User ID not found in token for category {CategoryId}", categoryId);
                return Unauthorized(new { message = "Unauthorized" });
            }

            try
            {
                var lessons = await _learnService.GetLessonsByCategoryAsync(categoryId, userId);
                return Ok(lessons);
            }
            catch (Exception ex)
            {
                _logger.LogError(ex, "Error fetching lessons for category {CategoryId}", categoryId);
                return StatusCode(500, new { message = "Error fetching lessons" });
            }
        }

        /// <summary>
        /// Get details of a specific lesson for the authenticated user.
        /// </summary>
        [HttpGet("lessons/{lessonId}")]
        public async Task<ActionResult<LessonDto>> GetLesson(int lessonId)
        {
            string userId;
            try
            {
                userId = GetUserId();
            }
            catch (InvalidOperationException ex)
            {
                _logger.LogError(ex, "User ID not found in token for lesson {LessonId}", lessonId);
                return Unauthorized(new { message = "Unauthorized" });
            }

            try
            {
                var lesson = await _learnService.GetLessonAsync(lessonId, userId);

                if (lesson == null)
                    return NotFound(new { message = "Lesson not found" });

                return Ok(lesson);
            }
            catch (Exception ex)
            {
                _logger.LogError(ex, "Error fetching lesson {LessonId}", lessonId);
                return StatusCode(500, new { message = "Error fetching lesson" });
            }
        }

        /// <summary>
        /// Update lesson progress for the authenticated user.
        /// </summary>
        [HttpPut("lessons/{lessonId}/progress")]
        public async Task<ActionResult<object>> UpdateLessonProgress(int lessonId, [FromQuery] int completionPercentage)
        {
            if (completionPercentage < 0 || completionPercentage > 100)
                return BadRequest(new { message = "Completion percentage must be between 0 and 100" });

            string userId;
            try
            {
                userId = GetUserId();
            }
            catch (InvalidOperationException ex)
            {
                _logger.LogError(ex, "User ID not found in token for lesson {LessonId}", lessonId);
                return Unauthorized(new { message = "Unauthorized" });
            }

            try
            {
                await _learnService.UpdateLessonProgressAsync(userId, lessonId, completionPercentage);
                return Ok(new { message = "Lesson progress updated successfully" });
            }
            catch (KeyNotFoundException ex)
            {
                _logger.LogWarning(ex, "Lesson {LessonId} not found while updating progress", lessonId);
                return NotFound(new { message = "Lesson not found" });
            }
            catch (Exception ex)
            {
                _logger.LogError(ex, "Error updating lesson progress for lesson {LessonId}", lessonId);
                return StatusCode(500, new { message = "Error updating lesson progress" });
            }
        }

        /// <summary>
        /// Mark a lesson as complete for the authenticated user.
        /// </summary>
        [HttpPost("lessons/{lessonId}/complete")]
        public async Task<ActionResult<object>> CompleteLesson(int lessonId)
        {
            string userId;
            try
            {
                userId = GetUserId();
            }
            catch (InvalidOperationException ex)
            {
                _logger.LogError(ex, "User ID not found in token for lesson {LessonId}", lessonId);
                return Unauthorized(new { message = "Unauthorized" });
            }

            try
            {
                await _learnService.CompleteLessonAsync(userId, lessonId);
                return Ok(new { message = "Lesson completed successfully" });
            }
            catch (KeyNotFoundException ex)
            {
                _logger.LogWarning(ex, "Lesson {LessonId} not found while completing lesson", lessonId);
                return NotFound(new { message = "Lesson not found" });
            }
            catch (Exception ex)
            {
                _logger.LogError(ex, "Error completing lesson {LessonId}", lessonId);
                return StatusCode(500, new { message = "Error completing lesson" });
            }
        }

        /// <summary>
        /// Get daily review lessons due today for the authenticated user.
        /// </summary>
        [HttpGet("daily-reviews")]
        public async Task<ActionResult<List<SpacedRepetitionLessonDto>>> GetDailyReviews()
        {
            string userId;
            try
            {
                userId = GetUserId();
            }
            catch (InvalidOperationException ex)
            {
                _logger.LogError(ex, "Error fetching daily reviews");
                return Unauthorized(new { message = "Unauthorized" });
            }

            try
            {
                var reviews = await _learnService.GetDailyReviewLessonsAsync(userId);
                return Ok(reviews);
            }
            catch (Exception ex)
            {
                _logger.LogError(ex, "Error fetching daily reviews");
                return StatusCode(500, new { message = "Error fetching daily reviews" });
            }
        }

        /// <summary>
        /// Review a lesson using spaced repetition for the authenticated user.
        /// </summary>
        [HttpPost("daily-reviews/{spacedRepetitionId}/review")]
        public async Task<ActionResult<object>> ReviewLesson(int spacedRepetitionId, [FromQuery] double qualityRating)
        {
            try
            {
                if (qualityRating < 0 || qualityRating > 5)
                    return BadRequest(new { message = "Quality rating must be between 0 and 5" });

                var userId = GetUserId();
                await _learnService.ReviewLessonAsync(userId, spacedRepetitionId, qualityRating);

                return Ok(new { message = "Review recorded successfully" });
            }
            catch (InvalidOperationException ex)
            {
                _logger.LogWarning(ex, "Invalid operation for spaced repetition {SpacedRepetitionId}", spacedRepetitionId);
                return BadRequest(new { message = ex.Message });
            }
            catch (KeyNotFoundException ex)
            {
                _logger.LogWarning(ex, "Spaced repetition lesson {SpacedRepetitionId} not found", spacedRepetitionId);
                return NotFound(new { message = "Spaced repetition lesson not found" });
            }
            catch (Exception ex)
            {
                _logger.LogError(ex, "Error recording review for spaced repetition {SpacedRepetitionId}", spacedRepetitionId);
                return StatusCode(500, new { message = "Error recording review" });
            }
        }

        /// <summary>
        /// Get all lesson categories for the authenticated user.
        /// </summary>
        [HttpGet("categories")]
        public async Task<ActionResult<List<LessonCategoryDto>>> GetAllCategories()
        {
            string userId;
            try
            {
                userId = GetUserId();
            }
            catch (InvalidOperationException ex)
            {
                _logger.LogError(ex, "User ID not found in token");
                return Unauthorized(new { message = "Unauthorized" });
            }

            try
            {
                var categories = await _learnService.GetAllCategoriesAsync(userId);
                return Ok(categories);
            }
            catch (Exception ex)
            {
                _logger.LogError(ex, "Error fetching categories");
                return StatusCode(500, new { message = "Error fetching categories" });
            }
        }

        /// <summary>
        /// Get a specific lesson category with its progress for the authenticated user.
        /// </summary>
        [HttpGet("categories/{categoryId}")]
        public async Task<ActionResult<LessonCategoryDto>> GetCategory(int categoryId)
        {
            string userId;
            try
            {
                userId = GetUserId();
            }
            catch (InvalidOperationException ex)
            {
                _logger.LogError(ex, "User ID not found in token for category {CategoryId}", categoryId);
                return Unauthorized(new { message = "Unauthorized" });
            }

            try
            {
                var category = await _learnService.GetCategoryAsync(categoryId, userId);

                if (category == null)
                    return NotFound(new { message = "Category not found" });

                return Ok(category);
            }
            catch (Exception ex)
            {
                _logger.LogError(ex, "Error fetching category {CategoryId}", categoryId);
                return StatusCode(500, new { message = "Error fetching category" });
            }
        }

        /// <summary>
        /// Get the daily learning goal progress for the authenticated user.
        /// </summary>
        [HttpGet("daily-goal")]
        public async Task<ActionResult<DailyGoalDto>> GetDailyGoal()
        {
            string userId;
            try
            {
                userId = GetUserId();
            }
            catch (InvalidOperationException ex)
            {
                _logger.LogError(ex, "Error fetching daily goal");
                return Unauthorized(new { message = "Unauthorized" });
            }

            try
            {
                var dailyGoal = await _learnService.GetDailyGoalAsync(userId);
                return Ok(dailyGoal);
            }
            catch (Exception ex)
            {
                _logger.LogError(ex, "Error fetching daily goal");
                return StatusCode(500, new { message = "Error fetching daily goal" });
            }
        }

        /// <summary>
        /// Get upcoming reviews schedule for the authenticated user.
        /// </summary>
        [HttpGet("upcoming-reviews")]
        public async Task<ActionResult<UpcomingReviewsDto>> GetUpcomingReviews()
        {
            string userId;
            try
            {
                userId = GetUserId();
            }
            catch (InvalidOperationException ex)
            {
                _logger.LogError(ex, "User ID not found in token. Error fetching upcoming reviews");
                return Unauthorized(new { message = "Unauthorized" });
            }

            try
            {
                var upcomingReviews = await _learnService.GetUpcomingReviewsAsync(userId);
                return Ok(upcomingReviews);
            }
            catch (Exception ex)
            {
                _logger.LogError(ex, "Error fetching upcoming reviews");
                return StatusCode(500, new { message = "Error fetching upcoming reviews" });
            }
        }

        /// <summary>
        /// Get personalized learning recommendation for the authenticated user.
        /// </summary>
        [HttpGet("recommendations")]
        public async Task<ActionResult<PersonalizedRecommendationDto>> GetRecommendation()
        {
            string userId;
            try
            {
                userId = GetUserId();
            }
            catch (InvalidOperationException ex)
            {
                _logger.LogError(ex, "User ID not found in token");
                return Unauthorized(new { message = "Unauthorized" });
            }

            try
            {
                var recommendation = await _learnService.GetPersonalizedRecommendationAsync(userId);
                return Ok(recommendation);
            }
            catch (Exception ex)
            {
                _logger.LogError(ex, "Error fetching recommendation");
                return StatusCode(500, new { message = "Error fetching recommendation" });
            }
        }
    }
}
