using Microsoft.AspNetCore.Authorization;
using Microsoft.AspNetCore.Mvc;
using SignLanguageApi.Data;
using SignLanguageApi.Dtos;
using System.Security.Claims;

namespace SignLanguageApi.Controllers
{
    [ApiController]
    [Route("api/[controller]")]
    [Authorize]
    public class FeedbackController : ControllerBase
    {
        private readonly AppDbContext _context;
        private readonly ILogger<FeedbackController> _logger;

        public FeedbackController(AppDbContext context, ILogger<FeedbackController> logger)
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

        [HttpPost]
        public async Task<ActionResult> SubmitFeedback([FromBody] FeedbackRequestDto request)
        {
            try
            {
                var userId = GetUserId();
                var feedback = new UserFeedback
                {
                    UserId = userId,
                    Subject = request.Subject,
                    Message = request.Message,
                    Rating = request.Rating,
                    SubmittedAt = DateTime.UtcNow
                };

                _context.UserFeedbacks.Add(feedback);
                await _context.SaveChangesAsync();

                return Ok(new { message = "Feedback submitted successfully" });
            }
            catch (Exception ex)
            {
                _logger.LogError(ex, "Error submitting feedback");
                return StatusCode(500, new { message = "Error submitting feedback" });
            }
        }
    }
}
