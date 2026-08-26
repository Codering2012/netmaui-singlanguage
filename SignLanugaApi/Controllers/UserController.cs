using Microsoft.AspNetCore.Authorization;
using Microsoft.AspNetCore.Mvc;
using SignLanguageApi.Data;
using SignLanguageApi.Dtos;
using SignLanguageApi.Services;
using System.Security.Claims;
using Microsoft.EntityFrameworkCore;

namespace SignLanguageApi.Controllers
{
    [Authorize]
    [ApiController]
    [Route("api/[controller]")]
    public class UserController : ControllerBase
    {
        private readonly AppDbContext _context;
        private readonly IAuthService _authService;
        private readonly IPasswordValidator _passwordValidator;
        private readonly ILogger<UserController> _logger;

        public UserController(
            AppDbContext context,
            IAuthService authService,
            IPasswordValidator passwordValidator,
            ILogger<UserController> logger)
        {
            _context = context;
            _authService = authService;
            _passwordValidator = passwordValidator;
            _logger = logger;
        }

        [HttpGet("profile")]
        public async Task<ActionResult<UserProfileDto>> GetProfile()
        {
            var userId = User.FindFirst(ClaimTypes.NameIdentifier)?.Value;
            if (string.IsNullOrEmpty(userId)) return Unauthorized();

            var user = await _context.Users.FindAsync(userId);
            if (user == null) return NotFound();

            return Ok(new UserProfileDto
            {
                Id = user.Id,
                Email = user.Email,
                Name = user.Name,
                AvatarUrl = user.AvatarUrl,
                ProfileDescription = user.ProfileDescription,
                LearningStreak = user.LearningStreak,
                TotalXp = user.TotalXp
            });
        }

        [HttpPut("name")]
        public async Task<ActionResult> UpdateName([FromBody] UpdateNameRequest request)
        {
            if (string.IsNullOrWhiteSpace(request.NewName))
                return BadRequest(new ApiMessageDto { message = "Name cannot be empty." });

            var userId = User.FindFirst(ClaimTypes.NameIdentifier)?.Value;
            var user = await _context.Users.FindAsync(userId);
            if (user == null) return NotFound();

            user.Name = request.NewName;
            await _context.SaveChangesAsync();

            return Ok(new ApiMessageDto { message = "Name updated successfully." });
        }

        [HttpPut("password")]
        public async Task<ActionResult> UpdatePassword([FromBody] UpdatePasswordRequest request)
        {
            var userId = User.FindFirst(ClaimTypes.NameIdentifier)?.Value;
            var user = await _context.Users.FindAsync(userId);
            if (user == null) return NotFound();

            if (!_authService.VerifyPassword(request.OldPassword, user.PasswordHash))
                return BadRequest(new ApiMessageDto { message = "Incorrect old password." });

            var (isValid, errorMessage) = _passwordValidator.ValidatePassword(request.NewPassword);
            if (!isValid) return BadRequest(new ApiMessageDto { message = errorMessage });

            user.PasswordHash = _authService.HashPassword(request.NewPassword);
            await _context.SaveChangesAsync();

            return Ok(new ApiMessageDto { message = "Password updated successfully." });
        }

        [HttpPut("avatar")]
        public async Task<ActionResult> UpdateAvatar([FromBody] UpdateAvatarRequest request)
        {
            var userId = User.FindFirst(ClaimTypes.NameIdentifier)?.Value;
            var user = await _context.Users.FindAsync(userId);
            if (user == null) return NotFound();

            user.AvatarUrl = request.AvatarUrl;
            await _context.SaveChangesAsync();

            return Ok(new ApiMessageDto { message = "Profile picture updated successfully." });
        }

        [HttpPut("description")]
        public async Task<ActionResult> UpdateDescription([FromBody] UpdateDescriptionRequest request)
        {
            var userId = User.FindFirst(ClaimTypes.NameIdentifier)?.Value;
            var user = await _context.Users.FindAsync(userId);
            if (user == null) return NotFound();

            user.ProfileDescription = request.Description;
            await _context.SaveChangesAsync();

            return Ok(new ApiMessageDto { message = "Description updated successfully." });
        }
        [HttpGet("achievements")]
        public async Task<ActionResult<IEnumerable<AchievementBadgeDto>>> GetAchievements()
        {
            var userId = User.FindFirst(ClaimTypes.NameIdentifier)?.Value;
            if (string.IsNullOrEmpty(userId)) return Unauthorized();

            var allAchievements = await _context.Set<Achievement>().ToListAsync();
            var userAchievements = await _context.Set<UserAchievement>()
                .Where(ua => ua.UserId == userId)
                .Select(ua => ua.AchievementId)
                .ToListAsync();

            return Ok(allAchievements.Select(a => new AchievementBadgeDto
            {
                Id = a.Id,
                Name = a.Title,
                Description = a.Description,
                ImageUrl = a.IconChar,
                IsUnlocked = userAchievements.Contains(a.Id)
            }));
        }
    }
}
