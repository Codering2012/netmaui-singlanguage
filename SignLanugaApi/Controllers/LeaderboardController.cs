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
    public class LeaderboardController : ControllerBase
    {
        private readonly AppDbContext _context;
        private readonly ILogger<LeaderboardController> _logger;

        public LeaderboardController(AppDbContext context, ILogger<LeaderboardController> logger)
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
        public async Task<ActionResult<LeaderboardDto>> GetLeaderboard([FromQuery] int count = 10)
        {
            try
            {
                var currentUserId = GetUserId();
                
                var topUsers = await _context.Users
                    .OrderByDescending(u => u.TotalXp)
                    .Take(count)
                    .ToListAsync();

                var entries = topUsers.Select((u, index) => new LeaderboardEntryDto
                {
                    UserId = u.Id,
                    Name = u.Name,
                    AvatarUrl = u.AvatarUrl,
                    TotalXp = u.TotalXp,
                    Rank = index + 1,
                    IsCurrentUser = u.Id == currentUserId
                }).ToList();

                var currentUser = await _context.Users.FindAsync(currentUserId);
                LeaderboardEntryDto? currentUserEntry = null;

                if (currentUser != null)
                {
                    // Find actual rank
                    var rank = await _context.Users
                        .CountAsync(u => u.TotalXp > currentUser.TotalXp) + 1;

                    currentUserEntry = new LeaderboardEntryDto
                    {
                        UserId = currentUser.Id,
                        Name = currentUser.Name,
                        AvatarUrl = currentUser.AvatarUrl,
                        TotalXp = currentUser.TotalXp,
                        Rank = rank,
                        IsCurrentUser = true
                    };
                }

                return Ok(new LeaderboardDto
                {
                    TopEntries = entries,
                    CurrentUserEntry = currentUserEntry
                });
            }
            catch (Exception ex)
            {
                _logger.LogError(ex, "Error fetching leaderboard");
                return StatusCode(500, new { message = "Error fetching leaderboard" });
            }
        }
    }
}
