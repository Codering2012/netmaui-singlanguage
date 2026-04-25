using System.Collections.Concurrent;

namespace SignLanguageApi.Services;

public class TokenBlacklistService : ITokenBlacklistService
{
    private readonly ConcurrentDictionary<string, DateTime> _blacklistedTokens = 
        new ConcurrentDictionary<string, DateTime>();
    private readonly ILogger<TokenBlacklistService> _logger;
    private readonly Timer _cleanupTimer;

    public TokenBlacklistService(ILogger<TokenBlacklistService> logger)
    {
        _logger = logger;
        
        // Start a background timer to clean up expired tokens every 5 minutes
        _cleanupTimer = new Timer(async _ => await RemoveExpiredTokensAsync(), 
            null, 
            TimeSpan.FromMinutes(5), 
            TimeSpan.FromMinutes(5));
    }

    public Task<bool> IsTokenBlacklistedAsync(string token)
    {
        var isBlacklisted = _blacklistedTokens.TryGetValue(token, out var expiryTime);
        
        if (isBlacklisted && expiryTime < DateTime.UtcNow)
        {
            // Token is blacklisted but has expired, remove it
            _blacklistedTokens.TryRemove(token, out _);
            return Task.FromResult(false);
        }

        return Task.FromResult(isBlacklisted && expiryTime >= DateTime.UtcNow);
    }

    public Task BlacklistTokenAsync(string token, DateTime expiryTime)
    {
        if (string.IsNullOrWhiteSpace(token))
        {
            throw new ArgumentException("Token cannot be null or empty", nameof(token));
        }

        _blacklistedTokens.AddOrUpdate(token, expiryTime, (_, _) => expiryTime);
        _logger.LogInformation("Token blacklisted until {ExpiryTime}", expiryTime);
        return Task.CompletedTask;
    }

    public Task RemoveExpiredTokensAsync()
    {
        var expiredTokens = _blacklistedTokens
            .Where(kvp => kvp.Value < DateTime.UtcNow)
            .Select(kvp => kvp.Key)
            .ToList();

        foreach (var token in expiredTokens)
        {
            _blacklistedTokens.TryRemove(token, out _);
        }

        if (expiredTokens.Count > 0)
        {
            _logger.LogInformation("Removed {Count} expired tokens from blacklist", expiredTokens.Count);
        }

        return Task.CompletedTask;
    }

    ~TokenBlacklistService()
    {
        _cleanupTimer?.Dispose();
    }
}
