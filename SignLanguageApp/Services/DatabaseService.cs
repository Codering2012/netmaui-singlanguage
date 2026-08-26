using System.Diagnostics;
using SQLite;
using SignLanguageApp.Model;

namespace SignLanguageApp.Services;

/// <summary>
/// Interface for local database operations
/// </summary>
public interface IDatabaseService
{
    Task<User?> GetUserAsync();
    Task SaveUserAsync(User user);
    Task<bool> DeleteUserAsync();
    Task<string?> GetAccessTokenAsync();
    Task SaveAccessTokenAsync(string token);
    Task<string?> GetRefreshTokenAsync();
    Task SaveRefreshTokenAsync(string token);
    Task<SignPerformance?> GetSignPerformanceAsync(string signId);
    Task SaveSignPerformanceAsync(SignPerformance performance);
    Task<List<SignPerformance>> GetAllSignPerformancesAsync();
    Task ClearAllAsync();
    
    // New methods for Gamification Download Tracking
    Task SaveDownloadStateAsync(int mediaId, string mediaType, bool isDownloaded, string localPath, string context);
    Task<bool> IsDownloadedAsync(int mediaId, string mediaType);
    Task<string?> GetLocalPathAsync(int mediaId, string mediaType);
}

// Internal SQLite Entities
public class UserEntity
{
    [PrimaryKey]
    public string Id { get; set; } = string.Empty;
    public string Email { get; set; } = string.Empty;
    public string Name { get; set; } = string.Empty;
    public string? AvatarUrl { get; set; }
    public int LearningStreak { get; set; }
    public int TotalXP { get; set; }
}

public class TokenEntity
{
    [PrimaryKey]
    public string Key { get; set; } = string.Empty;
    public string Token { get; set; } = string.Empty;
}

public class SignPerformanceEntity
{
    [PrimaryKey]
    public string SignId { get; set; } = string.Empty;
    public DateTime LastReviewed { get; set; }
    public DateTime NextReviewDate { get; set; }
    public int Interval { get; set; }
    public int Repetitions { get; set; }
    public double EaseFactor { get; set; }
    public int CorrectCount { get; set; }
    public int IncorrectCount { get; set; }
    public double AverageAccuracy { get; set; }
}

public class DownloadedMediaEntity
{
    [PrimaryKey, AutoIncrement]
    public int Id { get; set; }
    
    // The ID from the API (LessonId or VideoId)
    public int MediaId { get; set; }
    
    // "Lesson" or "Video"
    public string MediaType { get; set; } = string.Empty;
    
    public bool IsDownloaded { get; set; }
    public string LocalPath { get; set; } = string.Empty;
    
    // "Local", "InApp", "All"
    public string DownloadContext { get; set; } = string.Empty;
}

/// <summary>
/// Database service for secure local storage using sqlite-net-pcl
/// </summary>
public class DatabaseService : IDatabaseService
{
    private SQLiteAsyncConnection? _db;
    private const string AccessTokenKey = "access_token";
    private const string RefreshTokenKey = "refresh_token";

    private async Task InitAsync()
    {
        if (_db != null) return;

        var databasePath = Path.Combine(FileSystem.AppDataDirectory, "SignAppSecure.db3");
        
        var flags = SQLiteOpenFlags.ReadWrite | SQLiteOpenFlags.Create | SQLiteOpenFlags.SharedCache;
        _db = new SQLiteAsyncConnection(databasePath, flags);

        await _db.CreateTableAsync<UserEntity>();
        await _db.CreateTableAsync<TokenEntity>();
        await _db.CreateTableAsync<SignPerformanceEntity>();
        await _db.CreateTableAsync<DownloadedMediaEntity>();
    }

    public async Task<User?> GetUserAsync()
    {
        await InitAsync();
        try
        {
            var entity = await _db!.Table<UserEntity>().FirstOrDefaultAsync();
            if (entity == null) return null;

            return new User
            {
                Id = entity.Id,
                Email = entity.Email,
                Name = entity.Name,
                AvatarUrl = entity.AvatarUrl ?? string.Empty,
                LearningStreak = entity.LearningStreak,
                TotalXP = entity.TotalXP
            };
        }
        catch (Exception ex)
        {
            Debug.WriteLine($"GetUser error: {ex.Message}");
            return null;
        }
    }

    public async Task SaveUserAsync(User user)
    {
        await InitAsync();
        try
        {
            var entity = new UserEntity
            {
                Id = user.Id,
                Email = user.Email,
                Name = user.Name,
                AvatarUrl = user.AvatarUrl,
                LearningStreak = user.LearningStreak,
                TotalXP = user.TotalXP
            };
            await _db!.InsertOrReplaceAsync(entity);
        }
        catch (Exception ex)
        {
            Debug.WriteLine($"SaveUser error: {ex.Message}");
        }
    }

    public async Task<bool> DeleteUserAsync()
    {
        await InitAsync();
        try
        {
            await _db!.DeleteAllAsync<UserEntity>();
            await _db.DeleteAllAsync<TokenEntity>();
            return true;
        }
        catch (Exception ex)
        {
            Debug.WriteLine($"DeleteUser error: {ex.Message}");
            return false;
        }
    }

    public async Task<string?> GetAccessTokenAsync()
    {
        await InitAsync();
        try
        {
            var token = await _db!.FindAsync<TokenEntity>(AccessTokenKey);
            return token?.Token;
        }
        catch (Exception ex)
        {
            Debug.WriteLine($"GetAccessToken error: {ex.Message}");
            return null;
        }
    }

    public async Task SaveAccessTokenAsync(string token)
    {
        await InitAsync();
        try
        {
            if (!string.IsNullOrEmpty(token))
            {
                await _db!.InsertOrReplaceAsync(new TokenEntity { Key = AccessTokenKey, Token = token });
            }
            else
            {
                await _db!.DeleteAsync<TokenEntity>(AccessTokenKey);
            }
        }
        catch (Exception ex)
        {
            Debug.WriteLine($"SaveAccessToken error: {ex.Message}");
        }
    }

    public async Task<string?> GetRefreshTokenAsync()
    {
        await InitAsync();
        try
        {
            var token = await _db!.FindAsync<TokenEntity>(RefreshTokenKey);
            return token?.Token;
        }
        catch (Exception ex)
        {
            Debug.WriteLine($"GetRefreshToken error: {ex.Message}");
            return null;
        }
    }

    public async Task SaveRefreshTokenAsync(string token)
    {
        await InitAsync();
        try
        {
            if (!string.IsNullOrEmpty(token))
            {
                await _db!.InsertOrReplaceAsync(new TokenEntity { Key = RefreshTokenKey, Token = token });
            }
            else
            {
                await _db!.DeleteAsync<TokenEntity>(RefreshTokenKey);
            }
        }
        catch (Exception ex)
        {
            Debug.WriteLine($"SaveRefreshToken error: {ex.Message}");
        }
    }

    public async Task<SignPerformance?> GetSignPerformanceAsync(string signId)
    {
        await InitAsync();
        try
        {
            var entity = await _db!.FindAsync<SignPerformanceEntity>(signId);
            if (entity == null) return null;

            return new SignPerformance
            {
                SignId = entity.SignId,
                LastReviewed = entity.LastReviewed,
                NextReviewDate = entity.NextReviewDate,
                Interval = entity.Interval,
                Repetitions = entity.Repetitions,
                EaseFactor = entity.EaseFactor,
                CorrectCount = entity.CorrectCount,
                IncorrectCount = entity.IncorrectCount,
                AverageAccuracy = entity.AverageAccuracy
            };
        }
        catch (Exception ex)
        {
            Debug.WriteLine($"GetSignPerformance error: {ex.Message}");
            return null;
        }
    }

    public async Task SaveSignPerformanceAsync(SignPerformance performance)
    {
        await InitAsync();
        try
        {
            var entity = new SignPerformanceEntity
            {
                SignId = performance.SignId,
                LastReviewed = performance.LastReviewed,
                NextReviewDate = performance.NextReviewDate,
                Interval = performance.Interval,
                Repetitions = performance.Repetitions,
                EaseFactor = performance.EaseFactor,
                CorrectCount = performance.CorrectCount,
                IncorrectCount = performance.IncorrectCount,
                AverageAccuracy = performance.AverageAccuracy
            };
            await _db!.InsertOrReplaceAsync(entity);
        }
        catch (Exception ex)
        {
            Debug.WriteLine($"SaveSignPerformance error: {ex.Message}");
        }
    }

    public async Task<List<SignPerformance>> GetAllSignPerformancesAsync()
    {
        await InitAsync();
        try
        {
            var entities = await _db!.Table<SignPerformanceEntity>().ToListAsync();
            return entities.Select(e => new SignPerformance
            {
                SignId = e.SignId,
                LastReviewed = e.LastReviewed,
                NextReviewDate = e.NextReviewDate,
                Interval = e.Interval,
                Repetitions = e.Repetitions,
                EaseFactor = e.EaseFactor,
                CorrectCount = e.CorrectCount,
                IncorrectCount = e.IncorrectCount,
                AverageAccuracy = e.AverageAccuracy
            }).ToList();
        }
        catch (Exception ex)
        {
            Debug.WriteLine($"GetAllSignPerformances error: {ex.Message}");
            return new List<SignPerformance>();
        }
    }

    public async Task ClearAllAsync()
    {
        await InitAsync();
        try
        {
            await _db!.DeleteAllAsync<UserEntity>();
            await _db.DeleteAllAsync<TokenEntity>();
            await _db.DeleteAllAsync<SignPerformanceEntity>();
            await _db.DeleteAllAsync<DownloadedMediaEntity>();
        }
        catch (Exception ex)
        {
            Debug.WriteLine($"ClearAll error: {ex.Message}");
        }
    }

    // Media Download Tracking
    public async Task SaveDownloadStateAsync(int mediaId, string mediaType, bool isDownloaded, string localPath, string context)
    {
        await InitAsync();
        try
        {
            var existing = await _db!.Table<DownloadedMediaEntity>().Where(m => m.MediaId == mediaId && m.MediaType == mediaType).FirstOrDefaultAsync();
            if (existing != null)
            {
                existing.IsDownloaded = isDownloaded;
                existing.LocalPath = localPath;
                existing.DownloadContext = context;
                await _db.UpdateAsync(existing);
            }
            else
            {
                await _db.InsertAsync(new DownloadedMediaEntity
                {
                    MediaId = mediaId,
                    MediaType = mediaType,
                    IsDownloaded = isDownloaded,
                    LocalPath = localPath,
                    DownloadContext = context
                });
            }
        }
        catch (Exception ex)
        {
            Debug.WriteLine($"SaveDownloadState error: {ex.Message}");
        }
    }

    public async Task<bool> IsDownloadedAsync(int mediaId, string mediaType)
    {
        await InitAsync();
        try
        {
            var existing = await _db!.Table<DownloadedMediaEntity>().Where(m => m.MediaId == mediaId && m.MediaType == mediaType).FirstOrDefaultAsync();
            return existing?.IsDownloaded ?? false;
        }
        catch (Exception ex)
        {
            Debug.WriteLine($"IsDownloaded error: {ex.Message}");
            return false;
        }
    }

    public async Task<string?> GetLocalPathAsync(int mediaId, string mediaType)
    {
        await InitAsync();
        try
        {
            var existing = await _db!.Table<DownloadedMediaEntity>().Where(m => m.MediaId == mediaId && m.MediaType == mediaType).FirstOrDefaultAsync();
            return existing?.LocalPath;
        }
        catch (Exception ex)
        {
            Debug.WriteLine($"GetLocalPath error: {ex.Message}");
            return null;
        }
    }
}
