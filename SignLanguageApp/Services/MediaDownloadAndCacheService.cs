using System.Security.Cryptography;
using System.Text;

namespace SignLanguageApp.Services;

public interface IMediaDownloadAndCacheService
{
    Task<string> GetCachedMediaAsync(string remoteUrl, CancellationToken cancellationToken = default);
    Task ClearOldCacheAsync(long maxSizeBytes = 500 * 1024 * 1024); // 500MB default
}

public class MediaDownloadAndCacheService : IMediaDownloadAndCacheService
{
    private readonly HttpClient _httpClient;
    private readonly string _cacheDirectory;

    public MediaDownloadAndCacheService(HttpClient httpClient)
    {
        _httpClient = httpClient;
        _cacheDirectory = Path.Combine(FileSystem.CacheDirectory, "SignLanguageMedia");
        if (!Directory.Exists(_cacheDirectory))
        {
            Directory.CreateDirectory(_cacheDirectory);
        }
    }

    public async Task<string> GetCachedMediaAsync(string remoteUrl, CancellationToken cancellationToken = default)
    {
        if (string.IsNullOrWhiteSpace(remoteUrl)) return string.Empty;

        // Generate a safe local filename using SHA256 of the URL
        string extension = Path.GetExtension(remoteUrl);
        string safeFileName = ComputeSha256Hash(remoteUrl) + extension;
        string localFilePath = Path.Combine(_cacheDirectory, safeFileName);

        // If it already exists, update its access time and return it
        if (File.Exists(localFilePath))
        {
            File.SetLastAccessTimeUtc(localFilePath, DateTime.UtcNow);
            return localFilePath;
        }

        // Download and cache it
        try
        {
            using var response = await _httpClient.GetAsync(remoteUrl, HttpCompletionOption.ResponseHeadersRead, cancellationToken);
            response.EnsureSuccessStatusCode();

            using var stream = await response.Content.ReadAsStreamAsync(cancellationToken);
            using var fileStream = new FileStream(localFilePath, FileMode.Create, FileAccess.Write, FileShare.None, 8192, true);
            await stream.CopyToAsync(fileStream, cancellationToken);

            return localFilePath;
        }
        catch (Exception ex)
        {
            System.Diagnostics.Debug.WriteLine($"Failed to download media: {ex.Message}");
            return string.Empty;
        }
    }

    public Task ClearOldCacheAsync(long maxSizeBytes = 500 * 1024 * 1024)
    {
        return Task.Run(() =>
        {
            var dirInfo = new DirectoryInfo(_cacheDirectory);
            if (!dirInfo.Exists) return;

            var files = dirInfo.GetFiles().OrderBy(f => f.LastAccessTimeUtc).ToList();
            long totalSize = files.Sum(f => f.Length);

            foreach (var file in files)
            {
                if (totalSize <= maxSizeBytes) break;
                
                try
                {
                    totalSize -= file.Length;
                    file.Delete();
                }
                catch { /* Ignore locked files */ }
            }
        });
    }

    private static string ComputeSha256Hash(string rawData)
    {
        using SHA256 sha256Hash = SHA256.Create();
        byte[] bytes = sha256Hash.ComputeHash(Encoding.UTF8.GetBytes(rawData));
        
        var builder = new StringBuilder();
        foreach (byte b in bytes)
        {
            builder.Append(b.ToString("x2"));
        }
        return builder.ToString();
    }
}
