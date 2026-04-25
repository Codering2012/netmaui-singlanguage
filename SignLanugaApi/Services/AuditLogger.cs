using System.Text.Json;

namespace SignLanguageApi.Services;

public class AuditLogger : IAuditLogger
{
    private readonly ILogger<AuditLogger> _logger;
    private readonly string _auditLogPath;

    public AuditLogger(ILogger<AuditLogger> logger)
    {
        _logger = logger;
        _auditLogPath = Path.Combine(
            Environment.GetFolderPath(Environment.SpecialFolder.ApplicationData),
            "SignLanguageApp",
            "Logs",
            "Audit"
        );
        
        // Ensure the audit log directory exists
        Directory.CreateDirectory(_auditLogPath);
    }

    public async Task LogLoginAttemptAsync(string email, bool success, string ipAddress)
    {
        var logEntry = new
        {
            EventType = "LOGIN_ATTEMPT",
            Email = email,
            Success = success,
            IpAddress = ipAddress,
            Timestamp = DateTime.UtcNow.ToString("O"),
            Status = success ? "SUCCESS" : "FAILED"
        };

        await WriteAuditLogAsync(logEntry, "login");
    }

    public async Task LogLogoutAsync(string userId, string email, string ipAddress)
    {
        var logEntry = new
        {
            EventType = "LOGOUT",
            UserId = userId,
            Email = email,
            IpAddress = ipAddress,
            Timestamp = DateTime.UtcNow.ToString("O"),
            Status = "SUCCESS"
        };

        await WriteAuditLogAsync(logEntry, "logout");
    }

    public async Task LogRegisterAttemptAsync(string email, bool success, string ipAddress)
    {
        var logEntry = new
        {
            EventType = "REGISTER_ATTEMPT",
            Email = email,
            Success = success,
            IpAddress = ipAddress,
            Timestamp = DateTime.UtcNow.ToString("O"),
            Status = success ? "SUCCESS" : "FAILED"
        };

        await WriteAuditLogAsync(logEntry, "register");
    }

    public async Task LogUnauthorizedAccessAsync(string ipAddress, string endpoint)
    {
        var logEntry = new
        {
            EventType = "UNAUTHORIZED_ACCESS",
            IpAddress = ipAddress,
            Endpoint = endpoint,
            Timestamp = DateTime.UtcNow.ToString("O"),
            Status = "BLOCKED"
        };

        await WriteAuditLogAsync(logEntry, "unauthorized");
    }

    private async Task WriteAuditLogAsync(object logEntry, string fileName)
    {
        try
        {
            var logFile = Path.Combine(_auditLogPath, $"{fileName}_{DateTime.UtcNow:yyyy-MM-dd}.log");
            var json = JsonSerializer.Serialize(logEntry, new JsonSerializerOptions { WriteIndented = true });
            
            await File.AppendAllTextAsync(logFile, json + Environment.NewLine + new string('-', 50) + Environment.NewLine);
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Error writing audit log");
        }
    }
}
