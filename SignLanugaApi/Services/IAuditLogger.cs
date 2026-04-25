namespace SignLanguageApi.Services;

public interface IAuditLogger
{
    Task LogLoginAttemptAsync(string email, bool success, string ipAddress);
    Task LogLogoutAsync(string userId, string email, string ipAddress);
    Task LogRegisterAttemptAsync(string email, bool success, string ipAddress);
    Task LogUnauthorizedAccessAsync(string ipAddress, string endpoint);
}
