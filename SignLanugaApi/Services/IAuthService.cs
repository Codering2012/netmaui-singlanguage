using SignLanguageApi.Data;

namespace SignLanguageApi.Services
{
    public interface IAuthService
    {
        string HashPassword(string password);

        bool VerifyPassword(string password, string hash);

        string GenerateJwtToken(User user);

        string GenerateRefreshToken();
    }
}
