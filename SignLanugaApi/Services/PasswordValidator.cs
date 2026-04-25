using System.Text.RegularExpressions;

namespace SignLanguageApi.Services;

public class PasswordValidator : IPasswordValidator
{
    private readonly ILogger<PasswordValidator> _logger;

    public PasswordValidator(ILogger<PasswordValidator> logger)
    {
        _logger = logger;
    }

    public (bool IsValid, string ErrorMessage) ValidatePassword(string password)
    {
        if (string.IsNullOrWhiteSpace(password))
        {
            return (false, "Password cannot be empty.");
        }

        if (password.Length < 8)
        {
            return (false, "Password must be at least 8 characters long.");
        }

        if (password.Length > 128)
        {
            return (false, "Password must not exceed 128 characters.");
        }

        if (!password.Any(char.IsUpper))
        {
            return (false, "Password must contain at least one uppercase letter.");
        }

        if (!password.Any(char.IsLower))
        {
            return (false, "Password must contain at least one lowercase letter.");
        }

        if (!password.Any(char.IsDigit))
        {
            return (false, "Password must contain at least one digit.");
        }

        // Check for special characters
        if (!Regex.IsMatch(password, @"[!@#$%^&*()_+\-=\[\]{};':""\\|,.<>\/?]"))
        {
            return (false, "Password must contain at least one special character (!@#$%^&*...).");
        }

        _logger.LogInformation("Password validation successful");
        return (true, "Password is valid.");
    }
}
