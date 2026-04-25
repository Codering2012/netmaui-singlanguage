namespace SignLanguageApi.Services;

public interface IPasswordValidator
{
    (bool IsValid, string ErrorMessage) ValidatePassword(string password);
}
