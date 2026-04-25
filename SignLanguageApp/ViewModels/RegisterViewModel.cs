using System;
using System.ComponentModel;
using System.Runtime.CompilerServices;
using System.Threading.Tasks;
using System.Windows.Input;
using Microsoft.Maui.Controls;
using SignLanguageApp.Services;

namespace SignLanguageApp.ViewModels;

public class RegisterViewModel : INotifyPropertyChanged
{
    private readonly IApiService _apiService;
    private string _name = string.Empty;
    private string _email = string.Empty;
    private string _password = string.Empty;
    private string _confirmPassword = string.Empty;
    private string _errorMessage = string.Empty;
    private string _successMessage = string.Empty;
    private bool _hasError;
    private bool _hasSuccess;
    private bool _isLoading;

    public string Name
    {
        get => _name;
        set { _name = value; OnPropertyChanged(); }
    }

    public string Email
    {
        get => _email;
        set { _email = value; OnPropertyChanged(); }
    }

    public string Password
    {
        get => _password;
        set { _password = value; OnPropertyChanged(); }
    }

    public string ConfirmPassword
    {
        get => _confirmPassword;
        set { _confirmPassword = value; OnPropertyChanged(); }
    }

    public string ErrorMessage
    {
        get => _errorMessage;
        set { _errorMessage = value; OnPropertyChanged(); }
    }

    public string SuccessMessage
    {
        get => _successMessage;
        set { _successMessage = value; OnPropertyChanged(); }
    }

    public bool HasError
    {
        get => _hasError;
        set { _hasError = value; OnPropertyChanged(); }
    }

    public bool HasSuccess
    {
        get => _hasSuccess;
        set { _hasSuccess = value; OnPropertyChanged(); }
    }

    public bool IsLoading
    {
        get => _isLoading;
        set { _isLoading = value; OnPropertyChanged(); }
    }

    public ICommand RegisterCommand { get; }
    public ICommand LoginCommand { get; }

    public RegisterViewModel(IApiService apiService)
    {
        _apiService = apiService;

        RegisterCommand = new Command(async () => await RegisterAsync());
        LoginCommand = new Command(async () => await GoToLoginAsync());
    }

    private async Task RegisterAsync()
    {
        // Validation
        if (string.IsNullOrWhiteSpace(Name) || string.IsNullOrWhiteSpace(Email) || 
            string.IsNullOrWhiteSpace(Password) || string.IsNullOrWhiteSpace(ConfirmPassword))
        {
            ShowError("Please fill in all fields");
            return;
        }

        if (!Email.Contains("@"))
        {
            ShowError("Please enter a valid email address");
            return;
        }

        if (Password.Length < 6)
        {
            ShowError("Password must be at least 6 characters");
            return;
        }

        if (Password != ConfirmPassword)
        {
            ShowError("Passwords do not match");
            return;
        }

        IsLoading = true;
        HasError = false;
        HasSuccess = false;
        ErrorMessage = string.Empty;
        SuccessMessage = string.Empty;

        try
        {
            var (success, message) = await _apiService.RegisterAsync(Email, Password, Name);

            if (success)
            {
                ShowSuccess("Account created successfully! Redirecting to login...");
                
                // Navigate back to login after 2 seconds
                await Task.Delay(2000);
                await Shell.Current.GoToAsync("//login");
            }
            else
            {
                ShowError(message ?? "Registration failed. Please try again.");
            }
        }
        catch (Exception ex)
        {
            ShowError($"Registration failed: {ex.Message}");
        }
        finally
        {
            IsLoading = false;
        }
    }

    private async Task GoToLoginAsync()
    {
        await Shell.Current.GoToAsync("//login");
    }

    private void ShowError(string message)
    {
        ErrorMessage = message;
        HasError = true;
        HasSuccess = false;
    }

    private void ShowSuccess(string message)
    {
        SuccessMessage = message;
        HasSuccess = true;
        HasError = false;
    }

    public event PropertyChangedEventHandler? PropertyChanged;

    protected void OnPropertyChanged([CallerMemberName] string? name = null)
    {
        PropertyChanged?.Invoke(this, new PropertyChangedEventArgs(name));
    }
}
