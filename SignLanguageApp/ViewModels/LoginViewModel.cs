using System;
using System.ComponentModel;
using System.Linq;
using System.Runtime.CompilerServices;
using System.Threading.Tasks;
using System.Windows.Input;
using Microsoft.Maui.Controls;
using SignLanguageApp.Services;

namespace SignLanguageApp.ViewModels
{
    public partial class LoginViewModel : INotifyPropertyChanged
    {
        private readonly IAuthService _authService;
        private readonly IApiService _apiService;
        private string _email = string.Empty;
        private string _password = string.Empty;
        private string _errorMessage = string.Empty;
        private bool _hasError;
        private bool _isLoading;

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

        public string ErrorMessage
        {
            get => _errorMessage;
            set { _errorMessage = value; OnPropertyChanged(); }
        }

        public bool HasError
        {
            get => _hasError;
            set { _hasError = value; OnPropertyChanged(); }
        }

        public bool IsLoading
        {
            get => _isLoading;
            set { _isLoading = value; OnPropertyChanged(); }
        }

        public ICommand LoginCommand { get; }
        public ICommand RegisterCommand { get; }

        public LoginViewModel(IAuthService authService, IApiService apiService)
        {
            _authService = authService;
            _apiService = apiService;

            LoginCommand = new Command(async () => await LoginAsync());
            RegisterCommand = new Command(async () => await RegisterAsync());
        }

        private async Task LoginAsync()
        {
            try
            {
                if (string.IsNullOrWhiteSpace(Email) || string.IsNullOrWhiteSpace(Password))
                {
                    ShowError("Please enter email and password");
                    return;
                }

                IsLoading = true;
                HasError = false;
                ErrorMessage = string.Empty;

                try
                {
                    var response = await _authService.LoginAsync(Email, Password);

                    if (response?.AccessToken != null)
                    {
                        _apiService.SetAuthToken(response.AccessToken);
                        await SecureStorage.Default.SetAsync("access_token", response.AccessToken);
                        await SecureStorage.Default.SetAsync("jwt_token", response.AccessToken);

                        if (Application.Current != null)
                        {
#pragma warning disable CS0618
                            await Application.Current.MainPage!.DisplayAlert("Success", "Login successful", "OK");
                            MainThread.BeginInvokeOnMainThread(async () => 
                            {
                                await Task.Delay(50);
                                Application.Current.MainPage = new AppShell();
                            });
#pragma warning restore CS0618
                        }
                    }
                    else
                    {
                        ShowError("Invalid email or password");
                    }
                }
                catch (Exception ex)
                {
                    ShowError($"Login failed: {ex.Message}");
                }
                finally
                {
                    IsLoading = false;
                }
            }
            catch (Exception ex)
            {
                SignLanguageApp.Helpers.GlobalExceptionHandler.HandleException(ex);
            }
        }

        private async Task RegisterAsync()
        {
            try
            {
                // Navigate to register page
                await Shell.Current.GoToAsync("//register");
            }
            catch (Exception ex)
            {
                SignLanguageApp.Helpers.GlobalExceptionHandler.HandleException(ex);
            }
        }

        private void ShowError(string message)
        {
            ErrorMessage = message;
            HasError = true;
        }

        public event PropertyChangedEventHandler? PropertyChanged;

        protected void OnPropertyChanged([CallerMemberName] string? name = null)
        {
            PropertyChanged?.Invoke(this, new PropertyChangedEventArgs(name));
        }
    }
}
