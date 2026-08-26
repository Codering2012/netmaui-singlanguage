using CommunityToolkit.Mvvm.ComponentModel;
using System.ComponentModel;
using System.Runtime.CompilerServices;
using System.Threading.Tasks;
using SignLanguageApp.Services;


namespace SignLanguageApp.ViewModels
{
    public abstract partial class BaseViewModel : ObservableObject
    {
        [ObservableProperty]
        [NotifyPropertyChangedFor(nameof(IsNotBusy))]
        public partial bool IsBusy { get; set; }

        public bool IsNotBusy => !IsBusy;

        [ObservableProperty]
        public partial string Title { get; set; }

        [ObservableProperty]
        [NotifyPropertyChangedFor(nameof(HasError))]
        public partial string ErrorMessage { get; set; }

        public bool HasError => !string.IsNullOrEmpty(ErrorMessage);

        [ObservableProperty]
        public partial bool IsDisconnected { get; set; }

        [ObservableProperty]
        public partial bool IsServerDown { get; set; }



        protected async Task RunSafeAsync(Func<Task> action, bool showLoading = true)
        {
            if (IsBusy) return;

            try
            {
                if (showLoading) IsBusy = true;
                ErrorMessage = string.Empty;
                IsDisconnected = false;
                IsServerDown = false;

                await action();
            }
            catch (NoInternetException)
            {
                IsDisconnected = true;
                ErrorMessage = "Can't connect to network. Please check your WiFi or mobile data.";
            }
            catch (ServerUnreachableException ex)
            {
                IsServerDown = true;
                ErrorMessage = "Can't connect to server. Our sign language experts are looking into it!";
                System.Diagnostics.Debug.WriteLine($"Server unreachable: {ex.Message}");
            }
            catch (UnauthorizedException)
            {
                ErrorMessage = "Your session has expired. Please log in again.";
                // Logic to navigate to login could be added here
            }
            catch (ApiException ex)
            {
                ErrorMessage = ex.Message;
                System.Diagnostics.Debug.WriteLine($"API Error ({ex.StatusCode}): {ex.ResponseContent}");
            }
            catch (Exception ex)
            {
                ErrorMessage = $"Login failed: {ex.Message}";
                System.Diagnostics.Debug.WriteLine($"Generic Error: {ex.Message}");
            }
            finally
            {
                if (showLoading) IsBusy = false;
            }
        }
    }
}

