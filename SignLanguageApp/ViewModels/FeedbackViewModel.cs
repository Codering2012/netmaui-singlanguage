using CommunityToolkit.Mvvm.ComponentModel;
using CommunityToolkit.Mvvm.Input;
using SignLanguageApp.Model;
using SignLanguageApp.Services;

namespace SignLanguageApp.ViewModels
{
    public partial class FeedbackViewModel : BaseViewModel
    {
        private readonly IApiService _apiService;

        [ObservableProperty]
        public partial string Subject { get; set; }

        [ObservableProperty]
        public partial string Message { get; set; }

        [ObservableProperty]
        public partial int Rating { get; set; }

        [ObservableProperty]
        public partial bool IsSubmitted { get; set; }

        public FeedbackViewModel(IApiService apiService)
        {
            Subject = string.Empty;
            Message = string.Empty;
            Rating = 5;
            _apiService = apiService;
            Title = "Feedback";
        }

        [RelayCommand]
        private async Task SubmitFeedbackAsync()
        {
            if (string.IsNullOrWhiteSpace(Message))
            {
                ErrorMessage = "Please enter a message.";
                return;
            }

            await RunSafeAsync(async () =>
            {
                var success = await _apiService.SubmitFeedbackAsync(new FeedbackRequest
                {
                    Subject = Subject,
                    Message = Message,
                    Rating = Rating
                });

                if (success)
                {
                    IsSubmitted = true;
                    Subject = string.Empty;
                    Message = string.Empty;
                }
                else
                {
                    ErrorMessage = "Failed to submit feedback. Please try again later.";
                }
            });
        }

        [RelayCommand]
        private void Reset()
        {
            IsSubmitted = false;
        }
    }
}


