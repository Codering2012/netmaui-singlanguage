using CommunityToolkit.Mvvm.ComponentModel;
using CommunityToolkit.Mvvm.Input;
using SignLanguageApp.Services;
using System.Diagnostics;

namespace SignLanguageApp.ViewModels;

public partial class EditProfileViewModel : ObservableObject
{
    private readonly IApiService _apiService;

    [ObservableProperty]
    public partial string Name { get; set; }

    [ObservableProperty]
    public partial string AvatarUrl { get; set; }

    [ObservableProperty]
    public partial string Description { get; set; }

    [ObservableProperty]
    public partial string OldPassword { get; set; }

    [ObservableProperty]
    public partial string NewPassword { get; set; }

    [ObservableProperty]
    public partial string ConfirmPassword { get; set; }

    [ObservableProperty]
    public partial bool IsLoading { get; set; }

    [ObservableProperty]
    public partial string StatusMessage { get; set; }

    [ObservableProperty]
    public partial bool IsSuccess { get; set; }

    public EditProfileViewModel(IApiService apiService)
    {
        _apiService = apiService;
    }

    public async Task InitializeAsync()
    {
        if (IsLoading) return;
        IsLoading = true;

        try
        {
            var profile = await _apiService.GetUserProfileAsync();
            if (profile?.Data != null)
            {
                Name = profile.Data.Name;
                AvatarUrl = profile.Data.AvatarUrl ?? string.Empty;
                Description = profile.Data.ProfileDescription ?? string.Empty;
            }
        }
        catch (Exception ex)
        {
            Debug.WriteLine($"Error initializing edit profile: {ex.Message}");
        }
        finally
        {
            IsLoading = false;
        }
    }

    [RelayCommand]
    public async Task SaveProfile()
    {
        if (IsLoading) return;
        IsLoading = true;
        StatusMessage = "Saving profile...";
        IsSuccess = false;

        try
        {
            bool nameUpdated = await _apiService.UpdateNameAsync(Name);
            bool avatarUpdated = await _apiService.UpdateAvatarAsync(AvatarUrl);
            bool descriptionUpdated = await _apiService.UpdateDescriptionAsync(Description);

            if (nameUpdated && avatarUpdated && descriptionUpdated)
            {
                StatusMessage = "Profile updated successfully!";
                IsSuccess = true;
                await Task.Delay(2000);
                await Helpers.NavigationHelper.SafeNavigateAsync("..");
            }
            else
            {
                StatusMessage = "Failed to update some parts of the profile.";
            }
        }
        catch (Exception ex)
        {
            StatusMessage = $"Error: {ex.Message}";
        }
        finally
        {
            IsLoading = false;
        }
    }

    [RelayCommand]
    public async Task ChangePassword()
    {
        if (string.IsNullOrWhiteSpace(OldPassword) || string.IsNullOrWhiteSpace(NewPassword))
        {
            StatusMessage = "Please enter both old and new passwords.";
            return;
        }

        if (NewPassword != ConfirmPassword)
        {
            StatusMessage = "Passwords do not match.";
            return;
        }

        if (IsLoading) return;
        IsLoading = true;
        StatusMessage = "Updating password...";

        try
        {
            bool success = await _apiService.UpdatePasswordAsync(OldPassword, NewPassword);
            if (success)
            {
                StatusMessage = "Password updated successfully!";
                OldPassword = string.Empty;
                NewPassword = string.Empty;
                ConfirmPassword = string.Empty;
            }
            else
            {
                StatusMessage = "Failed to update password. Check your old password.";
            }
        }
        catch (Exception ex)
        {
            StatusMessage = $"Error: {ex.Message}";
        }
        finally
        {
            IsLoading = false;
        }
    }

    [RelayCommand]
    public async Task Cancel()
    {
        await Helpers.NavigationHelper.SafeNavigateAsync("..");
    }
}


