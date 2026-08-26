using CommunityToolkit.Mvvm.ComponentModel;
using CommunityToolkit.Mvvm.Input;
using SignLanguageApp.Services;
using SignLanguageApp.Model;
using System.Collections.ObjectModel;
using System.Diagnostics;

namespace SignLanguageApp.ViewModels;

public partial class DifficultyCalibrationViewModel : ObservableObject
{
    private readonly IApiService _apiService;
    private readonly IGesturePredictionService _gestureService;
    private readonly IDatabaseService _databaseService;

    [ObservableProperty]
    public partial string CurrentSignToTest { get; set; }

    [ObservableProperty]
    public partial int CurrentStep { get; set; }

    [ObservableProperty]
    public partial int TotalSteps { get; set; }

    [ObservableProperty]
    public partial double Progress { get; set; }

    [ObservableProperty]
    public partial bool IsProcessing { get; set; }

    [ObservableProperty]
    public partial string StatusMessage { get; set; }


    private readonly List<string> _calibrationSigns = new() { "A", "L", "W", "Y", "V" };
    private int _correctAnswers = 0;

    public DifficultyCalibrationViewModel(IApiService apiService, IGesturePredictionService gestureService, IDatabaseService databaseService)
    {
        _apiService = apiService;
        _gestureService = gestureService;
        _databaseService = databaseService;
        
        CurrentSignToTest = "A";
        CurrentStep = 1;
        TotalSteps = 5;
        StatusMessage = "Sign the letter shown below";
        
        UpdateStep();
    }

    private void UpdateStep()
    {
        if (CurrentStep <= _calibrationSigns.Count)
        {
            CurrentSignToTest = _calibrationSigns[CurrentStep - 1];
            Progress = (double)CurrentStep / TotalSteps;
        }
    }

    public async Task ProcessFrameAsync(byte[] frameBytes)
    {
        if (IsProcessing || CurrentStep > TotalSteps) return;

        IsProcessing = true;
        try
        {
            var result = await _gestureService.PerformGestureInferenceAsync(frameBytes, CurrentSignToTest);
            if (result.Gesture != null && string.Equals(result.Gesture.GestureLabel, CurrentSignToTest, StringComparison.OrdinalIgnoreCase))
            {
                _correctAnswers++;
                StatusMessage = "Great! Moving to next sign...";
                await Task.Delay(1500);
                await NextStep();
            }
        }
        catch (Exception ex)
        {
            Debug.WriteLine($"Calibration error: {ex.Message}");
        }
        finally
        {
            IsProcessing = false;
        }
    }

    private async Task NextStep()
    {
        if (CurrentStep >= TotalSteps)
        {
            await FinishCalibration();
            return;
        }

        CurrentStep++;
        StatusMessage = "Sign the letter shown below";
        UpdateStep();
    }

    [RelayCommand]
    private async Task SkipCalibration()
    {
        // Default to Beginner
        await SaveLevelAndNavigate(1);
    }

    private async Task FinishCalibration()
    {
        // Simple level estimation
        int estimatedLevel = 1; // Beginner
        if (_correctAnswers >= 4) estimatedLevel = 3; // Advanced
        else if (_correctAnswers >= 2) estimatedLevel = 2; // Intermediate

        StatusMessage = $"Calibration complete! Estimated level: {GetLevelName(estimatedLevel)}";
        await Task.Delay(2000);
        await SaveLevelAndNavigate(estimatedLevel);
    }

    private string GetLevelName(int level) => level switch
    {
        1 => "Beginner",
        2 => "Intermediate",
        3 => "Advanced",
        _ => "Beginner"
    };

    private async Task SaveLevelAndNavigate(int level)
    {
        var user = await _databaseService.GetUserAsync();
        if (user != null)
        {
            // user.Level = level; // Assuming User model has Level
            await _databaseService.SaveUserAsync(user);
        }
        
        await Helpers.NavigationHelper.SafeNavigateAsync("//home");
    }
}

