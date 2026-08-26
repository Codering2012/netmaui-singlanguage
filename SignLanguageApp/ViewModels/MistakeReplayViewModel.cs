using CommunityToolkit.Mvvm.ComponentModel;
using CommunityToolkit.Mvvm.Input;
using System.Diagnostics;

namespace SignLanguageApp.ViewModels;

public partial class MistakeReplayViewModel : ObservableObject, IQueryAttributable
{
    [ObservableProperty]
    public partial string TargetGesture { get; set; }

    [ObservableProperty]
    public partial byte[]? UserFrame { get; set; }

    [ObservableProperty]
    public partial string CorrectVideoUrl { get; set; }

    public MistakeReplayViewModel()
    {
        TargetGesture = string.Empty;
        CorrectVideoUrl = string.Empty;
    }

    public void ApplyQueryAttributes(IDictionary<string, object> query)
    {
        if (query.TryGetValue("targetGesture", out var targetGestureObj) && targetGestureObj != null)
        {
            TargetGesture = targetGestureObj.ToString() ?? string.Empty;
        }

        if (query.TryGetValue("userFrame", out var userFrameObj) && userFrameObj is byte[] frame)
        {
            UserFrame = frame;
        }
    }

    [RelayCommand]
    private async Task BackToLesson()
    {
        await Helpers.NavigationHelper.SafeNavigateAsync("..");
    }
}

