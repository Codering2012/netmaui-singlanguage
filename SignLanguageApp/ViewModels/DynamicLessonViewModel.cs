using CommunityToolkit.Mvvm.ComponentModel;
using CommunityToolkit.Mvvm.Input;

namespace SignLanguageApp.ViewModels;

#pragma warning disable MVVMTK0045
public partial class DynamicLessonViewModel : ObservableObject
{
    [ObservableProperty]
    public partial string LessonTitle { get; set; } = string.Empty;

    [ObservableProperty]
    public partial string LessonDescription { get; set; } = string.Empty;

    [RelayCommand]
    public async Task StartPractice()
    {
        await Shell.Current.GoToAsync("//translation");
    }
}
#pragma warning restore MVVMTK0045
