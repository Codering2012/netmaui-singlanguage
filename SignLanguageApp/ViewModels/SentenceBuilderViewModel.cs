using CommunityToolkit.Mvvm.ComponentModel;
using CommunityToolkit.Mvvm.Input;
using System.Collections.ObjectModel;

namespace SignLanguageApp.ViewModels;

public partial class SentenceBuilderViewModel : ObservableObject
{
    [ObservableProperty]
    public partial string CurrentSentence { get; set; }

    public SentenceBuilderViewModel()
    {
    }


    public ObservableCollection<string> SelectedSigns { get; } = new();
    public ObservableCollection<string> AvailableSigns { get; } = new() { "Hello", "I", "Love", "Sign", "Language", "You", "Want", "Eat" };

    [RelayCommand]
    private void AddSign(string sign)
    {
        SelectedSigns.Add(sign);
        UpdateSentence();
    }

    [RelayCommand]
    private void RemoveSign(string sign)
    {
        SelectedSigns.Remove(sign);
        UpdateSentence();
    }

    [RelayCommand]
    private void Clear()
    {
        SelectedSigns.Clear();
        UpdateSentence();
    }

    private void UpdateSentence()
    {
        CurrentSentence = string.Join(" ", SelectedSigns);
    }
}

