using System;
using System.Collections.Generic;
using System.Collections.ObjectModel;
using System.Linq;
using System.Threading.Tasks;
using CommunityToolkit.Mvvm.ComponentModel;
using CommunityToolkit.Mvvm.Input;
using Microsoft.Maui.Controls;
using SignLanguageApp.Services;

namespace SignLanguageApp.ViewModels;

public partial class DictionaryItemViewModel : ObservableObject
{
    public int Id { get; set; }
    public string SignName { get; set; } = string.Empty;
    public string Category { get; set; } = "General";
    public string Description { get; set; } = string.Empty;
    public string HandShapeInstruction { get; set; } = string.Empty;
    public string VideoUrl { get; set; } = string.Empty;
    public string ThumbnailUrl { get; set; } = string.Empty;
    public string Difficulty { get; set; } = "Beginner";
    
    [ObservableProperty]
    public partial bool IsFavorite { get; set; }

    [ObservableProperty]
    public partial bool IsMastered { get; set; }
}

#pragma warning disable MVVMTK0045 // Field using [ObservableProperty] not AOT compatible for WinRT
public partial class DictionaryViewModel : ObservableObject
{
    private readonly IApiService _apiService;
    private List<DictionaryItemViewModel> _allDictionaryItems = new();

    public ObservableCollection<DictionaryItemViewModel> FilteredItems { get; } = new();
    public ObservableCollection<string> Categories { get; } = new() { "All", "Alphabet", "Greetings", "Numbers", "Daily Words" };

    [ObservableProperty]
    public partial string SearchQuery { get; set; } = string.Empty;

    [ObservableProperty]
    public partial string SelectedCategory { get; set; } = "All";

    [ObservableProperty]
    public partial DictionaryItemViewModel? SelectedSign { get; set; }

    [ObservableProperty]
    public partial bool IsDetailModalVisible { get; set; }

    [ObservableProperty]
    public partial bool IsLoading { get; set; }

    public DictionaryViewModel(IApiService apiService)
    {
        _apiService = apiService;
        _ = InitializeDictionaryAsync();
    }

    partial void OnSearchQueryChanged(string value)
    {
        ApplyFilter();
    }

    partial void OnSelectedCategoryChanged(string value)
    {
        ApplyFilter();
    }

    [RelayCommand]
    public async Task SelectCategoryFilter(string category)
    {
        SelectedCategory = category;
        await Task.CompletedTask;
    }

    [RelayCommand]
    public async Task OpenSignDetail(DictionaryItemViewModel? item)
    {
        if (item == null) return;
        SelectedSign = item;
        IsDetailModalVisible = true;
        await Task.CompletedTask;
    }

    [RelayCommand]
    public async Task CloseSignDetail()
    {
        IsDetailModalVisible = false;
        await Task.CompletedTask;
    }

    [RelayCommand]
    public async Task PracticeSign(DictionaryItemViewModel? item)
    {
        IsDetailModalVisible = false;
        if (Shell.Current != null)
        {
            await Shell.Current.GoToAsync("//translation");
        }
    }

    [RelayCommand]
    public async Task ToggleFavorite(DictionaryItemViewModel? item)
    {
        if (item == null) return;
        item.IsFavorite = !item.IsFavorite;
        await Task.CompletedTask;
    }

    [RelayCommand]
    public async Task ToggleMastered(DictionaryItemViewModel? item)
    {
        if (item == null) return;
        item.IsMastered = !item.IsMastered;
        await Task.CompletedTask;
    }

    public async Task InitializeDictionaryAsync()
    {
        if (IsLoading) return;
        IsLoading = true;

        try
        {
            _allDictionaryItems.Clear();

            // 1. Populate 26 ASL Alphabet items
            string[] alphabet = { "A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K", "L", "M", "N", "O", "P", "Q", "R", "S", "T", "U", "V", "W", "X", "Y", "Z" };
            foreach (var letter in alphabet)
            {
                _allDictionaryItems.Add(new DictionaryItemViewModel
                {
                    Id = letter[0] - 'A' + 1,
                    SignName = $"Letter {letter}",
                    Category = "Alphabet",
                    Description = $"Fingerspelling letter '{letter}' in American Sign Language.",
                    HandShapeInstruction = $"Form the static hand shape for letter '{letter}'. Keep palm forward at chest height.",
                    VideoUrl = $"/api/videos/letter/{letter}",
                    ThumbnailUrl = "https://images.unsplash.com/photo-1516979187457-637abb4f9353?auto=format&fit=crop&q=80&w=300&h=200",
                    Difficulty = "Beginner"
                });
            }

            // 2. Populate Core Vocabulary
            var vocabulary = new[]
            {
                new { Name = "Hello", Cat = "Greetings", Desc = "Standard polite greeting in ASL.", Instruction = "Open palm placed near forehead, moving gently outward in a slight wave gesture.", Diff = "Beginner" },
                new { Name = "Thank You", Cat = "Greetings", Desc = "Expressing gratitude to someone.", Instruction = "Touch fingers of open hand to chin, then move hand forward and down toward person.", Diff = "Beginner" },
                new { Name = "Please", Cat = "Greetings", Desc = "Polite request in ASL.", Instruction = "Open palm flat on chest, moving in smooth clockwise circular motion.", Diff = "Beginner" },
                new { Name = "Sorry", Cat = "Greetings", Desc = "Apologizing or expressing regret.", Instruction = "Form 'S' fist over heart, moving in small circular motion.", Diff = "Beginner" },
                new { Name = "Family", Cat = "Daily Words", Desc = "Referring to family members.", Instruction = "Form 'F' hand shapes with both hands touching index fingers, circling out to meet pinkies.", Diff = "Intermediate" },
                new { Name = "Good Morning", Cat = "Greetings", Desc = "Morning greeting in ASL.", Instruction = "Touch chin with flat palm ('Good'), then place elbow on non-dominant hand and raise forearm like rising sun ('Morning').", Diff = "Intermediate" },
                new { Name = "Numbers 1-10", Cat = "Numbers", Desc = "Counting from 1 to 10.", Instruction = "Hold dominant hand palm inward for 1-5, palm outward for 6-10 with finger configurations.", Diff = "Beginner" },
                new { Name = "Love", Cat = "Daily Words", Desc = "Expressing affection or love.", Instruction = "Cross both arms over chest in fists ('hugging' position).", Diff = "Beginner" },
                new { Name = "Friend", Cat = "Daily Words", Desc = "Referring to a friend.", Instruction = "Hook index fingers together, then reverse and hook them again.", Diff = "Beginner" },
                new { Name = "Water", Cat = "Daily Words", Desc = "Sign for water or drink.", Instruction = "Form 'W' hand shape with index/middle/ring fingers, tap index finger on chin twice.", Diff = "Beginner" },
                new { Name = "Help", Cat = "Daily Words", Desc = "Requesting or offering assistance.", Instruction = "Closed fist with thumb up placed on open flat non-dominant palm, moving upwards together.", Diff = "Intermediate" }
            };

            int nextId = 100;
            foreach (var vocab in vocabulary)
            {
                _allDictionaryItems.Add(new DictionaryItemViewModel
                {
                    Id = nextId++,
                    SignName = vocab.Name,
                    Category = vocab.Cat,
                    Description = vocab.Desc,
                    HandShapeInstruction = vocab.Instruction,
                    VideoUrl = "/api/videos/1",
                    ThumbnailUrl = "https://images.unsplash.com/photo-1620336655052-a549d414a1a5?auto=format&fit=crop&q=80&w=300&h=200",
                    Difficulty = vocab.Diff
                });
            }

            ApplyFilter();
        }
        catch (Exception ex)
        {
            System.Diagnostics.Debug.WriteLine($"Error initializing dictionary: {ex.Message}");
        }
        finally
        {
            IsLoading = false;
        }
    }

    private void ApplyFilter()
    {
        var items = _allDictionaryItems.AsEnumerable();

        if (!string.IsNullOrWhiteSpace(SelectedCategory) && SelectedCategory != "All")
        {
            items = items.Where(x => x.Category.Equals(SelectedCategory, StringComparison.OrdinalIgnoreCase));
        }

        if (!string.IsNullOrWhiteSpace(SearchQuery))
        {
            var query = SearchQuery.Trim();
            items = items.Where(x =>
                x.SignName.Contains(query, StringComparison.OrdinalIgnoreCase) ||
                x.Description.Contains(query, StringComparison.OrdinalIgnoreCase) ||
                x.HandShapeInstruction.Contains(query, StringComparison.OrdinalIgnoreCase));
        }

        FilteredItems.Clear();
        foreach (var item in items)
        {
            FilteredItems.Add(item);
        }
    }
}
#pragma warning restore MVVMTK0045
