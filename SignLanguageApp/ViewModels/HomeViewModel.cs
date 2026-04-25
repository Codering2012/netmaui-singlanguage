using System.Collections.ObjectModel;
using CommunityToolkit.Mvvm.ComponentModel;
using CommunityToolkit.Mvvm.Input;
using Microsoft.Maui.Controls;
using SignLanguageApp.Services;

namespace SignLanguageApp.ViewModels;

public class ShortItem
{
    public int Id { get; set; }
    public string Title { get; set; } = string.Empty;
    public string Thumbnail { get; set; } = string.Empty;
    public string VideoId { get; set; } = string.Empty;
    public int ViewCount { get; set; }
    public string Duration { get; set; } = string.Empty;
}

public class LessonItem
{
    public int Id { get; set; }
    public string Title { get; set; } = string.Empty;
    public string Subtitle { get; set; } = string.Empty;
    public string Thumbnail { get; set; } = string.Empty;
    public string InstructorName { get; set; } = string.Empty;
    public string InstructorAvatar { get; set; } = string.Empty;
    public int ViewCount { get; set; }
    public string Difficulty { get; set; } = string.Empty;
}

public class SignOfTheDayItem
{
    public int Id { get; set; }
    public string SignName { get; set; } = string.Empty;
    public string Description { get; set; } = string.Empty;
    public string ImageUrl { get; set; } = string.Empty;
    public string VideoUrl { get; set; } = string.Empty;
}

public class CommunityItem
{
    public int Id { get; set; }
    public string Title { get; set; } = string.Empty;
    public string Contributor { get; set; } = string.Empty;
    public string Thumbnail { get; set; } = string.Empty;
}

/// <summary>
/// Modern MVVM Toolkit refactor with [ObservableProperty] and [RelayCommand]
/// </summary>
#pragma warning disable MVVMTK0045 // Field using [ObservableProperty] not AOT compatible for WinRT
public partial class HomeViewModel : ObservableObject
{
    private int _recommendedPage = 1;
    private const int PageSize = 10;

    // ============ Observable Collections ============
    public ObservableCollection<ShortItem> Shorts { get; } = [];
    public ObservableCollection<LessonItem> RecommendedLessons { get; } = [];
    public ObservableCollection<CommunityItem> Community { get; } = [];

    // ============ Observable Properties ============
    [ObservableProperty]
    private string greetingMessage = string.Empty;

    [ObservableProperty]
    private int learningStreak;

    [ObservableProperty]
    private SignOfTheDayItem? signOfTheDay;

    [ObservableProperty]
    private bool isSignOfTheDayExpanded;

    [ObservableProperty]
    private bool isCameraPreviewVisible;

    [ObservableProperty]
    private bool isLoadingShorts;

    [ObservableProperty]
    private bool isLoadingLessons;

    [ObservableProperty]
    private bool isLoadingMore;

    // ============ Constructor ============
    private readonly IApiService _apiService;

    public HomeViewModel(IApiService apiService)
    {
        _apiService = apiService;
        _ = InitializeHomeAsync();
    }

    // ============ Relay Commands ============

    [RelayCommand]
    public async Task LoadShorts()
    {
        IsLoadingShorts = true;
        try
        {
            var videos = await _apiService.GetVideosAsync();
            if (videos?.Data != null && videos.Data.Any())
            {
                Shorts.Clear();
                foreach (var video in videos.Data.Take(6))
                {
                    Shorts.Add(new ShortItem
                    {
                        Id = video.Id,
                        Title = video.Title,
                        Thumbnail = video.ThumbnailUrl,
                        VideoId = video.Id.ToString(),
                        ViewCount = video.ViewCount,
                        Duration = TimeSpan.FromSeconds(video.DurationSeconds).ToString(@"m\:ss")
                    });
                }
            }
            else
            {
                // Add sample shorts if API returns no data
                AddSampleShorts();
            }
        }
        catch (Exception ex)
        {
            System.Diagnostics.Debug.WriteLine($"Error loading shorts: {ex.Message}");
            // Add sample shorts on error
            AddSampleShorts();
        }
        finally
        {
            IsLoadingShorts = false;
        }
    }

    private void AddSampleShorts()
    {
        if (Shorts.Count > 0) return;

        Shorts.Add(new ShortItem
        {
            Id = 1,
            Title = "Alphabet Basics",
            Thumbnail = "https://via.placeholder.com/140x140?text=Alphabet",
            VideoId = "1",
            ViewCount = 2500,
            Duration = "0:45"
        });
        Shorts.Add(new ShortItem
        {
            Id = 2,
            Title = "Numbers 1-10",
            Thumbnail = "https://via.placeholder.com/140x140?text=Numbers",
            VideoId = "2",
            ViewCount = 1890,
            Duration = "1:20"
        });
        Shorts.Add(new ShortItem
        {
            Id = 3,
            Title = "Common Words",
            Thumbnail = "https://via.placeholder.com/140x140?text=Words",
            VideoId = "3",
            ViewCount = 3200,
            Duration = "2:15"
        });
        Shorts.Add(new ShortItem
        {
            Id = 4,
            Title = "Greetings",
            Thumbnail = "https://via.placeholder.com/140x140?text=Greetings",
            VideoId = "4",
            ViewCount = 4100,
            Duration = "1:50"
        });
    }

    [RelayCommand]
    public async Task LoadRecommendedLessons()
    {
        IsLoadingLessons = true;
        try
        {
            var recommendation = await _apiService.GetPersonalizedRecommendationAsync();
            if (recommendation?.Data != null)
            {
                var lesson = new LessonItem
                {
                    Id = recommendation.Data.RecommendedLessonId,
                    Title = !string.IsNullOrWhiteSpace(recommendation.Data.LessonTitle)
                        ? recommendation.Data.LessonTitle
                        : !string.IsNullOrWhiteSpace(recommendation.Data.RecommendedCategoryTitle)
                            ? recommendation.Data.RecommendedCategoryTitle
                            : recommendation.Data.CategoryName,
                    Subtitle = recommendation.Data.Reason,
                    Difficulty = "Intermediate"
                };
                RecommendedLessons.Clear();
                RecommendedLessons.Add(lesson);
            }
            else
            {
                AddSampleLessons();
            }
        }
        catch (Exception ex)
        {
            System.Diagnostics.Debug.WriteLine($"Error loading recommended lessons: {ex.Message}");
            AddSampleLessons();
        }
        finally
        {
            IsLoadingLessons = false;
        }
    }

    private void AddSampleLessons()
    {
        if (RecommendedLessons.Count > 0) return;

        RecommendedLessons.Add(new LessonItem
        {
            Id = 1,
            Title = "Fingerspelling 101",
            Subtitle = "Master the basics of fingerspelling",
            Thumbnail = "https://via.placeholder.com/110x80?text=Fingerspell",
            InstructorName = "Alex Johnson",
            InstructorAvatar = "https://via.placeholder.com/32x32?text=AJ",
            ViewCount = 5200,
            Difficulty = "Beginner"
        });
        RecommendedLessons.Add(new LessonItem
        {
            Id = 2,
            Title = "Daily Conversations",
            Subtitle = "Learn phrases for everyday communication",
            Thumbnail = "https://via.placeholder.com/110x80?text=Conversations",
            InstructorName = "Sarah Smith",
            InstructorAvatar = "https://via.placeholder.com/32x32?text=SS",
            ViewCount = 3800,
            Difficulty = "Intermediate"
        });
        RecommendedLessons.Add(new LessonItem
        {
            Id = 3,
            Title = "Advanced Grammar",
            Subtitle = "Complex sentence structures in ASL",
            Thumbnail = "https://via.placeholder.com/110x80?text=Grammar",
            InstructorName = "Mike Davis",
            InstructorAvatar = "https://via.placeholder.com/32x32?text=MD",
            ViewCount = 2100,
            Difficulty = "Advanced"
        });
    }

    [RelayCommand]
    public async Task LoadMoreLessons()
    {
        if (IsLoadingMore) return;

        IsLoadingMore = true;
        try
        {
            _recommendedPage++;
            var recommendation = await _apiService.GetPersonalizedRecommendationAsync();
            if (recommendation?.Data != null)
            {
                var lesson = new LessonItem
                {
                    Id = recommendation.Data.RecommendedLessonId,
                    Title = !string.IsNullOrWhiteSpace(recommendation.Data.LessonTitle)
                        ? recommendation.Data.LessonTitle
                        : !string.IsNullOrWhiteSpace(recommendation.Data.RecommendedCategoryTitle)
                            ? recommendation.Data.RecommendedCategoryTitle
                            : recommendation.Data.CategoryName,
                    Subtitle = recommendation.Data.Reason
                };
                RecommendedLessons.Add(lesson);
            }
        }
        catch (Exception ex)
        {
            System.Diagnostics.Debug.WriteLine($"Error loading more lessons: {ex.Message}");
        }
        finally
        {
            IsLoadingMore = false;
        }
    }

    [RelayCommand]
    public async Task ToggleSignOfTheDay()
    {
        IsSignOfTheDayExpanded = !IsSignOfTheDayExpanded;
        await Task.CompletedTask;
    }

    [RelayCommand]
    public async Task OpenCamera()
    {
        IsCameraPreviewVisible = true;
        await Shell.Current.GoToAsync("//translation");
    }

    [RelayCommand]
    public async Task NavigateToLesson(LessonItem lesson)
    {
        if (lesson?.Id > 0)
        {
            await Shell.Current.GoToAsync("//learn");
            return;
        }

        await Shell.Current.GoToAsync("//learn");
    }

    // ============ Private Methods ============

    private async Task InitializeHomeAsync()
    {
        var hour = DateTime.Now.Hour;
        GreetingMessage = hour switch
        {
            < 12 => "Good Morning",
            < 18 => "Good Afternoon",
            _ => "Good Evening"
        };

        // Set default sign of the day
        if (SignOfTheDay == null)
        {
            SignOfTheDay = new SignOfTheDayItem
            {
                Id = 1,
                SignName = "Hello",
                Description = "Learn how to greet someone in American Sign Language",
                ImageUrl = "https://via.placeholder.com/400x200?text=Hello+Sign",
                VideoUrl = ""
            };
        }

        try
        {
            var stats = await _apiService.GetUserStatsAsync();
            if (stats?.Data != null)
            {
                LearningStreak = stats.Data.CurrentStreak;
            }
        }
        catch (Exception ex)
        {
            System.Diagnostics.Debug.WriteLine($"Error loading user stats: {ex.Message}");
            LearningStreak = 0;
        }

        // Load data in parallel
        _ = LoadShorts();
        _ = LoadRecommendedLessons();
        _ = LoadCommunityTranslations();
    }

    private async Task LoadCommunityTranslations()
    {
        try
        {
            Community.Clear();
            // Add sample community items since API might not have this
            Community.Add(new CommunityItem
            {
                Id = 1,
                Title = "Coffee Order",
                Contributor = "Sarah M.",
                Thumbnail = "https://via.placeholder.com/100x80?text=Coffee"
            });
            Community.Add(new CommunityItem
            {
                Id = 2,
                Title = "Thank You",
                Contributor = "Mike D.",
                Thumbnail = "https://via.placeholder.com/100x80?text=Thanks"
            });
        }
        catch (Exception ex)
        {
            System.Diagnostics.Debug.WriteLine($"Error loading community: {ex.Message}");
        }
    }

    }
#pragma warning restore MVVMTK0045
