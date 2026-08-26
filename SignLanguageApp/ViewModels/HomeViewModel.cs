using System.Collections.ObjectModel;
using CommunityToolkit.Mvvm.ComponentModel;
using CommunityToolkit.Mvvm.Input;
using Microsoft.Maui.Controls;
using SignLanguageApp.Services;

namespace SignLanguageApp.ViewModels;

public partial class ShortItem
{
    public int Id { get; set; }
    public string Title { get; set; } = string.Empty;
    public string Thumbnail { get; set; } = string.Empty;
    public string VideoId { get; set; } = string.Empty;
    public int ViewCount { get; set; }
    public string Duration { get; set; } = string.Empty;
}

public partial class LessonItem
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

public partial class SignOfTheDayItem
{
    public int Id { get; set; }
    public string SignName { get; set; } = string.Empty;
    public string Description { get; set; } = string.Empty;
    public string ImageUrl { get; set; } = string.Empty;
    public string VideoUrl { get; set; } = string.Empty;
}

public partial class CommunityItem
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
    public ObservableCollection<CommunityItem> SourceCreators { get; } = [];

    // ============ Observable Properties ============
    [ObservableProperty]
    public partial string GreetingMessage { get; set; } = string.Empty;

    [ObservableProperty]
    public partial int LearningStreak { get; set; }

    [ObservableProperty]
    public partial SignOfTheDayItem? SignOfTheDay { get; set; }

    [ObservableProperty]
    public partial bool IsSignOfTheDayExpanded { get; set; }

    [ObservableProperty]
    public partial bool IsCameraPreviewVisible { get; set; }

    [ObservableProperty]
    public partial bool IsLoadingShorts { get; set; }

    [ObservableProperty]
    public partial bool IsLoadingLessons { get; set; }

    [ObservableProperty]
    public partial bool IsLoadingMore { get; set; }

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
            Thumbnail = "https://images.unsplash.com/photo-1516979187457-637abb4f9353?auto=format&fit=crop&q=80&w=300&h=200",
            VideoId = "1",
            ViewCount = 2500,
            Duration = "0:45"
        });
        Shorts.Add(new ShortItem
        {
            Id = 2,
            Title = "Numbers 1-10",
            Thumbnail = "https://images.unsplash.com/photo-1620336655052-a549d414a1a5?auto=format&fit=crop&q=80&w=300&h=200",
            VideoId = "2",
            ViewCount = 1890,
            Duration = "1:20"
        });
        Shorts.Add(new ShortItem
        {
            Id = 3,
            Title = "Common Words",
            Thumbnail = "https://images.unsplash.com/photo-1516979187457-637abb4f9353?auto=format&fit=crop&q=80&w=300&h=200",
            VideoId = "3",
            ViewCount = 3200,
            Duration = "2:15"
        });
        Shorts.Add(new ShortItem
        {
            Id = 4,
            Title = "Greetings",
            Thumbnail = "https://images.unsplash.com/photo-1620336655052-a549d414a1a5?auto=format&fit=crop&q=80&w=300&h=200",
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
            RecommendedLessons.Clear();

            var recommendation = await _apiService.GetPersonalizedRecommendationAsync();
            if (recommendation?.Data != null)
            {
                var thumb = !string.IsNullOrWhiteSpace(recommendation.Data.ThumbnailUrl)
                    ? recommendation.Data.ThumbnailUrl
                    : "https://images.unsplash.com/photo-1516979187457-637abb4f9353?auto=format&fit=crop&q=80&w=300&h=200";

                var lesson = new LessonItem
                {
                    Id = recommendation.Data.RecommendedLessonId > 0 ? recommendation.Data.RecommendedLessonId : 1,
                    Title = !string.IsNullOrWhiteSpace(recommendation.Data.LessonTitle)
                        ? recommendation.Data.LessonTitle
                        : !string.IsNullOrWhiteSpace(recommendation.Data.RecommendedCategoryTitle)
                            ? recommendation.Data.RecommendedCategoryTitle
                            : "Classifier Basics (CL:1, CL:3)",
                    Subtitle = recommendation.Data.Reason,
                    Thumbnail = thumb,
                    Difficulty = "Intermediate",
                    ViewCount = 1420
                };
                RecommendedLessons.Add(lesson);
            }

            // Also fetch lessons from API to populate additional recommended items
            var lessonsResponse = await _apiService.GetLessonsAsync();
            if (lessonsResponse?.Data != null && lessonsResponse.Data.Any())
            {
                foreach (var l in lessonsResponse.Data.Where(x => !RecommendedLessons.Any(existing => existing.Id == x.Id)).Take(3))
                {
                    var thumb = !string.IsNullOrWhiteSpace(l.ThumbnailUrl)
                        ? l.ThumbnailUrl
                        : "https://images.unsplash.com/photo-1620336655052-a549d414a1a5?auto=format&fit=crop&q=80&w=300&h=200";

                    RecommendedLessons.Add(new LessonItem
                    {
                        Id = l.Id,
                        Title = l.Title,
                        Subtitle = l.Description,
                        Thumbnail = thumb,
                        InstructorName = l.InstructorName,
                        ViewCount = l.DurationSeconds * 2 + 150,
                        Difficulty = l.Difficulty
                    });
                }
            }

            if (RecommendedLessons.Count == 0)
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
            Thumbnail = "https://images.unsplash.com/photo-1516979187457-637abb4f9353?auto=format&fit=crop&q=80&w=300&h=200",
            InstructorName = "Alex Johnson",
            InstructorAvatar = "https://images.unsplash.com/photo-1534528741775-53994a69daeb?auto=format&fit=crop&w=64&h=64",
            ViewCount = 5200,
            Difficulty = "Beginner"
        });
        RecommendedLessons.Add(new LessonItem
        {
            Id = 2,
            Title = "Daily Conversations",
            Subtitle = "Learn phrases for everyday communication",
            Thumbnail = "https://images.unsplash.com/photo-1620336655052-a549d414a1a5?auto=format&fit=crop&q=80&w=300&h=200",
            InstructorName = "Sarah Smith",
            InstructorAvatar = "https://images.unsplash.com/photo-1534528741775-53994a69daeb?auto=format&fit=crop&w=64&h=64",
            ViewCount = 3800,
            Difficulty = "Intermediate"
        });
        RecommendedLessons.Add(new LessonItem
        {
            Id = 3,
            Title = "Advanced Grammar",
            Subtitle = "Complex sentence structures in ASL",
            Thumbnail = "https://images.unsplash.com/photo-1516979187457-637abb4f9353?auto=format&fit=crop&q=80&w=300&h=200",
            InstructorName = "Mike Davis",
            InstructorAvatar = "https://images.unsplash.com/photo-1534528741775-53994a69daeb?auto=format&fit=crop&w=64&h=64",
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
                ImageUrl = "https://images.unsplash.com/photo-1516979187457-637abb4f9353?auto=format&fit=crop&q=80&w=300&h=200",
                VideoUrl = ""
            };
        }

        try
        {
            var stats = await _apiService.GetUserStatsAsync();
            if (stats != null)
            {
                LearningStreak = stats.CurrentStreak;
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
        _ = LoadSourceCreators();
    }

    private async Task LoadSourceCreators()
    {
        try
        {
            var credits = await _apiService.GetSignerCreditsAsync();
            SourceCreators.Clear();
            if (credits?.Data != null && credits.Data.Any())
            {
                int id = 1;
                foreach (var credit in credits.Data)
                {
                    SourceCreators.Add(new CommunityItem
                    {
                        Id = id++,
                        Title = credit.SignerName,
                        Contributor = credit.Bio,
                        Thumbnail = credit.AvatarUrl
                    });
                }
            }
            else
            {
                SourceCreators.Add(new CommunityItem
                {
                    Id = 1,
                    Title = "SignSchool",
                    Contributor = "An online platform offering free ASL resources.",
                    Thumbnail = "https://images.unsplash.com/photo-1531427186611-ecfd6d936c79?w=200"
                });
            }
        }
        catch (Exception ex)
        {
            System.Diagnostics.Debug.WriteLine($"Error loading source creators: {ex.Message}");
        }
    }

    }
#pragma warning restore MVVMTK0045
