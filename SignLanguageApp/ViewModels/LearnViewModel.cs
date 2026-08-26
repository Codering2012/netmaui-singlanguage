using System.Collections.ObjectModel;
using System.Windows.Input;
using CommunityToolkit.Mvvm.ComponentModel;
using CommunityToolkit.Mvvm.Input;
using Microsoft.Maui.Controls;
using SignLanguageApp.Model;
using SignLanguageApp.Pages;
using SignLanguageApp.Services;

namespace SignLanguageApp.ViewModels;

public partial class LessonCategory : ObservableObject
{
    public int Id { get; set; }
    public string Title { get; set; } = string.Empty;
    public string Description { get; set; } = string.Empty;
    public string Difficulty { get; set; } = string.Empty;
    public double Progress { get; set; }
    public string IconUrl { get; set; } = string.Empty;
}

public partial class Lesson : ObservableObject
{
    public int Id { get; set; }
    public string Title { get; set; } = string.Empty;
    public string Description { get; set; } = string.Empty;
    public string Thumbnail { get; set; } = string.Empty;
    public int DurationSeconds { get; set; }
    public string Difficulty { get; set; } = string.Empty;
    public double CompletionPercentage { get; set; }
    public bool IsCompleted { get; set; }

    [ObservableProperty]
    public partial bool IsDownloaded { get; set; }
}

public partial class SpacedRepetitionLesson : ObservableObject
{
    public int Id { get; set; }
    public string Title { get; set; } = string.Empty;
    public string DueDate { get; set; } = string.Empty;
    public int RepetitionCount { get; set; }
    public double RetentionPercentage { get; set; }
    public string Difficulty { get; set; } = string.Empty;
}

public partial class AchievementBadge : ObservableObject
{
    public int Id { get; set; }
    public string Title { get; set; } = string.Empty;
    public string Icon { get; set; } = string.Empty;
    public bool IsUnlocked { get; set; }
    public string Color { get; set; } = string.Empty;
}

public partial class LearnViewModel : ObservableObject
{
    private readonly IApiService _apiService;
    private readonly ILessonPayloadSecurityService _lessonPayloadSecurityService;
    private readonly IMediaDownloadAndCacheService _mediaCache;
    private readonly IDatabaseService _databaseService;

    private double progressPercentage = 0;
    public double ProgressPercentage
    {
        get => progressPercentage;
        set => SetProperty(ref progressPercentage, value);
    }

    private int currentStreak = 0;
    public int CurrentStreak
    {
        get => currentStreak;
        set => SetProperty(ref currentStreak, value);
    }

    private int totalXp = 0;
    public int TotalXp
    {
        get => totalXp;
        set => SetProperty(ref totalXp, value);
    }

    private string recommendationTitle = string.Empty;
    public string RecommendationTitle
    {
        get => recommendationTitle;
        set => SetProperty(ref recommendationTitle, value);
    }

    private string recommendationReason = string.Empty;
    public string RecommendationReason
    {
        get => recommendationReason;
        set => SetProperty(ref recommendationReason, value);
    }

    private int recommendedLessonId = 0;
    public int RecommendedLessonId
    {
        get => recommendedLessonId;
        set => SetProperty(ref recommendedLessonId, value);
    }

    private ObservableCollection<LessonCategory> categories = new();
    public ObservableCollection<LessonCategory> Categories
    {
        get => categories;
        set => SetProperty(ref categories, value);
    }

    private LessonCategory? selectedCategory;
    public LessonCategory? SelectedCategory
    {
        get => selectedCategory;
        set
        {
            if (SetProperty(ref selectedCategory, value))
            {
                OnSelectedCategoryChanged(value);
            }
        }
    }

    private ObservableCollection<Lesson> selectedLessons = new();
    public ObservableCollection<Lesson> SelectedLessons
    {
        get => selectedLessons;
        set => SetProperty(ref selectedLessons, value);
    }

    private LessonDetailDto? selectedLessonDetail;
    public LessonDetailDto? SelectedLessonDetail
    {
        get => selectedLessonDetail;
        set
        {
            if (SetProperty(ref selectedLessonDetail, value))
            {
                OnPropertyChanged(nameof(HasSelectedLessonDetail));
            }
        }
    }

    public bool HasSelectedLessonDetail => SelectedLessonDetail != null;

    private string selectedLessonLayoutFile = string.Empty;
    public string SelectedLessonLayoutFile
    {
        get => selectedLessonLayoutFile;
        set => SetProperty(ref selectedLessonLayoutFile, value);
    }

    private bool isCameraPracticeLesson;
    public bool IsCameraPracticeLesson
    {
        get => isCameraPracticeLesson;
        set => SetProperty(ref isCameraPracticeLesson, value);
    }

    private bool isLessonLayoutTrusted;
    public bool IsLessonLayoutTrusted
    {
        get => isLessonLayoutTrusted;
        set => SetProperty(ref isLessonLayoutTrusted, value);
    }

    private string lessonLayoutSecurityMessage = "No lesson payload loaded.";
    public string LessonLayoutSecurityMessage
    {
        get => lessonLayoutSecurityMessage;
        set => SetProperty(ref lessonLayoutSecurityMessage, value);
    }

    private int selectedTabIndex = 0;
    public int SelectedTabIndex
    {
        get => selectedTabIndex;
        set => SetProperty(ref selectedTabIndex, value);
    }

    private int dailyGoalCompleted = 0;
    public int DailyGoalCompleted
    {
        get => dailyGoalCompleted;
        set => SetProperty(ref dailyGoalCompleted, value);
    }

    private int dailyGoalTotal = 5;
    public int DailyGoalTotal
    {
        get => dailyGoalTotal;
        set => SetProperty(ref dailyGoalTotal, value);
    }

    private string dailyGoalMessage = string.Empty;
    public string DailyGoalMessage
    {
        get => dailyGoalMessage;
        set => SetProperty(ref dailyGoalMessage, value);
    }

    private double dailyGoalProgress = 0;
    public double DailyGoalProgress
    {
        get => dailyGoalProgress;
        set => SetProperty(ref dailyGoalProgress, value);
    }

    private ObservableCollection<SpacedRepetitionLesson> dailyReviewLessons = new();
    public ObservableCollection<SpacedRepetitionLesson> DailyReviewLessons
    {
        get => dailyReviewLessons;
        set => SetProperty(ref dailyReviewLessons, value);
    }

    private int tomorrowReviewCount = 0;
    public int TomorrowReviewCount
    {
        get => tomorrowReviewCount;
        set => SetProperty(ref tomorrowReviewCount, value);
    }

    private int thisWeekReviewCount = 0;
    public int ThisWeekReviewCount
    {
        get => thisWeekReviewCount;
        set => SetProperty(ref thisWeekReviewCount, value);
    }

    private double upcomingReviewProgress = 0;
    public double UpcomingReviewProgress
    {
        get => upcomingReviewProgress;
        set => SetProperty(ref upcomingReviewProgress, value);
    }

    private ObservableCollection<AchievementBadge> achievements = new();
    public ObservableCollection<AchievementBadge> Achievements
    {
        get => achievements;
        set => SetProperty(ref achievements, value);
    }

    [ObservableProperty]
    public partial SpacedRepetitionLesson? ActiveReviewItem { get; set; }

    [ObservableProperty]
    public partial bool IsReviewModalVisible { get; set; }

    [ObservableProperty]
    public partial bool IsReviewAnswerRevealed { get; set; }

    private bool isPracticeModeActive = false;
    public bool IsPracticeModeActive
    {
        get => isPracticeModeActive;
        set => SetProperty(ref isPracticeModeActive, value);
    }

    private bool isLoading = false;
    public bool IsLoading
    {
        get => isLoading;
        set => SetProperty(ref isLoading, value);
    }

    public LearnViewModel(IApiService apiService, ILessonPayloadSecurityService lessonPayloadSecurityService, IMediaDownloadAndCacheService mediaCache, IDatabaseService databaseService)
    {
        _apiService = apiService;
        _lessonPayloadSecurityService = lessonPayloadSecurityService;
        _mediaCache = mediaCache;
        _databaseService = databaseService;
    }

    public async Task InitializeAsync()
    {
        if (IsLoading) return;
        IsLoading = true;

        try
        {
            await Task.WhenAll(
                LoadUserStatsAsync(),
                LoadCategoriesAsync(),
                LoadRecommendationAsync(),
                LoadAchievementsAsync(),
                LoadDailyReviewDataAsync(),
                LoadUpcomingReviewsAsync()
            );
        }
        catch (Exception ex)
        {
            System.Diagnostics.Debug.WriteLine($"Error initializing Learn page: {ex.Message}");
            await ShowAlertAsync("Error", "Failed to load learning data");
        }
        finally
        {
            IsLoading = false;
        }
    }

    private async Task LoadUserStatsAsync()
    {
        try
        {
            var stats = await _apiService.GetUserStatsAsync();
            if (stats != null)
            {
                ProgressPercentage = stats.TotalProgress / 100.0;
                CurrentStreak = stats.CurrentStreak;
                TotalXp = stats.TotalXP;
            }
        }
        catch (Exception ex)
        {
            System.Diagnostics.Debug.WriteLine($"Error loading user stats: {ex.Message}");
        }
    }

    private async Task LoadCategoriesAsync()
    {
        try
        {
            var categoriesResponse = await _apiService.GetCategoriesAsync();
            Categories.Clear();
            if (categoriesResponse?.Data != null && categoriesResponse.Data.Any())
            {
                foreach (var category in categoriesResponse.Data)
                {
                    Categories.Add(new LessonCategory
                    {
                        Id = category.Id,
                        Title = category.Title,
                        Description = category.Description,
                        Difficulty = category.Difficulty,
                        Progress = category.Progress,
                        IconUrl = category.IconUrl ?? category.Icon ?? string.Empty
                    });
                }
            }

            if (Categories.Count == 0)
            {
                AddSampleCategories();
            }

            if (Categories.Count > 0 && SelectedCategory == null)
            {
                SelectedCategory = Categories.FirstOrDefault();
            }
        }
        catch (Exception ex)
        {
            System.Diagnostics.Debug.WriteLine($"Error loading categories: {ex.Message}");
            AddSampleCategories();
            SelectedCategory = Categories.FirstOrDefault();
        }
    }

    private void AddSampleCategories()
    {
        if (Categories.Count > 0) return;

        Categories.Add(new LessonCategory
        {
            Id = 1,
            Title = "Alphabet Basics",
            Description = "Master fingerspelling letters A through Z",
            Difficulty = "Beginner",
            Progress = 0.4
        });
        Categories.Add(new LessonCategory
        {
            Id = 2,
            Title = "Daily Greetings",
            Description = "Learn essential ASL greetings and farewells",
            Difficulty = "Beginner",
            Progress = 0.2
        });
        Categories.Add(new LessonCategory
        {
            Id = 3,
            Title = "Numbers & Counting",
            Description = "Express numbers, prices, and quantities",
            Difficulty = "Beginner",
            Progress = 0.1
        });
        Categories.Add(new LessonCategory
        {
            Id = 4,
            Title = "Real-Time Camera Drills",
            Description = "Practice sign recognition with your live webcam",
            Difficulty = "All Levels",
            Progress = 0.6
        });
    }

    private async Task LoadRecommendationAsync()
    {
        try
        {
            var recommendation = await _apiService.GetPersonalizedRecommendationAsync();
            if (recommendation?.Data != null)
            {
                RecommendationTitle =
                    !string.IsNullOrWhiteSpace(recommendation.Data.RecommendedCategoryTitle)
                        ? recommendation.Data.RecommendedCategoryTitle
                        : !string.IsNullOrWhiteSpace(recommendation.Data.CategoryName)
                            ? recommendation.Data.CategoryName
                            : recommendation.Data.LessonTitle;
                RecommendationReason = recommendation.Data.Reason;
                RecommendedLessonId = recommendation.Data.RecommendedLessonId;
            }
        }
        catch (Exception ex)
        {
            System.Diagnostics.Debug.WriteLine($"Error loading recommendation: {ex.Message}");
        }
    }

    private async Task LoadAchievementsAsync()
    {
        try
        {
            Achievements.Clear();
        }
        catch (Exception ex)
        {
            System.Diagnostics.Debug.WriteLine($"Error loading achievements: {ex.Message}");
        }
    }

    private async Task LoadDailyReviewDataAsync()
    {
        try
        {
            var dailyGoal = await _apiService.GetDailyGoalAsync();
            if (dailyGoal?.Data != null)
            {
                DailyGoalCompleted = dailyGoal.Data.CompletedToday;
                DailyGoalTotal = dailyGoal.Data.TotalRequired;
                DailyGoalProgress = DailyGoalTotal > 0 ? DailyGoalCompleted / (double)DailyGoalTotal : 0;
                var remaining = Math.Max(0, DailyGoalTotal - DailyGoalCompleted);
                DailyGoalMessage = remaining == 0
                    ? "Daily goal completed. Great work."
                    : $"Keep it up! Just {remaining} more review{(remaining == 1 ? string.Empty : "s")} to reach today's goal.";
            }

            var reviewLessons = await _apiService.GetDailyReviewLessonsAsync();
            DailyReviewLessons.Clear();
            if (reviewLessons?.Data != null)
            {
                foreach (var lesson in reviewLessons.Data)
                {
                    DailyReviewLessons.Add(new SpacedRepetitionLesson
                    {
                        Id = lesson.Id,
                        Title = lesson.Title,
                        DueDate = lesson.DueDate.ToString("g"),
                        RepetitionCount = lesson.RepetitionCount,
                        RetentionPercentage = lesson.RetentionPercentage,
                        Difficulty = lesson.Difficulty
                    });
                }
            }
        }
        catch (Exception ex)
        {
            System.Diagnostics.Debug.WriteLine($"Error loading daily review data: {ex.Message}");
        }
    }

    private async Task LoadUpcomingReviewsAsync()
    {
        try
        {
            var upcomingReviews = await _apiService.GetUpcomingReviewsAsync();
            if (upcomingReviews?.Data != null)
            {
                TomorrowReviewCount = upcomingReviews.Data.TomorrowCount;
                ThisWeekReviewCount = upcomingReviews.Data.ThisWeekCount;
                var totalUpcoming = upcomingReviews.Data.TomorrowCount + upcomingReviews.Data.ThisWeekCount + upcomingReviews.Data.NextWeekCount;
                UpcomingReviewProgress = totalUpcoming > 0 ? (upcomingReviews.Data.TomorrowCount + upcomingReviews.Data.ThisWeekCount) / (double)totalUpcoming : 0;
            }
        }
        catch (Exception ex)
        {
            System.Diagnostics.Debug.WriteLine($"Error loading upcoming reviews: {ex.Message}");
        }
    }

    private async Task LoadLessonsForCategoryAsync(int categoryId)
    {
        try
        {
            var lessons = await _apiService.GetLessonsByCategoryAsync(categoryId);
            SelectedLessons.Clear();
            if (lessons?.Data != null && lessons.Data.Any())
            {
                foreach (var lesson in lessons.Data)
                {
                    var thumb = !string.IsNullOrWhiteSpace(lesson.ThumbnailUrl)
                        ? lesson.ThumbnailUrl
                        : "https://images.unsplash.com/photo-1516979187457-637abb4f9353?auto=format&fit=crop&q=80&w=300&h=200";

                    SelectedLessons.Add(new Lesson
                    {
                        Id = lesson.Id,
                        Title = lesson.Title,
                        Description = lesson.Description,
                        Thumbnail = thumb,
                        DurationSeconds = lesson.DurationSeconds,
                        Difficulty = lesson.Difficulty,
                        CompletionPercentage = lesson.CompletionPercentage,
                        IsCompleted = lesson.CompletionPercentage >= 1.0,
                        IsDownloaded = lesson.IsDownloaded
                    });
                }
            }

            if (SelectedLessons.Count == 0)
            {
                AddSampleLessonsForCategory(categoryId);
            }
        }
        catch (Exception ex)
        {
            System.Diagnostics.Debug.WriteLine($"Error loading lessons: {ex.Message}");
            AddSampleLessonsForCategory(categoryId);
        }
    }

    private void AddSampleLessonsForCategory(int categoryId)
    {
        if (SelectedLessons.Count > 0) return;

        SelectedLessons.Add(new Lesson
        {
            Id = 1,
            Title = "Alphabet Drills (A - M)",
            Description = "Master fingerspelling hand shapes for letters A through M with live feedback.",
            Thumbnail = "https://images.unsplash.com/photo-1516979187457-637abb4f9353?auto=format&fit=crop&q=80&w=300&h=200",
            DurationSeconds = 120,
            Difficulty = "Beginner",
            CompletionPercentage = 0.5
        });

        SelectedLessons.Add(new Lesson
        {
            Id = 2,
            Title = "Alphabet Drills (N - Z)",
            Description = "Master the second half of the ASL alphabet in real-time context.",
            Thumbnail = "https://images.unsplash.com/photo-1620336655052-a549d414a1a5?auto=format&fit=crop&q=80&w=300&h=200",
            DurationSeconds = 150,
            Difficulty = "Beginner",
            CompletionPercentage = 0.0
        });

        SelectedLessons.Add(new Lesson
        {
            Id = 3,
            Title = "Common Phrases & Greetings",
            Description = "Learn Hello, Thank You, Please, and Goodbye in sign language.",
            Thumbnail = "https://images.unsplash.com/photo-1534528741775-53994a69daeb?auto=format&fit=crop&w=300&h=200",
            DurationSeconds = 180,
            Difficulty = "Beginner",
            CompletionPercentage = 0.0
        });
    }

    [RelayCommand]
    public async Task SelectTab(string tabIndex)
    {
        if (int.TryParse(tabIndex, out var index))
        {
            SelectedTabIndex = index;
            if (index == 1 && DailyReviewLessons.Count == 0)
            {
                await LoadDailyReviewDataAsync();
                await LoadUpcomingReviewsAsync();
            }
        }
    }

    [RelayCommand]
    public async Task Practice(object parameter)
    {
        IsPracticeModeActive = true;
        await Task.CompletedTask;
    }

    [RelayCommand]
    public async Task ClosePracticeMode()
    {
        IsPracticeModeActive = false;
        await Task.CompletedTask;
    }

    [RelayCommand]
    public async Task TriggerSuccessHaptic()
    {
        IsPracticeModeActive = false;
        await Task.CompletedTask;
    }

    [RelayCommand]
    public async Task CompleteLesson(object parameter)
    {
        if (parameter is Lesson lesson)
        {
            try
            {
                // Mark lesson as completed in API
                var result = await _apiService.MarkLessonCompleteAsync(lesson.Id);

                if (result?.Data == true)
                {
                    lesson.IsCompleted = true;
                    lesson.CompletionPercentage = 1.0;

                    await ShowAlertAsync("Success", $"Completed: {lesson.Title}");
                }
                else
                {
                    await ShowAlertAsync("Error", "Failed to mark lesson complete");
                }
            }
            catch (Exception ex)
            {
                System.Diagnostics.Debug.WriteLine($"Error completing lesson: {ex.Message}");
                await ShowAlertAsync("Error", $"Failed to complete lesson: {ex.Message}");
            }
        }
    }

    [RelayCommand]
    public async Task DownloadLesson(Lesson? lesson)
    {
        if (lesson == null || lesson.IsDownloaded)
            return;

        try
        {
            var page = GetCurrentPage();
            if (page == null) return;

            var action = await page.DisplayActionSheetAsync(
                "Download Options", 
                "Cancel", 
                null, 
                "Download for Local Storage", 
                "Download for In-App Use", 
                "Local and In-App"
            );

            if (action == "Cancel" || string.IsNullOrEmpty(action)) return;

            var lessonDetailResponse = await _apiService.GetLessonAsync(lesson.Id);
            var videoUrl = lessonDetailResponse?.Data?.VideoUrl;
            
            if (!string.IsNullOrWhiteSpace(videoUrl))
            {
                var cachedPath = await _mediaCache.GetCachedMediaAsync(videoUrl);
                if (!string.IsNullOrEmpty(cachedPath))
                {
                    lesson.IsDownloaded = true;
                    // Save to SQLite
                    await _databaseService.SaveDownloadStateAsync(lesson.Id, "Lesson", true, cachedPath, action);
                    
                    await ShowAlertAsync("Success", $"Downloaded successfully for: {action}");
                }
                else
                {
                    await ShowAlertAsync("Download Failed", "Could not cache the video.");
                }
            }
            else
            {
                await ShowAlertAsync("No Video", "This lesson does not have a video available for download.");
            }
        }
        catch (Exception ex)
        {
            System.Diagnostics.Debug.WriteLine($"Error downloading lesson: {ex.Message}");
            await ShowAlertAsync("Download Error", $"Failed to download: {ex.Message}");
        }
    }

    [RelayCommand]
    public async Task StartLessonFromRecommendation()
    {
        if (RecommendedLessonId > 0)
        {
            await OpenLessonByIdAsync(RecommendedLessonId);
        }
    }

    [RelayCommand]
    public async Task OpenLesson(Lesson? lesson)
    {
        if (lesson == null)
        {
            return;
        }

        await OpenLessonByIdAsync(lesson.Id);
    }

    [RelayCommand]
    public async Task SubmitDailyReview(SpacedRepetitionLesson? review)
    {
        if (review == null) return;
        ActiveReviewItem = review;
        IsReviewAnswerRevealed = false;
        IsReviewModalVisible = true;
        await Task.CompletedTask;
    }

    [RelayCommand]
    public async Task RevealReviewAnswer()
    {
        IsReviewAnswerRevealed = true;
        await Task.CompletedTask;
    }

    [RelayCommand]
    public async Task CompleteReviewWithRating(string ratingStr)
    {
        if (ActiveReviewItem == null) return;
        int rating = int.TryParse(ratingStr, out var r) ? r : 4;

        try
        {
            _ = await _apiService.MarkReviewCompleteAsync(ActiveReviewItem.Id, rating);
            
            DailyReviewLessons.Remove(ActiveReviewItem);
            DailyGoalCompleted++;
            TotalXp += 10;
            DailyGoalProgress = DailyGoalTotal > 0 ? DailyGoalCompleted / (double)DailyGoalTotal : 0;
            
            var remaining = Math.Max(0, DailyGoalTotal - DailyGoalCompleted);
            DailyGoalMessage = remaining == 0
                ? "Daily goal completed. Great work!"
                : $"Keep it up! Just {remaining} more review{(remaining == 1 ? string.Empty : "s")} to reach today's goal.";

            await ShowAlertAsync("Review Completed! 🎉", $"+10 XP earned for reviewing {ActiveReviewItem.Title}!");
        }
        catch (Exception ex)
        {
            System.Diagnostics.Debug.WriteLine($"Error completing review rating: {ex.Message}");
        }
        finally
        {
            IsReviewModalVisible = false;
            ActiveReviewItem = null;
        }
    }

    [RelayCommand]
    public async Task CloseReviewModal()
    {
        IsReviewModalVisible = false;
        ActiveReviewItem = null;
        await Task.CompletedTask;
    }

    private async Task OpenLessonByIdAsync(int lessonId)
    {
        await LoadLessonDetailsAsync(lessonId);

        if (!IsLessonLayoutTrusted)
        {
            var details = string.IsNullOrWhiteSpace(LessonLayoutSecurityMessage)
                ? "This lesson payload was blocked by local security validation."
                : LessonLayoutSecurityMessage;
            await ShowAlertAsync("Blocked", details);

            return;
        }

        if (IsCameraPracticeLesson)
        {
            await Shell.Current.GoToAsync("//translation");
            return;
        }

        var lessonDetail = SelectedLessonDetail;
        var xamlContent = lessonDetail?.Data?.UiLayout?.XamlContent;
        if (string.IsNullOrWhiteSpace(xamlContent))
        {
            await ShowAlertAsync("Error", "No dynamic layout content was available for this lesson.");

            return;
        }

        var dynamicViewModel = new DynamicLessonViewModel
        {
            LessonTitle = lessonDetail?.Title ?? $"Lesson {lessonId}",
            LessonDescription = lessonDetail?.Description ?? "Follow the prompts on screen to practice."
        };

        await Shell.Current.Navigation.PushAsync(new DynamicLessonPage(xamlContent, dynamicViewModel));
    }

    private void OnSelectedCategoryChanged(LessonCategory? value)
    {
        if (value != null)
        {
            _ = LoadLessonsForCategoryAsync(value.Id);
        }
    }

    private async Task LoadLessonDetailsAsync(int lessonId)
    {
        try
        {
            var lessonResponse = await _apiService.GetLessonAsync(lessonId);
            var lesson = lessonResponse?.Data;
            if (lesson == null)
            {
                return;
            }

            SelectedLessonDetail = lesson;
            var securityResult = _lessonPayloadSecurityService.Evaluate(lesson);

            SelectedLessonLayoutFile = securityResult.SafeLayoutFileName;
            IsCameraPracticeLesson = securityResult.IsTrusted && securityResult.IsCameraPracticeLesson;
            IsLessonLayoutTrusted = securityResult.IsTrusted;
            LessonLayoutSecurityMessage = securityResult.StatusMessage;
        }
        catch (Exception ex)
        {
            System.Diagnostics.Debug.WriteLine($"Error loading lesson details: {ex.Message}");
        }
    }

    private static Page? GetCurrentPage()
    {
        return Application.Current?.Windows?.FirstOrDefault()?.Page;
    }

    private static async Task ShowAlertAsync(string title, string message)
    {
        var page = GetCurrentPage();
        if (page != null)
        {
            await page.DisplayAlertAsync(title, message, "OK");
        }
    }
}
