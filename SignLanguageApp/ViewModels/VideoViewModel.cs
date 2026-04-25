using System.Collections.ObjectModel;
using System.Linq;
using System.Windows.Input;
using CommunityToolkit.Mvvm.ComponentModel;
using CommunityToolkit.Mvvm.Input;
using Microsoft.Maui.Controls;
using SignLanguageApp.Services;

namespace SignLanguageApp.ViewModels;

/// <summary>
/// Represents a single video with metadata
/// </summary>
public partial class VideoItem : ObservableObject
{
    public int Id { get; set; }
    public string Title { get; set; } = string.Empty;
    public string Description { get; set; } = string.Empty;
    public string ThumbnailUrl { get; set; } = string.Empty;
    public string VideoUrl { get; set; } = string.Empty;
    public DateTime UploadDate { get; set; }
    public int ViewCount { get; set; }
    public int LikeCount { get; set; }
    public double DurationSeconds { get; set; }
    public string Category { get; set; } = string.Empty;
    public bool IsLiked { get; set; }
    public string Instructor { get; set; } = string.Empty;
}

/// <summary>
/// ViewModel for video listing and management (YouTube/TikTok style)
/// </summary>
#pragma warning disable MVVMTK0045
public partial class VideoViewModel : ObservableObject
{
    private readonly IApiService _apiService;
    private readonly List<VideoItem> _allVideos = [];

    [ObservableProperty]
    private ObservableCollection<VideoItem> videos = new();

    [ObservableProperty]
    private ObservableCollection<string> categories = new();

    [ObservableProperty]
    private string selectedCategory = "All";

    [ObservableProperty]
    private bool isLoading;

    [ObservableProperty]
    private string searchQuery = string.Empty;

    [ObservableProperty]
    private int totalVideos = 0;

    public VideoViewModel(IApiService apiService)
    {
        _apiService = apiService;
    }

    [RelayCommand]
    public async Task LoadVideos()
    {
        if (IsLoading) return;
        IsLoading = true;

        try
        {
            var videosData = await _apiService.GetVideosAsync();

            if (videosData?.Data != null)
            {
                var videoList = videosData.Data.ToList();
                var videoItems = videoList.ConvertAll(dto => new VideoItem
                {
                    Id = dto.Id,
                    Title = dto.Title,
                    Description = dto.Description,
                    ThumbnailUrl = dto.ThumbnailUrl,
                    VideoUrl = dto.VideoUrl,
                    UploadDate = dto.UploadDate,
                    ViewCount = dto.ViewCount,
                    LikeCount = dto.LikeCount,
                    DurationSeconds = dto.DurationSeconds,
                    Category = dto.Category,
                    Instructor = dto.Instructor,
                    IsLiked = dto.IsLiked
                });
                _allVideos.Clear();
                _allVideos.AddRange(videoItems);
                Videos = new ObservableCollection<VideoItem>(videoItems);
                TotalVideos = videoItems.Count;
            }

            LoadCategories();
            ApplyFilters();
        }
        catch (Exception ex)
        {
            await ShowAlertAsync("Error", $"Failed to load videos: {ex.Message}");
        }
        finally
        {
            IsLoading = false;
        }
    }

    [RelayCommand]
    public async Task FilterByCategory()
    {
        if (_allVideos.Count == 0)
        {
            await LoadVideos();
            return;
        }

        ApplyFilters();
    }

    [RelayCommand]
    public async Task SearchVideos()
    {
        if (_allVideos.Count == 0)
        {
            await LoadVideos();
            return;
        }

        ApplyFilters();
    }

    [RelayCommand]
    public async Task LikeVideo(VideoItem video)
    {
        if (video == null) return;

        video.IsLiked = !video.IsLiked;
        video.LikeCount += video.IsLiked ? 1 : -1;

        try
        {
            // Persist like to API
            await _apiService.LikeVideoAsync(video.Id, video.IsLiked);
        }
        catch
        {
            // Revert on error
            video.IsLiked = !video.IsLiked;
            video.LikeCount += video.IsLiked ? 1 : -1;
        }
    }

    [RelayCommand]
    public async Task WatchVideo(VideoItem video)
    {
        if (video == null) return;

        try
        {
            // Track video view in API
            await _apiService.WatchVideoAsync(video.Id);

            if (!string.IsNullOrWhiteSpace(video.VideoUrl) && Uri.TryCreate(video.VideoUrl, UriKind.Absolute, out var videoUri))
            {
                await Launcher.Default.OpenAsync(videoUri);
                return;
            }

            await ShowAlertAsync("Unavailable", "This video does not have a playable URL yet.");
        }
        catch (Exception ex)
        {
            await ShowAlertAsync("Error", $"Could not open video: {ex.Message}");
        }
    }

    [RelayCommand]
    public async Task ShareVideo(VideoItem video)
    {
        if (video == null) return;

        try
        {
            await Share.Default.RequestAsync(new ShareTextRequest
            {
                Title = video.Title,
                Text = $"{video.Title}: {video.Description}",
                Uri = video.VideoUrl,
            });
        }
        catch
        {
            // Gracefully handle if share is not available
        }
    }

    private void LoadCategories()
    {
        var categoryList = new List<string> { "All" };
        categoryList.AddRange(_allVideos.Select(v => v.Category).Distinct().OrderBy(c => c));

        Categories = new ObservableCollection<string>(categoryList);
    }

    partial void OnSelectedCategoryChanged(string value)
    {
        if (IsLoading)
        {
            return;
        }

        ApplyFilters();
    }

    private void ApplyFilters()
    {
        IEnumerable<VideoItem> filtered = _allVideos;

        if (!string.IsNullOrWhiteSpace(SelectedCategory) &&
            !string.Equals(SelectedCategory, "All", StringComparison.OrdinalIgnoreCase))
        {
            filtered = filtered.Where(v => string.Equals(v.Category, SelectedCategory, StringComparison.OrdinalIgnoreCase));
        }

        if (!string.IsNullOrWhiteSpace(SearchQuery))
        {
            filtered = filtered.Where(v =>
                v.Title.Contains(SearchQuery, StringComparison.OrdinalIgnoreCase) ||
                v.Description.Contains(SearchQuery, StringComparison.OrdinalIgnoreCase));
        }

        Videos = new ObservableCollection<VideoItem>(filtered);
    }

    private static async Task ShowAlertAsync(string title, string message)
    {
        var page = Application.Current?.Windows?.FirstOrDefault()?.Page;
        if (page != null)
        {
            await page.DisplayAlertAsync(title, message, "OK");
        }
    }
}
#pragma warning restore MVVMTK0045
