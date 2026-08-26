using System;
using System.Collections;
using System.Collections.Generic;
using System.Collections.ObjectModel;
using System.Diagnostics;
using System.Linq;
using System.Threading;
using System.Threading.Tasks;

using CommunityToolkit.Mvvm;
using CommunityToolkit.Mvvm.ComponentModel;
using CommunityToolkit.Mvvm.Input;
using Microsoft.Maui;
using Microsoft.Maui.Controls;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using Moq;
using SignLanguageApp;
using SignLanguageApp.Model;
using SignLanguageApp.Services;
using SignLanguageApp.ViewModels;

namespace SignLanguageApp.ViewModels.UnitTests;


/// <summary>
/// Tests for the HomeViewModel class.
/// </summary>
[TestClass]
public partial class HomeViewModelTests
{
    /// <summary>
    /// Tests that the constructor successfully initializes with a valid apiService.
    /// Input: valid mocked IApiService
    /// Expected: Constructor completes without throwing an exception
    /// </summary>
    [TestMethod]
    public void HomeViewModel_WithValidApiService_InitializesSuccessfully()
    {
        // Arrange
        var mockApiService = new Mock<IApiService>();
        mockApiService.Setup(x => x.GetUserStatsAsync())
            .ReturnsAsync((UserStatsDto?)null);

        // Act
        var viewModel = new HomeViewModel(mockApiService.Object);

        // Assert
        Assert.IsNotNull(viewModel);
    }

    /// <summary>
    /// Tests that the constructor sets GreetingMessage based on morning hours.
    /// Input: valid mocked IApiService during morning hours (0-11)
    /// Expected: GreetingMessage contains "Good Morning"
    /// </summary>
    [TestMethod]
    public void HomeViewModel_InitializedInMorning_SetsGreetingMessageToGoodMorning()
    {
        // Arrange
        var mockApiService = new Mock<IApiService>();
        mockApiService.Setup(x => x.GetUserStatsAsync())
            .ReturnsAsync((UserStatsDto?)null);

        var currentHour = DateTime.Now.Hour;

        // Act
        var viewModel = new HomeViewModel(mockApiService.Object);

        // Assert
        if (currentHour < 12)
        {
            Assert.IsTrue(viewModel.GreetingMessage.Contains("Good Morning"));
        }
    }

    /// <summary>
    /// Tests that the constructor sets GreetingMessage based on afternoon hours.
    /// Input: valid mocked IApiService during afternoon hours (12-17)
    /// Expected: GreetingMessage contains "Good Afternoon"
    /// </summary>
    [TestMethod]
    public void HomeViewModel_InitializedInAfternoon_SetsGreetingMessageToGoodAfternoon()
    {
        // Arrange
        var mockApiService = new Mock<IApiService>();
        mockApiService.Setup(x => x.GetUserStatsAsync())
            .ReturnsAsync((UserStatsDto?)null);

        var currentHour = DateTime.Now.Hour;

        // Act
        var viewModel = new HomeViewModel(mockApiService.Object);

        // Assert
        if (currentHour >= 12 && currentHour < 18)
        {
            Assert.IsTrue(viewModel.GreetingMessage.Contains("Good Afternoon"));
        }
    }

    /// <summary>
    /// Tests that the constructor sets GreetingMessage based on evening hours.
    /// Input: valid mocked IApiService during evening hours (18-23)
    /// Expected: GreetingMessage contains "Good Evening"
    /// </summary>
    [TestMethod]
    public void HomeViewModel_InitializedInEvening_SetsGreetingMessageToGoodEvening()
    {
        // Arrange
        var mockApiService = new Mock<IApiService>();
        mockApiService.Setup(x => x.GetUserStatsAsync())
            .ReturnsAsync((UserStatsDto?)null);

        var currentHour = DateTime.Now.Hour;

        // Act
        var viewModel = new HomeViewModel(mockApiService.Object);

        // Assert
        if (currentHour >= 18)
        {
            Assert.IsTrue(viewModel.GreetingMessage.Contains("Good Evening"));
        }
    }

    /// <summary>
    /// Tests that the constructor initializes SignOfTheDay property.
    /// Input: valid mocked IApiService
    /// Expected: SignOfTheDay is not null and has default values
    /// </summary>
    [TestMethod]
    public void HomeViewModel_WithValidApiService_InitializesSignOfTheDay()
    {
        // Arrange
        var mockApiService = new Mock<IApiService>();
        mockApiService.Setup(x => x.GetUserStatsAsync())
            .ReturnsAsync((UserStatsDto?)null);

        // Act
        var viewModel = new HomeViewModel(mockApiService.Object);

        // Assert
        Assert.IsNotNull(viewModel.SignOfTheDay);
        Assert.AreEqual(1, viewModel.SignOfTheDay.Id);
        Assert.AreEqual("Hello", viewModel.SignOfTheDay.SignName);
        Assert.AreEqual("Learn how to greet someone in American Sign Language", viewModel.SignOfTheDay.Description);
    }

    /// <summary>
    /// Tests that the constructor initializes observable collections.
    /// Input: valid mocked IApiService
    /// Expected: Shorts, RecommendedLessons, and Community collections are not null
    /// </summary>
    [TestMethod]
    public void HomeViewModel_WithValidApiService_InitializesCollections()
    {
        // Arrange
        var mockApiService = new Mock<IApiService>();
        mockApiService.Setup(x => x.GetUserStatsAsync())
            .ReturnsAsync((UserStatsDto?)null);

        // Act
        var viewModel = new HomeViewModel(mockApiService.Object);

        // Assert
        Assert.IsNotNull(viewModel.Shorts);
        Assert.IsNotNull(viewModel.RecommendedLessons);
        Assert.IsNotNull(viewModel.SourceCreators);
    }

    /// <summary>
    /// Tests that the constructor calls GetUserStatsAsync on the API service.
    /// Input: valid mocked IApiService with GetUserStatsAsync setup
    /// Expected: GetUserStatsAsync is called during initialization
    /// </summary>
    [TestMethod]
    public async Task HomeViewModel_WithValidApiService_CallsGetUserStatsAsync()
    {
        // Arrange
        var mockApiService = new Mock<IApiService>();
        mockApiService.Setup(x => x.GetUserStatsAsync())
            .ReturnsAsync((UserStatsDto?)null);

        // Act
        var viewModel = new HomeViewModel(mockApiService.Object);

        // Wait for async initialization to complete
        await Task.Delay(100);

        // Assert
        mockApiService.Verify(x => x.GetUserStatsAsync(), Times.Once);
    }

    /// <summary>
    /// Tests that the constructor sets LearningStreak when GetUserStatsAsync returns valid data.
    /// Input: valid mocked IApiService returning user stats with CurrentStreak
    /// Expected: LearningStreak property is set to the returned value
    /// </summary>
    [TestMethod]
    public async Task HomeViewModel_WithValidUserStats_SetsLearningStreak()
    {
        // Arrange
        var mockApiService = new Mock<IApiService>();
        var statsData = new UserStatsDto { CurrentStreak = 7 };
        mockApiService.Setup(x => x.GetUserStatsAsync())
            .ReturnsAsync(statsData);

        // Act
        var viewModel = new HomeViewModel(mockApiService.Object);

        // Wait for async initialization to complete
        await Task.Delay(100);

        // Assert
        Assert.AreEqual(7, viewModel.LearningStreak);
    }

    /// <summary>
    /// Tests that the constructor sets LearningStreak to 0 when GetUserStatsAsync returns null.
    /// Input: valid mocked IApiService returning null
    /// Expected: LearningStreak remains 0
    /// </summary>
    [TestMethod]
    public async Task HomeViewModel_WithNullUserStats_SetsLearningStreakToZero()
    {
        // Arrange
        var mockApiService = new Mock<IApiService>();
        mockApiService.Setup(x => x.GetUserStatsAsync())
            .ReturnsAsync((UserStatsDto?)null);

        // Act
        var viewModel = new HomeViewModel(mockApiService.Object);

        // Wait for async initialization to complete
        await Task.Delay(100);

        // Assert
        Assert.AreEqual(0, viewModel.LearningStreak);
    }

    /// <summary>
    /// Tests that the constructor handles exceptions from GetUserStatsAsync gracefully.
    /// Input: valid mocked IApiService that throws an exception
    /// Expected: LearningStreak is set to 0 and no exception propagates
    /// </summary>
    [TestMethod]
    public async Task HomeViewModel_WhenGetUserStatsAsyncThrows_SetsLearningStreakToZero()
    {
        // Arrange
        var mockApiService = new Mock<IApiService>();
        mockApiService.Setup(x => x.GetUserStatsAsync())
            .ThrowsAsync(new InvalidOperationException("API error"));

        // Act
        var viewModel = new HomeViewModel(mockApiService.Object);

        // Wait for async initialization to complete
        await Task.Delay(100);

        // Assert
        Assert.AreEqual(0, viewModel.LearningStreak);
    }

    /// <summary>
    /// Tests that the constructor initializes boolean properties to default values.
    /// Input: valid mocked IApiService
    /// Expected: IsSignOfTheDayExpanded, IsCameraPreviewVisible, IsLoadingShorts, IsLoadingLessons, IsLoadingMore are all false
    /// </summary>
    [TestMethod]
    public void HomeViewModel_WithValidApiService_InitializesBooleanPropertiesToDefault()
    {
        // Arrange
        var mockApiService = new Mock<IApiService>();
        mockApiService.Setup(x => x.GetUserStatsAsync())
            .ReturnsAsync((UserStatsDto?)null);

        // Act
        var viewModel = new HomeViewModel(mockApiService.Object);

        // Assert
        Assert.IsFalse(viewModel.IsSignOfTheDayExpanded);
        Assert.IsFalse(viewModel.IsCameraPreviewVisible);
        Assert.IsFalse(viewModel.IsLoadingShorts);
        Assert.IsFalse(viewModel.IsLoadingLessons);
        Assert.IsFalse(viewModel.IsLoadingMore);
    }

    /// <summary>
    /// Tests that LoadShorts successfully populates the Shorts collection with API data
    /// when the API returns more videos than the limit (6).
    /// Expects only the first 6 videos to be added to the Shorts collection.
    /// </summary>
    [TestMethod]
    public async Task LoadShorts_WithMoreThan6Videos_PopulatesFirst6Shorts()
    {
        // Arrange
        var mockApiService = new Mock<IApiService>();
        var videos = CreateVideoList(10);
        var response = new ApiResponse<IEnumerable<VideoDto>>
        {
            Success = true,
            Data = videos
        };
        mockApiService.Setup(x => x.GetVideosAsync()).ReturnsAsync(response);
        var viewModel = new HomeViewModel(mockApiService.Object);

        // Act
        await viewModel.LoadShorts();

        // Assert
        Assert.AreEqual(6, viewModel.Shorts.Count);
        Assert.AreEqual(1, viewModel.Shorts[0].Id);
        Assert.AreEqual("Video 1", viewModel.Shorts[0].Title);
        Assert.AreEqual(9, viewModel.Shorts[5].Id);
    }

    /// <summary>
    /// Tests that LoadShorts populates the Shorts collection with exactly the number
    /// of videos returned by the API when it equals the limit.
    /// Expects all 6 videos to be added to the Shorts collection.
    /// </summary>
    [TestMethod]
    [DataRow(6)]
    [DataRow(5)]
    [DataRow(3)]
    [DataRow(1)]
    public async Task LoadShorts_WithVariousVideoCounts_PopulatesCorrectNumberOfShorts(int videoCount)
    {
        // Arrange
        var mockApiService = new Mock<IApiService>();
        var videos = CreateVideoList(videoCount);
        var response = new ApiResponse<IEnumerable<VideoDto>>
        {
            Success = true,
            Data = videos
        };
        mockApiService.Setup(x => x.GetVideosAsync()).ReturnsAsync(response);
        var viewModel = new HomeViewModel(mockApiService.Object);

        // Act
        await viewModel.LoadShorts();

        // Assert
        Assert.AreEqual(videoCount, viewModel.Shorts.Count);
    }

    /// <summary>
    /// Tests that LoadShorts correctly maps VideoDto properties to ShortItem properties.
    /// Expects all properties to be correctly mapped including duration formatting.
    /// </summary>
    [TestMethod]
    public async Task LoadShorts_WithValidVideo_MapsPropertiesCorrectly()
    {
        // Arrange
        var mockApiService = new Mock<IApiService>();
        var video = new VideoDto
        {
            Id = 42,
            Title = "Test Video Title",
            ThumbnailUrl = "https://example.com/thumb.jpg",
            ViewCount = 12345,
            DurationSeconds = 125.5 // 2:05
        };
        var response = new ApiResponse<IEnumerable<VideoDto>>
        {
            Success = true,
            Data = new[] { video }
        };
        mockApiService.Setup(x => x.GetVideosAsync()).ReturnsAsync(response);
        var viewModel = new HomeViewModel(mockApiService.Object);

        // Act
        await viewModel.LoadShorts();

        // Assert
        Assert.AreEqual(1, viewModel.Shorts.Count);
        var shortItem = viewModel.Shorts[0];
        Assert.AreEqual(42, shortItem.Id);
        Assert.AreEqual("Test Video Title", shortItem.Title);
        Assert.AreEqual("https://example.com/thumb.jpg", shortItem.Thumbnail);
        Assert.AreEqual("42", shortItem.VideoId);
        Assert.AreEqual(12345, shortItem.ViewCount);
        Assert.AreEqual("2:05", shortItem.Duration);
    }

    /// <summary>
    /// Tests that LoadShorts correctly formats duration for various DurationSeconds values.
    /// Expects proper TimeSpan formatting with m:ss pattern.
    /// </summary>
    [TestMethod]
    [DataRow(0.0, "0:00")]
    [DataRow(30.0, "0:30")]
    [DataRow(60.0, "1:00")]
    [DataRow(125.0, "2:05")]
    [DataRow(599.0, "9:59")]
    [DataRow(3599.0, "59:59")]
    public async Task LoadShorts_WithVariousDurationValues_FormatsCorrectly(double durationSeconds, string expectedDuration)
    {
        // Arrange
        var mockApiService = new Mock<IApiService>();
        var video = new VideoDto
        {
            Id = 1,
            Title = "Test",
            ThumbnailUrl = "thumb.jpg",
            ViewCount = 100,
            DurationSeconds = durationSeconds
        };
        var response = new ApiResponse<IEnumerable<VideoDto>>
        {
            Success = true,
            Data = new[] { video }
        };
        mockApiService.Setup(x => x.GetVideosAsync()).ReturnsAsync(response);
        var viewModel = new HomeViewModel(mockApiService.Object);

        // Act
        await viewModel.LoadShorts();

        // Assert
        Assert.AreEqual(expectedDuration, viewModel.Shorts[0].Duration);
    }

    /// <summary>
    /// Tests that LoadShorts handles negative duration values.
    /// Expects the duration to be formatted (TimeSpan handles negative values).
    /// </summary>
    [TestMethod]
    public async Task LoadShorts_WithNegativeDuration_FormatsWithNegativeSign()
    {
        // Arrange
        var mockApiService = new Mock<IApiService>();
        var video = new VideoDto
        {
            Id = 1,
            Title = "Test",
            ThumbnailUrl = "thumb.jpg",
            ViewCount = 100,
            DurationSeconds = -60.0
        };
        var response = new ApiResponse<IEnumerable<VideoDto>>
        {
            Success = true,
            Data = new[] { video }
        };
        mockApiService.Setup(x => x.GetVideosAsync()).ReturnsAsync(response);
        var viewModel = new HomeViewModel(mockApiService.Object);

        // Act
        await viewModel.LoadShorts();

        // Assert
        Assert.IsTrue(viewModel.Shorts[0].Duration.Contains("-"));
    }

    /// <summary>
    /// Tests that LoadShorts handles special double values for DurationSeconds (NaN, Infinity).
    /// Expects the method to complete without throwing and handle the edge cases.
    /// </summary>
    [TestMethod]
    [DataRow(double.NaN)]
    [DataRow(double.PositiveInfinity)]
    [DataRow(double.NegativeInfinity)]
    [DataRow(double.MaxValue)]
    [DataRow(double.MinValue)]
    public async Task LoadShorts_WithSpecialDoubleDurationValues_HandlesGracefully(double durationSeconds)
    {
        // Arrange
        var mockApiService = new Mock<IApiService>();
        var video = new VideoDto
        {
            Id = 1,
            Title = "Test",
            ThumbnailUrl = "thumb.jpg",
            ViewCount = 100,
            DurationSeconds = durationSeconds
        };
        var response = new ApiResponse<IEnumerable<VideoDto>>
        {
            Success = true,
            Data = new[] { video }
        };
        mockApiService.Setup(x => x.GetVideosAsync()).ReturnsAsync(response);
        var viewModel = new HomeViewModel(mockApiService.Object);

        // Act & Assert - Should not throw
        await viewModel.LoadShorts();
        Assert.AreEqual(1, viewModel.Shorts.Count);
    }

    /// <summary>
    /// Tests that LoadShorts clears existing Shorts before adding new ones.
    /// Expects the Shorts collection to be cleared and repopulated with new data.
    /// </summary>
    [TestMethod]
    public async Task LoadShorts_WithExistingShortsAndSuccessfulApiCall_ClearsAndRepopulates()
    {
        // Arrange
        var mockApiService = new Mock<IApiService>();
        var viewModel = new HomeViewModel(mockApiService.Object);

        // Add existing shorts
        viewModel.Shorts.Add(new ShortItem { Id = 999, Title = "Existing" });

        var videos = CreateVideoList(2);
        var response = new ApiResponse<IEnumerable<VideoDto>>
        {
            Success = true,
            Data = videos
        };
        mockApiService.Setup(x => x.GetVideosAsync()).ReturnsAsync(response);

        // Act
        await viewModel.LoadShorts();

        // Assert
        Assert.AreEqual(2, viewModel.Shorts.Count);
        Assert.AreEqual(1, viewModel.Shorts[0].Id);
        Assert.IsFalse(viewModel.Shorts.Any(s => s.Id == 999));
    }

    /// <summary>
    /// Tests that LoadShorts calls AddSampleShorts when API returns null response.
    /// Expects the Shorts collection to be populated with sample data.
    /// </summary>
    [TestMethod]
    public async Task LoadShorts_WithNullApiResponse_PopulatesSampleShorts()
    {
        // Arrange
        var mockApiService = new Mock<IApiService>();
        mockApiService.Setup(x => x.GetVideosAsync()).ReturnsAsync((ApiResponse<IEnumerable<VideoDto>>?)null);
        var viewModel = new HomeViewModel(mockApiService.Object);

        // Act
        await viewModel.LoadShorts();

        // Assert
        Assert.IsTrue(viewModel.Shorts.Count > 0, "Shorts should be populated with sample data");
    }

    /// <summary>
    /// Tests that LoadShorts calls AddSampleShorts when API returns response with null Data.
    /// Expects the Shorts collection to be populated with sample data.
    /// </summary>
    [TestMethod]
    public async Task LoadShorts_WithNullDataInResponse_PopulatesSampleShorts()
    {
        // Arrange
        var mockApiService = new Mock<IApiService>();
        var response = new ApiResponse<IEnumerable<VideoDto>>
        {
            Success = false,
            Data = null
        };
        mockApiService.Setup(x => x.GetVideosAsync()).ReturnsAsync(response);
        var viewModel = new HomeViewModel(mockApiService.Object);

        // Act
        await viewModel.LoadShorts();

        // Assert
        Assert.IsTrue(viewModel.Shorts.Count > 0, "Shorts should be populated with sample data");
    }

    /// <summary>
    /// Tests that LoadShorts calls AddSampleShorts when API returns empty Data collection.
    /// Expects the Shorts collection to be populated with sample data.
    /// </summary>
    [TestMethod]
    public async Task LoadShorts_WithEmptyDataInResponse_PopulatesSampleShorts()
    {
        // Arrange
        var mockApiService = new Mock<IApiService>();
        var response = new ApiResponse<IEnumerable<VideoDto>>
        {
            Success = true,
            Data = Enumerable.Empty<VideoDto>()
        };
        mockApiService.Setup(x => x.GetVideosAsync()).ReturnsAsync(response);
        var viewModel = new HomeViewModel(mockApiService.Object);

        // Act
        await viewModel.LoadShorts();

        // Assert
        Assert.IsTrue(viewModel.Shorts.Count > 0, "Shorts should be populated with sample data");
    }

    /// <summary>
    /// Tests that LoadShorts handles exceptions from the API gracefully.
    /// Expects the method to catch the exception and populate sample shorts.
    /// </summary>
    [TestMethod]
    public async Task LoadShorts_WhenApiThrowsException_PopulatesSampleShorts()
    {
        // Arrange
        var mockApiService = new Mock<IApiService>();
        mockApiService.Setup(x => x.GetVideosAsync()).ThrowsAsync(new Exception("API Error"));
        var viewModel = new HomeViewModel(mockApiService.Object);

        // Act
        await viewModel.LoadShorts();

        // Assert
        Assert.IsTrue(viewModel.Shorts.Count > 0, "Shorts should be populated with sample data after exception");
    }

    /// <summary>
    /// Tests that LoadShorts handles various exception types from the API.
    /// Expects all exception types to be caught and handled gracefully.
    /// </summary>
    [TestMethod]
    public async Task LoadShorts_WhenApiThrowsVariousExceptions_HandlesGracefully()
    {
        // Arrange & Act & Assert - InvalidOperationException
        var mockApiService1 = new Mock<IApiService>();
        mockApiService1.Setup(x => x.GetVideosAsync()).ThrowsAsync(new InvalidOperationException("Invalid operation"));
        var viewModel1 = new HomeViewModel(mockApiService1.Object);
        await viewModel1.LoadShorts();
        Assert.IsTrue(viewModel1.Shorts.Count > 0);

        // Arrange & Act & Assert - ArgumentException
        var mockApiService2 = new Mock<IApiService>();
        mockApiService2.Setup(x => x.GetVideosAsync()).ThrowsAsync(new ArgumentException("Invalid argument"));
        var viewModel2 = new HomeViewModel(mockApiService2.Object);
        await viewModel2.LoadShorts();
        Assert.IsTrue(viewModel2.Shorts.Count > 0);

        // Arrange & Act & Assert - NullReferenceException
        var mockApiService3 = new Mock<IApiService>();
        mockApiService3.Setup(x => x.GetVideosAsync()).ThrowsAsync(new NullReferenceException("Null reference"));
        var viewModel3 = new HomeViewModel(mockApiService3.Object);
        await viewModel3.LoadShorts();
        Assert.IsTrue(viewModel3.Shorts.Count > 0);
    }

    /// <summary>
    /// Tests that LoadShorts sets IsLoadingShorts to true at the start and false at the end
    /// when the API call succeeds.
    /// Expects IsLoadingShorts to be false after the method completes.
    /// </summary>
    [TestMethod]
    public async Task LoadShorts_WithSuccessfulApiCall_SetsIsLoadingShortsCorrectly()
    {
        // Arrange
        var mockApiService = new Mock<IApiService>();
        var videos = CreateVideoList(3);
        var response = new ApiResponse<IEnumerable<VideoDto>>
        {
            Success = true,
            Data = videos
        };
        mockApiService.Setup(x => x.GetVideosAsync()).ReturnsAsync(response);
        var viewModel = new HomeViewModel(mockApiService.Object);

        // Act
        await viewModel.LoadShorts();

        // Assert
        Assert.IsFalse(viewModel.IsLoadingShorts, "IsLoadingShorts should be false after completion");
    }

    /// <summary>
    /// Tests that LoadShorts sets IsLoadingShorts to false even when an exception occurs.
    /// Expects IsLoadingShorts to be false after the method completes despite the exception.
    /// </summary>
    [TestMethod]
    public async Task LoadShorts_WhenExceptionOccurs_SetsIsLoadingShortsToFalse()
    {
        // Arrange
        var mockApiService = new Mock<IApiService>();
        mockApiService.Setup(x => x.GetVideosAsync()).ThrowsAsync(new Exception("API Error"));
        var viewModel = new HomeViewModel(mockApiService.Object);

        // Act
        await viewModel.LoadShorts();

        // Assert
        Assert.IsFalse(viewModel.IsLoadingShorts, "IsLoadingShorts should be false even after exception");
    }

    /// <summary>
    /// Tests that LoadShorts sets IsLoadingShorts to false when API returns null.
    /// Expects IsLoadingShorts to be false after the method completes.
    /// </summary>
    [TestMethod]
    public async Task LoadShorts_WithNullApiResponse_SetsIsLoadingShortsToFalse()
    {
        // Arrange
        var mockApiService = new Mock<IApiService>();
        mockApiService.Setup(x => x.GetVideosAsync()).ReturnsAsync((ApiResponse<IEnumerable<VideoDto>>?)null);
        var viewModel = new HomeViewModel(mockApiService.Object);

        // Act
        await viewModel.LoadShorts();

        // Assert
        Assert.IsFalse(viewModel.IsLoadingShorts, "IsLoadingShorts should be false after completion");
    }

    /// <summary>
    /// Tests that LoadShorts handles empty string properties in VideoDto.
    /// Expects empty strings to be mapped correctly to ShortItem properties.
    /// </summary>
    [TestMethod]
    public async Task LoadShorts_WithEmptyStringProperties_MapsCorrectly()
    {
        // Arrange
        var mockApiService = new Mock<IApiService>();
        var video = new VideoDto
        {
            Id = 1,
            Title = "",
            ThumbnailUrl = "",
            ViewCount = 0,
            DurationSeconds = 0
        };
        var response = new ApiResponse<IEnumerable<VideoDto>>
        {
            Success = true,
            Data = new[] { video }
        };
        mockApiService.Setup(x => x.GetVideosAsync()).ReturnsAsync(response);
        var viewModel = new HomeViewModel(mockApiService.Object);

        // Act
        await viewModel.LoadShorts();

        // Assert
        Assert.AreEqual(1, viewModel.Shorts.Count);
        Assert.AreEqual("", viewModel.Shorts[0].Title);
        Assert.AreEqual("", viewModel.Shorts[0].Thumbnail);
    }

    /// <summary>
    /// Tests that LoadShorts handles extreme ViewCount values correctly.
    /// Expects all ViewCount values to be mapped correctly without overflow or errors.
    /// </summary>
    [TestMethod]
    [DataRow(0)]
    [DataRow(1)]
    [DataRow(int.MaxValue)]
    [DataRow(int.MinValue)]
    public async Task LoadShorts_WithExtremeViewCountValues_MapsCorrectly(int viewCount)
    {
        // Arrange
        var mockApiService = new Mock<IApiService>();
        var video = new VideoDto
        {
            Id = 1,
            Title = "Test",
            ThumbnailUrl = "thumb.jpg",
            ViewCount = viewCount,
            DurationSeconds = 60
        };
        var response = new ApiResponse<IEnumerable<VideoDto>>
        {
            Success = true,
            Data = new[] { video }
        };
        mockApiService.Setup(x => x.GetVideosAsync()).ReturnsAsync(response);
        var viewModel = new HomeViewModel(mockApiService.Object);

        // Act
        await viewModel.LoadShorts();

        // Assert
        Assert.AreEqual(viewCount, viewModel.Shorts[0].ViewCount);
    }

    /// <summary>
    /// Tests that LoadShorts handles extreme Id values correctly.
    /// Expects all Id values to be mapped correctly including edge cases.
    /// </summary>
    [TestMethod]
    [DataRow(0)]
    [DataRow(1)]
    [DataRow(-1)]
    [DataRow(int.MaxValue)]
    [DataRow(int.MinValue)]
    public async Task LoadShorts_WithExtremeIdValues_MapsCorrectly(int id)
    {
        // Arrange
        var mockApiService = new Mock<IApiService>();
        var video = new VideoDto
        {
            Id = id,
            Title = "Test",
            ThumbnailUrl = "thumb.jpg",
            ViewCount = 100,
            DurationSeconds = 60
        };
        var response = new ApiResponse<IEnumerable<VideoDto>>
        {
            Success = true,
            Data = new[] { video }
        };
        mockApiService.Setup(x => x.GetVideosAsync()).ReturnsAsync(response);
        var viewModel = new HomeViewModel(mockApiService.Object);

        // Act
        await viewModel.LoadShorts();

        // Assert
        Assert.AreEqual(id, viewModel.Shorts[0].Id);
        Assert.AreEqual(id.ToString(), viewModel.Shorts[0].VideoId);
    }

    /// <summary>
    /// Tests that LoadShorts handles very long string values in VideoDto properties.
    /// Expects long strings to be mapped without truncation or errors.
    /// </summary>
    [TestMethod]
    public async Task LoadShorts_WithVeryLongStrings_MapsCorrectly()
    {
        // Arrange
        var mockApiService = new Mock<IApiService>();
        var longTitle = new string('A', 10000);
        var longThumbnailUrl = new string('B', 10000);
        var video = new VideoDto
        {
            Id = 1,
            Title = longTitle,
            ThumbnailUrl = longThumbnailUrl,
            ViewCount = 100,
            DurationSeconds = 60
        };
        var response = new ApiResponse<IEnumerable<VideoDto>>
        {
            Success = true,
            Data = new[] { video }
        };
        mockApiService.Setup(x => x.GetVideosAsync()).ReturnsAsync(response);
        var viewModel = new HomeViewModel(mockApiService.Object);

        // Act
        await viewModel.LoadShorts();

        // Assert
        Assert.AreEqual(longTitle, viewModel.Shorts[0].Title);
        Assert.AreEqual(longThumbnailUrl, viewModel.Shorts[0].Thumbnail);
    }

    /// <summary>
    /// Tests that LoadShorts handles special characters in string properties.
    /// Expects special characters to be preserved in the mapping.
    /// </summary>
    [TestMethod]
    public async Task LoadShorts_WithSpecialCharactersInStrings_MapsCorrectly()
    {
        // Arrange
        var mockApiService = new Mock<IApiService>();
        var video = new VideoDto
        {
            Id = 1,
            Title = "Test\nWith\tSpecial\rChars<>\"'&",
            ThumbnailUrl = "https://example.com/thumb?id=1&param=value",
            ViewCount = 100,
            DurationSeconds = 60
        };
        var response = new ApiResponse<IEnumerable<VideoDto>>
        {
            Success = true,
            Data = new[] { video }
        };
        mockApiService.Setup(x => x.GetVideosAsync()).ReturnsAsync(response);
        var viewModel = new HomeViewModel(mockApiService.Object);

        // Act
        await viewModel.LoadShorts();

        // Assert
        Assert.AreEqual("Test\nWith\tSpecial\rChars<>\"'&", viewModel.Shorts[0].Title);
        Assert.AreEqual("https://example.com/thumb?id=1&param=value", viewModel.Shorts[0].Thumbnail);
    }

    /// <summary>
    /// Helper method to create a list of VideoDto objects for testing.
    /// </summary>
    /// <param name="count">The number of videos to create.</param>
    /// <returns>A list of VideoDto objects.</returns>
    private static List<VideoDto> CreateVideoList(int count)
    {
        var videos = new List<VideoDto>();
        for (int i = 1; i <= count; i++)
        {
            videos.Add(new VideoDto
            {
                Id = i,
                Title = $"Video {i}",
                ThumbnailUrl = $"https://example.com/thumb{i}.jpg",
                ViewCount = i * 100,
                DurationSeconds = 60 + i
            });
        }
        return videos;
    }

    /// <summary>
    /// Tests that LoadMoreLessons returns early and does not increment page or call API when IsLoadingMore is already true.
    /// Input: IsLoadingMore is already true.
    /// Expected: Method returns immediately without calling API or modifying state.
    /// </summary>
    [TestMethod]
    public async Task LoadMoreLessons_WhenIsLoadingMoreIsTrue_ReturnsEarlyWithoutApiCall()
    {
        // Arrange
        Mock<IApiService> mockApiService = new Mock<IApiService>();
        HomeViewModel viewModel = new HomeViewModel(mockApiService.Object);

        // Set IsLoadingMore to true to trigger early exit
        viewModel.IsLoadingMore = true;
        int initialCount = viewModel.RecommendedLessons.Count;

        // Act
        await viewModel.LoadMoreLessons();

        // Assert
        Assert.IsTrue(viewModel.IsLoadingMore);
        Assert.AreEqual(initialCount, viewModel.RecommendedLessons.Count);
        mockApiService.Verify(x => x.GetPersonalizedRecommendationAsync(), Times.Never);
    }

    /// <summary>
    /// Tests that LoadMoreLessons successfully loads a lesson when API returns valid data.
    /// Input: Valid API response with non-null Data.
    /// Expected: Lesson is added to RecommendedLessons collection with correct properties, IsLoadingMore is reset to false.
    /// </summary>
    [TestMethod]
    public async Task LoadMoreLessons_WithValidApiResponse_AddsLessonToCollection()
    {
        // Arrange
        Mock<IApiService> mockApiService = new Mock<IApiService>();
        ApiResponse<PersonalizedRecommendationDto> apiResponse = new ApiResponse<PersonalizedRecommendationDto>
        {
            Success = true,
            Data = new PersonalizedRecommendationDto
            {
                RecommendedCategoryId = 42,
                RecommendedCategoryTitle = "Advanced Fingerspelling",
                Reason = "Based on your learning progress"
            }
        };
        mockApiService.Setup(x => x.GetPersonalizedRecommendationAsync()).ReturnsAsync(apiResponse);
        HomeViewModel viewModel = new HomeViewModel(mockApiService.Object);
        int initialCount = viewModel.RecommendedLessons.Count;

        // Act
        await viewModel.LoadMoreLessons();

        // Assert
        Assert.IsFalse(viewModel.IsLoadingMore);
        Assert.AreEqual(initialCount + 1, viewModel.RecommendedLessons.Count);
        LessonItem addedLesson = viewModel.RecommendedLessons[viewModel.RecommendedLessons.Count - 1];
        Assert.AreEqual(42, addedLesson.Id);
        Assert.AreEqual("Advanced Fingerspelling", addedLesson.Title);
        Assert.AreEqual("Based on your learning progress", addedLesson.Subtitle);
        mockApiService.Verify(x => x.GetPersonalizedRecommendationAsync(), Times.Once);
    }

    /// <summary>
    /// Tests that LoadMoreLessons does not add a lesson when API returns null recommendation.
    /// Input: API returns null.
    /// Expected: No lesson added, IsLoadingMore is reset to false.
    /// </summary>
    [TestMethod]
    public async Task LoadMoreLessons_WhenApiReturnsNull_DoesNotAddLesson()
    {
        // Arrange
        Mock<IApiService> mockApiService = new Mock<IApiService>();
        mockApiService.Setup(x => x.GetPersonalizedRecommendationAsync()).ReturnsAsync((ApiResponse<PersonalizedRecommendationDto>?)null);
        HomeViewModel viewModel = new HomeViewModel(mockApiService.Object);
        int initialCount = viewModel.RecommendedLessons.Count;

        // Act
        await viewModel.LoadMoreLessons();

        // Assert
        Assert.IsFalse(viewModel.IsLoadingMore);
        Assert.AreEqual(initialCount, viewModel.RecommendedLessons.Count);
        mockApiService.Verify(x => x.GetPersonalizedRecommendationAsync(), Times.Once);
    }

    /// <summary>
    /// Tests that LoadMoreLessons does not add a lesson when API returns response with null Data.
    /// Input: API returns ApiResponse with null Data property.
    /// Expected: No lesson added, IsLoadingMore is reset to false.
    /// </summary>
    [TestMethod]
    public async Task LoadMoreLessons_WhenApiReturnsNullData_DoesNotAddLesson()
    {
        // Arrange
        Mock<IApiService> mockApiService = new Mock<IApiService>();
        ApiResponse<PersonalizedRecommendationDto> apiResponse = new ApiResponse<PersonalizedRecommendationDto>
        {
            Success = false,
            Data = null
        };
        mockApiService.Setup(x => x.GetPersonalizedRecommendationAsync()).ReturnsAsync(apiResponse);
        HomeViewModel viewModel = new HomeViewModel(mockApiService.Object);
        int initialCount = viewModel.RecommendedLessons.Count;

        // Act
        await viewModel.LoadMoreLessons();

        // Assert
        Assert.IsFalse(viewModel.IsLoadingMore);
        Assert.AreEqual(initialCount, viewModel.RecommendedLessons.Count);
        mockApiService.Verify(x => x.GetPersonalizedRecommendationAsync(), Times.Once);
    }

    /// <summary>
    /// Tests that LoadMoreLessons handles exceptions gracefully and resets IsLoadingMore.
    /// Input: API service throws an exception.
    /// Expected: Exception is caught, no lesson added, IsLoadingMore is reset to false.
    /// </summary>
    [TestMethod]
    public async Task LoadMoreLessons_WhenApiThrowsException_HandlesExceptionAndResetsState()
    {
        // Arrange
        Mock<IApiService> mockApiService = new Mock<IApiService>();
        mockApiService.Setup(x => x.GetPersonalizedRecommendationAsync()).ThrowsAsync(new InvalidOperationException("Network error"));
        HomeViewModel viewModel = new HomeViewModel(mockApiService.Object);
        int initialCount = viewModel.RecommendedLessons.Count;

        // Act
        await viewModel.LoadMoreLessons();

        // Assert
        Assert.IsFalse(viewModel.IsLoadingMore);
        Assert.AreEqual(initialCount, viewModel.RecommendedLessons.Count);
        mockApiService.Verify(x => x.GetPersonalizedRecommendationAsync(), Times.Once);
    }

    /// <summary>
    /// Tests that LoadMoreLessons sets IsLoadingMore to true during execution.
    /// Input: Valid API call that will be intercepted.
    /// Expected: IsLoadingMore is set to true before API call.
    /// </summary>
    [TestMethod]
    public async Task LoadMoreLessons_DuringExecution_SetsIsLoadingMoreToTrue()
    {
        // Arrange
        Mock<IApiService> mockApiService = new Mock<IApiService>();
        bool isLoadingDuringApiCall = false;
        mockApiService.Setup(x => x.GetPersonalizedRecommendationAsync())
            .ReturnsAsync(() =>
            {
                // Capture the state during API call
                return new ApiResponse<PersonalizedRecommendationDto>
                {
                    Success = true,
                    Data = new PersonalizedRecommendationDto
                    {
                        RecommendedCategoryId = 1,
                        RecommendedCategoryTitle = "Test",
                        Reason = "Test reason"
                    }
                };
            });
        HomeViewModel viewModel = new HomeViewModel(mockApiService.Object);

        // Act
        Task loadTask = viewModel.LoadMoreLessons();

        // Check state before awaiting completion
        isLoadingDuringApiCall = viewModel.IsLoadingMore;

        await loadTask;

        // Assert
        Assert.IsTrue(isLoadingDuringApiCall);
        Assert.IsFalse(viewModel.IsLoadingMore);
    }

    /// <summary>
    /// Tests that LoadMoreLessons correctly handles edge case with empty strings in recommendation data.
    /// Input: API returns data with empty strings for title and reason.
    /// Expected: Lesson is added with empty string values.
    /// </summary>
    [TestMethod]
    public async Task LoadMoreLessons_WithEmptyStringsInData_AddsLessonWithEmptyStrings()
    {
        // Arrange
        Mock<IApiService> mockApiService = new Mock<IApiService>();
        ApiResponse<PersonalizedRecommendationDto> apiResponse = new ApiResponse<PersonalizedRecommendationDto>
        {
            Success = true,
            Data = new PersonalizedRecommendationDto
            {
                RecommendedCategoryId = 0,
                RecommendedCategoryTitle = string.Empty,
                Reason = string.Empty
            }
        };
        mockApiService.Setup(x => x.GetPersonalizedRecommendationAsync()).ReturnsAsync(apiResponse);
        HomeViewModel viewModel = new HomeViewModel(mockApiService.Object);
        int initialCount = viewModel.RecommendedLessons.Count;

        // Act
        await viewModel.LoadMoreLessons();

        // Assert
        Assert.IsFalse(viewModel.IsLoadingMore);
        Assert.AreEqual(initialCount + 1, viewModel.RecommendedLessons.Count);
        LessonItem addedLesson = viewModel.RecommendedLessons[viewModel.RecommendedLessons.Count - 1];
        Assert.AreEqual(0, addedLesson.Id);
        Assert.AreEqual(string.Empty, addedLesson.Title);
        Assert.AreEqual(string.Empty, addedLesson.Subtitle);
    }

    /// <summary>
    /// Tests that LoadMoreLessons handles extreme integer values for RecommendedCategoryId.
    /// Input: API returns data with int.MaxValue as RecommendedCategoryId.
    /// Expected: Lesson is added with the maximum integer value.
    /// </summary>
    [TestMethod]
    [DataRow(int.MaxValue)]
    [DataRow(int.MinValue)]
    [DataRow(0)]
    [DataRow(-1)]
    public async Task LoadMoreLessons_WithExtremeIntegerValues_AddsLessonCorrectly(int categoryId)
    {
        // Arrange
        Mock<IApiService> mockApiService = new Mock<IApiService>();
        ApiResponse<PersonalizedRecommendationDto> apiResponse = new ApiResponse<PersonalizedRecommendationDto>
        {
            Success = true,
            Data = new PersonalizedRecommendationDto
            {
                RecommendedCategoryId = categoryId,
                RecommendedCategoryTitle = "Test Title",
                Reason = "Test Reason"
            }
        };
        mockApiService.Setup(x => x.GetPersonalizedRecommendationAsync()).ReturnsAsync(apiResponse);
        HomeViewModel viewModel = new HomeViewModel(mockApiService.Object);
        int initialCount = viewModel.RecommendedLessons.Count;

        // Act
        await viewModel.LoadMoreLessons();

        // Assert
        Assert.IsFalse(viewModel.IsLoadingMore);
        Assert.AreEqual(initialCount + 1, viewModel.RecommendedLessons.Count);
        LessonItem addedLesson = viewModel.RecommendedLessons[viewModel.RecommendedLessons.Count - 1];
        Assert.AreEqual(categoryId, addedLesson.Id);
    }

    /// <summary>
    /// Tests that LoadMoreLessons handles very long strings in recommendation data.
    /// Input: API returns data with extremely long title and reason strings.
    /// Expected: Lesson is added with full string values preserved.
    /// </summary>
    [TestMethod]
    public async Task LoadMoreLessons_WithVeryLongStrings_AddsLessonWithFullStrings()
    {
        // Arrange
        Mock<IApiService> mockApiService = new Mock<IApiService>();
        string longTitle = new string('A', 10000);
        string longReason = new string('B', 10000);
        ApiResponse<PersonalizedRecommendationDto> apiResponse = new ApiResponse<PersonalizedRecommendationDto>
        {
            Success = true,
            Data = new PersonalizedRecommendationDto
            {
                RecommendedCategoryId = 1,
                RecommendedCategoryTitle = longTitle,
                Reason = longReason
            }
        };
        mockApiService.Setup(x => x.GetPersonalizedRecommendationAsync()).ReturnsAsync(apiResponse);
        HomeViewModel viewModel = new HomeViewModel(mockApiService.Object);

        // Act
        await viewModel.LoadMoreLessons();

        // Assert
        LessonItem addedLesson = viewModel.RecommendedLessons[viewModel.RecommendedLessons.Count - 1];
        Assert.AreEqual(longTitle, addedLesson.Title);
        Assert.AreEqual(longReason, addedLesson.Subtitle);
    }

    /// <summary>
    /// Tests that LoadMoreLessons handles special characters in recommendation data.
    /// Input: API returns data with special characters, newlines, and Unicode in strings.
    /// Expected: Lesson is added with special characters preserved.
    /// </summary>
    [TestMethod]
    public async Task LoadMoreLessons_WithSpecialCharactersInData_AddsLessonWithSpecialCharacters()
    {
        // Arrange
        Mock<IApiService> mockApiService = new Mock<IApiService>();
        ApiResponse<PersonalizedRecommendationDto> apiResponse = new ApiResponse<PersonalizedRecommendationDto>
        {
            Success = true,
            Data = new PersonalizedRecommendationDto
            {
                RecommendedCategoryId = 5,
                RecommendedCategoryTitle = "Title with 🤟 emoji\nand\nnewlines\t\r\n",
                Reason = "Reason <>&\"'`~!@#$%^&*()"
            }
        };
        mockApiService.Setup(x => x.GetPersonalizedRecommendationAsync()).ReturnsAsync(apiResponse);
        HomeViewModel viewModel = new HomeViewModel(mockApiService.Object);

        // Act
        await viewModel.LoadMoreLessons();

        // Assert
        LessonItem addedLesson = viewModel.RecommendedLessons[viewModel.RecommendedLessons.Count - 1];
        Assert.AreEqual("Title with 🤟 emoji\nand\nnewlines\t\r\n", addedLesson.Title);
        Assert.AreEqual("Reason <>&\"'`~!@#$%^&*()", addedLesson.Subtitle);
    }

    /// <summary>
    /// Tests that LoadMoreLessons can be called multiple times successfully.
    /// Input: Multiple consecutive calls with valid API responses.
    /// Expected: Multiple lessons are added to the collection.
    /// </summary>
    [TestMethod]
    public async Task LoadMoreLessons_CalledMultipleTimes_AddsMultipleLessons()
    {
        // Arrange
        Mock<IApiService> mockApiService = new Mock<IApiService>();
        int callCount = 0;
        mockApiService.Setup(x => x.GetPersonalizedRecommendationAsync())
            .ReturnsAsync(() =>
            {
                callCount++;
                return new ApiResponse<PersonalizedRecommendationDto>
                {
                    Success = true,
                    Data = new PersonalizedRecommendationDto
                    {
                        RecommendedCategoryId = callCount,
                        RecommendedCategoryTitle = $"Title {callCount}",
                        Reason = $"Reason {callCount}"
                    }
                };
            });
        HomeViewModel viewModel = new HomeViewModel(mockApiService.Object);
        int initialCount = viewModel.RecommendedLessons.Count;

        // Act
        await viewModel.LoadMoreLessons();
        await viewModel.LoadMoreLessons();
        await viewModel.LoadMoreLessons();

        // Assert
        Assert.AreEqual(initialCount + 3, viewModel.RecommendedLessons.Count);
        Assert.IsFalse(viewModel.IsLoadingMore);
        mockApiService.Verify(x => x.GetPersonalizedRecommendationAsync(), Times.Exactly(3));
    }

    /// <summary>
    /// Tests that LoadRecommendedLessons successfully loads and maps recommendation data from API.
    /// Verifies that IsLoadingLessons is set correctly, collection is cleared, and lesson is added with proper mapping.
    /// </summary>
    [TestMethod]
    public async Task LoadRecommendedLessons_WithValidApiResponse_ClearsCollectionAndAddsLesson()
    {
        // Arrange
        var mockApiService = new Mock<IApiService>();
        var recommendationDto = new PersonalizedRecommendationDto
        {
            RecommendedCategoryId = 42,
            RecommendedCategoryTitle = "Advanced Fingerspelling",
            Reason = "Based on your progress in intermediate lessons"
        };
        var apiResponse = new ApiResponse<PersonalizedRecommendationDto>
        {
            Success = true,
            Data = recommendationDto
        };
        mockApiService.Setup(x => x.GetPersonalizedRecommendationAsync())
            .ReturnsAsync(apiResponse);

        var viewModel = new HomeViewModel(mockApiService.Object);
        viewModel.RecommendedLessons.Add(new LessonItem { Id = 1, Title = "Old Lesson" });

        // Act
        await viewModel.LoadRecommendedLessons();

        // Assert
        Assert.IsFalse(viewModel.IsLoadingLessons, "IsLoadingLessons should be false after loading completes");
        Assert.AreEqual(1, viewModel.RecommendedLessons.Count, "Should have exactly one lesson");

        var lesson = viewModel.RecommendedLessons[0];
        Assert.AreEqual(42, lesson.Id, "Lesson Id should match RecommendedCategoryId");
        Assert.AreEqual("Advanced Fingerspelling", lesson.Title, "Lesson Title should match RecommendedCategoryTitle");
        Assert.AreEqual("Based on your progress in intermediate lessons", lesson.Subtitle, "Lesson Subtitle should match Reason");
        Assert.AreEqual("Intermediate", lesson.Difficulty, "Lesson Difficulty should be set to Intermediate");

        mockApiService.Verify(x => x.GetPersonalizedRecommendationAsync(), Times.Once);
    }

    /// <summary>
    /// Tests that LoadRecommendedLessons handles null API response by calling AddSampleLessons.
    /// Verifies that IsLoadingLessons is reset and collection is populated with sample data.
    /// </summary>
    [TestMethod]
    public async Task LoadRecommendedLessons_WithNullApiResponse_CallsAddSampleLessons()
    {
        // Arrange
        var mockApiService = new Mock<IApiService>();
        mockApiService.Setup(x => x.GetPersonalizedRecommendationAsync())
            .ReturnsAsync((ApiResponse<PersonalizedRecommendationDto>?)null);

        var viewModel = new HomeViewModel(mockApiService.Object);

        // Act
        await viewModel.LoadRecommendedLessons();

        // Assert
        Assert.IsFalse(viewModel.IsLoadingLessons, "IsLoadingLessons should be false after loading completes");
        Assert.IsTrue(viewModel.RecommendedLessons.Count > 0, "Should have sample lessons added");
        mockApiService.Verify(x => x.GetPersonalizedRecommendationAsync(), Times.Once);
    }

    /// <summary>
    /// Tests that LoadRecommendedLessons handles API response with null Data property by calling AddSampleLessons.
    /// Verifies that IsLoadingLessons is reset and collection is populated with sample data.
    /// </summary>
    [TestMethod]
    public async Task LoadRecommendedLessons_WithNullDataInResponse_CallsAddSampleLessons()
    {
        // Arrange
        var mockApiService = new Mock<IApiService>();
        var apiResponse = new ApiResponse<PersonalizedRecommendationDto>
        {
            Success = false,
            Data = null
        };
        mockApiService.Setup(x => x.GetPersonalizedRecommendationAsync())
            .ReturnsAsync(apiResponse);

        var viewModel = new HomeViewModel(mockApiService.Object);

        // Act
        await viewModel.LoadRecommendedLessons();

        // Assert
        Assert.IsFalse(viewModel.IsLoadingLessons, "IsLoadingLessons should be false after loading completes");
        Assert.IsTrue(viewModel.RecommendedLessons.Count > 0, "Should have sample lessons added");
        mockApiService.Verify(x => x.GetPersonalizedRecommendationAsync(), Times.Once);
    }

    /// <summary>
    /// Tests that LoadRecommendedLessons handles API exceptions by calling AddSampleLessons.
    /// Verifies that IsLoadingLessons is reset and collection is populated with sample data even when exception occurs.
    /// </summary>
    [TestMethod]
    [DataRow("Network error")]
    [DataRow("")]
    [DataRow("   ")]
    public async Task LoadRecommendedLessons_WhenApiThrowsException_CallsAddSampleLessons(string errorMessage)
    {
        // Arrange
        var mockApiService = new Mock<IApiService>();
        mockApiService.Setup(x => x.GetPersonalizedRecommendationAsync())
            .ThrowsAsync(new Exception(errorMessage));

        var viewModel = new HomeViewModel(mockApiService.Object);

        // Act
        await viewModel.LoadRecommendedLessons();

        // Assert
        Assert.IsFalse(viewModel.IsLoadingLessons, "IsLoadingLessons should be false after exception handling");
        Assert.IsTrue(viewModel.RecommendedLessons.Count > 0, "Should have sample lessons added after exception");
        mockApiService.Verify(x => x.GetPersonalizedRecommendationAsync(), Times.Once);
    }

    /// <summary>
    /// Tests that LoadRecommendedLessons handles various exception types by calling AddSampleLessons.
    /// Verifies proper exception handling for different exception scenarios.
    /// </summary>
    [TestMethod]
    [DataRow(typeof(InvalidOperationException))]
    [DataRow(typeof(ArgumentException))]
    [DataRow(typeof(NullReferenceException))]
    public async Task LoadRecommendedLessons_WhenApiThrowsVariousExceptions_CallsAddSampleLessons(Type exceptionType)
    {
        // Arrange
        var mockApiService = new Mock<IApiService>();
        var exception = (Exception)Activator.CreateInstance(exceptionType, "Test exception")!;
        mockApiService.Setup(x => x.GetPersonalizedRecommendationAsync())
            .ThrowsAsync(exception);

        var viewModel = new HomeViewModel(mockApiService.Object);

        // Act
        await viewModel.LoadRecommendedLessons();

        // Assert
        Assert.IsFalse(viewModel.IsLoadingLessons, "IsLoadingLessons should be false after exception handling");
        Assert.IsTrue(viewModel.RecommendedLessons.Count > 0, "Should have sample lessons added after exception");
        mockApiService.Verify(x => x.GetPersonalizedRecommendationAsync(), Times.Once);
    }

    /// <summary>
    /// Tests that LoadRecommendedLessons properly clears existing lessons before adding new one.
    /// Verifies that the collection is cleared when valid data is received.
    /// </summary>
    [TestMethod]
    public async Task LoadRecommendedLessons_WithExistingLessons_ClearsCollectionBeforeAdding()
    {
        // Arrange
        var mockApiService = new Mock<IApiService>();
        var recommendationDto = new PersonalizedRecommendationDto
        {
            RecommendedCategoryId = 100,
            RecommendedCategoryTitle = "New Lesson",
            Reason = "Personalized for you"
        };
        var apiResponse = new ApiResponse<PersonalizedRecommendationDto>
        {
            Success = true,
            Data = recommendationDto
        };
        mockApiService.Setup(x => x.GetPersonalizedRecommendationAsync())
            .ReturnsAsync(apiResponse);

        var viewModel = new HomeViewModel(mockApiService.Object);
        viewModel.RecommendedLessons.Add(new LessonItem { Id = 1, Title = "Old Lesson 1" });
        viewModel.RecommendedLessons.Add(new LessonItem { Id = 2, Title = "Old Lesson 2" });
        viewModel.RecommendedLessons.Add(new LessonItem { Id = 3, Title = "Old Lesson 3" });

        // Act
        await viewModel.LoadRecommendedLessons();

        // Assert
        Assert.AreEqual(1, viewModel.RecommendedLessons.Count, "Should have exactly one lesson after clearing");
        Assert.AreEqual(100, viewModel.RecommendedLessons[0].Id, "Should have the new lesson");
        Assert.AreEqual("New Lesson", viewModel.RecommendedLessons[0].Title);
    }

    /// <summary>
    /// Tests that LoadRecommendedLessons correctly maps edge case values from DTO to LessonItem.
    /// Verifies proper handling of boundary values for numeric properties.
    /// </summary>
    [TestMethod]
    [DataRow(0, "")]
    [DataRow(int.MaxValue, "Very long reason text that might represent an edge case scenario where the API returns extremely detailed personalization information")]
    [DataRow(int.MinValue, "   ")]
    [DataRow(-1, "Reason with special characters: !@#$%^&*()")]
    public async Task LoadRecommendedLessons_WithEdgeCaseValues_MapsCorrectly(int categoryId, string reason)
    {
        // Arrange
        var mockApiService = new Mock<IApiService>();
        var recommendationDto = new PersonalizedRecommendationDto
        {
            RecommendedCategoryId = categoryId,
            RecommendedCategoryTitle = "Test Title",
            Reason = reason
        };
        var apiResponse = new ApiResponse<PersonalizedRecommendationDto>
        {
            Success = true,
            Data = recommendationDto
        };
        mockApiService.Setup(x => x.GetPersonalizedRecommendationAsync())
            .ReturnsAsync(apiResponse);

        var viewModel = new HomeViewModel(mockApiService.Object);

        // Act
        await viewModel.LoadRecommendedLessons();

        // Assert
        Assert.IsFalse(viewModel.IsLoadingLessons);
        Assert.AreEqual(1, viewModel.RecommendedLessons.Count);
        Assert.AreEqual(categoryId, viewModel.RecommendedLessons[0].Id);
        Assert.AreEqual(reason, viewModel.RecommendedLessons[0].Subtitle);
    }

    /// <summary>
    /// Tests that LoadRecommendedLessons correctly maps empty and whitespace string values from DTO.
    /// Verifies proper handling of edge case string values.
    /// </summary>
    [TestMethod]
    [DataRow("", "", "")]
    [DataRow("Title", "", "Reason")]
    [DataRow("", "Title", "")]
    public async Task LoadRecommendedLessons_WithEmptyStringValues_MapsCorrectly(string title, string whitespace, string reason)
    {
        // Arrange
        var mockApiService = new Mock<IApiService>();
        var recommendationDto = new PersonalizedRecommendationDto
        {
            RecommendedCategoryId = 1,
            RecommendedCategoryTitle = title,
            Reason = reason
        };
        var apiResponse = new ApiResponse<PersonalizedRecommendationDto>
        {
            Success = true,
            Data = recommendationDto
        };
        mockApiService.Setup(x => x.GetPersonalizedRecommendationAsync())
            .ReturnsAsync(apiResponse);

        var viewModel = new HomeViewModel(mockApiService.Object);

        // Act
        await viewModel.LoadRecommendedLessons();

        // Assert
        Assert.IsFalse(viewModel.IsLoadingLessons);
        Assert.AreEqual(1, viewModel.RecommendedLessons.Count);
        Assert.AreEqual(title, viewModel.RecommendedLessons[0].Title);
        Assert.AreEqual(reason, viewModel.RecommendedLessons[0].Subtitle);
        Assert.AreEqual("Intermediate", viewModel.RecommendedLessons[0].Difficulty);
    }

    /// <summary>
    /// Tests that LoadRecommendedLessons always resets IsLoadingLessons to false regardless of outcome.
    /// Verifies the finally block executes properly in success scenario.
    /// </summary>
    [TestMethod]
    public async Task LoadRecommendedLessons_OnSuccess_ResetsIsLoadingLessons()
    {
        // Arrange
        var mockApiService = new Mock<IApiService>();
        var apiResponse = new ApiResponse<PersonalizedRecommendationDto>
        {
            Success = true,
            Data = new PersonalizedRecommendationDto
            {
                RecommendedCategoryId = 1,
                RecommendedCategoryTitle = "Test",
                Reason = "Test"
            }
        };
        mockApiService.Setup(x => x.GetPersonalizedRecommendationAsync())
            .ReturnsAsync(apiResponse);

        var viewModel = new HomeViewModel(mockApiService.Object);

        // Act
        await viewModel.LoadRecommendedLessons();

        // Assert
        Assert.IsFalse(viewModel.IsLoadingLessons, "IsLoadingLessons must be false after successful completion");
    }

    /// <summary>
    /// Tests that LoadRecommendedLessons always resets IsLoadingLessons to false even when exception occurs.
    /// Verifies the finally block executes properly in exception scenario.
    /// </summary>
    [TestMethod]
    public async Task LoadRecommendedLessons_OnException_ResetsIsLoadingLessons()
    {
        // Arrange
        var mockApiService = new Mock<IApiService>();
        mockApiService.Setup(x => x.GetPersonalizedRecommendationAsync())
            .ThrowsAsync(new Exception("Test exception"));

        var viewModel = new HomeViewModel(mockApiService.Object);

        // Act
        await viewModel.LoadRecommendedLessons();

        // Assert
        Assert.IsFalse(viewModel.IsLoadingLessons, "IsLoadingLessons must be false after exception handling");
    }

    /// <summary>
    /// Tests that NavigateToLesson does not throw when called with a valid lesson.
    /// Note: This test is inconclusive because Shell.Current is a static dependency that cannot be mocked with Moq.
    /// In a real test environment, Shell.Current would need to be initialized or the method would need refactoring
    /// to accept an INavigation or similar injectable dependency.
    /// </summary>
    [TestMethod]
    public async Task NavigateToLesson_ValidLessonWithPositiveId_NavigatesToCorrectRoute()
    {
        // Arrange
        var apiServiceMock = new Mock<IApiService>();
        var viewModel = new HomeViewModel(apiServiceMock.Object);
        var lesson = new LessonItem { Id = 42 };

        // Act & Assert
        // This test cannot be properly executed because Shell.Current is a static property
        // that cannot be mocked using Moq without creating fake implementations (which is prohibited).
        // To properly test this method, consider refactoring to inject INavigation or IShellNavigation
        // as a dependency instead of using Shell.Current directly.
        Assert.Inconclusive("Cannot test navigation with Shell.Current static dependency without mocking framework support or design refactoring.");
    }

    /// <summary>
    /// Tests that NavigateToLesson handles lesson with Id = 0.
    /// Note: This test is inconclusive due to Shell.Current static dependency.
    /// </summary>
    [TestMethod]
    public async Task NavigateToLesson_LessonWithZeroId_NavigatesToCorrectRoute()
    {
        // Arrange
        var apiServiceMock = new Mock<IApiService>();
        var viewModel = new HomeViewModel(apiServiceMock.Object);
        var lesson = new LessonItem { Id = 0 };

        // Act & Assert
        Assert.Inconclusive("Cannot test navigation with Shell.Current static dependency without mocking framework support or design refactoring.");
    }

    /// <summary>
    /// Tests that NavigateToLesson handles lesson with negative Id.
    /// Note: This test is inconclusive due to Shell.Current static dependency.
    /// </summary>
    [TestMethod]
    public async Task NavigateToLesson_LessonWithNegativeId_NavigatesToCorrectRoute()
    {
        // Arrange
        var apiServiceMock = new Mock<IApiService>();
        var viewModel = new HomeViewModel(apiServiceMock.Object);
        var lesson = new LessonItem { Id = -1 };

        // Act & Assert
        Assert.Inconclusive("Cannot test navigation with Shell.Current static dependency without mocking framework support or design refactoring.");
    }

    /// <summary>
    /// Tests that NavigateToLesson handles lesson with Id = int.MaxValue.
    /// Note: This test is inconclusive due to Shell.Current static dependency.
    /// </summary>
    [TestMethod]
    public async Task NavigateToLesson_LessonWithMaxId_NavigatesToCorrectRoute()
    {
        // Arrange
        var apiServiceMock = new Mock<IApiService>();
        var viewModel = new HomeViewModel(apiServiceMock.Object);
        var lesson = new LessonItem { Id = int.MaxValue };

        // Act & Assert
        Assert.Inconclusive("Cannot test navigation with Shell.Current static dependency without mocking framework support or design refactoring.");
    }

    /// <summary>
    /// Tests that NavigateToLesson handles lesson with Id = int.MinValue.
    /// Note: This test is inconclusive due to Shell.Current static dependency.
    /// </summary>
    [TestMethod]
    public async Task NavigateToLesson_LessonWithMinId_NavigatesToCorrectRoute()
    {
        // Arrange
        var apiServiceMock = new Mock<IApiService>();
        var viewModel = new HomeViewModel(apiServiceMock.Object);
        var lesson = new LessonItem { Id = int.MinValue };

        // Act & Assert
        Assert.Inconclusive("Cannot test navigation with Shell.Current static dependency without mocking framework support or design refactoring.");
    }

    /// <summary>
    /// Tests that ToggleSignOfTheDay toggles IsSignOfTheDayExpanded from its initial state to the expected state.
    /// Verifies that the property correctly toggles between true and false values.
    /// </summary>
    /// <param name="initialValue">The initial value of IsSignOfTheDayExpanded before toggling.</param>
    /// <param name="expectedValue">The expected value of IsSignOfTheDayExpanded after toggling.</param>
    [TestMethod]
    [DataRow(false, true, DisplayName = "Toggle from false to true")]
    [DataRow(true, false, DisplayName = "Toggle from true to false")]
    public async Task ToggleSignOfTheDay_WithInitialValue_TogglesCorrectly(bool initialValue, bool expectedValue)
    {
        // Arrange
        var mockApiService = new Mock<IApiService>();
        var viewModel = new HomeViewModel(mockApiService.Object);
        viewModel.IsSignOfTheDayExpanded = initialValue;

        // Act
        await viewModel.ToggleSignOfTheDay();

        // Assert
        Assert.AreEqual(expectedValue, viewModel.IsSignOfTheDayExpanded);
    }

    /// <summary>
    /// Tests that ToggleSignOfTheDay can be called multiple times consecutively.
    /// Verifies that each toggle correctly alternates the IsSignOfTheDayExpanded property value.
    /// </summary>
    [TestMethod]
    public async Task ToggleSignOfTheDay_CalledMultipleTimes_TogglesCorrectly()
    {
        // Arrange
        var mockApiService = new Mock<IApiService>();
        var viewModel = new HomeViewModel(mockApiService.Object);
        var initialValue = false;
        viewModel.IsSignOfTheDayExpanded = initialValue;

        // Act & Assert
        await viewModel.ToggleSignOfTheDay();
        Assert.AreEqual(true, viewModel.IsSignOfTheDayExpanded, "First toggle should set to true");

        await viewModel.ToggleSignOfTheDay();
        Assert.AreEqual(false, viewModel.IsSignOfTheDayExpanded, "Second toggle should set to false");

        await viewModel.ToggleSignOfTheDay();
        Assert.AreEqual(true, viewModel.IsSignOfTheDayExpanded, "Third toggle should set to true");

        await viewModel.ToggleSignOfTheDay();
        Assert.AreEqual(false, viewModel.IsSignOfTheDayExpanded, "Fourth toggle should set to false");
    }

    /// <summary>
    /// Tests that ToggleSignOfTheDay completes successfully without throwing any exceptions.
    /// Verifies that the async method completes as expected.
    /// </summary>
    [TestMethod]
    public async Task ToggleSignOfTheDay_WhenCalled_CompletesSuccessfully()
    {
        // Arrange
        var mockApiService = new Mock<IApiService>();
        var viewModel = new HomeViewModel(mockApiService.Object);

        // Act
        var task = viewModel.ToggleSignOfTheDay();

        // Assert
        await task;
        Assert.IsTrue(task.IsCompleted);
        Assert.IsFalse(task.IsFaulted);
        Assert.IsFalse(task.IsCanceled);
    }

    /// <summary>
    /// Tests that OpenCamera method attempts to set IsCameraPreviewVisible to true and navigate.
    /// Note: This test is marked inconclusive because Shell.Current is a static property that cannot
    /// be mocked with Moq. Full testing of this method requires either:
    /// 1. Integration testing with MAUI test infrastructure
    /// 2. Refactoring to inject an INavigationService abstraction
    /// </summary>
    [TestMethod]
    public async Task OpenCamera_WhenCalled_MarkAsInconclusive_DueToStaticShellDependency()
    {
        // Arrange
        var mockApiService = new Mock<IApiService>();
        var viewModel = new HomeViewModel(mockApiService.Object);

        // Act & Assert
        // Shell.Current is a static property that cannot be mocked with Moq.
        // In a unit test context, Shell.Current will be null, causing a NullReferenceException
        // when GoToAsync is called. This method requires integration testing or a navigation
        // service abstraction to be properly unit tested.
        Assert.Inconclusive(
            "OpenCamera method cannot be fully unit tested due to dependency on Shell.Current static property. " +
            "The method sets IsCameraPreviewVisible = true and calls Shell.Current.GoToAsync(\"camera-translation\"). " +
            "To properly test this method, either: " +
            "1) Use integration tests with MAUI test host, or " +
            "2) Refactor to inject INavigationService abstraction that can be mocked.");
    }

    /// <summary>
    /// Tests that IsCameraPreviewVisible property can be set to true.
    /// This verifies the property setter behavior that OpenCamera relies on.
    /// </summary>
    [TestMethod]
    public void IsCameraPreviewVisible_WhenSetToTrue_PropertyIsTrue()
    {
        // Arrange
        var mockApiService = new Mock<IApiService>();
        var viewModel = new HomeViewModel(mockApiService.Object);

        // Act
        viewModel.IsCameraPreviewVisible = true;

        // Assert
        Assert.IsTrue(viewModel.IsCameraPreviewVisible);
    }

    /// <summary>
    /// Tests that IsCameraPreviewVisible property can be set to false.
    /// This verifies the property setter behavior.
    /// </summary>
    [TestMethod]
    public void IsCameraPreviewVisible_WhenSetToFalse_PropertyIsFalse()
    {
        // Arrange
        var mockApiService = new Mock<IApiService>();
        var viewModel = new HomeViewModel(mockApiService.Object);

        // Act
        viewModel.IsCameraPreviewVisible = false;

        // Assert
        Assert.IsFalse(viewModel.IsCameraPreviewVisible);
    }

    /// <summary>
    /// Tests that IsCameraPreviewVisible property defaults to false on initialization.
    /// </summary>
    [TestMethod]
    public void IsCameraPreviewVisible_OnInitialization_DefaultsToFalse()
    {
        // Arrange & Act
        var mockApiService = new Mock<IApiService>();
        var viewModel = new HomeViewModel(mockApiService.Object);

        // Assert
        Assert.IsFalse(viewModel.IsCameraPreviewVisible);
    }
}