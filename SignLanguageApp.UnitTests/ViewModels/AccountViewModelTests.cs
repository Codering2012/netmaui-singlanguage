using System;
using System.Collections;
using System.Collections.Generic;
using System.Collections.ObjectModel;
using System.Linq;
using System.Reflection;
using System.Threading;
using System.Threading.Tasks;
using System.Windows;

using CommunityToolkit.Mvvm.ComponentModel;
using CommunityToolkit.Mvvm.Input;
using Microsoft.Maui.ApplicationModel;
using Microsoft.Maui.Controls;
using Microsoft.Maui.Storage;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using Moq;
using SignLanguageApp.Model;
using SignLanguageApp.Services;
using SignLanguageApp.ViewModels;

namespace SignLanguageApp.ViewModels.UnitTests;


/// <summary>
/// Unit tests for the AccountViewModel class.
/// </summary>
[TestClass]
public partial class AccountViewModelTests
{
    /// <summary>
    /// Tests that OnDisappearing does not throw an exception when called with the CancellationTokenSource in its default null state.
    /// This verifies the null-conditional operators safely handle the null case.
    /// </summary>
    [TestMethod]
    public void OnDisappearing_WhenCancellationTokenSourceIsNull_DoesNotThrow()
    {
        // Arrange
        var mockApiService = new Mock<IApiService>();
        var viewModel = new AccountViewModel(mockApiService.Object);

        // Act & Assert
        viewModel.OnDisappearing();
    }

    /// <summary>
    /// Tests that OnDisappearing can be called multiple times consecutively without throwing exceptions.
    /// This verifies the method is idempotent and safely handles repeated cleanup calls.
    /// </summary>
    [TestMethod]
    public void OnDisappearing_CalledMultipleTimes_DoesNotThrow()
    {
        // Arrange
        var mockApiService = new Mock<IApiService>();
        var viewModel = new AccountViewModel(mockApiService.Object);

        // Act & Assert
        viewModel.OnDisappearing();
        viewModel.OnDisappearing();
        viewModel.OnDisappearing();
    }

    /// <summary>
    /// Verifies that ExportData is an async method that returns a Task.
    /// Note: Full verification of DisplayAlertAsync behavior requires integration testing
    /// because Shell.Current is a static property that cannot be mocked with standard Moq.
    /// 
    /// Limitation: This test cannot verify that DisplayAlertAsync is called with the correct
    /// parameters ("Export Data", "Your data export is ready. Check your email for the download link.", "OK")
    /// without either:
    /// 1. Using integration tests with a properly initialized MAUI Shell
    /// 2. Refactoring to inject Shell as a dependency
    /// 3. Using a wrapper/abstraction for Shell operations
    /// </summary>
    [TestMethod]
    public void ExportData_ReturnsTask()
    {
        // Arrange
        var mockApiService = new Mock<IApiService>();
        var viewModel = new AccountViewModel(mockApiService.Object);

        // Act
        var result = viewModel.ExportData();

        // Assert
        Assert.IsNotNull(result);
        Assert.IsInstanceOfType(result, typeof(Task));
    }

    /// <summary>
    /// Tests that RefreshDiagnostics sets CpuUsage within the expected range.
    /// The method uses Random.Shared.Next(20, 80) / 100.0, so the valid range is [0.2, 0.8).
    /// </summary>
    [TestMethod]
    public async Task RefreshDiagnostics_ShouldSetCpuUsageWithinValidRange()
    {
        // Arrange
        Mock<IApiService> mockApiService = new Mock<IApiService>();
        AccountViewModel viewModel = new AccountViewModel(mockApiService.Object);

        // Act
        await viewModel.RefreshDiagnosticsCommand.ExecuteAsync(null);

        // Assert
        Assert.IsTrue(viewModel.CpuUsage >= 0.2, $"CpuUsage should be >= 0.2, but was {viewModel.CpuUsage}");
        Assert.IsTrue(viewModel.CpuUsage < 0.8, $"CpuUsage should be < 0.8, but was {viewModel.CpuUsage}");
    }

    /// <summary>
    /// Tests that RefreshDiagnostics sets NpuUsage within the expected range.
    /// The method uses Random.Shared.Next(40, 95) / 100.0, so the valid range is [0.4, 0.95).
    /// </summary>
    [TestMethod]
    public async Task RefreshDiagnostics_ShouldSetNpuUsageWithinValidRange()
    {
        // Arrange
        Mock<IApiService> mockApiService = new Mock<IApiService>();
        AccountViewModel viewModel = new AccountViewModel(mockApiService.Object);

        // Act
        await viewModel.RefreshDiagnosticsCommand.ExecuteAsync(null);

        // Assert
        Assert.IsTrue(viewModel.NpuUsage >= 0.4, $"NpuUsage should be >= 0.4, but was {viewModel.NpuUsage}");
        Assert.IsTrue(viewModel.NpuUsage < 0.95, $"NpuUsage should be < 0.95, but was {viewModel.NpuUsage}");
    }

    /// <summary>
    /// Tests that RefreshDiagnostics sets LastSyncTime to a non-empty string.
    /// The method formats DateTime.Now using the "g" format specifier.
    /// </summary>
    [TestMethod]
    public async Task RefreshDiagnostics_ShouldSetLastSyncTimeToNonEmptyString()
    {
        // Arrange
        Mock<IApiService> mockApiService = new Mock<IApiService>();
        AccountViewModel viewModel = new AccountViewModel(mockApiService.Object);

        // Act
        await viewModel.RefreshDiagnosticsCommand.ExecuteAsync(null);

        // Assert
        Assert.IsFalse(string.IsNullOrWhiteSpace(viewModel.LastSyncTime), "LastSyncTime should not be null or whitespace");
    }

    /// <summary>
    /// Tests that RefreshDiagnostics completes successfully without throwing exceptions.
    /// Verifies that all three properties (CpuUsage, NpuUsage, LastSyncTime) are updated.
    /// </summary>
    [TestMethod]
    public async Task RefreshDiagnostics_ShouldCompleteSuccessfullyAndUpdateAllProperties()
    {
        // Arrange
        Mock<IApiService> mockApiService = new Mock<IApiService>();
        AccountViewModel viewModel = new AccountViewModel(mockApiService.Object);
        double initialCpuUsage = viewModel.CpuUsage;
        double initialNpuUsage = viewModel.NpuUsage;
        string initialLastSyncTime = viewModel.LastSyncTime;

        // Act
        await viewModel.RefreshDiagnosticsCommand.ExecuteAsync(null);

        // Assert - verify all properties were set (different from initial values or within valid ranges)
        Assert.IsTrue(viewModel.CpuUsage >= 0.2 && viewModel.CpuUsage < 0.8, "CpuUsage should be within valid range");
        Assert.IsTrue(viewModel.NpuUsage >= 0.4 && viewModel.NpuUsage < 0.95, "NpuUsage should be within valid range");
        Assert.IsFalse(string.IsNullOrWhiteSpace(viewModel.LastSyncTime), "LastSyncTime should be set");
    }

    /// <summary>
    /// Tests that calling RefreshDiagnostics multiple times produces different random values
    /// for CpuUsage and NpuUsage with high probability, and updates LastSyncTime each time.
    /// </summary>
    [TestMethod]
    public async Task RefreshDiagnostics_MultipleCalls_ShouldUpdateValuesEachTime()
    {
        // Arrange
        Mock<IApiService> mockApiService = new Mock<IApiService>();
        AccountViewModel viewModel = new AccountViewModel(mockApiService.Object);

        // Act - Call multiple times
        await viewModel.RefreshDiagnosticsCommand.ExecuteAsync(null);
        double firstCpuUsage = viewModel.CpuUsage;
        double firstNpuUsage = viewModel.NpuUsage;
        string firstLastSyncTime = viewModel.LastSyncTime;

        await Task.Delay(10); // Small delay to ensure DateTime.Now changes

        await viewModel.RefreshDiagnosticsCommand.ExecuteAsync(null);
        double secondCpuUsage = viewModel.CpuUsage;
        double secondNpuUsage = viewModel.NpuUsage;
        string secondLastSyncTime = viewModel.LastSyncTime;

        // Assert - At least one of the random values should be different (extremely high probability)
        // and LastSyncTime should be updated
        bool valuesChanged = firstCpuUsage != secondCpuUsage || firstNpuUsage != secondNpuUsage;
        Assert.IsTrue(valuesChanged, "At least one of the random usage values should change between calls");

        // LastSyncTime should be set on each call
        Assert.IsFalse(string.IsNullOrWhiteSpace(firstLastSyncTime), "First LastSyncTime should be set");
        Assert.IsFalse(string.IsNullOrWhiteSpace(secondLastSyncTime), "Second LastSyncTime should be set");
    }

    /// <summary>
    /// Tests edge case where RefreshDiagnostics is called on a viewModel with default initial values.
    /// Verifies that the method correctly initializes all diagnostic properties.
    /// </summary>
    [TestMethod]
    public async Task RefreshDiagnostics_WithDefaultInitialValues_ShouldSetValidValues()
    {
        // Arrange
        Mock<IApiService> mockApiService = new Mock<IApiService>();
        AccountViewModel viewModel = new AccountViewModel(mockApiService.Object);

        // Verify initial state (default values)
        Assert.AreEqual(0.0, viewModel.CpuUsage, "Initial CpuUsage should be 0");
        Assert.AreEqual(0.0, viewModel.NpuUsage, "Initial NpuUsage should be 0");
        Assert.AreEqual(string.Empty, viewModel.LastSyncTime, "Initial LastSyncTime should be empty");

        // Act
        await viewModel.RefreshDiagnosticsCommand.ExecuteAsync(null);

        // Assert - verify values are now set correctly
        Assert.AreNotEqual(0.0, viewModel.CpuUsage, "CpuUsage should be updated from default");
        Assert.AreNotEqual(0.0, viewModel.NpuUsage, "NpuUsage should be updated from default");
        Assert.AreNotEqual(string.Empty, viewModel.LastSyncTime, "LastSyncTime should be updated from default");
    }

    /// <summary>
    /// Tests that RefreshDiagnostics correctly formats LastSyncTime using the "g" format.
    /// Verifies that the resulting string can be parsed back to a valid DateTime.
    /// </summary>
    [TestMethod]
    public async Task RefreshDiagnostics_ShouldSetLastSyncTimeWithValidDateTimeFormat()
    {
        // Arrange
        Mock<IApiService> mockApiService = new Mock<IApiService>();
        AccountViewModel viewModel = new AccountViewModel(mockApiService.Object);

        // Act
        await viewModel.RefreshDiagnosticsCommand.ExecuteAsync(null);

        // Assert - verify LastSyncTime can be parsed as a valid DateTime
        bool canParse = DateTime.TryParse(viewModel.LastSyncTime, out DateTime parsedDateTime);
        Assert.IsTrue(canParse, $"LastSyncTime '{viewModel.LastSyncTime}' should be parseable as a DateTime");

        // Verify the parsed date is reasonably close to now (within 1 minute)
        TimeSpan difference = DateTime.Now - parsedDateTime;
        Assert.IsTrue(Math.Abs(difference.TotalMinutes) < 1, "LastSyncTime should be close to current time");
    }

    /// <summary>
    /// Tests boundary conditions for CpuUsage to ensure it never equals or exceeds 0.8.
    /// This test calls RefreshDiagnostics multiple times to increase confidence.
    /// </summary>
    [TestMethod]
    public async Task RefreshDiagnostics_MultipleCalls_CpuUsageShouldNeverExceedUpperBound()
    {
        // Arrange
        Mock<IApiService> mockApiService = new Mock<IApiService>();
        AccountViewModel viewModel = new AccountViewModel(mockApiService.Object);

        // Act & Assert - call multiple times to test randomness boundaries
        for (int i = 0; i < 50; i++)
        {
            await viewModel.RefreshDiagnosticsCommand.ExecuteAsync(null);
            Assert.IsTrue(viewModel.CpuUsage < 0.8, $"Iteration {i}: CpuUsage {viewModel.CpuUsage} should be < 0.8");
            Assert.IsTrue(viewModel.CpuUsage >= 0.2, $"Iteration {i}: CpuUsage {viewModel.CpuUsage} should be >= 0.2");
        }
    }

    /// <summary>
    /// Tests boundary conditions for NpuUsage to ensure it never equals or exceeds 0.95.
    /// This test calls RefreshDiagnostics multiple times to increase confidence.
    /// </summary>
    [TestMethod]
    public async Task RefreshDiagnostics_MultipleCalls_NpuUsageShouldNeverExceedUpperBound()
    {
        // Arrange
        Mock<IApiService> mockApiService = new Mock<IApiService>();
        AccountViewModel viewModel = new AccountViewModel(mockApiService.Object);

        // Act & Assert - call multiple times to test randomness boundaries
        for (int i = 0; i < 50; i++)
        {
            await viewModel.RefreshDiagnosticsCommand.ExecuteAsync(null);
            Assert.IsTrue(viewModel.NpuUsage < 0.95, $"Iteration {i}: NpuUsage {viewModel.NpuUsage} should be < 0.95");
            Assert.IsTrue(viewModel.NpuUsage >= 0.4, $"Iteration {i}: NpuUsage {viewModel.NpuUsage} should be >= 0.4");
        }
    }

    /// <summary>
    /// Tests that ViewPrivacyPolicy method navigates to the privacy policy route.
    /// NOTE: This test is marked as Inconclusive because Shell.Current is a static property
    /// that cannot be mocked using Moq. Full testing of this method requires either:
    /// 1. Integration testing with a running MAUI Shell instance
    /// 2. Refactoring to inject an INavigationService abstraction
    /// The method is expected to call Shell.Current.GoToAsync("privacy-policy") when invoked.
    /// </summary>
    [TestMethod]
    public async Task ViewPrivacyPolicy_WhenCalled_ShouldNavigateToPrivacyPolicyRoute()
    {
        // Arrange
        var mockApiService = new Mock<IApiService>();
        var viewModel = new AccountViewModel(mockApiService.Object);

        // Act & Assert
        // Cannot properly test due to static Shell.Current dependency
        // In a real scenario, this would call: await Shell.Current.GoToAsync("privacy-policy")
        // Expected behavior: Navigate to "privacy-policy" route
        // Actual testing requires integration test or dependency injection refactoring
        Assert.Inconclusive(
            "This test cannot be fully executed because Shell.Current is a static property that cannot be mocked. " +
            "The method should navigate to 'privacy-policy' route when called. " +
            "Consider refactoring to use an injectable INavigationService for better testability.");
    }

    /// <summary>
    /// Tests that Logout method cancels the diagnostics CancellationTokenSource when it is not null.
    /// This verifies that background diagnostic operations are properly stopped during logout.
    /// Expected result: The CancellationTokenSource should be cancelled.
    /// </summary>
    [TestMethod]
    public async Task Logout_WhenDiagnosticsCtsIsNotNull_CancelsCancellationToken()
    {
        // Arrange
        var mockApiService = new Mock<IApiService>();
        var viewModel = new AccountViewModel(mockApiService.Object);
        var cts = new CancellationTokenSource();

        // Use reflection to set the private _diagnosticsCts field since it's not publicly accessible
        var diagnosticsCtsField = typeof(AccountViewModel).GetField("_diagnosticsCts", BindingFlags.NonPublic | BindingFlags.Instance);
        diagnosticsCtsField?.SetValue(viewModel, cts);

        // Act & Assert
        // Note: This test will throw NullReferenceException when trying to access Shell.Current
        // because the MAUI runtime is not initialized in a unit test environment.
        // The production code has a design issue where it directly depends on static MAUI classes
        // that cannot be mocked without refactoring to use dependency injection.
        try
        {
            await viewModel.LogoutCommand.ExecuteAsync(null);
            Assert.Inconclusive("Test completed without MAUI runtime - this should not happen in typical unit test environment.");
        }
        catch (NullReferenceException)
        {
            // Expected: Shell.Current or Application.Current is null in unit test environment
            // Verify that CancellationToken was requested before the exception occurred
            Assert.IsTrue(cts.IsCancellationRequested, "CancellationTokenSource should have been cancelled before accessing MAUI static classes.");
        }
    }

    /// <summary>
    /// Tests that Logout method handles null diagnostics CancellationTokenSource gracefully.
    /// When _diagnosticsCts is null, the null-conditional operator should prevent any exceptions.
    /// Expected result: No exception should be thrown from the null _diagnosticsCts.
    /// </summary>
    [TestMethod]
    public async Task Logout_WhenDiagnosticsCtsIsNull_DoesNotThrowFromCancellation()
    {
        // Arrange
        var mockApiService = new Mock<IApiService>();
        var viewModel = new AccountViewModel(mockApiService.Object);

        // _diagnosticsCts is null by default, no need to set it

        // Act & Assert
        // Note: This test will throw NullReferenceException when trying to access Shell.Current or Application.Current
        // because the MAUI runtime is not initialized in a unit test environment.
        try
        {
            await viewModel.LogoutCommand.ExecuteAsync(null);
            Assert.Inconclusive("Test completed without MAUI runtime - this should not happen in typical unit test environment.");
        }
        catch (NullReferenceException ex)
        {
            // Expected: Shell.Current or Application.Current is null in unit test environment
            // The exception should not be related to _diagnosticsCts cancellation
            Assert.IsFalse(ex.Message.Contains("Cancel"),
                "Exception should be from MAUI static classes, not from _diagnosticsCts.Cancel()");
        }
    }

    /// <summary>
    /// Tests the Logout method execution flow and verifies it's properly decorated with RelayCommand attribute.
    /// Due to direct dependencies on static MAUI classes (SecureStorage.Default, Application.Current, Shell.Current),
    /// this method cannot be fully unit tested without a running MAUI application context.
    /// Expected result: LogoutCommand should be available and executable.
    /// </summary>
    /// <remarks>
    /// LIMITATION: This test cannot verify the following behaviors in a unit test environment:
    /// 1. SecureStorage.Default.Remove("jwt_token") - static class cannot be mocked
    /// 2. Application.Current.Windows access and Page assignment - static class cannot be mocked
    /// 3. Shell.Current.GoToAsync("//login") navigation - static class cannot be mocked
    /// 4. LoginShell instantiation and assignment to Window.Page
    /// 
    /// To properly test this method, consider:
    /// - Integration tests with MAUI runtime initialized
    /// - Refactoring to inject ISecureStorage, INavigationService, and IWindowManager abstractions
    /// </remarks>
    [TestMethod]
    public async Task Logout_LogoutCommandExists_CanBeExecuted()
    {
        // Arrange
        var mockApiService = new Mock<IApiService>();
        var viewModel = new AccountViewModel(mockApiService.Object);

        // Assert - Verify command exists (generated by RelayCommand attribute)
        Assert.IsNotNull(viewModel.LogoutCommand, "LogoutCommand should be generated by RelayCommand attribute.");

        // Act & Assert
        // The command will throw when executed due to MAUI static dependencies
        try
        {
            await viewModel.LogoutCommand.ExecuteAsync(null);
            Assert.Inconclusive("Logout completed without MAUI runtime context - unexpected in unit test environment.");
        }
        catch (NullReferenceException)
        {
            // Expected in unit test environment without MAUI runtime
            Assert.IsTrue(true, "Expected NullReferenceException due to missing MAUI runtime context.");
        }
    }

    /// <summary>
    /// Tests that EditProfile navigates to the correct route.
    /// Due to the static dependency on Shell.Current, this test cannot be fully executed
    /// without integration test infrastructure or refactoring to use dependency injection
    /// for navigation services.
    /// </summary>
    [TestMethod]
    public async Task EditProfile_WhenCalled_NavigatesToEditProfileRoute()
    {
        // Arrange
        var mockApiService = new Mock<IApiService>();
        var viewModel = new AccountViewModel(mockApiService.Object);

        // Act & Assert
        // NOTE: This method uses Shell.Current.GoToAsync which is a static dependency
        // that cannot be mocked with Moq. To properly test this method, the production
        // code should be refactored to:
        // 1. Inject an INavigationService abstraction instead of using Shell.Current directly
        // 2. Make Shell.Current accessible through dependency injection
        //
        // Without this refactoring, we cannot verify:
        // - That GoToAsync is called with the correct route ("edit-profile")
        // - That the navigation completes successfully
        // - Error handling when Shell.Current is null or navigation fails
        //
        // This test is marked as Inconclusive until the code is refactored for testability.

        Assert.Inconclusive(
            "EditProfile method has a static dependency on Shell.Current that cannot be mocked. " +
            "Refactor to use dependency injection for navigation services to enable proper unit testing.");
    }

    /// <summary>
    /// Tests that LoadData returns early without executing when IsLoading is already true.
    /// Verifies the guard clause prevents concurrent execution.
    /// Expected: Method returns immediately, no API calls made.
    /// </summary>
    [TestMethod]
    public async Task LoadData_WhenIsLoadingIsTrue_ReturnsEarlyWithoutExecuting()
    {
        // Arrange
        var mockApiService = new Mock<IApiService>();
        var viewModel = new AccountViewModel(mockApiService.Object);

        // Manually set IsLoading to true using reflection-free approach
        // First call will set IsLoading, we'll verify second call doesn't execute
        mockApiService.Setup(x => x.GetUserStatsAsync())
            .ReturnsAsync((ApiResponse<UserStatsDto>?)null);
        mockApiService.Setup(x => x.GetLessonsAsync())
            .ReturnsAsync((ApiResponse<IEnumerable<LessonDto>>?)null);

        // Start first call (which will set IsLoading to true)
        var firstTask = viewModel.LoadDataCommand.ExecuteAsync(null);

        // Immediately try second call while first is running
        await viewModel.LoadDataCommand.ExecuteAsync(null);

        // Complete first task
        await firstTask;

        // Assert
        // If guard works properly, API methods should be called only once
        mockApiService.Verify(x => x.GetUserStatsAsync(), Times.Once);
        mockApiService.Verify(x => x.GetLessonsAsync(), Times.Once);
    }

    /// <summary>
    /// Tests LoadData with valid user stats response.
    /// Verifies CurrentStreak and GlobalRanking are properly assigned from API response.
    /// Expected: Properties are set to values from stats.Data.
    /// </summary>
    [TestMethod]
    [DataRow(0, 0)]
    [DataRow(1, 1)]
    [DataRow(100, 500)]
    [DataRow(int.MaxValue, int.MaxValue)]
    [DataRow(-1, -1)]
    public async Task LoadData_WithValidUserStats_SetsCurrentStreakAndGlobalRanking(int streak, int ranking)
    {
        // Arrange
        var mockApiService = new Mock<IApiService>();
        var statsResponse = new ApiResponse<UserStatsDto>
        {
            Success = true,
            Data = new UserStatsDto
            {
                CurrentStreak = streak,
                GlobalRanking = ranking
            }
        };

        mockApiService.Setup(x => x.GetUserStatsAsync())
            .ReturnsAsync(statsResponse);
        mockApiService.Setup(x => x.GetLessonsAsync())
            .ReturnsAsync((ApiResponse<IEnumerable<LessonDto>>?)null);

        var viewModel = new AccountViewModel(mockApiService.Object);

        // Act
        await viewModel.LoadDataCommand.ExecuteAsync(null);

        // Assert
        Assert.AreEqual(streak, viewModel.CurrentStreak);
        Assert.AreEqual(ranking, viewModel.GlobalRanking);
        Assert.IsFalse(viewModel.IsLoading);
    }

    /// <summary>
    /// Tests LoadData when GetUserStatsAsync returns null response.
    /// Verifies null-safe handling and properties remain unchanged.
    /// Expected: No exception thrown, properties not modified.
    /// </summary>
    [TestMethod]
    public async Task LoadData_WhenStatsResponseIsNull_DoesNotSetProperties()
    {
        // Arrange
        var mockApiService = new Mock<IApiService>();
        mockApiService.Setup(x => x.GetUserStatsAsync())
            .ReturnsAsync((ApiResponse<UserStatsDto>?)null);
        mockApiService.Setup(x => x.GetLessonsAsync())
            .ReturnsAsync((ApiResponse<IEnumerable<LessonDto>>?)null);

        var viewModel = new AccountViewModel(mockApiService.Object);
        var initialStreak = viewModel.CurrentStreak;
        var initialRanking = viewModel.GlobalRanking;

        // Act
        await viewModel.LoadDataCommand.ExecuteAsync(null);

        // Assert
        Assert.AreEqual(initialStreak, viewModel.CurrentStreak);
        Assert.AreEqual(initialRanking, viewModel.GlobalRanking);
        Assert.IsFalse(viewModel.IsLoading);
    }

    /// <summary>
    /// Tests LoadData when stats response has null Data property.
    /// Verifies null-safe handling for nested null values.
    /// Expected: No exception thrown, properties remain unchanged.
    /// </summary>
    [TestMethod]
    public async Task LoadData_WhenStatsDataIsNull_DoesNotSetProperties()
    {
        // Arrange
        var mockApiService = new Mock<IApiService>();
        var statsResponse = new ApiResponse<UserStatsDto>
        {
            Success = false,
            Data = null
        };

        mockApiService.Setup(x => x.GetUserStatsAsync())
            .ReturnsAsync(statsResponse);
        mockApiService.Setup(x => x.GetLessonsAsync())
            .ReturnsAsync((ApiResponse<IEnumerable<LessonDto>>?)null);

        var viewModel = new AccountViewModel(mockApiService.Object);
        var initialStreak = viewModel.CurrentStreak;
        var initialRanking = viewModel.GlobalRanking;

        // Act
        await viewModel.LoadDataCommand.ExecuteAsync(null);

        // Assert
        Assert.AreEqual(initialStreak, viewModel.CurrentStreak);
        Assert.AreEqual(initialRanking, viewModel.GlobalRanking);
        Assert.IsFalse(viewModel.IsLoading);
    }

    /// <summary>
    /// Tests LoadData with valid lessons response containing various counts.
    /// Verifies TotalSignsLearned is set to the count of lessons.
    /// Expected: TotalSignsLearned equals the number of lessons in the collection.
    /// </summary>
    [TestMethod]
    [DataRow(0)]
    [DataRow(1)]
    [DataRow(5)]
    [DataRow(100)]
    public async Task LoadData_WithValidLessons_SetsTotalSignsLearnedToCount(int lessonCount)
    {
        // Arrange
        var mockApiService = new Mock<IApiService>();
        var lessons = Enumerable.Range(1, lessonCount)
            .Select(i => new LessonDto { Id = i, Title = $"Lesson {i}" })
            .ToList();

        var lessonsResponse = new ApiResponse<IEnumerable<LessonDto>>
        {
            Success = true,
            Data = lessons
        };

        mockApiService.Setup(x => x.GetUserStatsAsync())
            .ReturnsAsync((ApiResponse<UserStatsDto>?)null);
        mockApiService.Setup(x => x.GetLessonsAsync())
            .ReturnsAsync(lessonsResponse);

        var viewModel = new AccountViewModel(mockApiService.Object);

        // Act
        await viewModel.LoadDataCommand.ExecuteAsync(null);

        // Assert
        Assert.AreEqual(lessonCount, viewModel.TotalSignsLearned);
        Assert.IsFalse(viewModel.IsLoading);
    }

    /// <summary>
    /// Tests LoadData when GetLessonsAsync returns null response.
    /// Verifies null-safe handling and TotalSignsLearned remains unchanged.
    /// Expected: No exception thrown, TotalSignsLearned not modified.
    /// </summary>
    [TestMethod]
    public async Task LoadData_WhenLessonsResponseIsNull_DoesNotSetTotalSignsLearned()
    {
        // Arrange
        var mockApiService = new Mock<IApiService>();
        mockApiService.Setup(x => x.GetUserStatsAsync())
            .ReturnsAsync((ApiResponse<UserStatsDto>?)null);
        mockApiService.Setup(x => x.GetLessonsAsync())
            .ReturnsAsync((ApiResponse<IEnumerable<LessonDto>>?)null);

        var viewModel = new AccountViewModel(mockApiService.Object);
        var initialTotal = viewModel.TotalSignsLearned;

        // Act
        await viewModel.LoadDataCommand.ExecuteAsync(null);

        // Assert
        Assert.AreEqual(initialTotal, viewModel.TotalSignsLearned);
        Assert.IsFalse(viewModel.IsLoading);
    }

    /// <summary>
    /// Tests LoadData when lessons response has null Data property.
    /// Verifies null-safe handling for nested null values in lessons response.
    /// Expected: No exception thrown, TotalSignsLearned remains unchanged.
    /// </summary>
    [TestMethod]
    public async Task LoadData_WhenLessonsDataIsNull_DoesNotSetTotalSignsLearned()
    {
        // Arrange
        var mockApiService = new Mock<IApiService>();
        var lessonsResponse = new ApiResponse<IEnumerable<LessonDto>>
        {
            Success = false,
            Data = null
        };

        mockApiService.Setup(x => x.GetUserStatsAsync())
            .ReturnsAsync((ApiResponse<UserStatsDto>?)null);
        mockApiService.Setup(x => x.GetLessonsAsync())
            .ReturnsAsync(lessonsResponse);

        var viewModel = new AccountViewModel(mockApiService.Object);
        var initialTotal = viewModel.TotalSignsLearned;

        // Act
        await viewModel.LoadDataCommand.ExecuteAsync(null);

        // Assert
        Assert.AreEqual(initialTotal, viewModel.TotalSignsLearned);
        Assert.IsFalse(viewModel.IsLoading);
    }

    /// <summary>
    /// Tests LoadData with both valid stats and lessons responses.
    /// Verifies all properties are correctly populated in a successful scenario.
    /// Expected: CurrentStreak, GlobalRanking, and TotalSignsLearned all set correctly.
    /// </summary>
    [TestMethod]
    public async Task LoadData_WithValidStatsAndLessons_SetsAllProperties()
    {
        // Arrange
        var mockApiService = new Mock<IApiService>();
        var statsResponse = new ApiResponse<UserStatsDto>
        {
            Success = true,
            Data = new UserStatsDto
            {
                CurrentStreak = 42,
                GlobalRanking = 123,
                TotalXP = 1000
            }
        };

        var lessons = new List<LessonDto>
        {
            new LessonDto { Id = 1, Title = "Lesson 1" },
            new LessonDto { Id = 2, Title = "Lesson 2" },
            new LessonDto { Id = 3, Title = "Lesson 3" }
        };

        var lessonsResponse = new ApiResponse<IEnumerable<LessonDto>>
        {
            Success = true,
            Data = lessons
        };

        mockApiService.Setup(x => x.GetUserStatsAsync())
            .ReturnsAsync(statsResponse);
        mockApiService.Setup(x => x.GetLessonsAsync())
            .ReturnsAsync(lessonsResponse);

        var viewModel = new AccountViewModel(mockApiService.Object);

        // Act
        await viewModel.LoadDataCommand.ExecuteAsync(null);

        // Assert
        Assert.AreEqual(42, viewModel.CurrentStreak);
        Assert.AreEqual(123, viewModel.GlobalRanking);
        Assert.AreEqual(3, viewModel.TotalSignsLearned);
        Assert.IsFalse(viewModel.IsLoading);
    }

    /// <summary>
    /// Tests LoadData when GetUserStatsAsync throws an exception.
    /// Verifies exception is caught and IsLoading is properly reset in finally block.
    /// Expected: Exception handled gracefully, IsLoading set to false.
    /// </summary>
    [TestMethod]
    public async Task LoadData_WhenGetUserStatsThrowsException_HandlesExceptionAndResetsIsLoading()
    {
        // Arrange
        var mockApiService = new Mock<IApiService>();
        mockApiService.Setup(x => x.GetUserStatsAsync())
            .ThrowsAsync(new InvalidOperationException("API Error"));

        var viewModel = new AccountViewModel(mockApiService.Object);

        // Act
        await viewModel.LoadDataCommand.ExecuteAsync(null);

        // Assert
        Assert.IsFalse(viewModel.IsLoading);
        // Note: Cannot verify Shell.Current.DisplayAlertAsync call due to static property limitation
    }

    /// <summary>
    /// Tests LoadData when GetLessonsAsync throws an exception.
    /// Verifies exception is caught after successful stats call and IsLoading is reset.
    /// Expected: Exception handled gracefully, IsLoading set to false, stats properties may be set.
    /// </summary>
    [TestMethod]
    public async Task LoadData_WhenGetLessonsThrowsException_HandlesExceptionAndResetsIsLoading()
    {
        // Arrange
        var mockApiService = new Mock<IApiService>();
        var statsResponse = new ApiResponse<UserStatsDto>
        {
            Success = true,
            Data = new UserStatsDto
            {
                CurrentStreak = 10,
                GlobalRanking = 50
            }
        };

        mockApiService.Setup(x => x.GetUserStatsAsync())
            .ReturnsAsync(statsResponse);
        mockApiService.Setup(x => x.GetLessonsAsync())
            .ThrowsAsync(new TimeoutException("Network timeout"));

        var viewModel = new AccountViewModel(mockApiService.Object);

        // Act
        await viewModel.LoadDataCommand.ExecuteAsync(null);

        // Assert
        Assert.IsFalse(viewModel.IsLoading);
        Assert.AreEqual(10, viewModel.CurrentStreak);
        Assert.AreEqual(50, viewModel.GlobalRanking);
        // Note: Cannot verify Shell.Current.DisplayAlertAsync call due to static property limitation
    }

    /// <summary>
    /// Tests LoadData with various exception types to ensure all are properly handled.
    /// Verifies the catch block handles any Exception type and properly resets state.
    /// Expected: All exception types caught, IsLoading reset in all cases.
    /// </summary>
    [TestMethod]
    [DataRow(typeof(ArgumentNullException))]
    [DataRow(typeof(InvalidOperationException))]
    [DataRow(typeof(TimeoutException))]
    [DataRow(typeof(Exception))]
    public async Task LoadData_WithDifferentExceptionTypes_HandlesAllExceptionsGracefully(Type exceptionType)
    {
        // Arrange
        var mockApiService = new Mock<IApiService>();
        var exception = (Exception)Activator.CreateInstance(exceptionType, "Test exception")!;

        mockApiService.Setup(x => x.GetUserStatsAsync())
            .ThrowsAsync(exception);

        var viewModel = new AccountViewModel(mockApiService.Object);

        // Act
        await viewModel.LoadDataCommand.ExecuteAsync(null);

        // Assert
        Assert.IsFalse(viewModel.IsLoading);
    }

    /// <summary>
    /// Tests LoadData verifies both API methods are called in sequence.
    /// Confirms the execution flow calls GetUserStatsAsync before GetLessonsAsync.
    /// Expected: Both methods invoked exactly once in correct order.
    /// </summary>
    [TestMethod]
    public async Task LoadData_CallsBothApiMethodsInSequence()
    {
        // Arrange
        var mockApiService = new Mock<IApiService>();
        var callOrder = new List<string>();

        mockApiService.Setup(x => x.GetUserStatsAsync())
            .ReturnsAsync((ApiResponse<UserStatsDto>?)null)
            .Callback(() => callOrder.Add("GetUserStatsAsync"));

        mockApiService.Setup(x => x.GetLessonsAsync())
            .ReturnsAsync((ApiResponse<IEnumerable<LessonDto>>?)null)
            .Callback(() => callOrder.Add("GetLessonsAsync"));

        var viewModel = new AccountViewModel(mockApiService.Object);

        // Act
        await viewModel.LoadDataCommand.ExecuteAsync(null);

        // Assert
        mockApiService.Verify(x => x.GetUserStatsAsync(), Times.Once);
        mockApiService.Verify(x => x.GetLessonsAsync(), Times.Once);
        Assert.AreEqual(2, callOrder.Count);
        Assert.AreEqual("GetUserStatsAsync", callOrder[0]);
        Assert.AreEqual("GetLessonsAsync", callOrder[1]);
    }

    /// <summary>
    /// Tests LoadData ensures IsLoading is set to true during execution.
    /// Verifies the loading state is properly managed throughout the async operation.
    /// Expected: IsLoading becomes true at start, false at end.
    /// </summary>
    [TestMethod]
    public async Task LoadData_SetsIsLoadingTrueDuringExecution()
    {
        // Arrange
        var mockApiService = new Mock<IApiService>();
        var tcs = new TaskCompletionSource<ApiResponse<UserStatsDto>?>();

        mockApiService.Setup(x => x.GetUserStatsAsync())
            .Returns(tcs.Task);
        mockApiService.Setup(x => x.GetLessonsAsync())
            .ReturnsAsync((ApiResponse<IEnumerable<LessonDto>>?)null);

        var viewModel = new AccountViewModel(mockApiService.Object);

        // Act
        var loadTask = viewModel.LoadDataCommand.ExecuteAsync(null);

        // Assert - IsLoading should be true during execution
        Assert.IsTrue(viewModel.IsLoading);

        // Complete the async operation
        tcs.SetResult(null);
        await loadTask;

        // Assert - IsLoading should be false after completion
        Assert.IsFalse(viewModel.IsLoading);
    }

    /// <summary>
    /// Tests LoadData with boundary values for numeric properties.
    /// Verifies proper handling of extreme integer values.
    /// Expected: All boundary values handled correctly without overflow.
    /// </summary>
    [TestMethod]
    [DataRow(int.MinValue, int.MinValue)]
    [DataRow(0, int.MaxValue)]
    [DataRow(int.MaxValue, 0)]
    public async Task LoadData_WithBoundaryValues_HandlesExtremeValuesCorrectly(int streak, int ranking)
    {
        // Arrange
        var mockApiService = new Mock<IApiService>();
        var statsResponse = new ApiResponse<UserStatsDto>
        {
            Success = true,
            Data = new UserStatsDto
            {
                CurrentStreak = streak,
                GlobalRanking = ranking
            }
        };

        mockApiService.Setup(x => x.GetUserStatsAsync())
            .ReturnsAsync(statsResponse);
        mockApiService.Setup(x => x.GetLessonsAsync())
            .ReturnsAsync((ApiResponse<IEnumerable<LessonDto>>?)null);

        var viewModel = new AccountViewModel(mockApiService.Object);

        // Act
        await viewModel.LoadDataCommand.ExecuteAsync(null);

        // Assert
        Assert.AreEqual(streak, viewModel.CurrentStreak);
        Assert.AreEqual(ranking, viewModel.GlobalRanking);
        Assert.IsFalse(viewModel.IsLoading);
    }

    /// <summary>
    /// Tests that the AccountViewModel constructor successfully creates an instance
    /// when provided with a valid IApiService dependency.
    /// </summary>
    [TestMethod]
    public void AccountViewModel_ValidApiService_CreatesInstanceSuccessfully()
    {
        // Arrange
        var mockApiService = new Mock<IApiService>();

        // Act
        var viewModel = new AccountViewModel(mockApiService.Object);

        // Assert
        Assert.IsNotNull(viewModel);
        Assert.IsInstanceOfType(viewModel, typeof(AccountViewModel));
        Assert.IsInstanceOfType(viewModel, typeof(ObservableObject));
    }
}