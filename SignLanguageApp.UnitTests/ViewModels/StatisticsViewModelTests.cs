using Microsoft.VisualStudio.TestTools.UnitTesting;
using Moq;
using SignLanguageApp.Model;
using SignLanguageApp.Services;
using SignLanguageApp.ViewModels;

namespace SignLanguageApp.UnitTests.ViewModels
{
    [TestClass]
    public class StatisticsViewModelTests
    {
        private Mock<IApiService> _apiServiceMock = null!;
        private StatisticsViewModel _viewModel = null!;

        [TestInitialize]
        public void Setup()
        {
            _apiServiceMock = new Mock<IApiService>();
            _viewModel = new StatisticsViewModel(_apiServiceMock.Object);
        }

        [TestMethod]
        public async Task LoadStatsAsync_ShouldPopulateData_WhenSuccessful()
        {
            // Arrange
            var statsDto = new UserStatsDto
            {
                TotalXp = 500,
                LearningStreak = 5,
                LessonsCompleted = 10,
                WeeklyXp = new List<DailyXpDto> { new DailyXpDto { Date = "2026-05-01", Xp = 50 } },
                CategoryProgress = new List<CategoryProgressDto> { new CategoryProgressDto { CategoryName = "Basics", Progress = 0.5 } }
            };

            _apiServiceMock.Setup(s => s.GetUserStatsAsync()).ReturnsAsync(statsDto);

            // Act
            await _viewModel.LoadStatsCommand.ExecuteAsync(null);

            // Assert
            Assert.AreEqual(500, _viewModel.TotalXp);
            Assert.AreEqual(5, _viewModel.LearningStreak);
            Assert.AreEqual(1, _viewModel.WeeklyXp.Count);
            Assert.AreEqual(1, _viewModel.CategoryProgress.Count);
        }
    }
}
