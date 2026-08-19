using Microsoft.VisualStudio.TestTools.UnitTesting;
using Moq;
using SignLanguageApp.Model;
using SignLanguageApp.Services;
using SignLanguageApp.ViewModels;
using System.Collections.Generic;
using System.Threading.Tasks;

namespace SignLanguageApp.UnitTests.ViewModels
{
    [TestClass]
    public class LearnViewModelTests
    {
        private Mock<IApiService> _apiServiceMock = null!;
        private Mock<ILessonPayloadSecurityService> _securityServiceMock = null!;
        private LearnViewModel _viewModel = null!;

        [TestInitialize]
        public void Setup()
        {
            _apiServiceMock = new Mock<IApiService>();
            _securityServiceMock = new Mock<ILessonPayloadSecurityService>();
            _viewModel = new LearnViewModel(_apiServiceMock.Object, _securityServiceMock.Object);
        }

        [TestMethod]
        public async Task InitializeAsync_WhenApiFails_SetsIsApiDisconnectedTrueAndClearsCategories()
        {
            // Arrange
            _apiServiceMock.Setup(s => s.GetUserStatsAsync()).ReturnsAsync((UserStatsDto?)null);
            _apiServiceMock.Setup(s => s.GetCategoriesAsync()).ReturnsAsync((ApiResponse<IEnumerable<LessonCategoryDto>>?)null);
            _apiServiceMock.Setup(s => s.GetPersonalizedRecommendationAsync()).ReturnsAsync((ApiResponse<PersonalizedRecommendationDto>?)null);

            // Act
            await _viewModel.InitializeAsync();

            // Assert
            Assert.IsTrue(_viewModel.IsApiDisconnected, new Moq.Mock<SignLanguageApp.Services.IMediaDownloadAndCacheService>().Object, "IsApiDisconnected should be true when API returns null");
            Assert.IsTrue(_viewModel.Categories == null || _viewModel.Categories.Count == 0, "Categories should be empty when API fails");
        }

        [TestMethod]
        public async Task InitializeAsync_WhenApiReturnsCategories_PopulatesCategories()
        {
            // Arrange
            var categories = new List<LessonCategoryDto>
            {
                new LessonCategoryDto { Id = 10, Title = "ASL Basics", Description = "Basics", Difficulty = "Beginner", Progress = 0.5 },
                new LessonCategoryDto { Id = 11, Title = "ASL Advanced", Description = "Advanced", Difficulty = "Advanced", Progress = 0.1 }
            };

            var categoriesResponse = new ApiResponse<IEnumerable<LessonCategoryDto>>
            {
                Success = true,
                Data = categories
            };

            _apiServiceMock.Setup(s => s.GetCategoriesAsync()).ReturnsAsync(categoriesResponse);

            // Act
            await _viewModel.InitializeAsync();

            // Assert
            Assert.IsNotNull(_viewModel.Categories);
            Assert.AreEqual(2, _viewModel.Categories.Count);
            Assert.AreEqual(10, _viewModel.SelectedCategory?.Id);
        }

        [TestMethod]
        public void SelectCategory_UpdatesSelectedCategory()
        {
            // Arrange
            var category = new LessonCategory { Id = 5, Title = "Family" };

            // Act
            _.Execute(category);

            // Assert
            Assert.AreEqual(category, _viewModel.SelectedCategory);
        }

        [TestMethod]
        public void HasDailyReviews_ReturnsTrue_WhenDailyReviewsExist()
        {
            // Act
            _viewModel.DailyReviewLessons = new System.Collections.ObjectModel.ObservableCollection<SpacedRepetitionLesson>
            {
                new SpacedRepetitionLesson { Id = 1, Title = "Alphabet A" }
            };

            // Assert
            Assert.IsTrue(_viewModel.HasDailyReviews);
        }

        [TestMethod]
        public void HasDailyReviews_ReturnsFalse_WhenEmpty()
        {
            // Act
            _viewModel.DailyReviewLessons = new System.Collections.ObjectModel.ObservableCollection<SpacedRepetitionLesson>();

            // Assert
            Assert.IsFalse(_viewModel.HasDailyReviews);
        }
    }
}
