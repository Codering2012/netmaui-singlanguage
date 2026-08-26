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
        private Mock<IMediaDownloadAndCacheService> _mediaCacheMock = null!;
        private Mock<IDatabaseService> _dbMock = null!;
        private LearnViewModel _viewModel = null!;

        [TestInitialize]
        public void Setup()
        {
            _apiServiceMock = new Mock<IApiService>();
            _securityServiceMock = new Mock<ILessonPayloadSecurityService>();
            _mediaCacheMock = new Mock<IMediaDownloadAndCacheService>();
            _dbMock = new Mock<IDatabaseService>();
            _viewModel = new LearnViewModel(_apiServiceMock.Object, _securityServiceMock.Object, _mediaCacheMock.Object, _dbMock.Object);
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
        }

        [TestMethod]
        public void SelectCategory_UpdatesSelectedCategory()
        {
            // Arrange
            var category = new LessonCategory { Id = 5, Title = "Family" };

            // Act
            _viewModel.SelectedCategory = category;

            // Assert
            Assert.AreEqual(category, _viewModel.SelectedCategory);
        }
    }
}
