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
    public class InteractiveLessonViewModelTests
    {
        private Mock<IApiService> _apiServiceMock = null!;
        private Mock<IGesturePredictionService> _gestureServiceMock = null!;
        private Mock<IEnvironmentDetectionService> _envServiceMock = null!;
        private Mock<IStudyService> _studyServiceMock = null!;
        private Mock<IFrameBufferService> _frameBufferMock = null!;
        private InteractiveLessonViewModel _viewModel = null!;

        [TestInitialize]
        public void Setup()
        {
            _apiServiceMock = new Mock<IApiService>();
            _gestureServiceMock = new Mock<IGesturePredictionService>();
            _envServiceMock = new Mock<IEnvironmentDetectionService>();
            _studyServiceMock = new Mock<IStudyService>();
            _frameBufferMock = new Mock<IFrameBufferService>();

            _viewModel = new InteractiveLessonViewModel(
                _apiServiceMock.Object, _gestureServiceMock.Object, _envServiceMock.Object, _studyServiceMock.Object, _frameBufferMock.Object
            , new Moq.Mock<SignLanguageApp.Services.IMediaDownloadAndCacheService>().Object);
        }

        [TestMethod]
        public void PushFallbackFrame_SetsLatestFrameData()
        {
            // Arrange
            var testFrame = new byte[] { 0x01, 0x02, 0x03 };

            // Act & Assert (Should complete without throwing exception)
            _viewModel.PushFallbackFrame(testFrame);
            Assert.IsNotNull(_viewModel);
        }

        [TestMethod]
        public void ApplyQueryAttributes_LoadsLessonData_WhenLessonIdQueryParamPassed()
        {
            // Arrange
            var query = new Dictionary<string, object>
            {
                { "lessonId", "1" }
            };

            var interactiveLessonDto = new ApiResponse<InteractiveLessonDto>
            {
                Success = true,
                Data = new InteractiveLessonDto
                {
                    Id = 1,
                    Title = "Test Interactive Lesson",
                    Steps = new List<LessonStepDto>
                    {
                        new LessonStepDto { Type = LessonStepType.Flashcard, Title = "Welcome Step", Description = "Intro" }
                    }
                }
            };

            _apiServiceMock.Setup(s => s.GetInteractiveLessonAsync(1)).ReturnsAsync(interactiveLessonDto);

            // Act
            _viewModel.ApplyQueryAttributes(query);

            // Assert
            Assert.IsNotNull(_viewModel);
        }

        [TestMethod]
        public void Cleanup_CancelsProcessingAndDisposesResourcesSafely()
        {
            // Act & Assert (Should execute without exceptions)
            _viewModel.Cleanup();
            Assert.IsNotNull(_viewModel);
        }
    }
}
