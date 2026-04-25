using Moq;
using SignLanguageApp.Model;
using SignLanguageApp.Services;
using SignLanguageApp.ViewModels;
using System.Threading.Tasks;
using Microsoft.VisualStudio.TestTools.UnitTesting;

namespace SignLanguageApp.UnitTests.ViewModels
{
    [TestClass]
    public class CameraTranslationViewModelTests
    {
        private Mock<IApiService> _apiServiceMock = null!;
        private CameraTranslationViewModel _viewModel = null!;

        [TestInitialize]
        public void Setup()
        {
            _apiServiceMock = new Mock<IApiService>();
            _viewModel = new CameraTranslationViewModel(_apiServiceMock.Object);
        }

        [TestMethod]
        public void StartCameraCaptureCommand_CanExecute()
        {
            // Act
            _viewModel.StartCameraCaptureCommand.Execute(null);

            // Assert
            Assert.IsTrue(_viewModel.IsProcessingFrames);
        }

        [TestMethod]
        public void StopCameraCaptureCommand_DisablesProcessing()
        {
            // Arrange
            _viewModel.StartCameraCaptureCommand.Execute(null);
            
            // Act
            _viewModel.StopCameraCaptureCommand.Execute(null);

            // Assert
            Assert.IsFalse(_viewModel.IsProcessingFrames);
        }
        
        [TestMethod]
        public void InitialState_ValuesAreEmptyAndNoError()
        {
            // Assert
            Assert.AreEqual(string.Empty, _viewModel.CurrentGestureLabel);
            Assert.AreEqual(string.Empty, _viewModel.TranslatedText);
            Assert.AreEqual(string.Empty, _viewModel.ConfidenceText);
            Assert.IsFalse(_viewModel.IsProcessingFrames);
        }
    }
}
