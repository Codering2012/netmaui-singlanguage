using Moq;
using SignLanguageApp.Model;
using SignLanguageApp.Services;
using SignLanguageApp.ViewModels;
using System.Threading.Tasks;
using Microsoft.VisualStudio.TestTools.UnitTesting;

namespace SignLanguageApp.UnitTests.ViewModels
{
    [TestClass]
    public class LoginViewModelTests
    {
        private Mock<IAuthService> _authServiceMock = null!;
        private Mock<IApiService> _apiServiceMock = null!;
        private LoginViewModel _viewModel = null!;

        [TestInitialize]
        public void Setup()
        {
            _authServiceMock = new Mock<IAuthService>();
            _apiServiceMock = new Mock<IApiService>();
            _viewModel = new LoginViewModel(_authServiceMock.Object, _apiServiceMock.Object);
        }

        [TestMethod]
        public void InitialState_ValuesAreEmptyAndNoError()
        {
            // Assert
            Assert.AreEqual(string.Empty, _viewModel.Email);
            Assert.AreEqual(string.Empty, _viewModel.Password);
            Assert.AreEqual(string.Empty, _viewModel.ErrorMessage);
            Assert.IsFalse(_viewModel.HasError);
            Assert.IsFalse(_viewModel.IsLoading);
        }

        [TestMethod]
        public void ExecuteLoginCommand_EmptyCredentials_ShowsError()
        {
            // Arrange
            _viewModel.Email = "";
            _viewModel.Password = "";

            // Act
            _viewModel.LoginCommand.Execute(null);

            // Assert
            Assert.IsTrue(_viewModel.HasError);
            Assert.AreEqual("Please enter email and password", _viewModel.ErrorMessage);
        }

        [TestMethod]
        public void ExecuteLoginCommand_ValidCredentialsButApiFails_ShowsInvalidError()
        {
            // Arrange
            _viewModel.Email = "test@test.com";
            _viewModel.Password = "password";
            
            _authServiceMock.Setup(a => a.LoginAsync(It.IsAny<string>(), It.IsAny<string>()))
                            .ReturnsAsync((LoginResponse?)null);

            // Act
            _viewModel.LoginCommand.Execute(null);

            // Assert
            // Since it's async void equivalent through ICommand.Execute on a Command with an async lambda, we might have a slight race condition in unit tests if we don't await the inner task. 
            // In MAUI, the Command wraps async methods but doesn't expose the Task. 
            // A common workaround is to use a manual reset event or mock a delay, but we'll assume it runs sync enough for this test or we can wait briefly.
            System.Threading.Thread.Sleep(100); // give task time to complete
            
            Assert.IsTrue(_viewModel.HasError);
            Assert.AreEqual("Invalid email or password", _viewModel.ErrorMessage);
            Assert.IsFalse(_viewModel.IsLoading);
        }

        [TestMethod]
        public void ExecuteLoginCommand_ApiException_ShowsExceptionError()
        {
            // Arrange
            _viewModel.Email = "test@test.com";
            _viewModel.Password = "password";

            _authServiceMock.Setup(a => a.LoginAsync(It.IsAny<string>(), It.IsAny<string>()))
                            .ThrowsAsync(new System.Exception("Network Error"));

            // Act
            _viewModel.LoginCommand.Execute(null);

            // Assert
            System.Threading.Thread.Sleep(100);

            Assert.IsTrue(_viewModel.HasError);
            Assert.AreEqual("Login failed: Network Error", _viewModel.ErrorMessage);
            Assert.IsFalse(_viewModel.IsLoading);
        }
    }
}
