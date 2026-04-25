using Moq;
using SignLanguageApp.Services;
using SignLanguageApp.ViewModels;
using System.Threading.Tasks;
using Microsoft.VisualStudio.TestTools.UnitTesting;

namespace SignLanguageApp.UnitTests.ViewModels
{
    [TestClass]
    public class RegisterViewModelTests
    {
        private Mock<IApiService> _apiServiceMock = null!;
        private RegisterViewModel _viewModel = null!;

        [TestInitialize]
        public void Setup()
        {
            _apiServiceMock = new Mock<IApiService>();
            _viewModel = new RegisterViewModel(_apiServiceMock.Object);
        }

        [TestMethod]
        public void InitialState_ValuesAreEmptyAndNoError()
        {
            Assert.AreEqual(string.Empty, _viewModel.Name);
            Assert.AreEqual(string.Empty, _viewModel.Email);
            Assert.AreEqual(string.Empty, _viewModel.Password);
            Assert.AreEqual(string.Empty, _viewModel.ConfirmPassword);
            Assert.IsFalse(_viewModel.HasError);
            Assert.IsFalse(_viewModel.HasSuccess);
            Assert.IsFalse(_viewModel.IsLoading);
        }

        [TestMethod]
        public void ExecuteRegisterCommand_EmptyFields_ShowsError()
        {
            // Act
            _viewModel.RegisterCommand.Execute(null);

            // Assert
            Assert.IsTrue(_viewModel.HasError);
            Assert.AreEqual("Please fill in all fields", _viewModel.ErrorMessage);
        }

        [TestMethod]
        public void ExecuteRegisterCommand_InvalidEmail_ShowsError()
        {
            // Arrange
            _viewModel.Name = "Test";
            _viewModel.Email = "invalid";
            _viewModel.Password = "password";
            _viewModel.ConfirmPassword = "password";

            // Act
            _viewModel.RegisterCommand.Execute(null);

            // Assert
            Assert.IsTrue(_viewModel.HasError);
            Assert.AreEqual("Please enter a valid email address", _viewModel.ErrorMessage);
        }

        [TestMethod]
        public void ExecuteRegisterCommand_ShortPassword_ShowsError()
        {
            // Arrange
            _viewModel.Name = "Test";
            _viewModel.Email = "test@test.com";
            _viewModel.Password = "pass";
            _viewModel.ConfirmPassword = "pass";

            // Act
            _viewModel.RegisterCommand.Execute(null);

            // Assert
            Assert.IsTrue(_viewModel.HasError);
            Assert.AreEqual("Password must be at least 6 characters", _viewModel.ErrorMessage);
        }

        [TestMethod]
        public void ExecuteRegisterCommand_PasswordMismatch_ShowsError()
        {
            // Arrange
            _viewModel.Name = "Test";
            _viewModel.Email = "test@test.com";
            _viewModel.Password = "password";
            _viewModel.ConfirmPassword = "different";

            // Act
            _viewModel.RegisterCommand.Execute(null);

            // Assert
            Assert.IsTrue(_viewModel.HasError);
            Assert.AreEqual("Passwords do not match", _viewModel.ErrorMessage);
        }

        [TestMethod]
        public void ExecuteRegisterCommand_ValidDataButApiFails_ShowsError()
        {
            // Arrange
            _viewModel.Name = "Test";
            _viewModel.Email = "test@test.com";
            _viewModel.Password = "password";
            _viewModel.ConfirmPassword = "password";

            _apiServiceMock.Setup(a => a.RegisterAsync("test@test.com", "password", "Test"))
                           .ReturnsAsync((false, "Email already in use"));

            // Act
            _viewModel.RegisterCommand.Execute(null);

            // Assert
            System.Threading.Thread.Sleep(100);
            
            Assert.IsTrue(_viewModel.HasError);
            Assert.AreEqual("Email already in use", _viewModel.ErrorMessage);
        }

        [TestMethod]
        public void ExecuteRegisterCommand_Exception_ShowsExceptionError()
        {
            // Arrange
            _viewModel.Name = "Test";
            _viewModel.Email = "test@test.com";
            _viewModel.Password = "password";
            _viewModel.ConfirmPassword = "password";

            _apiServiceMock.Setup(a => a.RegisterAsync("test@test.com", "password", "Test"))
                           .ThrowsAsync(new System.Exception("Network Error"));

            // Act
            _viewModel.RegisterCommand.Execute(null);

            // Assert
            System.Threading.Thread.Sleep(100);

            Assert.IsTrue(_viewModel.HasError);
            Assert.AreEqual("Registration failed: Network Error", _viewModel.ErrorMessage);
        }
    }
}
