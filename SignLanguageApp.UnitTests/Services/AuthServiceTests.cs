using Moq;
using SignLanguageApp.Model;
using SignLanguageApp.Services;

namespace SignLanguageApp.UnitTests.Services
{
    [TestClass]
    public class AuthServiceTests
    {
        private Mock<IApiService> _apiServiceMock = null!;
        private Mock<IDatabaseService> _databaseServiceMock = null!;
        private AuthService _authService = null!;

        [TestInitialize]
        public void Setup()
        {
            _apiServiceMock = new Mock<IApiService>();
            _databaseServiceMock = new Mock<IDatabaseService>();
            _authService = new AuthService(_apiServiceMock.Object, _databaseServiceMock.Object);
        }

        [TestMethod]
        public async Task LoginAsync_SuccessfulLogin_ReturnsResponseAndSavesData()
        {
            // Arrange
            var response = new LoginResponse
            {
                AccessToken = "access_token",
                RefreshToken = "refresh_token",
                User = new UserDto { Id = "1", Email = "test@test.com", Name = "Test User" }
            };

            _apiServiceMock.Setup(a => a.LoginAsync(It.IsAny<string>(), It.IsAny<string>()))
                           .ReturnsAsync(response);

            // Act
            var result = await _authService.LoginAsync("email", "password");

            // Assert
            Assert.IsNotNull(result);
            Assert.AreEqual("access_token", result.AccessToken);
            _databaseServiceMock.Verify(d => d.SaveAccessTokenAsync("access_token"), Times.Once);
            _databaseServiceMock.Verify(d => d.SaveRefreshTokenAsync("refresh_token"), Times.Once);
            _databaseServiceMock.Verify(d => d.SaveUserAsync(It.IsAny<User>()), Times.Once);
            _apiServiceMock.Verify(a => a.SetAuthToken("access_token"), Times.Once);
        }

        [TestMethod]
        public async Task LoginAsync_FailedLogin_ReturnsNull()
        {
            // Arrange
            _apiServiceMock.Setup(a => a.LoginAsync(It.IsAny<string>(), It.IsAny<string>()))
                           .ReturnsAsync((LoginResponse?)null);

            // Act
            var result = await _authService.LoginAsync("email", "password");

            // Assert
            Assert.IsNull(result);
            _databaseServiceMock.Verify(d => d.SaveAccessTokenAsync(It.IsAny<string>()), Times.Never);
        }

        [TestMethod]
        public async Task LoginAsync_ExceptionThrown_ReturnsNull()
        {
            // Arrange
            _apiServiceMock.Setup(a => a.LoginAsync(It.IsAny<string>(), It.IsAny<string>()))
                           .ThrowsAsync(new System.Exception("API Error"));

            // Act
            var result = await _authService.LoginAsync("email", "password");

            // Assert
            Assert.IsNull(result);
            _databaseServiceMock.Verify(d => d.SaveAccessTokenAsync(It.IsAny<string>()), Times.Never);
        }

        [TestMethod]
        public async Task RegisterAsync_SuccessfulRegistration_ReturnsBasicUserDto()
        {
            // Arrange
            _apiServiceMock.Setup(a => a.RegisterAsync(It.IsAny<string>(), It.IsAny<string>(), It.IsAny<string>()))
                           .ReturnsAsync((true, "Success"));

            // Act
            var result = await _authService.RegisterAsync("email", "password", "name");

            // Assert
            Assert.IsNotNull(result);
            Assert.AreEqual("email", result.Email);
            Assert.AreEqual("name", result.Name);
        }

        [TestMethod]
        public async Task RegisterAsync_FailedRegistration_ReturnsNull()
        {
            // Arrange
            _apiServiceMock.Setup(a => a.RegisterAsync(It.IsAny<string>(), It.IsAny<string>(), It.IsAny<string>()))
                           .ReturnsAsync((false, "Error"));

            // Act
            var result = await _authService.RegisterAsync("email", "password", "name");

            // Assert
            Assert.IsNull(result);
        }

        [TestMethod]
        public async Task RegisterAsync_ExceptionThrown_ReturnsNull()
        {
            // Arrange
            _apiServiceMock.Setup(a => a.RegisterAsync(It.IsAny<string>(), It.IsAny<string>(), It.IsAny<string>()))
                           .ThrowsAsync(new System.Exception("API Error"));

            // Act
            var result = await _authService.RegisterAsync("email", "password", "name");

            // Assert
            Assert.IsNull(result);
        }

        [TestMethod]
        public async Task LogoutAsync_SuccessfulLogout_ClearsDatabaseAndToken()
        {
            // Arrange
            _databaseServiceMock.Setup(d => d.ClearAllAsync()).Returns(Task.CompletedTask);

            // Act
            var result = await _authService.LogoutAsync();

            // Assert
            Assert.IsTrue(result);
            _databaseServiceMock.Verify(d => d.ClearAllAsync(), Times.Once);
            _apiServiceMock.Verify(a => a.SetAuthToken(string.Empty), Times.Once);
        }

        [TestMethod]
        public async Task GetCurrentUserAsync_ReturnsUserFromDatabase()
        {
            // Arrange
            var user = new User { Id = "1", Email = "test@test.com" };
            _databaseServiceMock.Setup(d => d.GetUserAsync()).ReturnsAsync(user);

            // Act
            var result = await _authService.GetCurrentUserAsync();

            // Assert
            Assert.IsNotNull(result);
            Assert.AreEqual(user.Id, result.Id);
        }

        [TestMethod]
        public async Task IsAuthenticatedAsync_TokenExists_ReturnsTrue()
        {
            // Arrange
            _databaseServiceMock.Setup(d => d.GetAccessTokenAsync()).ReturnsAsync("token");

            // Act
            var result = await _authService.IsAuthenticatedAsync();

            // Assert
            Assert.IsTrue(result);
        }

        [TestMethod]
        public async Task IsAuthenticatedAsync_TokenEmpty_ReturnsFalse()
        {
            // Arrange
            _databaseServiceMock.Setup(d => d.GetAccessTokenAsync()).ReturnsAsync(string.Empty);

            // Act
            var result = await _authService.IsAuthenticatedAsync();

            // Assert
            Assert.IsFalse(result);
        }

        [TestMethod]
        public async Task RefreshTokenAsync_RefreshTokenExistsAndRefreshSucceeds_ReturnsTrue()
        {
            // Arrange
            _databaseServiceMock.Setup(d => d.GetRefreshTokenAsync()).ReturnsAsync("refresh_token");
            _apiServiceMock.Setup(a => a.RefreshTokenAsync("refresh_token")).ReturnsAsync(true);

            // Act
            var result = await _authService.RefreshTokenAsync();

            // Assert
            Assert.IsTrue(result);
            _apiServiceMock.Verify(a => a.RefreshTokenAsync("refresh_token"), Times.Once);
        }

        [TestMethod]
        public async Task RefreshTokenAsync_RefreshTokenEmpty_ReturnsFalse()
        {
            // Arrange
            _databaseServiceMock.Setup(d => d.GetRefreshTokenAsync()).ReturnsAsync(string.Empty);

            // Act
            var result = await _authService.RefreshTokenAsync();

            // Assert
            Assert.IsFalse(result);
            _apiServiceMock.Verify(a => a.RefreshTokenAsync(It.IsAny<string>()), Times.Never);
        }
    }
}
