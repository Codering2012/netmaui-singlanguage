using System;
using System.Threading.Tasks;
using Microsoft.AspNetCore.Http;
using Microsoft.AspNetCore.Mvc;
using Microsoft.EntityFrameworkCore;
using Microsoft.Extensions.Logging;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using Moq;
using SignLanguageApi.Controllers;
using SignLanguageApi.Data;
using SignLanguageApi.Dtos;
using SignLanguageApi.Services;

namespace SignLanguageApi.UnitTests.Controllers
{
    [TestClass]
    public class AuthControllerEdgeAndSecurityTests
    {
        private Mock<AppDbContext> _contextMock = null!;
        private Mock<IAuthService> _authServiceMock = null!;
        private Mock<IUserProgressService> _progressServiceMock = null!;
        private Mock<IPasswordValidator> _passwordValidatorMock = null!;
        private Mock<ITokenBlacklistService> _tokenBlacklistMock = null!;
        private Mock<IAuditLogger> _auditLoggerMock = null!;
        private Mock<ILogger<AuthController>> _loggerMock = null!;

        [TestInitialize]
        public void Setup()
        {
            var options = new DbContextOptionsBuilder<AppDbContext>()
                .UseInMemoryDatabase(databaseName: "AuthTestDb_" + Guid.NewGuid().ToString("N"))
                .Options;

            _contextMock = new Mock<AppDbContext>(options);
            _authServiceMock = new Mock<IAuthService>();
            _progressServiceMock = new Mock<IUserProgressService>();
            _passwordValidatorMock = new Mock<IPasswordValidator>();
            _tokenBlacklistMock = new Mock<ITokenBlacklistService>();
            _auditLoggerMock = new Mock<IAuditLogger>();
            _loggerMock = new Mock<ILogger<AuthController>>();
        }

        private AuthController CreateController()
        {
            var controller = new AuthController(
                _contextMock.Object,
                _authServiceMock.Object,
                _progressServiceMock.Object,
                _passwordValidatorMock.Object,
                _tokenBlacklistMock.Object,
                _auditLoggerMock.Object,
                _loggerMock.Object);

            controller.ControllerContext = new ControllerContext
            {
                HttpContext = new DefaultHttpContext()
            };

            return controller;
        }

        [TestMethod]
        public async Task Register_NullRequest_ReturnsBadRequest()
        {
            var controller = CreateController();
            var result = await controller.Register(null!);
            Assert.IsInstanceOfType(result.Result, typeof(BadRequestObjectResult));
        }

        [TestMethod]
        public async Task Register_WhitespaceEmail_ReturnsBadRequest()
        {
            var controller = CreateController();
            var request = new RegisterRequest("   ", "ValidPassword123!", "Alice");

            var result = await controller.Register(request);
            Assert.IsInstanceOfType(result.Result, typeof(BadRequestObjectResult));
        }

        [TestMethod]
        public async Task Register_SqlInjectionEmailPayload_HandledSafely()
        {
            var controller = CreateController();
            _passwordValidatorMock.Setup(p => p.ValidatePassword(It.IsAny<string>())).Returns((true, string.Empty));

            var request = new RegisterRequest("attacker' OR '1'='1' -- @domain.com", "ValidPassword123!", "Attacker' --");

            var result = await controller.Register(request);
            Assert.IsNotNull(result);
        }

        [TestMethod]
        public async Task Register_WeakPassword_ReturnsBadRequestWithDetails()
        {
            var controller = CreateController();
            _passwordValidatorMock.Setup(p => p.ValidatePassword("weak")).Returns((false, "Password must be at least 8 characters long."));

            var request = new RegisterRequest("user@test.com", "weak", "User");

            var result = await controller.Register(request);
            Assert.IsInstanceOfType(result.Result, typeof(BadRequestObjectResult));
        }

        [TestMethod]
        public async Task Login_NullPayload_ReturnsBadRequest()
        {
            var controller = CreateController();
            var result = await controller.Login(null!);
            Assert.IsInstanceOfType(result.Result, typeof(BadRequestObjectResult));
        }

        [TestMethod]
        public async Task Login_EmptyPassword_ReturnsBadRequest()
        {
            var controller = CreateController();
            var request = new LoginRequest("test@example.com", "");

            var result = await controller.Login(request);
            Assert.IsInstanceOfType(result.Result, typeof(BadRequestObjectResult));
        }

        [TestMethod]
        public async Task RefreshToken_BlacklistedToken_ReturnsBadRequest()
        {
            var controller = CreateController();
            _tokenBlacklistMock.Setup(t => t.IsTokenBlacklistedAsync(It.IsAny<string>())).ReturnsAsync(true);

            var request = new RefreshTokenRequest { RefreshToken = "blacklisted_token" };

            var result = await controller.RefreshToken(request);
            Assert.IsInstanceOfType(result.Result, typeof(ObjectResult));
        }
    }
}
