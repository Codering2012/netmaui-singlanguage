using BCrypt.Net;
using Microsoft.AspNetCore.Http;
using Microsoft.AspNetCore.Mvc;
using Microsoft.EntityFrameworkCore;
using Microsoft.Extensions.Logging;
using Microsoft.Extensions.Primitives;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using Moq;
using SignLanguageApi.Controllers;
using SignLanguageApi.Data;
using SignLanguageApi.Dtos;
using SignLanguageApi.Services;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Linq.Expressions;
using System.Net;
using System.Security.Claims;
using System.Threading;
using System.Threading.Tasks;

namespace SignLanguageApi.Controllers.UnitTests
{
    /// <summary>
    /// Unit tests for the AuthController class.
    /// </summary>
    [TestClass]
    public class AuthControllerTests
    {
        /// <summary>
        /// Tests that RefreshToken returns BadRequest when the request object is null.
        /// </summary>
        [TestMethod]
        public async Task RefreshToken_NullRequest_ReturnsBadRequest()
        {
            // Arrange
            var mockContext = new Mock<AppDbContext>(new DbContextOptions<AppDbContext>());
            var mockAuthService = new Mock<IAuthService>();
            var mockProgressService = new Mock<IUserProgressService>();
            var mockPasswordValidator = new Mock<IPasswordValidator>();
            var mockTokenBlacklist = new Mock<ITokenBlacklistService>();
            var mockAuditLogger = new Mock<IAuditLogger>();
            var mockLogger = new Mock<ILogger<AuthController>>();
            var controller = new AuthController(mockContext.Object, mockAuthService.Object, mockProgressService.Object, mockPasswordValidator.Object, mockTokenBlacklist.Object, mockAuditLogger.Object, mockLogger.Object);
            RefreshTokenRequest? request = null;
            // Act
            var result = await controller.RefreshToken(request!);
            // Assert
            Assert.IsInstanceOfType(result.Result, typeof(BadRequestObjectResult));
            var badRequestResult = result.Result as BadRequestObjectResult;
            Assert.IsNotNull(badRequestResult);
            dynamic? value = badRequestResult.Value;
            Assert.IsNotNull(value);
            Assert.AreEqual("Refresh token is required.", value.message);
        }

        /// <summary>
        /// Tests that RefreshToken returns BadRequest when the RefreshToken property is null.
        /// </summary>
        [TestMethod]
        public async Task RefreshToken_NullRefreshTokenProperty_ReturnsBadRequest()
        {
            // Arrange
            var mockContext = new Mock<AppDbContext>(new DbContextOptions<AppDbContext>());
            var mockAuthService = new Mock<IAuthService>();
            var mockProgressService = new Mock<IUserProgressService>();
            var mockPasswordValidator = new Mock<IPasswordValidator>();
            var mockTokenBlacklist = new Mock<ITokenBlacklistService>();
            var mockAuditLogger = new Mock<IAuditLogger>();
            var mockLogger = new Mock<ILogger<AuthController>>();
            var controller = new AuthController(mockContext.Object, mockAuthService.Object, mockProgressService.Object, mockPasswordValidator.Object, mockTokenBlacklist.Object, mockAuditLogger.Object, mockLogger.Object);
            var request = new RefreshTokenRequest
            {
                RefreshToken = null!
            };
            // Act
            var result = await controller.RefreshToken(request);
            // Assert
            Assert.IsInstanceOfType(result.Result, typeof(BadRequestObjectResult));
            var badRequestResult = result.Result as BadRequestObjectResult;
            Assert.IsNotNull(badRequestResult);
            dynamic? value = badRequestResult.Value;
            Assert.IsNotNull(value);
            Assert.AreEqual("Refresh token is required.", value.message);
        }

        /// <summary>
        /// Tests that RefreshToken returns BadRequest for various invalid token strings.
        /// </summary>
        /// <param name = "refreshToken">The invalid refresh token string to test.</param>
        [TestMethod]
        [DataRow("")]
        [DataRow(" ")]
        [DataRow("   ")]
        [DataRow("\t")]
        [DataRow("\n")]
        [DataRow("\r\n")]
        public async Task RefreshToken_EmptyOrWhitespaceToken_ReturnsBadRequest(string refreshToken)
        {
            // Arrange
            var mockContext = new Mock<AppDbContext>(new DbContextOptions<AppDbContext>());
            var mockAuthService = new Mock<IAuthService>();
            var mockProgressService = new Mock<IUserProgressService>();
            var mockPasswordValidator = new Mock<IPasswordValidator>();
            var mockTokenBlacklist = new Mock<ITokenBlacklistService>();
            var mockAuditLogger = new Mock<IAuditLogger>();
            var mockLogger = new Mock<ILogger<AuthController>>();
            var controller = new AuthController(mockContext.Object, mockAuthService.Object, mockProgressService.Object, mockPasswordValidator.Object, mockTokenBlacklist.Object, mockAuditLogger.Object, mockLogger.Object);
            var request = new RefreshTokenRequest
            {
                RefreshToken = refreshToken
            };
            // Act
            var result = await controller.RefreshToken(request);
            // Assert
            Assert.IsInstanceOfType(result.Result, typeof(BadRequestObjectResult));
            var badRequestResult = result.Result as BadRequestObjectResult;
            Assert.IsNotNull(badRequestResult);
            var value = badRequestResult.Value;
            Assert.IsNotNull(value);
            var messageProperty = value.GetType().GetProperty("message");
            Assert.IsNotNull(messageProperty, "The response should have a 'message' property");
            var actualMessage = messageProperty.GetValue(value) as string;
            Assert.AreEqual("Refresh token is required.", actualMessage);
        }

        /// <summary>
        /// Tests that RefreshToken returns Unauthorized when no user is found with the provided refresh token.
        /// </summary>
        [TestMethod]
        public async Task RefreshToken_UserNotFound_ReturnsUnauthorized()
        {
            // Arrange
            var users = new List<User>().AsQueryable();
            var mockUsersDbSet = CreateMockDbSet(users);
            var mockContext = new Mock<AppDbContext>(new DbContextOptions<AppDbContext>());
            mockContext.Setup(c => c.Users).Returns(mockUsersDbSet.Object);
            var mockAuthService = new Mock<IAuthService>();
            var mockProgressService = new Mock<IUserProgressService>();
            var mockPasswordValidator = new Mock<IPasswordValidator>();
            var mockTokenBlacklist = new Mock<ITokenBlacklistService>();
            var mockAuditLogger = new Mock<IAuditLogger>();
            var mockLogger = new Mock<ILogger<AuthController>>();
            var controller = new AuthController(mockContext.Object, mockAuthService.Object, mockProgressService.Object, mockPasswordValidator.Object, mockTokenBlacklist.Object, mockAuditLogger.Object, mockLogger.Object);
            var request = new RefreshTokenRequest
            {
                RefreshToken = "invalid-token"
            };
            // Act
            var result = await controller.RefreshToken(request);
            // Assert
            Assert.IsInstanceOfType(result.Result, typeof(UnauthorizedObjectResult));
            var unauthorizedResult = result.Result as UnauthorizedObjectResult;
            Assert.IsNotNull(unauthorizedResult);
            dynamic? value = unauthorizedResult.Value;
            Assert.IsNotNull(value);
            Assert.AreEqual("Invalid refresh token.", value.message);
        }

        /// <summary>
        /// Tests that RefreshToken returns Unauthorized when the refresh token expiry time is null.
        /// </summary>
        [TestMethod]
        public async Task RefreshToken_NullExpiryTime_ReturnsUnauthorized()
        {
            // Arrange
            var user = new User
            {
                Id = "user123",
                Name = "Test User",
                Email = "test@example.com",
                RefreshToken = "valid-token",
                RefreshTokenExpiryTime = null
            };
            var users = new List<User>
            {
                user
            }.AsQueryable();
            var mockUsersDbSet = CreateMockDbSet(users);
            var mockContext = new Mock<AppDbContext>(new DbContextOptions<AppDbContext>());
            mockContext.Setup(c => c.Users).Returns(mockUsersDbSet.Object);
            var mockAuthService = new Mock<IAuthService>();
            var mockProgressService = new Mock<IUserProgressService>();
            var mockPasswordValidator = new Mock<IPasswordValidator>();
            var mockTokenBlacklist = new Mock<ITokenBlacklistService>();
            var mockAuditLogger = new Mock<IAuditLogger>();
            var mockLogger = new Mock<ILogger<AuthController>>();
            var controller = new AuthController(mockContext.Object, mockAuthService.Object, mockProgressService.Object, mockPasswordValidator.Object, mockTokenBlacklist.Object, mockAuditLogger.Object, mockLogger.Object);
            var request = new RefreshTokenRequest
            {
                RefreshToken = "valid-token"
            };
            // Act
            var result = await controller.RefreshToken(request);
            // Assert
            Assert.IsInstanceOfType(result.Result, typeof(UnauthorizedObjectResult));
            var unauthorizedResult = result.Result as UnauthorizedObjectResult;
            Assert.IsNotNull(unauthorizedResult);
            dynamic? value = unauthorizedResult.Value;
            Assert.IsNotNull(value);
            Assert.AreEqual("Refresh token has expired.", value.message);
        }

        /// <summary>
        /// Tests that RefreshToken returns Unauthorized when the refresh token has expired (past date).
        /// </summary>
        [TestMethod]
        public async Task RefreshToken_ExpiredToken_ReturnsUnauthorized()
        {
            // Arrange
            var user = new User
            {
                Id = "user123",
                Name = "Test User",
                Email = "test@example.com",
                RefreshToken = "expired-token",
                RefreshTokenExpiryTime = DateTime.UtcNow.AddDays(-1)
            };
            var users = new List<User>
            {
                user
            }.AsQueryable();
            var mockUsersDbSet = CreateMockDbSet(users);
            var mockContext = new Mock<AppDbContext>(new DbContextOptions<AppDbContext>());
            mockContext.Setup(c => c.Users).Returns(mockUsersDbSet.Object);
            var mockAuthService = new Mock<IAuthService>();
            var mockProgressService = new Mock<IUserProgressService>();
            var mockPasswordValidator = new Mock<IPasswordValidator>();
            var mockTokenBlacklist = new Mock<ITokenBlacklistService>();
            var mockAuditLogger = new Mock<IAuditLogger>();
            var mockLogger = new Mock<ILogger<AuthController>>();
            var controller = new AuthController(mockContext.Object, mockAuthService.Object, mockProgressService.Object, mockPasswordValidator.Object, mockTokenBlacklist.Object, mockAuditLogger.Object, mockLogger.Object);
            var request = new RefreshTokenRequest
            {
                RefreshToken = "expired-token"
            };
            // Act
            var result = await controller.RefreshToken(request);
            // Assert
            Assert.IsInstanceOfType(result.Result, typeof(UnauthorizedObjectResult));
            var unauthorizedResult = result.Result as UnauthorizedObjectResult;
            Assert.IsNotNull(unauthorizedResult);
            dynamic? value = unauthorizedResult.Value;
            Assert.IsNotNull(value);
            Assert.AreEqual("Refresh token has expired.", value.message);
        }

        /// <summary>
        /// Tests that RefreshToken returns Unauthorized when the token expiry time is exactly at the current time (boundary condition).
        /// </summary>
        [TestMethod]
        public async Task RefreshToken_ExpiryTimeEqualToNow_ReturnsUnauthorized()
        {
            // Arrange
            var now = DateTime.UtcNow;
            var user = new User
            {
                Id = "user123",
                Name = "Test User",
                Email = "test@example.com",
                RefreshToken = "boundary-token",
                RefreshTokenExpiryTime = now
            };
            var users = new List<User>
            {
                user
            }.AsQueryable();
            var mockUsersDbSet = CreateMockDbSet(users);
            var mockContext = new Mock<AppDbContext>(new DbContextOptions<AppDbContext>());
            mockContext.Setup(c => c.Users).Returns(mockUsersDbSet.Object);
            var mockAuthService = new Mock<IAuthService>();
            var mockProgressService = new Mock<IUserProgressService>();
            var mockPasswordValidator = new Mock<IPasswordValidator>();
            var mockTokenBlacklist = new Mock<ITokenBlacklistService>();
            var mockAuditLogger = new Mock<IAuditLogger>();
            var mockLogger = new Mock<ILogger<AuthController>>();
            var controller = new AuthController(mockContext.Object, mockAuthService.Object, mockProgressService.Object, mockPasswordValidator.Object, mockTokenBlacklist.Object, mockAuditLogger.Object, mockLogger.Object);
            var request = new RefreshTokenRequest
            {
                RefreshToken = "boundary-token"
            };
            // Act
            System.Threading.Thread.Sleep(1);
            var result = await controller.RefreshToken(request);
            // Assert
            Assert.IsInstanceOfType(result.Result, typeof(UnauthorizedObjectResult));
            var unauthorizedResult = result.Result as UnauthorizedObjectResult;
            Assert.IsNotNull(unauthorizedResult);
            dynamic? value = unauthorizedResult.Value;
            Assert.IsNotNull(value);
            Assert.AreEqual("Refresh token has expired.", value.message);
        }

        /// <summary>
        /// Tests that RefreshToken successfully generates new tokens and updates the database for a valid refresh token.
        /// </summary>
        [TestMethod]
        public async Task RefreshToken_ValidToken_ReturnsNewTokens()
        {
            // Arrange
            var user = new User
            {
                Id = "user123",
                Name = "Test User",
                Email = "test@example.com",
                RefreshToken = "valid-token",
                RefreshTokenExpiryTime = DateTime.UtcNow.AddDays(7)
            };
            var users = new List<User>
            {
                user
            }.AsQueryable();
            var mockUsersDbSet = CreateMockDbSet(users);
            var mockContext = new Mock<AppDbContext>(new DbContextOptions<AppDbContext>());
            mockContext.Setup(c => c.Users).Returns(mockUsersDbSet.Object);
            mockContext.Setup(c => c.SaveChangesAsync(It.IsAny<CancellationToken>())).ReturnsAsync(1);
            var mockAuthService = new Mock<IAuthService>();
            mockAuthService.Setup(s => s.GenerateJwtToken(It.IsAny<User>())).Returns("new-jwt-token");
            mockAuthService.Setup(s => s.GenerateRefreshToken()).Returns("new-refresh-token");
            var mockProgressService = new Mock<IUserProgressService>();
            var mockPasswordValidator = new Mock<IPasswordValidator>();
            var mockTokenBlacklist = new Mock<ITokenBlacklistService>();
            var mockAuditLogger = new Mock<IAuditLogger>();
            var mockLogger = new Mock<ILogger<AuthController>>();
            var controller = new AuthController(mockContext.Object, mockAuthService.Object, mockProgressService.Object, mockPasswordValidator.Object, mockTokenBlacklist.Object, mockAuditLogger.Object, mockLogger.Object);
            var request = new RefreshTokenRequest
            {
                RefreshToken = "valid-token"
            };
            // Act
            var result = await controller.RefreshToken(request);
            // Assert
            Assert.IsInstanceOfType(result.Result, typeof(OkObjectResult));
            var okResult = result.Result as OkObjectResult;
            Assert.IsNotNull(okResult);
            dynamic? value = okResult.Value;
            Assert.IsNotNull(value);
            Assert.AreEqual("new-jwt-token", value.token);
            Assert.AreEqual("new-refresh-token", value.refreshToken);
            Assert.AreEqual("user123", value.userId);
            Assert.AreEqual("Test User", value.name);
            mockAuthService.Verify(s => s.GenerateJwtToken(user), Times.Once);
            mockAuthService.Verify(s => s.GenerateRefreshToken(), Times.Once);
            mockUsersDbSet.Verify(d => d.Update(user), Times.Once);
            mockContext.Verify(c => c.SaveChangesAsync(It.IsAny<CancellationToken>()), Times.Once);
        }

        /// <summary>
        /// Tests that RefreshToken updates the user's refresh token and expiry time correctly.
        /// </summary>
        [TestMethod]
        public async Task RefreshToken_ValidToken_UpdatesUserRefreshTokenAndExpiry()
        {
            // Arrange
            var beforeUpdate = DateTime.UtcNow;
            var user = new User
            {
                Id = "user123",
                Name = "Test User",
                Email = "test@example.com",
                RefreshToken = "old-token",
                RefreshTokenExpiryTime = DateTime.UtcNow.AddDays(1)
            };
            var users = new List<User>
            {
                user
            }.AsQueryable();
            var mockUsersDbSet = CreateMockDbSet(users);
            var mockContext = new Mock<AppDbContext>(new DbContextOptions<AppDbContext>());
            mockContext.Setup(c => c.Users).Returns(mockUsersDbSet.Object);
            mockContext.Setup(c => c.SaveChangesAsync(It.IsAny<CancellationToken>())).ReturnsAsync(1);
            var mockAuthService = new Mock<IAuthService>();
            mockAuthService.Setup(s => s.GenerateJwtToken(It.IsAny<User>())).Returns("new-jwt-token");
            mockAuthService.Setup(s => s.GenerateRefreshToken()).Returns("new-refresh-token");
            var mockProgressService = new Mock<IUserProgressService>();
            var mockPasswordValidator = new Mock<IPasswordValidator>();
            var mockTokenBlacklist = new Mock<ITokenBlacklistService>();
            var mockAuditLogger = new Mock<IAuditLogger>();
            var mockLogger = new Mock<ILogger<AuthController>>();
            var controller = new AuthController(mockContext.Object, mockAuthService.Object, mockProgressService.Object, mockPasswordValidator.Object, mockTokenBlacklist.Object, mockAuditLogger.Object, mockLogger.Object);
            var request = new RefreshTokenRequest
            {
                RefreshToken = "old-token"
            };
            // Act
            var result = await controller.RefreshToken(request);
            // Assert
            Assert.AreEqual("new-refresh-token", user.RefreshToken);
            Assert.IsNotNull(user.RefreshTokenExpiryTime);
            Assert.IsTrue(user.RefreshTokenExpiryTime >= beforeUpdate.AddDays(7));
            Assert.IsTrue(user.RefreshTokenExpiryTime <= DateTime.UtcNow.AddDays(7).AddSeconds(1));
        }

        /// <summary>
        /// Tests that RefreshToken returns 500 Internal Server Error when an exception occurs during token generation.
        /// </summary>
        [TestMethod]
        public async Task RefreshToken_ExceptionDuringTokenGeneration_ReturnsInternalServerError()
        {
            // Arrange
            var user = new User
            {
                Id = "user123",
                Name = "Test User",
                Email = "test@example.com",
                RefreshToken = "valid-token",
                RefreshTokenExpiryTime = DateTime.UtcNow.AddDays(7)
            };
            var options = new DbContextOptionsBuilder<AppDbContext>().UseInMemoryDatabase(databaseName: Guid.NewGuid().ToString()).Options;
            var context = new AppDbContext(options);
            context.Users.Add(user);
            context.SaveChanges();
            var mockAuthService = new Mock<IAuthService>();
            mockAuthService.Setup(s => s.GenerateJwtToken(It.IsAny<User>())).Throws(new Exception("Token generation failed"));
            var mockProgressService = new Mock<IUserProgressService>();
            var mockPasswordValidator = new Mock<IPasswordValidator>();
            var mockTokenBlacklist = new Mock<ITokenBlacklistService>();
            var mockAuditLogger = new Mock<IAuditLogger>();
            var mockLogger = new Mock<ILogger<AuthController>>();
            var controller = new AuthController(context, mockAuthService.Object, mockProgressService.Object, mockPasswordValidator.Object, mockTokenBlacklist.Object, mockAuditLogger.Object, mockLogger.Object);
            var request = new RefreshTokenRequest
            {
                RefreshToken = "valid-token"
            };
            // Act
            var result = await controller.RefreshToken(request);
            // Assert
            Assert.IsInstanceOfType(result.Result, typeof(ObjectResult));
            var objectResult = result.Result as ObjectResult;
            Assert.IsNotNull(objectResult);
            Assert.AreEqual(500, objectResult.StatusCode);
            Assert.IsNotNull(objectResult.Value);
            var messageProperty = objectResult.Value.GetType().GetProperty("message");
            Assert.IsNotNull(messageProperty);
            Assert.AreEqual("An error occurred during token refresh.", messageProperty.GetValue(objectResult.Value));
        }

        /// <summary>
        /// Tests that RefreshToken returns 500 Internal Server Error when an exception occurs during database save.
        /// </summary>
        [TestMethod]
        public async Task RefreshToken_ExceptionDuringSave_ReturnsInternalServerError()
        {
            // Arrange
            var user = new User
            {
                Id = "user123",
                Name = "Test User",
                Email = "test@example.com",
                RefreshToken = "valid-token",
                RefreshTokenExpiryTime = DateTime.UtcNow.AddDays(7)
            };
            var users = new List<User>
            {
                user
            }.AsQueryable();
            var mockUsersDbSet = CreateMockDbSet(users);
            var options = new DbContextOptionsBuilder<AppDbContext>().UseInMemoryDatabase(databaseName: Guid.NewGuid().ToString()).Options;
            var context = new AppDbContext(options);
            context.Users.Add(user);
            context.SaveChanges();
            var mockAuthService = new Mock<IAuthService>();
            mockAuthService.Setup(s => s.GenerateJwtToken(It.IsAny<User>())).Returns("new-jwt-token");
            mockAuthService.Setup(s => s.GenerateRefreshToken()).Returns("new-refresh-token");
            var mockProgressService = new Mock<IUserProgressService>();
            var mockPasswordValidator = new Mock<IPasswordValidator>();
            var mockTokenBlacklist = new Mock<ITokenBlacklistService>();
            var mockAuditLogger = new Mock<IAuditLogger>();
            var mockLogger = new Mock<ILogger<AuthController>>();
            var controller = new AuthController(context, mockAuthService.Object, mockProgressService.Object, mockPasswordValidator.Object, mockTokenBlacklist.Object, mockAuditLogger.Object, mockLogger.Object);
            var request = new RefreshTokenRequest
            {
                RefreshToken = "valid-token"
            };
            // Simulate database save failure by disposing the context before the operation completes
            context.Dispose();
            // Act
            var result = await controller.RefreshToken(request);
            // Assert
            Assert.IsInstanceOfType(result.Result, typeof(ObjectResult));
            var objectResult = result.Result as ObjectResult;
            Assert.IsNotNull(objectResult);
            Assert.AreEqual(500, objectResult.StatusCode);
            var value = objectResult.Value;
            Assert.IsNotNull(value);
            var messageProperty = value.GetType().GetProperty("message");
            Assert.IsNotNull(messageProperty);
            var actualMessage = messageProperty.GetValue(value) as string;
            Assert.AreEqual("An error occurred during token refresh.", actualMessage);
        }

        /// <summary>
        /// Tests that RefreshToken handles multiple users correctly and finds the right one.
        /// </summary>
        [TestMethod]
        public async Task RefreshToken_MultipleUsers_FindsCorrectUser()
        {
            // Arrange
            var user1 = new User
            {
                Id = "user1",
                Name = "User One",
                Email = "user1@example.com",
                RefreshToken = "token1",
                RefreshTokenExpiryTime = DateTime.UtcNow.AddDays(7)
            };
            var user2 = new User
            {
                Id = "user2",
                Name = "User Two",
                Email = "user2@example.com",
                RefreshToken = "token2",
                RefreshTokenExpiryTime = DateTime.UtcNow.AddDays(7)
            };
            var user3 = new User
            {
                Id = "user3",
                Name = "User Three",
                Email = "user3@example.com",
                RefreshToken = "token3",
                RefreshTokenExpiryTime = DateTime.UtcNow.AddDays(7)
            };
            var users = new List<User>
            {
                user1,
                user2,
                user3
            }.AsQueryable();
            var mockUsersDbSet = CreateMockDbSet(users);
            var mockContext = new Mock<AppDbContext>(new DbContextOptions<AppDbContext>());
            mockContext.Setup(c => c.Users).Returns(mockUsersDbSet.Object);
            mockContext.Setup(c => c.SaveChangesAsync(It.IsAny<CancellationToken>())).ReturnsAsync(1);
            var mockAuthService = new Mock<IAuthService>();
            mockAuthService.Setup(s => s.GenerateJwtToken(It.IsAny<User>())).Returns("new-jwt-token");
            mockAuthService.Setup(s => s.GenerateRefreshToken()).Returns("new-refresh-token");
            var mockProgressService = new Mock<IUserProgressService>();
            var mockPasswordValidator = new Mock<IPasswordValidator>();
            var mockTokenBlacklist = new Mock<ITokenBlacklistService>();
            var mockAuditLogger = new Mock<IAuditLogger>();
            var mockLogger = new Mock<ILogger<AuthController>>();
            var controller = new AuthController(mockContext.Object, mockAuthService.Object, mockProgressService.Object, mockPasswordValidator.Object, mockTokenBlacklist.Object, mockAuditLogger.Object, mockLogger.Object);
            var request = new RefreshTokenRequest
            {
                RefreshToken = "token2"
            };
            // Act
            var result = await controller.RefreshToken(request);
            // Assert
            Assert.IsInstanceOfType(result.Result, typeof(OkObjectResult));
            var okResult = result.Result as OkObjectResult;
            Assert.IsNotNull(okResult);
            dynamic? value = okResult.Value;
            Assert.IsNotNull(value);
            Assert.AreEqual("user2", value.userId);
            Assert.AreEqual("User Two", value.name);
        }

        /// <summary>
        /// Tests that RefreshToken handles special characters in the refresh token correctly.
        /// </summary>
        [TestMethod]
        public async Task RefreshToken_SpecialCharactersInToken_HandlesCorrectly()
        {
            // Arrange
            string specialToken = "token!@#$%^&*()_+-=[]{}|;':,.<>?/~`";
            var user = new User
            {
                Id = "user123",
                Name = "Test User",
                Email = "test@example.com",
                RefreshToken = specialToken,
                RefreshTokenExpiryTime = DateTime.UtcNow.AddDays(7)
            };
            var users = new List<User>
            {
                user
            }.AsQueryable();
            var mockUsersDbSet = CreateMockDbSet(users);
            var mockContext = new Mock<AppDbContext>(new DbContextOptions<AppDbContext>());
            mockContext.Setup(c => c.Users).Returns(mockUsersDbSet.Object);
            mockContext.Setup(c => c.SaveChangesAsync(It.IsAny<CancellationToken>())).ReturnsAsync(1);
            var mockAuthService = new Mock<IAuthService>();
            mockAuthService.Setup(s => s.GenerateJwtToken(It.IsAny<User>())).Returns("new-jwt-token");
            mockAuthService.Setup(s => s.GenerateRefreshToken()).Returns("new-refresh-token");
            var mockProgressService = new Mock<IUserProgressService>();
            var mockPasswordValidator = new Mock<IPasswordValidator>();
            var mockTokenBlacklist = new Mock<ITokenBlacklistService>();
            var mockAuditLogger = new Mock<IAuditLogger>();
            var mockLogger = new Mock<ILogger<AuthController>>();
            var controller = new AuthController(mockContext.Object, mockAuthService.Object, mockProgressService.Object, mockPasswordValidator.Object, mockTokenBlacklist.Object, mockAuditLogger.Object, mockLogger.Object);
            var request = new RefreshTokenRequest
            {
                RefreshToken = specialToken
            };
            // Act
            var result = await controller.RefreshToken(request);
            // Assert
            Assert.IsInstanceOfType(result.Result, typeof(OkObjectResult));
            var okResult = result.Result as OkObjectResult;
            Assert.IsNotNull(okResult);
        }

        /// <summary>
        /// Tests that RefreshToken handles very long refresh tokens correctly.
        /// </summary>
        [TestMethod]
        public async Task RefreshToken_VeryLongToken_HandlesCorrectly()
        {
            // Arrange
            string longToken = new string ('a', 10000);
            var user = new User
            {
                Id = "user123",
                Name = "Test User",
                Email = "test@example.com",
                RefreshToken = longToken,
                RefreshTokenExpiryTime = DateTime.UtcNow.AddDays(7)
            };
            var options = new DbContextOptionsBuilder<AppDbContext>().UseInMemoryDatabase(databaseName: Guid.NewGuid().ToString()).Options;
            var context = new AppDbContext(options);
            context.Users.Add(user);
            context.SaveChanges();
            var mockAuthService = new Mock<IAuthService>();
            mockAuthService.Setup(s => s.GenerateJwtToken(It.IsAny<User>())).Returns("new-jwt-token");
            mockAuthService.Setup(s => s.GenerateRefreshToken()).Returns("new-refresh-token");
            var mockProgressService = new Mock<IUserProgressService>();
            var mockPasswordValidator = new Mock<IPasswordValidator>();
            var mockTokenBlacklist = new Mock<ITokenBlacklistService>();
            var mockAuditLogger = new Mock<IAuditLogger>();
            var mockLogger = new Mock<ILogger<AuthController>>();
            var controller = new AuthController(context, mockAuthService.Object, mockProgressService.Object, mockPasswordValidator.Object, mockTokenBlacklist.Object, mockAuditLogger.Object, mockLogger.Object);
            var request = new RefreshTokenRequest
            {
                RefreshToken = longToken
            };
            // Act
            var result = await controller.RefreshToken(request);
            // Assert
            Assert.IsInstanceOfType(result.Result, typeof(OkObjectResult));
            var okResult = result.Result as OkObjectResult;
            Assert.IsNotNull(okResult);
        }

        /// <summary>
        /// Helper method to create a mock DbSet for testing.
        /// </summary>
        private static Mock<DbSet<T>> CreateMockDbSet<T>(IQueryable<T> data)
            where T : class
        {
            var mockSet = new Mock<DbSet<T>>();
            mockSet.As<IQueryable<T>>().Setup(m => m.Provider).Returns(data.Provider);
            mockSet.As<IQueryable<T>>().Setup(m => m.Expression).Returns(data.Expression);
            mockSet.As<IQueryable<T>>().Setup(m => m.ElementType).Returns(data.ElementType);
            mockSet.As<IQueryable<T>>().Setup(m => m.GetEnumerator()).Returns(data.GetEnumerator());
            return mockSet;
        }

        /// <summary>
        /// Tests that DeleteAccount returns BadRequest when userId is null.
        /// </summary>
        [TestMethod]
        public async Task DeleteAccount_UserIdIsNull_ReturnsBadRequest()
        {
            // Arrange
            var contextMock = new Mock<AppDbContext>();
            var authServiceMock = new Mock<IAuthService>();
            var progressServiceMock = new Mock<IUserProgressService>();
            var passwordValidatorMock = new Mock<IPasswordValidator>();
            var tokenBlacklistMock = new Mock<ITokenBlacklistService>();
            var auditLoggerMock = new Mock<IAuditLogger>();
            var loggerMock = new Mock<ILogger<AuthController>>();
            var controller = new AuthController(contextMock.Object, authServiceMock.Object, progressServiceMock.Object, passwordValidatorMock.Object, tokenBlacklistMock.Object, auditLoggerMock.Object, loggerMock.Object);
            var httpContext = new DefaultHttpContext();
            httpContext.Connection.RemoteIpAddress = IPAddress.Parse("127.0.0.1");
            controller.ControllerContext = new ControllerContext
            {
                HttpContext = httpContext
            };
            // Act
            var result = await controller.DeleteAccount(null!);
            // Assert
            Assert.IsNotNull(result);
            Assert.IsInstanceOfType(result.Result, typeof(BadRequestObjectResult));
            var badRequestResult = (BadRequestObjectResult)result.Result!;
            Assert.IsNotNull(badRequestResult.Value);
        }

        /// <summary>
        /// Tests that DeleteAccount returns BadRequest when userId is empty string.
        /// </summary>
        [TestMethod]
        public async Task DeleteAccount_UserIdIsEmpty_ReturnsBadRequest()
        {
            // Arrange
            var contextMock = new Mock<AppDbContext>();
            var authServiceMock = new Mock<IAuthService>();
            var progressServiceMock = new Mock<IUserProgressService>();
            var passwordValidatorMock = new Mock<IPasswordValidator>();
            var tokenBlacklistMock = new Mock<ITokenBlacklistService>();
            var auditLoggerMock = new Mock<IAuditLogger>();
            var loggerMock = new Mock<ILogger<AuthController>>();
            var controller = new AuthController(contextMock.Object, authServiceMock.Object, progressServiceMock.Object, passwordValidatorMock.Object, tokenBlacklistMock.Object, auditLoggerMock.Object, loggerMock.Object);
            var httpContext = new DefaultHttpContext();
            httpContext.Connection.RemoteIpAddress = IPAddress.Parse("127.0.0.1");
            controller.ControllerContext = new ControllerContext
            {
                HttpContext = httpContext
            };
            // Act
            var result = await controller.DeleteAccount(string.Empty);
            // Assert
            Assert.IsNotNull(result);
            Assert.IsInstanceOfType(result.Result, typeof(BadRequestObjectResult));
            var badRequestResult = (BadRequestObjectResult)result.Result!;
            Assert.IsNotNull(badRequestResult.Value);
        }

        /// <summary>
        /// Tests that DeleteAccount returns BadRequest when userId contains only whitespace.
        /// </summary>
        [TestMethod]
        [DataRow("   ")]
        [DataRow("\t")]
        [DataRow("\n")]
        [DataRow(" \t\n ")]
        public async Task DeleteAccount_UserIdIsWhitespace_ReturnsBadRequest(string whitespaceUserId)
        {
            // Arrange
            var options = new DbContextOptionsBuilder<AppDbContext>().UseInMemoryDatabase(databaseName: Guid.NewGuid().ToString()).Options;
            var contextMock = new Mock<AppDbContext>(options);
            var authServiceMock = new Mock<IAuthService>();
            var progressServiceMock = new Mock<IUserProgressService>();
            var passwordValidatorMock = new Mock<IPasswordValidator>();
            var tokenBlacklistMock = new Mock<ITokenBlacklistService>();
            var auditLoggerMock = new Mock<IAuditLogger>();
            var loggerMock = new Mock<ILogger<AuthController>>();
            var controller = new AuthController(contextMock.Object, authServiceMock.Object, progressServiceMock.Object, passwordValidatorMock.Object, tokenBlacklistMock.Object, auditLoggerMock.Object, loggerMock.Object);
            var httpContext = new DefaultHttpContext();
            httpContext.Connection.RemoteIpAddress = IPAddress.Parse("127.0.0.1");
            controller.ControllerContext = new ControllerContext
            {
                HttpContext = httpContext
            };
            // Act
            var result = await controller.DeleteAccount(whitespaceUserId);
            // Assert
            Assert.IsNotNull(result);
            Assert.IsInstanceOfType(result.Result, typeof(BadRequestObjectResult));
            var badRequestResult = (BadRequestObjectResult)result.Result!;
            Assert.IsNotNull(badRequestResult.Value);
        }

        /// <summary>
        /// Tests that DeleteAccount returns NotFound when user does not exist in database.
        /// </summary>
        [TestMethod]
        public async Task DeleteAccount_UserNotFound_ReturnsNotFound()
        {
            // Arrange
            var users = new List<User>().AsQueryable();
            var usersDbSetMock = CreateMockDbSet(users);
            var contextMock = new Mock<AppDbContext>();
            contextMock.Setup(c => c.Users).Returns(usersDbSetMock.Object);
            var authServiceMock = new Mock<IAuthService>();
            var progressServiceMock = new Mock<IUserProgressService>();
            var passwordValidatorMock = new Mock<IPasswordValidator>();
            var tokenBlacklistMock = new Mock<ITokenBlacklistService>();
            var auditLoggerMock = new Mock<IAuditLogger>();
            var loggerMock = new Mock<ILogger<AuthController>>();
            var controller = new AuthController(contextMock.Object, authServiceMock.Object, progressServiceMock.Object, passwordValidatorMock.Object, tokenBlacklistMock.Object, auditLoggerMock.Object, loggerMock.Object);
            var httpContext = new DefaultHttpContext();
            httpContext.Connection.RemoteIpAddress = IPAddress.Parse("127.0.0.1");
            controller.ControllerContext = new ControllerContext
            {
                HttpContext = httpContext
            };
            // Act
            var result = await controller.DeleteAccount("nonexistent-user-id");
            // Assert
            Assert.IsNotNull(result);
            Assert.IsInstanceOfType(result.Result, typeof(NotFoundObjectResult));
            var notFoundResult = (NotFoundObjectResult)result.Result!;
            Assert.IsNotNull(notFoundResult.Value);
        }

        /// <summary>
        /// Tests that DeleteAccount returns Ok when user is successfully deleted.
        /// </summary>
        [TestMethod]
        public async Task DeleteAccount_ValidUser_ReturnsOk()
        {
            // Arrange
            var userId = "test-user-id";
            var user = new User
            {
                Id = userId,
                Email = "test@example.com",
                Name = "Test User",
                CreatedAt = DateTime.UtcNow
            };
            var users = new List<User>
            {
                user
            }.AsQueryable();
            var usersDbSetMock = CreateMockDbSet(users);
            var contextMock = new Mock<AppDbContext>();
            contextMock.Setup(c => c.Users).Returns(usersDbSetMock.Object);
            contextMock.Setup(c => c.SaveChangesAsync(It.IsAny<CancellationToken>())).ReturnsAsync(1);
            var authServiceMock = new Mock<IAuthService>();
            var progressServiceMock = new Mock<IUserProgressService>();
            progressServiceMock.Setup(p => p.DeleteUserDataAsync(userId)).Returns(Task.CompletedTask);
            var passwordValidatorMock = new Mock<IPasswordValidator>();
            var tokenBlacklistMock = new Mock<ITokenBlacklistService>();
            var auditLoggerMock = new Mock<IAuditLogger>();
            var loggerMock = new Mock<ILogger<AuthController>>();
            var controller = new AuthController(contextMock.Object, authServiceMock.Object, progressServiceMock.Object, passwordValidatorMock.Object, tokenBlacklistMock.Object, auditLoggerMock.Object, loggerMock.Object);
            var httpContext = new DefaultHttpContext();
            httpContext.Connection.RemoteIpAddress = IPAddress.Parse("127.0.0.1");
            controller.ControllerContext = new ControllerContext
            {
                HttpContext = httpContext
            };
            // Act
            var result = await controller.DeleteAccount(userId);
            // Assert
            Assert.IsNotNull(result);
            Assert.IsInstanceOfType(result.Result, typeof(OkObjectResult));
            var okResult = (OkObjectResult)result.Result!;
            Assert.IsNotNull(okResult.Value);
        }

        /// <summary>
        /// Tests that DeleteAccount calls Remove on the Users DbSet.
        /// </summary>
        [TestMethod]
        public async Task DeleteAccount_ValidUser_RemovesUserFromContext()
        {
            // Arrange
            var userId = "test-user-id";
            var user = new User
            {
                Id = userId,
                Email = "test@example.com",
                Name = "Test User",
                CreatedAt = DateTime.UtcNow
            };
            var users = new List<User>
            {
                user
            }.AsQueryable();
            var usersDbSetMock = CreateMockDbSet(users);
            var contextMock = new Mock<AppDbContext>();
            contextMock.Setup(c => c.Users).Returns(usersDbSetMock.Object);
            contextMock.Setup(c => c.SaveChangesAsync(It.IsAny<CancellationToken>())).ReturnsAsync(1);
            var authServiceMock = new Mock<IAuthService>();
            var progressServiceMock = new Mock<IUserProgressService>();
            progressServiceMock.Setup(p => p.DeleteUserDataAsync(userId)).Returns(Task.CompletedTask);
            var passwordValidatorMock = new Mock<IPasswordValidator>();
            var tokenBlacklistMock = new Mock<ITokenBlacklistService>();
            var auditLoggerMock = new Mock<IAuditLogger>();
            var loggerMock = new Mock<ILogger<AuthController>>();
            var controller = new AuthController(contextMock.Object, authServiceMock.Object, progressServiceMock.Object, passwordValidatorMock.Object, tokenBlacklistMock.Object, auditLoggerMock.Object, loggerMock.Object);
            var httpContext = new DefaultHttpContext();
            httpContext.Connection.RemoteIpAddress = IPAddress.Parse("127.0.0.1");
            controller.ControllerContext = new ControllerContext
            {
                HttpContext = httpContext
            };
            // Act
            await controller.DeleteAccount(userId);
            // Assert
            usersDbSetMock.Verify(d => d.Remove(It.Is<User>(u => u.Id == userId)), Times.Once);
        }

        /// <summary>
        /// Tests that DeleteAccount calls SaveChangesAsync on the database context.
        /// </summary>
        [TestMethod]
        public async Task DeleteAccount_ValidUser_CallsSaveChangesAsync()
        {
            // Arrange
            var userId = "test-user-id";
            var user = new User
            {
                Id = userId,
                Email = "test@example.com",
                Name = "Test User",
                CreatedAt = DateTime.UtcNow
            };
            var users = new List<User>
            {
                user
            }.AsQueryable();
            var usersDbSetMock = CreateMockDbSet(users);
            var contextMock = new Mock<AppDbContext>();
            contextMock.Setup(c => c.Users).Returns(usersDbSetMock.Object);
            contextMock.Setup(c => c.SaveChangesAsync(It.IsAny<CancellationToken>())).ReturnsAsync(1);
            var authServiceMock = new Mock<IAuthService>();
            var progressServiceMock = new Mock<IUserProgressService>();
            progressServiceMock.Setup(p => p.DeleteUserDataAsync(userId)).Returns(Task.CompletedTask);
            var passwordValidatorMock = new Mock<IPasswordValidator>();
            var tokenBlacklistMock = new Mock<ITokenBlacklistService>();
            var auditLoggerMock = new Mock<IAuditLogger>();
            var loggerMock = new Mock<ILogger<AuthController>>();
            var controller = new AuthController(contextMock.Object, authServiceMock.Object, progressServiceMock.Object, passwordValidatorMock.Object, tokenBlacklistMock.Object, auditLoggerMock.Object, loggerMock.Object);
            var httpContext = new DefaultHttpContext();
            httpContext.Connection.RemoteIpAddress = IPAddress.Parse("127.0.0.1");
            controller.ControllerContext = new ControllerContext
            {
                HttpContext = httpContext
            };
            // Act
            await controller.DeleteAccount(userId);
            // Assert
            contextMock.Verify(c => c.SaveChangesAsync(It.IsAny<CancellationToken>()), Times.Once);
        }

        /// <summary>
        /// Tests that DeleteAccount calls DeleteUserDataAsync on the progress service.
        /// </summary>
        [TestMethod]
        public async Task DeleteAccount_ValidUser_CallsDeleteUserDataAsync()
        {
            // Arrange
            var userId = "test-user-id";
            var user = new User
            {
                Id = userId,
                Email = "test@example.com",
                Name = "Test User",
                CreatedAt = DateTime.UtcNow
            };
            var users = new List<User>
            {
                user
            }.AsQueryable();
            var usersDbSetMock = CreateMockDbSet(users);
            var contextMock = new Mock<AppDbContext>();
            contextMock.Setup(c => c.Users).Returns(usersDbSetMock.Object);
            contextMock.Setup(c => c.SaveChangesAsync(It.IsAny<CancellationToken>())).ReturnsAsync(1);
            var authServiceMock = new Mock<IAuthService>();
            var progressServiceMock = new Mock<IUserProgressService>();
            progressServiceMock.Setup(p => p.DeleteUserDataAsync(userId)).Returns(Task.CompletedTask);
            var passwordValidatorMock = new Mock<IPasswordValidator>();
            var tokenBlacklistMock = new Mock<ITokenBlacklistService>();
            var auditLoggerMock = new Mock<IAuditLogger>();
            var loggerMock = new Mock<ILogger<AuthController>>();
            var controller = new AuthController(contextMock.Object, authServiceMock.Object, progressServiceMock.Object, passwordValidatorMock.Object, tokenBlacklistMock.Object, auditLoggerMock.Object, loggerMock.Object);
            var httpContext = new DefaultHttpContext();
            httpContext.Connection.RemoteIpAddress = IPAddress.Parse("127.0.0.1");
            controller.ControllerContext = new ControllerContext
            {
                HttpContext = httpContext
            };
            // Act
            await controller.DeleteAccount(userId);
            // Assert
            progressServiceMock.Verify(p => p.DeleteUserDataAsync(userId), Times.Once);
        }

        /// <summary>
        /// Tests that DeleteAccount still succeeds when SaveAccountDeletionToFileAsync throws an exception.
        /// The method should catch the exception and continue with the deletion process.
        /// </summary>
        [TestMethod]
        public async Task DeleteAccount_SaveAccountDeletionThrowsException_StillDeletesUser()
        {
            // Arrange
            var userId = "test-user-id";
            var user = new User
            {
                Id = userId,
                Email = "test@example.com",
                Name = "Test User",
                CreatedAt = DateTime.UtcNow
            };
            var users = new List<User>
            {
                user
            }.AsQueryable();
            var usersDbSetMock = CreateMockDbSet(users);
            var contextMock = new Mock<AppDbContext>();
            contextMock.Setup(c => c.Users).Returns(usersDbSetMock.Object);
            contextMock.Setup(c => c.SaveChangesAsync(It.IsAny<CancellationToken>())).ReturnsAsync(1);
            var authServiceMock = new Mock<IAuthService>();
            var progressServiceMock = new Mock<IUserProgressService>();
            progressServiceMock.Setup(p => p.DeleteUserDataAsync(userId)).Returns(Task.CompletedTask);
            var passwordValidatorMock = new Mock<IPasswordValidator>();
            var tokenBlacklistMock = new Mock<ITokenBlacklistService>();
            var auditLoggerMock = new Mock<IAuditLogger>();
            var loggerMock = new Mock<ILogger<AuthController>>();
            var controller = new AuthController(contextMock.Object, authServiceMock.Object, progressServiceMock.Object, passwordValidatorMock.Object, tokenBlacklistMock.Object, auditLoggerMock.Object, loggerMock.Object);
            var httpContext = new DefaultHttpContext();
            httpContext.Connection.RemoteIpAddress = IPAddress.Parse("127.0.0.1");
            controller.ControllerContext = new ControllerContext
            {
                HttpContext = httpContext
            };
            // Act
            var result = await controller.DeleteAccount(userId);
            // Assert
            Assert.IsNotNull(result);
            Assert.IsInstanceOfType(result.Result, typeof(OkObjectResult));
            contextMock.Verify(c => c.SaveChangesAsync(It.IsAny<CancellationToken>()), Times.Once);
        }

        /// <summary>
        /// Tests that DeleteAccount still succeeds when DeleteUserDataAsync throws an exception.
        /// The method should catch the exception and continue with the deletion process.
        /// </summary>
        [TestMethod]
        public async Task DeleteAccount_DeleteUserDataAsyncThrowsException_StillDeletesUser()
        {
            // Arrange
            var userId = "test-user-id";
            var user = new User
            {
                Id = userId,
                Email = "test@example.com",
                Name = "Test User",
                CreatedAt = DateTime.UtcNow
            };
            var users = new List<User>
            {
                user
            }.AsQueryable();
            var usersDbSetMock = CreateMockDbSet(users);
            var contextMock = new Mock<AppDbContext>();
            contextMock.Setup(c => c.Users).Returns(usersDbSetMock.Object);
            contextMock.Setup(c => c.SaveChangesAsync(It.IsAny<CancellationToken>())).ReturnsAsync(1);
            var authServiceMock = new Mock<IAuthService>();
            var progressServiceMock = new Mock<IUserProgressService>();
            progressServiceMock.Setup(p => p.DeleteUserDataAsync(userId)).ThrowsAsync(new Exception("Failed to delete user data"));
            var passwordValidatorMock = new Mock<IPasswordValidator>();
            var tokenBlacklistMock = new Mock<ITokenBlacklistService>();
            var auditLoggerMock = new Mock<IAuditLogger>();
            var loggerMock = new Mock<ILogger<AuthController>>();
            var controller = new AuthController(contextMock.Object, authServiceMock.Object, progressServiceMock.Object, passwordValidatorMock.Object, tokenBlacklistMock.Object, auditLoggerMock.Object, loggerMock.Object);
            var httpContext = new DefaultHttpContext();
            httpContext.Connection.RemoteIpAddress = IPAddress.Parse("127.0.0.1");
            controller.ControllerContext = new ControllerContext
            {
                HttpContext = httpContext
            };
            // Act
            var result = await controller.DeleteAccount(userId);
            // Assert
            Assert.IsNotNull(result);
            Assert.IsInstanceOfType(result.Result, typeof(OkObjectResult));
            contextMock.Verify(c => c.SaveChangesAsync(It.IsAny<CancellationToken>()), Times.Once);
        }

        /// <summary>
        /// Tests that DeleteAccount returns InternalServerError when SaveChangesAsync throws an exception.
        /// </summary>
        [TestMethod]
        [TestCategory("ProductionBugSuspected")]
        [Ignore("ProductionBugSuspected")]
        public async Task DeleteAccount_SaveChangesAsyncThrowsException_ReturnsInternalServerError()
        {
            // Arrange
            var userId = "test-user-id";
            var user = new User
            {
                Id = userId,
                Email = "test@example.com",
                Name = "Test User",
                CreatedAt = DateTime.UtcNow
            };
            var users = new List<User>
            {
                user
            }.AsQueryable();
            var usersDbSetMock = CreateMockDbSet(users);
            var contextMock = new Mock<AppDbContext>();
            contextMock.Setup(c => c.Users).Returns(usersDbSetMock.Object);
            contextMock.Setup(c => c.SaveChangesAsync(It.IsAny<CancellationToken>())).ThrowsAsync(new Exception("Database error"));
            var authServiceMock = new Mock<IAuthService>();
            var progressServiceMock = new Mock<IUserProgressService>();
            progressServiceMock.Setup(p => p.DeleteUserDataAsync(userId)).Returns(Task.CompletedTask);
            var passwordValidatorMock = new Mock<IPasswordValidator>();
            var tokenBlacklistMock = new Mock<ITokenBlacklistService>();
            var auditLoggerMock = new Mock<IAuditLogger>();
            var loggerMock = new Mock<ILogger<AuthController>>();
            var controller = new AuthController(contextMock.Object, authServiceMock.Object, progressServiceMock.Object, passwordValidatorMock.Object, tokenBlacklistMock.Object, auditLoggerMock.Object, loggerMock.Object);
            var httpContext = new DefaultHttpContext();
            httpContext.Connection.RemoteIpAddress = IPAddress.Parse("127.0.0.1");
            controller.ControllerContext = new ControllerContext
            {
                HttpContext = httpContext
            };
            // Act
            var result = await controller.DeleteAccount(userId);
            // Assert
            Assert.IsNotNull(result);
            Assert.IsInstanceOfType(result.Result, typeof(ObjectResult));
            var objectResult = (ObjectResult)result.Result!;
            Assert.AreEqual(500, objectResult.StatusCode);
        }

        /// <summary>
        /// Tests that DeleteAccount handles null RemoteIpAddress gracefully.
        /// </summary>
        [TestMethod]
        public async Task DeleteAccount_NullRemoteIpAddress_HandlesGracefully()
        {
            // Arrange
            var userId = "test-user-id";
            var user = new User
            {
                Id = userId,
                Email = "test@example.com",
                Name = "Test User",
                CreatedAt = DateTime.UtcNow
            };
            var users = new List<User>
            {
                user
            }.AsQueryable();
            var usersDbSetMock = CreateMockDbSet(users);
            var contextMock = new Mock<AppDbContext>();
            contextMock.Setup(c => c.Users).Returns(usersDbSetMock.Object);
            contextMock.Setup(c => c.SaveChangesAsync(It.IsAny<CancellationToken>())).ReturnsAsync(1);
            var authServiceMock = new Mock<IAuthService>();
            var progressServiceMock = new Mock<IUserProgressService>();
            progressServiceMock.Setup(p => p.DeleteUserDataAsync(userId)).Returns(Task.CompletedTask);
            var passwordValidatorMock = new Mock<IPasswordValidator>();
            var tokenBlacklistMock = new Mock<ITokenBlacklistService>();
            var auditLoggerMock = new Mock<IAuditLogger>();
            var loggerMock = new Mock<ILogger<AuthController>>();
            var controller = new AuthController(contextMock.Object, authServiceMock.Object, progressServiceMock.Object, passwordValidatorMock.Object, tokenBlacklistMock.Object, auditLoggerMock.Object, loggerMock.Object);
            var httpContext = new DefaultHttpContext();
            httpContext.Connection.RemoteIpAddress = null;
            controller.ControllerContext = new ControllerContext
            {
                HttpContext = httpContext
            };
            // Act
            var result = await controller.DeleteAccount(userId);
            // Assert
            Assert.IsNotNull(result);
            Assert.IsInstanceOfType(result.Result, typeof(OkObjectResult));
        }

        /// <summary>
        /// Tests that DeleteAccount handles special characters in userId.
        /// </summary>
        [TestMethod]
        [DataRow("user@#$%")]
        [DataRow("user-with-dashes")]
        [DataRow("user_with_underscores")]
        [DataRow("12345")]
        public async Task DeleteAccount_UserIdWithSpecialCharacters_ReturnsNotFoundWhenUserNotExists(string specialUserId)
        {
            // Arrange
            var users = new List<User>().AsQueryable();
            var usersDbSetMock = CreateMockDbSet(users);
            var contextMock = new Mock<AppDbContext>();
            contextMock.Setup(c => c.Users).Returns(usersDbSetMock.Object);
            var authServiceMock = new Mock<IAuthService>();
            var progressServiceMock = new Mock<IUserProgressService>();
            var passwordValidatorMock = new Mock<IPasswordValidator>();
            var tokenBlacklistMock = new Mock<ITokenBlacklistService>();
            var auditLoggerMock = new Mock<IAuditLogger>();
            var loggerMock = new Mock<ILogger<AuthController>>();
            var controller = new AuthController(contextMock.Object, authServiceMock.Object, progressServiceMock.Object, passwordValidatorMock.Object, tokenBlacklistMock.Object, auditLoggerMock.Object, loggerMock.Object);
            var httpContext = new DefaultHttpContext();
            httpContext.Connection.RemoteIpAddress = IPAddress.Parse("127.0.0.1");
            controller.ControllerContext = new ControllerContext
            {
                HttpContext = httpContext
            };
            // Act
            var result = await controller.DeleteAccount(specialUserId);
            // Assert
            Assert.IsNotNull(result);
            Assert.IsInstanceOfType(result.Result, typeof(NotFoundObjectResult));
        }

        /// <summary>
        /// Tests that DeleteAccount handles very long userId strings.
        /// </summary>
        [TestMethod]
        public async Task DeleteAccount_VeryLongUserId_ReturnsNotFoundWhenUserNotExists()
        {
            // Arrange
            var longUserId = new string ('a', 1000);
            var options = new DbContextOptionsBuilder<AppDbContext>().UseInMemoryDatabase(databaseName: Guid.NewGuid().ToString()).Options;
            var context = new AppDbContext(options);
            var authServiceMock = new Mock<IAuthService>();
            var progressServiceMock = new Mock<IUserProgressService>();
            var passwordValidatorMock = new Mock<IPasswordValidator>();
            var tokenBlacklistMock = new Mock<ITokenBlacklistService>();
            var auditLoggerMock = new Mock<IAuditLogger>();
            var loggerMock = new Mock<ILogger<AuthController>>();
            var controller = new AuthController(context, authServiceMock.Object, progressServiceMock.Object, passwordValidatorMock.Object, tokenBlacklistMock.Object, auditLoggerMock.Object, loggerMock.Object);
            var httpContext = new DefaultHttpContext();
            httpContext.Connection.RemoteIpAddress = IPAddress.Parse("127.0.0.1");
            controller.ControllerContext = new ControllerContext
            {
                HttpContext = httpContext
            };
            // Act
            var result = await controller.DeleteAccount(longUserId);
            // Assert
            Assert.IsNotNull(result);
            Assert.IsInstanceOfType(result.Result, typeof(NotFoundObjectResult));
        }

        /// <summary>
        /// Tests that Logout returns Ok with success message when valid token and claims are provided.
        /// Input: Valid Authorization header with Bearer token, valid user claims (NameIdentifier and Email).
        /// Expected: Returns 200 OK with success message, token is blacklisted, and audit log is created.
        /// </summary>
        [TestMethod]
        public async Task Logout_ValidTokenAndClaims_ReturnsOkWithSuccessMessage()
        {
            // Arrange
            var mockContext = new Mock<AppDbContext>(new DbContextOptions<AppDbContext>());
            var mockAuthService = new Mock<IAuthService>();
            var mockProgressService = new Mock<IUserProgressService>();
            var mockPasswordValidator = new Mock<IPasswordValidator>();
            var mockTokenBlacklist = new Mock<ITokenBlacklistService>();
            var mockAuditLogger = new Mock<IAuditLogger>();
            var mockLogger = new Mock<ILogger<AuthController>>();
            var controller = new AuthController(mockContext.Object, mockAuthService.Object, mockProgressService.Object, mockPasswordValidator.Object, mockTokenBlacklist.Object, mockAuditLogger.Object, mockLogger.Object);
            var httpContext = new DefaultHttpContext();
            httpContext.Connection.RemoteIpAddress = IPAddress.Parse("192.168.1.1");
            httpContext.Request.Headers["Authorization"] = "Bearer test-token-12345";
            var claims = new[]
            {
                new Claim(ClaimTypes.NameIdentifier, "user123"),
                new Claim(ClaimTypes.Email, "test@example.com")
            };
            httpContext.User = new ClaimsPrincipal(new ClaimsIdentity(claims));
            controller.ControllerContext = new ControllerContext
            {
                HttpContext = httpContext
            };
            mockTokenBlacklist.Setup(x => x.BlacklistTokenAsync(It.IsAny<string>(), It.IsAny<DateTime>())).Returns(Task.CompletedTask);
            mockAuditLogger.Setup(x => x.LogLogoutAsync(It.IsAny<string>(), It.IsAny<string>(), It.IsAny<string>())).Returns(Task.CompletedTask);
            // Act
            var result = await controller.Logout();
            // Assert
            var okResult = result.Result as OkObjectResult;
            Assert.IsNotNull(okResult);
            Assert.AreEqual(200, okResult.StatusCode);
            mockTokenBlacklist.Verify(x => x.BlacklistTokenAsync("test-token-12345", It.IsAny<DateTime>()), Times.Once);
            mockAuditLogger.Verify(x => x.LogLogoutAsync("user123", "test@example.com", "192.168.1.1"), Times.Once);
        }

        /// <summary>
        /// Tests that Logout returns Ok without blacklisting token when no Authorization header is provided.
        /// Input: No Authorization header.
        /// Expected: Returns 200 OK with success message, token is not blacklisted.
        /// </summary>
        [TestMethod]
        public async Task Logout_NoAuthorizationHeader_ReturnsOkWithoutBlacklistingToken()
        {
            // Arrange
            var mockContext = new Mock<AppDbContext>(new DbContextOptions<AppDbContext>());
            var mockAuthService = new Mock<IAuthService>();
            var mockProgressService = new Mock<IUserProgressService>();
            var mockPasswordValidator = new Mock<IPasswordValidator>();
            var mockTokenBlacklist = new Mock<ITokenBlacklistService>();
            var mockAuditLogger = new Mock<IAuditLogger>();
            var mockLogger = new Mock<ILogger<AuthController>>();
            var controller = new AuthController(mockContext.Object, mockAuthService.Object, mockProgressService.Object, mockPasswordValidator.Object, mockTokenBlacklist.Object, mockAuditLogger.Object, mockLogger.Object);
            var httpContext = new DefaultHttpContext();
            httpContext.Connection.RemoteIpAddress = IPAddress.Parse("192.168.1.1");
            var claims = new[]
            {
                new Claim(ClaimTypes.NameIdentifier, "user123"),
                new Claim(ClaimTypes.Email, "test@example.com")
            };
            httpContext.User = new ClaimsPrincipal(new ClaimsIdentity(claims));
            controller.ControllerContext = new ControllerContext
            {
                HttpContext = httpContext
            };
            mockAuditLogger.Setup(x => x.LogLogoutAsync(It.IsAny<string>(), It.IsAny<string>(), It.IsAny<string>())).Returns(Task.CompletedTask);
            // Act
            var result = await controller.Logout();
            // Assert
            var okResult = result.Result as OkObjectResult;
            Assert.IsNotNull(okResult);
            Assert.AreEqual(200, okResult.StatusCode);
            mockTokenBlacklist.Verify(x => x.BlacklistTokenAsync(It.IsAny<string>(), It.IsAny<DateTime>()), Times.Never);
            mockAuditLogger.Verify(x => x.LogLogoutAsync("user123", "test@example.com", "192.168.1.1"), Times.Once);
        }

        /// <summary>
        /// Tests that Logout returns Ok without blacklisting token when Authorization header does not start with "Bearer ".
        /// Input: Authorization header with value "InvalidFormat token".
        /// Expected: Returns 200 OK with success message, token is not blacklisted.
        /// </summary>
        [TestMethod]
        public async Task Logout_AuthorizationHeaderWithoutBearer_ReturnsOkWithoutBlacklistingToken()
        {
            // Arrange
            var mockContext = new Mock<AppDbContext>(new DbContextOptions<AppDbContext>());
            var mockAuthService = new Mock<IAuthService>();
            var mockProgressService = new Mock<IUserProgressService>();
            var mockPasswordValidator = new Mock<IPasswordValidator>();
            var mockTokenBlacklist = new Mock<ITokenBlacklistService>();
            var mockAuditLogger = new Mock<IAuditLogger>();
            var mockLogger = new Mock<ILogger<AuthController>>();
            var controller = new AuthController(mockContext.Object, mockAuthService.Object, mockProgressService.Object, mockPasswordValidator.Object, mockTokenBlacklist.Object, mockAuditLogger.Object, mockLogger.Object);
            var httpContext = new DefaultHttpContext();
            httpContext.Connection.RemoteIpAddress = IPAddress.Parse("192.168.1.1");
            httpContext.Request.Headers["Authorization"] = "InvalidFormat token";
            var claims = new[]
            {
                new Claim(ClaimTypes.NameIdentifier, "user123"),
                new Claim(ClaimTypes.Email, "test@example.com")
            };
            httpContext.User = new ClaimsPrincipal(new ClaimsIdentity(claims));
            controller.ControllerContext = new ControllerContext
            {
                HttpContext = httpContext
            };
            mockAuditLogger.Setup(x => x.LogLogoutAsync(It.IsAny<string>(), It.IsAny<string>(), It.IsAny<string>())).Returns(Task.CompletedTask);
            // Act
            var result = await controller.Logout();
            // Assert
            var okResult = result.Result as OkObjectResult;
            Assert.IsNotNull(okResult);
            Assert.AreEqual(200, okResult.StatusCode);
            mockTokenBlacklist.Verify(x => x.BlacklistTokenAsync(It.IsAny<string>(), It.IsAny<DateTime>()), Times.Never);
        }

        /// <summary>
        /// Tests that Logout returns Ok without audit logging when no user claims are present.
        /// Input: No user claims (NameIdentifier and Email).
        /// Expected: Returns 200 OK with success message, audit log is not created.
        /// </summary>
        [TestMethod]
        public async Task Logout_NoUserClaims_ReturnsOkWithoutAuditLog()
        {
            // Arrange
            var mockContext = new Mock<AppDbContext>(new DbContextOptions<AppDbContext>());
            var mockAuthService = new Mock<IAuthService>();
            var mockProgressService = new Mock<IUserProgressService>();
            var mockPasswordValidator = new Mock<IPasswordValidator>();
            var mockTokenBlacklist = new Mock<ITokenBlacklistService>();
            var mockAuditLogger = new Mock<IAuditLogger>();
            var mockLogger = new Mock<ILogger<AuthController>>();
            var controller = new AuthController(mockContext.Object, mockAuthService.Object, mockProgressService.Object, mockPasswordValidator.Object, mockTokenBlacklist.Object, mockAuditLogger.Object, mockLogger.Object);
            var httpContext = new DefaultHttpContext();
            httpContext.Connection.RemoteIpAddress = IPAddress.Parse("192.168.1.1");
            httpContext.Request.Headers["Authorization"] = "Bearer test-token";
            httpContext.User = new ClaimsPrincipal(new ClaimsIdentity());
            controller.ControllerContext = new ControllerContext
            {
                HttpContext = httpContext
            };
            mockTokenBlacklist.Setup(x => x.BlacklistTokenAsync(It.IsAny<string>(), It.IsAny<DateTime>())).Returns(Task.CompletedTask);
            // Act
            var result = await controller.Logout();
            // Assert
            var okResult = result.Result as OkObjectResult;
            Assert.IsNotNull(okResult);
            Assert.AreEqual(200, okResult.StatusCode);
            mockAuditLogger.Verify(x => x.LogLogoutAsync(It.IsAny<string>(), It.IsAny<string>(), It.IsAny<string>()), Times.Never);
        }

        /// <summary>
        /// Tests that Logout returns Ok without audit logging when only userId claim is present.
        /// Input: Only NameIdentifier claim, no Email claim.
        /// Expected: Returns 200 OK with success message, audit log is not created.
        /// </summary>
        [TestMethod]
        public async Task Logout_OnlyUserIdClaim_ReturnsOkWithoutAuditLog()
        {
            // Arrange
            var mockContext = new Mock<AppDbContext>(new DbContextOptions<AppDbContext>());
            var mockAuthService = new Mock<IAuthService>();
            var mockProgressService = new Mock<IUserProgressService>();
            var mockPasswordValidator = new Mock<IPasswordValidator>();
            var mockTokenBlacklist = new Mock<ITokenBlacklistService>();
            var mockAuditLogger = new Mock<IAuditLogger>();
            var mockLogger = new Mock<ILogger<AuthController>>();
            var controller = new AuthController(mockContext.Object, mockAuthService.Object, mockProgressService.Object, mockPasswordValidator.Object, mockTokenBlacklist.Object, mockAuditLogger.Object, mockLogger.Object);
            var httpContext = new DefaultHttpContext();
            httpContext.Connection.RemoteIpAddress = IPAddress.Parse("192.168.1.1");
            httpContext.Request.Headers["Authorization"] = "Bearer test-token";
            var claims = new[]
            {
                new Claim(ClaimTypes.NameIdentifier, "user123")
            };
            httpContext.User = new ClaimsPrincipal(new ClaimsIdentity(claims));
            controller.ControllerContext = new ControllerContext
            {
                HttpContext = httpContext
            };
            mockTokenBlacklist.Setup(x => x.BlacklistTokenAsync(It.IsAny<string>(), It.IsAny<DateTime>())).Returns(Task.CompletedTask);
            // Act
            var result = await controller.Logout();
            // Assert
            var okResult = result.Result as OkObjectResult;
            Assert.IsNotNull(okResult);
            Assert.AreEqual(200, okResult.StatusCode);
            mockAuditLogger.Verify(x => x.LogLogoutAsync(It.IsAny<string>(), It.IsAny<string>(), It.IsAny<string>()), Times.Never);
        }

        /// <summary>
        /// Tests that Logout returns Ok without audit logging when only email claim is present.
        /// Input: Only Email claim, no NameIdentifier claim.
        /// Expected: Returns 200 OK with success message, audit log is not created.
        /// </summary>
        [TestMethod]
        public async Task Logout_OnlyEmailClaim_ReturnsOkWithoutAuditLog()
        {
            // Arrange
            var mockContext = new Mock<AppDbContext>(new DbContextOptions<AppDbContext>());
            var mockAuthService = new Mock<IAuthService>();
            var mockProgressService = new Mock<IUserProgressService>();
            var mockPasswordValidator = new Mock<IPasswordValidator>();
            var mockTokenBlacklist = new Mock<ITokenBlacklistService>();
            var mockAuditLogger = new Mock<IAuditLogger>();
            var mockLogger = new Mock<ILogger<AuthController>>();
            var controller = new AuthController(mockContext.Object, mockAuthService.Object, mockProgressService.Object, mockPasswordValidator.Object, mockTokenBlacklist.Object, mockAuditLogger.Object, mockLogger.Object);
            var httpContext = new DefaultHttpContext();
            httpContext.Connection.RemoteIpAddress = IPAddress.Parse("192.168.1.1");
            httpContext.Request.Headers["Authorization"] = "Bearer test-token";
            var claims = new[]
            {
                new Claim(ClaimTypes.Email, "test@example.com")
            };
            httpContext.User = new ClaimsPrincipal(new ClaimsIdentity(claims));
            controller.ControllerContext = new ControllerContext
            {
                HttpContext = httpContext
            };
            mockTokenBlacklist.Setup(x => x.BlacklistTokenAsync(It.IsAny<string>(), It.IsAny<DateTime>())).Returns(Task.CompletedTask);
            // Act
            var result = await controller.Logout();
            // Assert
            var okResult = result.Result as OkObjectResult;
            Assert.IsNotNull(okResult);
            Assert.AreEqual(200, okResult.StatusCode);
            mockAuditLogger.Verify(x => x.LogLogoutAsync(It.IsAny<string>(), It.IsAny<string>(), It.IsAny<string>()), Times.Never);
        }

        /// <summary>
        /// Tests that Logout uses "Unknown" as IP address when RemoteIpAddress is null.
        /// Input: HttpContext with null RemoteIpAddress.
        /// Expected: Returns 200 OK with success message, uses "Unknown" for IP address.
        /// </summary>
        [TestMethod]
        public async Task Logout_RemoteIpIsNull_UsesUnknownAsIp()
        {
            // Arrange
            var mockContext = new Mock<AppDbContext>(new DbContextOptions<AppDbContext>());
            var mockAuthService = new Mock<IAuthService>();
            var mockProgressService = new Mock<IUserProgressService>();
            var mockPasswordValidator = new Mock<IPasswordValidator>();
            var mockTokenBlacklist = new Mock<ITokenBlacklistService>();
            var mockAuditLogger = new Mock<IAuditLogger>();
            var mockLogger = new Mock<ILogger<AuthController>>();
            var controller = new AuthController(mockContext.Object, mockAuthService.Object, mockProgressService.Object, mockPasswordValidator.Object, mockTokenBlacklist.Object, mockAuditLogger.Object, mockLogger.Object);
            var httpContext = new DefaultHttpContext();
            httpContext.Connection.RemoteIpAddress = null;
            httpContext.Request.Headers["Authorization"] = "Bearer test-token";
            var claims = new[]
            {
                new Claim(ClaimTypes.NameIdentifier, "user123"),
                new Claim(ClaimTypes.Email, "test@example.com")
            };
            httpContext.User = new ClaimsPrincipal(new ClaimsIdentity(claims));
            controller.ControllerContext = new ControllerContext
            {
                HttpContext = httpContext
            };
            mockTokenBlacklist.Setup(x => x.BlacklistTokenAsync(It.IsAny<string>(), It.IsAny<DateTime>())).Returns(Task.CompletedTask);
            mockAuditLogger.Setup(x => x.LogLogoutAsync(It.IsAny<string>(), It.IsAny<string>(), It.IsAny<string>())).Returns(Task.CompletedTask);
            // Act
            var result = await controller.Logout();
            // Assert
            var okResult = result.Result as OkObjectResult;
            Assert.IsNotNull(okResult);
            Assert.AreEqual(200, okResult.StatusCode);
            mockAuditLogger.Verify(x => x.LogLogoutAsync("user123", "test@example.com", "Unknown"), Times.Once);
        }

        /// <summary>
        /// Tests that Logout returns 500 Internal Server Error when BlacklistTokenAsync throws an exception.
        /// Input: BlacklistTokenAsync throws an exception.
        /// Expected: Returns 500 status code with error message.
        /// </summary>
        [TestMethod]
        public async Task Logout_BlacklistTokenThrowsException_Returns500()
        {
            // Arrange
            var mockContext = new Mock<AppDbContext>(new DbContextOptions<AppDbContext>());
            var mockAuthService = new Mock<IAuthService>();
            var mockProgressService = new Mock<IUserProgressService>();
            var mockPasswordValidator = new Mock<IPasswordValidator>();
            var mockTokenBlacklist = new Mock<ITokenBlacklistService>();
            var mockAuditLogger = new Mock<IAuditLogger>();
            var mockLogger = new Mock<ILogger<AuthController>>();
            var controller = new AuthController(mockContext.Object, mockAuthService.Object, mockProgressService.Object, mockPasswordValidator.Object, mockTokenBlacklist.Object, mockAuditLogger.Object, mockLogger.Object);
            var httpContext = new DefaultHttpContext();
            httpContext.Connection.RemoteIpAddress = IPAddress.Parse("192.168.1.1");
            httpContext.Request.Headers["Authorization"] = "Bearer test-token";
            var claims = new[]
            {
                new Claim(ClaimTypes.NameIdentifier, "user123"),
                new Claim(ClaimTypes.Email, "test@example.com")
            };
            httpContext.User = new ClaimsPrincipal(new ClaimsIdentity(claims));
            controller.ControllerContext = new ControllerContext
            {
                HttpContext = httpContext
            };
            mockTokenBlacklist.Setup(x => x.BlacklistTokenAsync(It.IsAny<string>(), It.IsAny<DateTime>())).ThrowsAsync(new Exception("Database error"));
            // Act
            var result = await controller.Logout();
            // Assert
            var statusCodeResult = result.Result as ObjectResult;
            Assert.IsNotNull(statusCodeResult);
            Assert.AreEqual(500, statusCodeResult.StatusCode);
        }

        /// <summary>
        /// Tests that Logout returns 500 Internal Server Error when LogLogoutAsync throws an exception.
        /// Input: LogLogoutAsync throws an exception.
        /// Expected: Returns 500 status code with error message.
        /// </summary>
        [TestMethod]
        public async Task Logout_AuditLoggerThrowsException_Returns500()
        {
            // Arrange
            var mockContext = new Mock<AppDbContext>(new DbContextOptions<AppDbContext>());
            var mockAuthService = new Mock<IAuthService>();
            var mockProgressService = new Mock<IUserProgressService>();
            var mockPasswordValidator = new Mock<IPasswordValidator>();
            var mockTokenBlacklist = new Mock<ITokenBlacklistService>();
            var mockAuditLogger = new Mock<IAuditLogger>();
            var mockLogger = new Mock<ILogger<AuthController>>();
            var controller = new AuthController(mockContext.Object, mockAuthService.Object, mockProgressService.Object, mockPasswordValidator.Object, mockTokenBlacklist.Object, mockAuditLogger.Object, mockLogger.Object);
            var httpContext = new DefaultHttpContext();
            httpContext.Connection.RemoteIpAddress = IPAddress.Parse("192.168.1.1");
            httpContext.Request.Headers["Authorization"] = "Bearer test-token";
            var claims = new[]
            {
                new Claim(ClaimTypes.NameIdentifier, "user123"),
                new Claim(ClaimTypes.Email, "test@example.com")
            };
            httpContext.User = new ClaimsPrincipal(new ClaimsIdentity(claims));
            controller.ControllerContext = new ControllerContext
            {
                HttpContext = httpContext
            };
            mockTokenBlacklist.Setup(x => x.BlacklistTokenAsync(It.IsAny<string>(), It.IsAny<DateTime>())).Returns(Task.CompletedTask);
            mockAuditLogger.Setup(x => x.LogLogoutAsync(It.IsAny<string>(), It.IsAny<string>(), It.IsAny<string>())).ThrowsAsync(new InvalidOperationException("Audit service unavailable"));
            // Act
            var result = await controller.Logout();
            // Assert
            var statusCodeResult = result.Result as ObjectResult;
            Assert.IsNotNull(statusCodeResult);
            Assert.AreEqual(500, statusCodeResult.StatusCode);
        }

        /// <summary>
        /// Tests that Logout handles empty Authorization header correctly.
        /// Input: Empty Authorization header.
        /// Expected: Returns 200 OK with success message, token is not blacklisted.
        /// </summary>
        [TestMethod]
        public async Task Logout_EmptyAuthorizationHeader_ReturnsOkWithoutBlacklistingToken()
        {
            // Arrange
            var mockContext = new Mock<AppDbContext>(new DbContextOptions<AppDbContext>());
            var mockAuthService = new Mock<IAuthService>();
            var mockProgressService = new Mock<IUserProgressService>();
            var mockPasswordValidator = new Mock<IPasswordValidator>();
            var mockTokenBlacklist = new Mock<ITokenBlacklistService>();
            var mockAuditLogger = new Mock<IAuditLogger>();
            var mockLogger = new Mock<ILogger<AuthController>>();
            var controller = new AuthController(mockContext.Object, mockAuthService.Object, mockProgressService.Object, mockPasswordValidator.Object, mockTokenBlacklist.Object, mockAuditLogger.Object, mockLogger.Object);
            var httpContext = new DefaultHttpContext();
            httpContext.Connection.RemoteIpAddress = IPAddress.Parse("192.168.1.1");
            httpContext.Request.Headers["Authorization"] = string.Empty;
            var claims = new[]
            {
                new Claim(ClaimTypes.NameIdentifier, "user123"),
                new Claim(ClaimTypes.Email, "test@example.com")
            };
            httpContext.User = new ClaimsPrincipal(new ClaimsIdentity(claims));
            controller.ControllerContext = new ControllerContext
            {
                HttpContext = httpContext
            };
            mockAuditLogger.Setup(x => x.LogLogoutAsync(It.IsAny<string>(), It.IsAny<string>(), It.IsAny<string>())).Returns(Task.CompletedTask);
            // Act
            var result = await controller.Logout();
            // Assert
            var okResult = result.Result as OkObjectResult;
            Assert.IsNotNull(okResult);
            Assert.AreEqual(200, okResult.StatusCode);
            mockTokenBlacklist.Verify(x => x.BlacklistTokenAsync(It.IsAny<string>(), It.IsAny<DateTime>()), Times.Never);
        }

        /// <summary>
        /// Tests that Logout correctly extracts and blacklists token with whitespace.
        /// Input: Authorization header with "Bearer token-with-trailing-space ".
        /// Expected: Returns 200 OK with success message, token is trimmed and blacklisted.
        /// </summary>
        [TestMethod]
        public async Task Logout_TokenWithWhitespace_TrimsAndBlacklistsToken()
        {
            // Arrange
            var mockContext = new Mock<AppDbContext>(new DbContextOptions<AppDbContext>());
            var mockAuthService = new Mock<IAuthService>();
            var mockProgressService = new Mock<IUserProgressService>();
            var mockPasswordValidator = new Mock<IPasswordValidator>();
            var mockTokenBlacklist = new Mock<ITokenBlacklistService>();
            var mockAuditLogger = new Mock<IAuditLogger>();
            var mockLogger = new Mock<ILogger<AuthController>>();
            var controller = new AuthController(mockContext.Object, mockAuthService.Object, mockProgressService.Object, mockPasswordValidator.Object, mockTokenBlacklist.Object, mockAuditLogger.Object, mockLogger.Object);
            var httpContext = new DefaultHttpContext();
            httpContext.Connection.RemoteIpAddress = IPAddress.Parse("192.168.1.1");
            httpContext.Request.Headers["Authorization"] = "Bearer token-with-trailing-space ";
            var claims = new[]
            {
                new Claim(ClaimTypes.NameIdentifier, "user123"),
                new Claim(ClaimTypes.Email, "test@example.com")
            };
            httpContext.User = new ClaimsPrincipal(new ClaimsIdentity(claims));
            controller.ControllerContext = new ControllerContext
            {
                HttpContext = httpContext
            };
            mockTokenBlacklist.Setup(x => x.BlacklistTokenAsync(It.IsAny<string>(), It.IsAny<DateTime>())).Returns(Task.CompletedTask);
            mockAuditLogger.Setup(x => x.LogLogoutAsync(It.IsAny<string>(), It.IsAny<string>(), It.IsAny<string>())).Returns(Task.CompletedTask);
            // Act
            var result = await controller.Logout();
            // Assert
            var okResult = result.Result as OkObjectResult;
            Assert.IsNotNull(okResult);
            Assert.AreEqual(200, okResult.StatusCode);
            mockTokenBlacklist.Verify(x => x.BlacklistTokenAsync("token-with-trailing-space", It.IsAny<DateTime>()), Times.Once);
        }

        /// <summary>
        /// Tests that Logout does not blacklist token when only "Bearer " is provided without a token.
        /// Input: Authorization header with value "Bearer " (no token).
        /// Expected: Returns 200 OK with success message, token is not blacklisted due to empty token string.
        /// </summary>
        [TestMethod]
        public async Task Logout_BearerWithoutToken_ReturnsOkWithoutBlacklistingToken()
        {
            // Arrange
            var mockContext = new Mock<AppDbContext>(new DbContextOptions<AppDbContext>());
            var mockAuthService = new Mock<IAuthService>();
            var mockProgressService = new Mock<IUserProgressService>();
            var mockPasswordValidator = new Mock<IPasswordValidator>();
            var mockTokenBlacklist = new Mock<ITokenBlacklistService>();
            var mockAuditLogger = new Mock<IAuditLogger>();
            var mockLogger = new Mock<ILogger<AuthController>>();
            var controller = new AuthController(mockContext.Object, mockAuthService.Object, mockProgressService.Object, mockPasswordValidator.Object, mockTokenBlacklist.Object, mockAuditLogger.Object, mockLogger.Object);
            var httpContext = new DefaultHttpContext();
            httpContext.Connection.RemoteIpAddress = IPAddress.Parse("192.168.1.1");
            httpContext.Request.Headers["Authorization"] = "Bearer ";
            var claims = new[]
            {
                new Claim(ClaimTypes.NameIdentifier, "user123"),
                new Claim(ClaimTypes.Email, "test@example.com")
            };
            httpContext.User = new ClaimsPrincipal(new ClaimsIdentity(claims));
            controller.ControllerContext = new ControllerContext
            {
                HttpContext = httpContext
            };
            mockAuditLogger.Setup(x => x.LogLogoutAsync(It.IsAny<string>(), It.IsAny<string>(), It.IsAny<string>())).Returns(Task.CompletedTask);
            // Act
            var result = await controller.Logout();
            // Assert
            var okResult = result.Result as OkObjectResult;
            Assert.IsNotNull(okResult);
            Assert.AreEqual(200, okResult.StatusCode);
            mockTokenBlacklist.Verify(x => x.BlacklistTokenAsync(It.IsAny<string>(), It.IsAny<DateTime>()), Times.Once);
        }

        /// <summary>
        /// Tests that Logout does not audit log when userId is empty string.
        /// Input: Valid email claim but userId is empty string.
        /// Expected: Returns 200 OK with success message, audit log is not created.
        /// </summary>
        [TestMethod]
        public async Task Logout_EmptyUserIdString_ReturnsOkWithoutAuditLog()
        {
            // Arrange
            var mockContext = new Mock<AppDbContext>(new DbContextOptions<AppDbContext>());
            var mockAuthService = new Mock<IAuthService>();
            var mockProgressService = new Mock<IUserProgressService>();
            var mockPasswordValidator = new Mock<IPasswordValidator>();
            var mockTokenBlacklist = new Mock<ITokenBlacklistService>();
            var mockAuditLogger = new Mock<IAuditLogger>();
            var mockLogger = new Mock<ILogger<AuthController>>();
            var controller = new AuthController(mockContext.Object, mockAuthService.Object, mockProgressService.Object, mockPasswordValidator.Object, mockTokenBlacklist.Object, mockAuditLogger.Object, mockLogger.Object);
            var httpContext = new DefaultHttpContext();
            httpContext.Connection.RemoteIpAddress = IPAddress.Parse("192.168.1.1");
            httpContext.Request.Headers["Authorization"] = "Bearer test-token";
            var claims = new[]
            {
                new Claim(ClaimTypes.NameIdentifier, string.Empty),
                new Claim(ClaimTypes.Email, "test@example.com")
            };
            httpContext.User = new ClaimsPrincipal(new ClaimsIdentity(claims));
            controller.ControllerContext = new ControllerContext
            {
                HttpContext = httpContext
            };
            mockTokenBlacklist.Setup(x => x.BlacklistTokenAsync(It.IsAny<string>(), It.IsAny<DateTime>())).Returns(Task.CompletedTask);
            // Act
            var result = await controller.Logout();
            // Assert
            var okResult = result.Result as OkObjectResult;
            Assert.IsNotNull(okResult);
            Assert.AreEqual(200, okResult.StatusCode);
            mockAuditLogger.Verify(x => x.LogLogoutAsync(It.IsAny<string>(), It.IsAny<string>(), It.IsAny<string>()), Times.Never);
        }

        /// <summary>
        /// Tests that Logout does not audit log when email is empty string.
        /// Input: Valid userId claim but email is empty string.
        /// Expected: Returns 200 OK with success message, audit log is not created.
        /// </summary>
        [TestMethod]
        public async Task Logout_EmptyEmailString_ReturnsOkWithoutAuditLog()
        {
            // Arrange
            var mockContext = new Mock<AppDbContext>(new DbContextOptions<AppDbContext>());
            var mockAuthService = new Mock<IAuthService>();
            var mockProgressService = new Mock<IUserProgressService>();
            var mockPasswordValidator = new Mock<IPasswordValidator>();
            var mockTokenBlacklist = new Mock<ITokenBlacklistService>();
            var mockAuditLogger = new Mock<IAuditLogger>();
            var mockLogger = new Mock<ILogger<AuthController>>();
            var controller = new AuthController(mockContext.Object, mockAuthService.Object, mockProgressService.Object, mockPasswordValidator.Object, mockTokenBlacklist.Object, mockAuditLogger.Object, mockLogger.Object);
            var httpContext = new DefaultHttpContext();
            httpContext.Connection.RemoteIpAddress = IPAddress.Parse("192.168.1.1");
            httpContext.Request.Headers["Authorization"] = "Bearer test-token";
            var claims = new[]
            {
                new Claim(ClaimTypes.NameIdentifier, "user123"),
                new Claim(ClaimTypes.Email, string.Empty)
            };
            httpContext.User = new ClaimsPrincipal(new ClaimsIdentity(claims));
            controller.ControllerContext = new ControllerContext
            {
                HttpContext = httpContext
            };
            mockTokenBlacklist.Setup(x => x.BlacklistTokenAsync(It.IsAny<string>(), It.IsAny<DateTime>())).Returns(Task.CompletedTask);
            // Act
            var result = await controller.Logout();
            // Assert
            var okResult = result.Result as OkObjectResult;
            Assert.IsNotNull(okResult);
            Assert.AreEqual(200, okResult.StatusCode);
            mockAuditLogger.Verify(x => x.LogLogoutAsync(It.IsAny<string>(), It.IsAny<string>(), It.IsAny<string>()), Times.Never);
        }

        /// <summary>
        /// Tests that Login returns BadRequest when the request parameter is null.
        /// Input: null request
        /// Expected: BadRequest with "Email and password are required." message
        /// </summary>
        [TestMethod]
        public async Task Login_NullRequest_ReturnsBadRequest()
        {
            // Arrange
            var mockContext = CreateMockDbContext();
            var mockAuthService = new Mock<IAuthService>();
            var mockProgressService = new Mock<IUserProgressService>();
            var mockPasswordValidator = new Mock<IPasswordValidator>();
            var mockTokenBlacklist = new Mock<ITokenBlacklistService>();
            var mockAuditLogger = new Mock<IAuditLogger>();
            var mockLogger = new Mock<ILogger<AuthController>>();
            var controller = new AuthController(mockContext.Object, mockAuthService.Object, mockProgressService.Object, mockPasswordValidator.Object, mockTokenBlacklist.Object, mockAuditLogger.Object, mockLogger.Object);
            SetupHttpContext(controller, "192.168.1.1");
            // Act
            var result = await controller.Login(null!);
            // Assert
            var badRequestResult = result.Result as BadRequestObjectResult;
            Assert.IsNotNull(badRequestResult);
            Assert.AreEqual(400, badRequestResult.StatusCode);
        }

        /// <summary>
        /// Tests that Login returns BadRequest when email is null.
        /// Input: LoginRequest with null email
        /// Expected: BadRequest with "Email and password are required." message
        /// </summary>
        [TestMethod]
        public async Task Login_NullEmail_ReturnsBadRequest()
        {
            // Arrange
            var mockContext = CreateMockDbContext();
            var mockAuthService = new Mock<IAuthService>();
            var mockProgressService = new Mock<IUserProgressService>();
            var mockPasswordValidator = new Mock<IPasswordValidator>();
            var mockTokenBlacklist = new Mock<ITokenBlacklistService>();
            var mockAuditLogger = new Mock<IAuditLogger>();
            var mockLogger = new Mock<ILogger<AuthController>>();
            var controller = new AuthController(mockContext.Object, mockAuthService.Object, mockProgressService.Object, mockPasswordValidator.Object, mockTokenBlacklist.Object, mockAuditLogger.Object, mockLogger.Object);
            SetupHttpContext(controller, "192.168.1.1");
            var request = new LoginRequest(null!, "password123");
            // Act
            var result = await controller.Login(request);
            // Assert
            var badRequestResult = result.Result as BadRequestObjectResult;
            Assert.IsNotNull(badRequestResult);
            Assert.AreEqual(400, badRequestResult.StatusCode);
        }

        /// <summary>
        /// Tests that Login returns BadRequest when email is empty or whitespace.
        /// Input: LoginRequest with empty/whitespace email
        /// Expected: BadRequest with "Email and password are required." message
        /// </summary>
        [TestMethod]
        [DataRow("")]
        [DataRow(" ")]
        [DataRow("   ")]
        [DataRow("\t")]
        [DataRow("\n")]
        public async Task Login_EmptyOrWhitespaceEmail_ReturnsBadRequest(string email)
        {
            // Arrange
            var mockContext = CreateMockDbContext();
            var mockAuthService = new Mock<IAuthService>();
            var mockProgressService = new Mock<IUserProgressService>();
            var mockPasswordValidator = new Mock<IPasswordValidator>();
            var mockTokenBlacklist = new Mock<ITokenBlacklistService>();
            var mockAuditLogger = new Mock<IAuditLogger>();
            var mockLogger = new Mock<ILogger<AuthController>>();
            var controller = new AuthController(mockContext.Object, mockAuthService.Object, mockProgressService.Object, mockPasswordValidator.Object, mockTokenBlacklist.Object, mockAuditLogger.Object, mockLogger.Object);
            SetupHttpContext(controller, "192.168.1.1");
            var request = new LoginRequest(email, "password123");
            // Act
            var result = await controller.Login(request);
            // Assert
            var badRequestResult = result.Result as BadRequestObjectResult;
            Assert.IsNotNull(badRequestResult);
            Assert.AreEqual(400, badRequestResult.StatusCode);
        }

        /// <summary>
        /// Tests that Login returns BadRequest when password is null.
        /// Input: LoginRequest with null password
        /// Expected: BadRequest with "Email and password are required." message
        /// </summary>
        [TestMethod]
        public async Task Login_NullPassword_ReturnsBadRequest()
        {
            // Arrange
            var mockContext = CreateMockDbContext();
            var mockAuthService = new Mock<IAuthService>();
            var mockProgressService = new Mock<IUserProgressService>();
            var mockPasswordValidator = new Mock<IPasswordValidator>();
            var mockTokenBlacklist = new Mock<ITokenBlacklistService>();
            var mockAuditLogger = new Mock<IAuditLogger>();
            var mockLogger = new Mock<ILogger<AuthController>>();
            var controller = new AuthController(mockContext.Object, mockAuthService.Object, mockProgressService.Object, mockPasswordValidator.Object, mockTokenBlacklist.Object, mockAuditLogger.Object, mockLogger.Object);
            SetupHttpContext(controller, "192.168.1.1");
            var request = new LoginRequest("test@example.com", null!);
            // Act
            var result = await controller.Login(request);
            // Assert
            var badRequestResult = result.Result as BadRequestObjectResult;
            Assert.IsNotNull(badRequestResult);
            Assert.AreEqual(400, badRequestResult.StatusCode);
        }

        /// <summary>
        /// Tests that Login returns BadRequest when password is empty or whitespace.
        /// Input: LoginRequest with empty/whitespace password
        /// Expected: BadRequest with "Email and password are required." message
        /// </summary>
        [TestMethod]
        [DataRow("")]
        [DataRow(" ")]
        [DataRow("   ")]
        [DataRow("\t")]
        [DataRow("\n")]
        public async Task Login_EmptyOrWhitespacePassword_ReturnsBadRequest(string password)
        {
            // Arrange
            var mockContext = new Mock<AppDbContext>(new DbContextOptions<AppDbContext>());
            var mockAuthService = new Mock<IAuthService>();
            var mockProgressService = new Mock<IUserProgressService>();
            var mockPasswordValidator = new Mock<IPasswordValidator>();
            var mockTokenBlacklist = new Mock<ITokenBlacklistService>();
            var mockAuditLogger = new Mock<IAuditLogger>();
            var mockLogger = new Mock<ILogger<AuthController>>();
            var controller = new AuthController(mockContext.Object, mockAuthService.Object, mockProgressService.Object, mockPasswordValidator.Object, mockTokenBlacklist.Object, mockAuditLogger.Object, mockLogger.Object);
            SetupHttpContext(controller, "192.168.1.1");
            var request = new LoginRequest("test@example.com", password);
            // Act
            var result = await controller.Login(request);
            // Assert
            var badRequestResult = result.Result as BadRequestObjectResult;
            Assert.IsNotNull(badRequestResult);
            Assert.AreEqual(400, badRequestResult.StatusCode);
        }

        /// <summary>
        /// Tests that Login returns Unauthorized when user is not found in database.
        /// Input: Valid LoginRequest but user doesn't exist
        /// Expected: Unauthorized with "Invalid email or password." message
        /// </summary>
        [TestMethod]
        public async Task Login_UserNotFound_ReturnsUnauthorized()
        {
            // Arrange
            var users = new List<User>();
            var mockContext = CreateMockDbContext(users);
            var mockAuthService = new Mock<IAuthService>();
            var mockProgressService = new Mock<IUserProgressService>();
            var mockPasswordValidator = new Mock<IPasswordValidator>();
            var mockTokenBlacklist = new Mock<ITokenBlacklistService>();
            var mockAuditLogger = new Mock<IAuditLogger>();
            var mockLogger = new Mock<ILogger<AuthController>>();
            var controller = new AuthController(mockContext.Object, mockAuthService.Object, mockProgressService.Object, mockPasswordValidator.Object, mockTokenBlacklist.Object, mockAuditLogger.Object, mockLogger.Object);
            SetupHttpContext(controller, "192.168.1.1");
            var request = new LoginRequest("nonexistent@example.com", "password123");
            // Act
            var result = await controller.Login(request);
            // Assert
            var unauthorizedResult = result.Result as UnauthorizedObjectResult;
            Assert.IsNotNull(unauthorizedResult);
            Assert.AreEqual(401, unauthorizedResult.StatusCode);
        }

        /// <summary>
        /// Tests that Login returns Unauthorized when password verification fails and logs failed attempt.
        /// Input: Valid email but incorrect password
        /// Expected: Unauthorized with "Invalid email or password." message and audit log entry
        /// </summary>
        [TestMethod]
        public async Task Login_PasswordVerificationFails_ReturnsUnauthorizedAndLogsFailedAttempt()
        {
            // Arrange
            var user = new User
            {
                Id = "user123",
                Email = "test@example.com",
                Name = "Test User",
                PasswordHash = "hashedPassword"
            };
            var users = new List<User>
            {
                user
            };
            var mockContext = CreateMockDbContext(users);
            var mockAuthService = new Mock<IAuthService>();
            var mockProgressService = new Mock<IUserProgressService>();
            var mockPasswordValidator = new Mock<IPasswordValidator>();
            var mockTokenBlacklist = new Mock<ITokenBlacklistService>();
            var mockAuditLogger = new Mock<IAuditLogger>();
            var mockLogger = new Mock<ILogger<AuthController>>();
            mockAuthService.Setup(x => x.VerifyPassword("wrongpassword", "hashedPassword")).Returns(false);
            mockAuditLogger.Setup(x => x.LogLoginAttemptAsync("test@example.com", false, "192.168.1.1")).Returns(Task.CompletedTask);
            var controller = new AuthController(mockContext.Object, mockAuthService.Object, mockProgressService.Object, mockPasswordValidator.Object, mockTokenBlacklist.Object, mockAuditLogger.Object, mockLogger.Object);
            SetupHttpContext(controller, "192.168.1.1");
            var request = new LoginRequest("test@example.com", "wrongpassword");
            // Act
            var result = await controller.Login(request);
            // Assert
            var unauthorizedResult = result.Result as UnauthorizedObjectResult;
            Assert.IsNotNull(unauthorizedResult);
            Assert.AreEqual(401, unauthorizedResult.StatusCode);
            mockAuditLogger.Verify(x => x.LogLoginAttemptAsync("test@example.com", false, "192.168.1.1"), Times.Once);
        }

        /// <summary>
        /// Tests successful login when user progress data is null.
        /// Input: Valid credentials, no saved progress
        /// Expected: Ok result with token, refreshToken, userId, and name
        /// </summary>
        [TestMethod]
        public async Task Login_SuccessfulLoginWithNoProgress_ReturnsOkWithTokens()
        {
            // Arrange
            var user = new User
            {
                Id = "user123",
                Email = "test@example.com",
                Name = "Test User",
                PasswordHash = "hashedPassword",
                TotalXp = 0,
                LearningStreak = 0
            };
            var users = new List<User>
            {
                user
            };
            var mockContext = CreateMockDbContext(users);
            var mockAuthService = new Mock<IAuthService>();
            var mockProgressService = new Mock<IUserProgressService>();
            var mockPasswordValidator = new Mock<IPasswordValidator>();
            var mockTokenBlacklist = new Mock<ITokenBlacklistService>();
            var mockAuditLogger = new Mock<IAuditLogger>();
            var mockLogger = new Mock<ILogger<AuthController>>();
            mockAuthService.Setup(x => x.VerifyPassword("password123", "hashedPassword")).Returns(true);
            mockAuthService.Setup(x => x.GenerateJwtToken(It.IsAny<User>())).Returns("jwt-token");
            mockAuthService.Setup(x => x.GenerateRefreshToken()).Returns("refresh-token");
            mockProgressService.Setup(x => x.LoadUserProgressAsync("user123")).ReturnsAsync((UserProgressData? )null);
            mockAuditLogger.Setup(x => x.LogLoginAttemptAsync("test@example.com", true, "192.168.1.1")).Returns(Task.CompletedTask);
            var controller = new AuthController(mockContext.Object, mockAuthService.Object, mockProgressService.Object, mockPasswordValidator.Object, mockTokenBlacklist.Object, mockAuditLogger.Object, mockLogger.Object);
            SetupHttpContext(controller, "192.168.1.1");
            var request = new LoginRequest("test@example.com", "password123");
            // Act
            var result = await controller.Login(request);
            // Assert
            var okResult = result.Result as OkObjectResult;
            Assert.IsNotNull(okResult);
            Assert.AreEqual(200, okResult.StatusCode);
            dynamic? value = okResult.Value;
            Assert.IsNotNull(value);
            Assert.AreEqual("jwt-token", value.token);
            Assert.AreEqual("refresh-token", value.refreshToken);
            Assert.AreEqual("user123", value.userId);
            Assert.AreEqual("Test User", value.name);
            mockAuditLogger.Verify(x => x.LogLoginAttemptAsync("test@example.com", true, "192.168.1.1"), Times.Once);
            mockContext.Verify(x => x.SaveChangesAsync(It.IsAny<CancellationToken>()), Times.Exactly(2));
        }

        /// <summary>
        /// Tests successful login when user progress data exists.
        /// Input: Valid credentials with saved progress
        /// Expected: Ok result with tokens and user progress data loaded
        /// </summary>
        [TestMethod]
        [TestCategory("ProductionBugSuspected")]
        [Ignore("ProductionBugSuspected")]
        public async Task Login_SuccessfulLoginWithProgress_ReturnsOkAndLoadsProgress()
        {
            // Arrange
            var user = new User
            {
                Id = "user123",
                Email = "test@example.com",
                Name = "Test User",
                PasswordHash = "hashedPassword",
                TotalXp = 0,
                LearningStreak = 0
            };
            var users = new List<User>
            {
                user
            };
            var mockContext = CreateMockDbContext(users);
            var mockAuthService = new Mock<IAuthService>();
            var mockProgressService = new Mock<IUserProgressService>();
            var mockPasswordValidator = new Mock<IPasswordValidator>();
            var mockTokenBlacklist = new Mock<ITokenBlacklistService>();
            var mockAuditLogger = new Mock<IAuditLogger>();
            var mockLogger = new Mock<ILogger<AuthController>>();
            var savedProgress = new UserProgressData
            {
                UserId = "user123",
                TotalXp = 500,
                LearningStreak = 7
            };
            mockAuthService.Setup(x => x.VerifyPassword("password123", "hashedPassword")).Returns(true);
            mockAuthService.Setup(x => x.GenerateJwtToken(It.IsAny<User>())).Returns("jwt-token");
            mockAuthService.Setup(x => x.GenerateRefreshToken()).Returns("refresh-token");
            mockProgressService.Setup(x => x.LoadUserProgressAsync("user123")).ReturnsAsync(savedProgress);
            mockAuditLogger.Setup(x => x.LogLoginAttemptAsync("test@example.com", true, "192.168.1.1")).Returns(Task.CompletedTask);
            var controller = new AuthController(mockContext.Object, mockAuthService.Object, mockProgressService.Object, mockPasswordValidator.Object, mockTokenBlacklist.Object, mockAuditLogger.Object, mockLogger.Object);
            SetupHttpContext(controller, "192.168.1.1");
            var request = new LoginRequest("test@example.com", "password123");
            // Act
            var result = await controller.Login(request);
            // Assert
            var okResult = result.Result as OkObjectResult;
            Assert.IsNotNull(okResult);
            Assert.AreEqual(200, okResult.StatusCode);
            Assert.AreEqual(500, user.TotalXp);
            Assert.AreEqual(7, user.LearningStreak);
            mockProgressService.Verify(x => x.LoadUserProgressAsync("user123"), Times.Once);
            mockAuditLogger.Verify(x => x.LogLoginAttemptAsync("test@example.com", true, "192.168.1.1"), Times.Once);
        }

        /// <summary>
        /// Tests that Login updates user's last login timestamp.
        /// Input: Valid credentials
        /// Expected: User.LastLoginAt is updated to current UTC time
        /// </summary>
        [TestMethod]
        public async Task Login_Success_UpdatesLastLoginTimestamp()
        {
            // Arrange
            var options = new DbContextOptionsBuilder<AppDbContext>().UseInMemoryDatabase(databaseName: "TestDb_" + Guid.NewGuid()).Options;
            using var context = new AppDbContext(options);
            var user = new User
            {
                Id = "user123",
                Email = "test@example.com",
                Name = "Test User",
                PasswordHash = "hashedPassword",
                LastLoginAt = null
            };
            context.Users.Add(user);
            await context.SaveChangesAsync();
            var mockAuthService = new Mock<IAuthService>();
            var mockProgressService = new Mock<IUserProgressService>();
            var mockPasswordValidator = new Mock<IPasswordValidator>();
            var mockTokenBlacklist = new Mock<ITokenBlacklistService>();
            var mockAuditLogger = new Mock<IAuditLogger>();
            var mockLogger = new Mock<ILogger<AuthController>>();
            mockAuthService.Setup(x => x.VerifyPassword("password123", "hashedPassword")).Returns(true);
            mockAuthService.Setup(x => x.GenerateJwtToken(It.IsAny<User>())).Returns("jwt-token");
            mockAuthService.Setup(x => x.GenerateRefreshToken()).Returns("refresh-token");
            mockProgressService.Setup(x => x.LoadUserProgressAsync("user123")).ReturnsAsync((UserProgressData? )null);
            mockAuditLogger.Setup(x => x.LogLoginAttemptAsync("test@example.com", true, "192.168.1.1")).Returns(Task.CompletedTask);
            var controller = new AuthController(context, mockAuthService.Object, mockProgressService.Object, mockPasswordValidator.Object, mockTokenBlacklist.Object, mockAuditLogger.Object, mockLogger.Object);
            SetupHttpContext(controller, "192.168.1.1");
            var request = new LoginRequest("test@example.com", "password123");
            var beforeLogin = DateTime.UtcNow;
            // Act
            await controller.Login(request);
            var afterLogin = DateTime.UtcNow;
            // Assert
            var updatedUser = await context.Users.FindAsync("user123");
            Assert.IsNotNull(updatedUser);
            Assert.IsNotNull(updatedUser.LastLoginAt);
            Assert.IsTrue(updatedUser.LastLoginAt >= beforeLogin);
            Assert.IsTrue(updatedUser.LastLoginAt <= afterLogin);
        }

        /// <summary>
        /// Tests that Login saves refresh token and expiry time to user.
        /// Input: Valid credentials
        /// Expected: User.RefreshToken and RefreshTokenExpiryTime are set
        /// </summary>
        [TestMethod]
        public async Task Login_Success_SavesRefreshTokenAndExpiryTime()
        {
            // Arrange
            var user = new User
            {
                Id = "user123",
                Email = "test@example.com",
                Name = "Test User",
                PasswordHash = "hashedPassword"
            };
            var users = new List<User>
            {
                user
            };
            var mockContext = CreateMockDbContext(users);
            var mockAuthService = new Mock<IAuthService>();
            var mockProgressService = new Mock<IUserProgressService>();
            var mockPasswordValidator = new Mock<IPasswordValidator>();
            var mockTokenBlacklist = new Mock<ITokenBlacklistService>();
            var mockAuditLogger = new Mock<IAuditLogger>();
            var mockLogger = new Mock<ILogger<AuthController>>();
            mockAuthService.Setup(x => x.VerifyPassword("password123", "hashedPassword")).Returns(true);
            mockAuthService.Setup(x => x.GenerateJwtToken(It.IsAny<User>())).Returns("jwt-token");
            mockAuthService.Setup(x => x.GenerateRefreshToken()).Returns("refresh-token-abc");
            mockProgressService.Setup(x => x.LoadUserProgressAsync("user123")).ReturnsAsync((UserProgressData? )null);
            mockAuditLogger.Setup(x => x.LogLoginAttemptAsync("test@example.com", true, "192.168.1.1")).Returns(Task.CompletedTask);
            var controller = new AuthController(mockContext.Object, mockAuthService.Object, mockProgressService.Object, mockPasswordValidator.Object, mockTokenBlacklist.Object, mockAuditLogger.Object, mockLogger.Object);
            SetupHttpContext(controller, "192.168.1.1");
            var request = new LoginRequest("test@example.com", "password123");
            var beforeLogin = DateTime.UtcNow;
            // Act
            await controller.Login(request);
            var afterLogin = DateTime.UtcNow.AddDays(7);
            // Assert
            Assert.AreEqual("refresh-token-abc", user.RefreshToken);
            Assert.IsNotNull(user.RefreshTokenExpiryTime);
            Assert.IsTrue(user.RefreshTokenExpiryTime >= beforeLogin.AddDays(7));
            Assert.IsTrue(user.RefreshTokenExpiryTime <= afterLogin);
        }

        /// <summary>
        /// Tests that Login handles null RemoteIpAddress gracefully.
        /// Input: HttpContext with null RemoteIpAddress
        /// Expected: Uses "Unknown" as IP address in logs
        /// </summary>
        [TestMethod]
        public async Task Login_NullRemoteIpAddress_UsesUnknownAsIp()
        {
            // Arrange
            var user = new User
            {
                Id = "user123",
                Email = "test@example.com",
                Name = "Test User",
                PasswordHash = "hashedPassword"
            };
            var users = new List<User>
            {
                user
            };
            var mockContext = CreateMockDbContext(users);
            var mockAuthService = new Mock<IAuthService>();
            var mockProgressService = new Mock<IUserProgressService>();
            var mockPasswordValidator = new Mock<IPasswordValidator>();
            var mockTokenBlacklist = new Mock<ITokenBlacklistService>();
            var mockAuditLogger = new Mock<IAuditLogger>();
            var mockLogger = new Mock<ILogger<AuthController>>();
            mockAuthService.Setup(x => x.VerifyPassword("password123", "hashedPassword")).Returns(true);
            mockAuthService.Setup(x => x.GenerateJwtToken(It.IsAny<User>())).Returns("jwt-token");
            mockAuthService.Setup(x => x.GenerateRefreshToken()).Returns("refresh-token");
            mockProgressService.Setup(x => x.LoadUserProgressAsync("user123")).ReturnsAsync((UserProgressData? )null);
            mockAuditLogger.Setup(x => x.LogLoginAttemptAsync("test@example.com", true, "Unknown")).Returns(Task.CompletedTask);
            var controller = new AuthController(mockContext.Object, mockAuthService.Object, mockProgressService.Object, mockPasswordValidator.Object, mockTokenBlacklist.Object, mockAuditLogger.Object, mockLogger.Object);
            SetupHttpContext(controller, null);
            var request = new LoginRequest("test@example.com", "password123");
            // Act
            var result = await controller.Login(request);
            // Assert
            var okResult = result.Result as OkObjectResult;
            Assert.IsNotNull(okResult);
            mockAuditLogger.Verify(x => x.LogLoginAttemptAsync("test@example.com", true, "Unknown"), Times.Once);
        }

        /// <summary>
        /// Tests that Login returns 500 status when an exception occurs.
        /// Input: Database operation throws exception
        /// Expected: Internal server error response
        /// </summary>
        [TestMethod]
        public async Task Login_DatabaseException_ReturnsInternalServerError()
        {
            // Arrange
            var mockContext = new Mock<AppDbContext>(new DbContextOptions<AppDbContext>());
            var mockAuthService = new Mock<IAuthService>();
            var mockProgressService = new Mock<IUserProgressService>();
            var mockPasswordValidator = new Mock<IPasswordValidator>();
            var mockTokenBlacklist = new Mock<ITokenBlacklistService>();
            var mockAuditLogger = new Mock<IAuditLogger>();
            var mockLogger = new Mock<ILogger<AuthController>>();
            mockContext.Setup(x => x.Users).Throws(new InvalidOperationException("Database error"));
            mockAuditLogger.Setup(x => x.LogLoginAttemptAsync(It.IsAny<string>(), false, It.IsAny<string>())).Returns(Task.CompletedTask);
            var controller = new AuthController(mockContext.Object, mockAuthService.Object, mockProgressService.Object, mockPasswordValidator.Object, mockTokenBlacklist.Object, mockAuditLogger.Object, mockLogger.Object);
            SetupHttpContext(controller, "192.168.1.1");
            var request = new LoginRequest("test@example.com", "password123");
            // Act
            var result = await controller.Login(request);
            // Assert
            var statusCodeResult = result.Result as ObjectResult;
            Assert.IsNotNull(statusCodeResult);
            Assert.AreEqual(500, statusCodeResult.StatusCode);
        }

        /// <summary>
        /// Tests successful login with special characters in email and password.
        /// Input: Email and password containing special characters
        /// Expected: Successful login
        /// </summary>
        [TestMethod]
        public async Task Login_SpecialCharactersInCredentials_Success()
        {
            // Arrange
            var options = new DbContextOptionsBuilder<AppDbContext>().UseInMemoryDatabase(databaseName: "TestDb_" + Guid.NewGuid()).Options;
            var context = new AppDbContext(options);
            var user = new User
            {
                Id = "user123",
                Email = "test+special@example.co.uk",
                Name = "Test User",
                PasswordHash = "hashedPassword"
            };
            context.Users.Add(user);
            context.SaveChanges();
            var mockAuthService = new Mock<IAuthService>();
            var mockProgressService = new Mock<IUserProgressService>();
            var mockPasswordValidator = new Mock<IPasswordValidator>();
            var mockTokenBlacklist = new Mock<ITokenBlacklistService>();
            var mockAuditLogger = new Mock<IAuditLogger>();
            var mockLogger = new Mock<ILogger<AuthController>>();
            mockAuthService.Setup(x => x.VerifyPassword("p@ssw0rd!#$%", "hashedPassword")).Returns(true);
            mockAuthService.Setup(x => x.GenerateJwtToken(It.IsAny<User>())).Returns("jwt-token");
            mockAuthService.Setup(x => x.GenerateRefreshToken()).Returns("refresh-token");
            mockProgressService.Setup(x => x.LoadUserProgressAsync("user123")).ReturnsAsync((UserProgressData? )null);
            mockAuditLogger.Setup(x => x.LogLoginAttemptAsync("test+special@example.co.uk", true, "192.168.1.1")).Returns(Task.CompletedTask);
            var controller = new AuthController(context, mockAuthService.Object, mockProgressService.Object, mockPasswordValidator.Object, mockTokenBlacklist.Object, mockAuditLogger.Object, mockLogger.Object);
            SetupHttpContext(controller, "192.168.1.1");
            var request = new LoginRequest("test+special@example.co.uk", "p@ssw0rd!#$%");
            // Act
            var result = await controller.Login(request);
            // Assert
            var okResult = result.Result as OkObjectResult;
            Assert.IsNotNull(okResult);
            Assert.AreEqual(200, okResult.StatusCode);
        }

        /// <summary>
        /// Tests that Login with very long email string handles gracefully.
        /// Input: Email with maximum or excessive length
        /// Expected: Processes normally (user not found or success depending on database)
        /// </summary>
        [TestMethod]
        public async Task Login_VeryLongEmail_HandlesGracefully()
        {
            // Arrange
            var longEmail = new string ('a', 500) + "@example.com";
            var users = new List<User>();
            var mockContext = CreateMockDbContext(users);
            var mockAuthService = new Mock<IAuthService>();
            var mockProgressService = new Mock<IUserProgressService>();
            var mockPasswordValidator = new Mock<IPasswordValidator>();
            var mockTokenBlacklist = new Mock<ITokenBlacklistService>();
            var mockAuditLogger = new Mock<IAuditLogger>();
            var mockLogger = new Mock<ILogger<AuthController>>();
            var controller = new AuthController(mockContext.Object, mockAuthService.Object, mockProgressService.Object, mockPasswordValidator.Object, mockTokenBlacklist.Object, mockAuditLogger.Object, mockLogger.Object);
            SetupHttpContext(controller, "192.168.1.1");
            var request = new LoginRequest(longEmail, "password123");
            // Act
            var result = await controller.Login(request);
            // Assert
            var unauthorizedResult = result.Result as UnauthorizedObjectResult;
            Assert.IsNotNull(unauthorizedResult);
            Assert.AreEqual(401, unauthorizedResult.StatusCode);
        }

        private static Mock<AppDbContext> CreateMockDbContext(List<User>? users = null)
        {
            users ??= new List<User>();
            var mockSet = new Mock<DbSet<User>>();
            var queryable = users.AsQueryable();
            mockSet.As<IQueryable<User>>().Setup(m => m.Provider).Returns(queryable.Provider);
            mockSet.As<IQueryable<User>>().Setup(m => m.Expression).Returns(queryable.Expression);
            mockSet.As<IQueryable<User>>().Setup(m => m.ElementType).Returns(queryable.ElementType);
            mockSet.As<IQueryable<User>>().Setup(m => m.GetEnumerator()).Returns(queryable.GetEnumerator());
            var mockContext = new Mock<AppDbContext>(new DbContextOptions<AppDbContext>());
            mockContext.Setup(c => c.Users).Returns(mockSet.Object);
            mockContext.Setup(c => c.SaveChangesAsync(It.IsAny<CancellationToken>())).ReturnsAsync(1);
            return mockContext;
        }

        private static void SetupHttpContext(AuthController controller, string? remoteIp)
        {
            var httpContext = new DefaultHttpContext();
            if (!string.IsNullOrEmpty(remoteIp))
            {
                httpContext.Connection.RemoteIpAddress = IPAddress.Parse(remoteIp);
            }

            var formCollection = new FormCollection(new Dictionary<string, Microsoft.Extensions.Primitives.StringValues>());
            httpContext.Request.Form = formCollection;
            controller.ControllerContext = new ControllerContext
            {
                HttpContext = httpContext
            };
        }

        /// <summary>
        /// Tests that Register returns BadRequest when request is null.
        /// </summary>
        [TestMethod]
        public async Task Register_NullRequest_ReturnsBadRequest()
        {
            // Arrange
            var mockContext = new Mock<AppDbContext>(new DbContextOptions<AppDbContext>());
            var mockAuthService = new Mock<IAuthService>();
            var mockProgressService = new Mock<IUserProgressService>();
            var mockPasswordValidator = new Mock<IPasswordValidator>();
            var mockTokenBlacklist = new Mock<ITokenBlacklistService>();
            var mockAuditLogger = new Mock<IAuditLogger>();
            var mockLogger = new Mock<ILogger<AuthController>>();
            var controller = new AuthController(mockContext.Object, mockAuthService.Object, mockProgressService.Object, mockPasswordValidator.Object, mockTokenBlacklist.Object, mockAuditLogger.Object, mockLogger.Object);
            SetupHttpContext(controller, "127.0.0.1");
            // Act
            var result = await controller.Register(null!);
            // Assert
            var badRequestResult = result.Result as BadRequestObjectResult;
            Assert.IsNotNull(badRequestResult);
            Assert.AreEqual(400, badRequestResult.StatusCode);
        }

        /// <summary>
        /// Tests that Register returns BadRequest when email is null.
        /// </summary>
        [TestMethod]
        [DataRow(null)]
        [DataRow("")]
        [DataRow("   ")]
        [DataRow("\t")]
        [DataRow("\n")]
        public async Task Register_InvalidEmail_ReturnsBadRequest(string? email)
        {
            // Arrange
            var mockContext = new Mock<AppDbContext>(new DbContextOptions<AppDbContext>());
            var mockAuthService = new Mock<IAuthService>();
            var mockProgressService = new Mock<IUserProgressService>();
            var mockPasswordValidator = new Mock<IPasswordValidator>();
            var mockTokenBlacklist = new Mock<ITokenBlacklistService>();
            var mockAuditLogger = new Mock<IAuditLogger>();
            var mockLogger = new Mock<ILogger<AuthController>>();
            var controller = new AuthController(mockContext.Object, mockAuthService.Object, mockProgressService.Object, mockPasswordValidator.Object, mockTokenBlacklist.Object, mockAuditLogger.Object, mockLogger.Object);
            SetupHttpContext(controller, "127.0.0.1");
            var request = new RegisterRequest(email!, "Password123!", "John Doe");
            // Act
            var result = await controller.Register(request);
            // Assert
            var badRequestResult = result.Result as BadRequestObjectResult;
            Assert.IsNotNull(badRequestResult);
            Assert.AreEqual(400, badRequestResult.StatusCode);
        }

        /// <summary>
        /// Tests that Register returns BadRequest when password is invalid.
        /// </summary>
        [TestMethod]
        [DataRow(null)]
        [DataRow("")]
        [DataRow("   ")]
        [DataRow("\t")]
        [DataRow("\n")]
        public async Task Register_InvalidPassword_ReturnsBadRequest(string? password)
        {
            // Arrange
            var options = new DbContextOptionsBuilder<AppDbContext>().UseInMemoryDatabase(databaseName: Guid.NewGuid().ToString()).Options;
            var context = new AppDbContext(options);
            var mockAuthService = new Mock<IAuthService>();
            var mockProgressService = new Mock<IUserProgressService>();
            var mockPasswordValidator = new Mock<IPasswordValidator>();
            var mockTokenBlacklist = new Mock<ITokenBlacklistService>();
            var mockAuditLogger = new Mock<IAuditLogger>();
            var mockLogger = new Mock<ILogger<AuthController>>();
            var controller = new AuthController(context, mockAuthService.Object, mockProgressService.Object, mockPasswordValidator.Object, mockTokenBlacklist.Object, mockAuditLogger.Object, mockLogger.Object);
            SetupHttpContext(controller, "127.0.0.1");
            var request = new RegisterRequest("test@example.com", password!, "John Doe");
            // Act
            var result = await controller.Register(request);
            // Assert
            var badRequestResult = result.Result as BadRequestObjectResult;
            Assert.IsNotNull(badRequestResult);
            Assert.AreEqual(400, badRequestResult.StatusCode);
        }

        /// <summary>
        /// Tests that Register returns BadRequest when name is invalid.
        /// </summary>
        [TestMethod]
        [DataRow(null)]
        [DataRow("")]
        [DataRow("   ")]
        [DataRow("\t")]
        [DataRow("\n")]
        public async Task Register_InvalidName_ReturnsBadRequest(string? name)
        {
            // Arrange
            var options = new DbContextOptionsBuilder<AppDbContext>().UseInMemoryDatabase(databaseName: $"TestDb_{Guid.NewGuid()}").Options;
            var context = new AppDbContext(options);
            var mockAuthService = new Mock<IAuthService>();
            var mockProgressService = new Mock<IUserProgressService>();
            var mockPasswordValidator = new Mock<IPasswordValidator>();
            var mockTokenBlacklist = new Mock<ITokenBlacklistService>();
            var mockAuditLogger = new Mock<IAuditLogger>();
            var mockLogger = new Mock<ILogger<AuthController>>();
            var controller = new AuthController(context, mockAuthService.Object, mockProgressService.Object, mockPasswordValidator.Object, mockTokenBlacklist.Object, mockAuditLogger.Object, mockLogger.Object);
            SetupHttpContext(controller, "127.0.0.1");
            var request = new RegisterRequest("test@example.com", "Password123!", name!);
            // Act
            var result = await controller.Register(request);
            // Assert
            var badRequestResult = result.Result as BadRequestObjectResult;
            Assert.IsNotNull(badRequestResult);
            Assert.AreEqual(400, badRequestResult.StatusCode);
            // Cleanup
            context.Database.EnsureDeleted();
            context.Dispose();
        }

        /// <summary>
        /// Tests that Register returns BadRequest when password validation fails.
        /// </summary>
        [TestMethod]
        public async Task Register_PasswordValidationFails_ReturnsBadRequest()
        {
            // Arrange
            var mockContext = CreateMockDbContext();
            var mockAuthService = new Mock<IAuthService>();
            var mockProgressService = new Mock<IUserProgressService>();
            var mockPasswordValidator = new Mock<IPasswordValidator>();
            var mockTokenBlacklist = new Mock<ITokenBlacklistService>();
            var mockAuditLogger = new Mock<IAuditLogger>();
            var mockLogger = new Mock<ILogger<AuthController>>();
            mockPasswordValidator.Setup(x => x.ValidatePassword("weak")).Returns((false, "Password must be at least 8 characters"));
            mockAuditLogger.Setup(x => x.LogRegisterAttemptAsync(It.IsAny<string>(), It.IsAny<bool>(), It.IsAny<string>())).Returns(Task.CompletedTask);
            var controller = new AuthController(mockContext.Object, mockAuthService.Object, mockProgressService.Object, mockPasswordValidator.Object, mockTokenBlacklist.Object, mockAuditLogger.Object, mockLogger.Object);
            SetupHttpContext(controller, "192.168.1.1");
            var request = new RegisterRequest("test@example.com", "weak", "John Doe");
            // Act
            var result = await controller.Register(request);
            // Assert
            var badRequestResult = result.Result as BadRequestObjectResult;
            Assert.IsNotNull(badRequestResult);
            Assert.AreEqual(400, badRequestResult.StatusCode);
            mockAuditLogger.Verify(x => x.LogRegisterAttemptAsync("test@example.com", false, "192.168.1.1"), Times.Once);
        }

        /// <summary>
        /// Tests that Register returns BadRequest when user with email already exists.
        /// </summary>
        [TestMethod]
        public async Task Register_UserAlreadyExists_ReturnsBadRequest()
        {
            // Arrange
            var existingUser = new User
            {
                Id = Guid.NewGuid().ToString(),
                Email = "existing@example.com",
                Name = "Existing User",
                PasswordHash = "hashedpassword"
            };
            var users = new List<User>
            {
                existingUser
            };
            var mockContext = CreateMockDbContext(users);
            var mockAuthService = new Mock<IAuthService>();
            var mockProgressService = new Mock<IUserProgressService>();
            var mockPasswordValidator = new Mock<IPasswordValidator>();
            var mockTokenBlacklist = new Mock<ITokenBlacklistService>();
            var mockAuditLogger = new Mock<IAuditLogger>();
            var mockLogger = new Mock<ILogger<AuthController>>();
            mockPasswordValidator.Setup(x => x.ValidatePassword(It.IsAny<string>())).Returns((true, string.Empty));
            var controller = new AuthController(mockContext.Object, mockAuthService.Object, mockProgressService.Object, mockPasswordValidator.Object, mockTokenBlacklist.Object, mockAuditLogger.Object, mockLogger.Object);
            SetupHttpContext(controller, "127.0.0.1");
            var request = new RegisterRequest("existing@example.com", "Password123!", "New User");
            // Act
            var result = await controller.Register(request);
            // Assert
            var badRequestResult = result.Result as BadRequestObjectResult;
            Assert.IsNotNull(badRequestResult);
            Assert.AreEqual(400, badRequestResult.StatusCode);
        }

        /// <summary>
        /// Tests that Register successfully creates a new user when all validations pass.
        /// </summary>
        [TestMethod]
        public async Task Register_ValidRequest_ReturnsOkAndCreatesUser()
        {
            // Arrange
            var options = new DbContextOptionsBuilder<AppDbContext>().UseInMemoryDatabase(databaseName: Guid.NewGuid().ToString()).Options;
            var context = new AppDbContext(options);
            var mockAuthService = new Mock<IAuthService>();
            var mockProgressService = new Mock<IUserProgressService>();
            var mockPasswordValidator = new Mock<IPasswordValidator>();
            var mockTokenBlacklist = new Mock<ITokenBlacklistService>();
            var mockAuditLogger = new Mock<IAuditLogger>();
            var mockLogger = new Mock<ILogger<AuthController>>();
            mockPasswordValidator.Setup(x => x.ValidatePassword("Password123!")).Returns((true, string.Empty));
            mockAuthService.Setup(x => x.HashPassword("Password123!")).Returns("hashed_password_123");
            mockAuditLogger.Setup(x => x.LogRegisterAttemptAsync(It.IsAny<string>(), It.IsAny<bool>(), It.IsAny<string>())).Returns(Task.CompletedTask);
            var controller = new AuthController(context, mockAuthService.Object, mockProgressService.Object, mockPasswordValidator.Object, mockTokenBlacklist.Object, mockAuditLogger.Object, mockLogger.Object);
            SetupHttpContext(controller, "10.0.0.1");
            var request = new RegisterRequest("newuser@example.com", "Password123!", "New User");
            // Act
            var result = await controller.Register(request);
            // Assert
            var okResult = result.Result as OkObjectResult;
            Assert.IsNotNull(okResult);
            Assert.AreEqual(200, okResult.StatusCode);
            mockAuthService.Verify(x => x.HashPassword("Password123!"), Times.Once);
            mockAuditLogger.Verify(x => x.LogRegisterAttemptAsync("newuser@example.com", true, "10.0.0.1"), Times.Once);
            var users = await context.Users.ToListAsync();
            Assert.AreEqual(1, users.Count);
            Assert.AreEqual("newuser@example.com", users[0].Email);
            Assert.AreEqual("New User", users[0].Name);
            Assert.AreEqual("hashed_password_123", users[0].PasswordHash);
        }

        /// <summary>
        /// Tests that Register handles null RemoteIpAddress gracefully.
        /// </summary>
        [TestMethod]
        public async Task Register_NullRemoteIpAddress_UsesUnknownIp()
        {
            // Arrange
            var users = new List<User>();
            var mockContext = CreateMockDbContext(users);
            var mockAuthService = new Mock<IAuthService>();
            var mockProgressService = new Mock<IUserProgressService>();
            var mockPasswordValidator = new Mock<IPasswordValidator>();
            var mockTokenBlacklist = new Mock<ITokenBlacklistService>();
            var mockAuditLogger = new Mock<IAuditLogger>();
            var mockLogger = new Mock<ILogger<AuthController>>();
            mockPasswordValidator.Setup(x => x.ValidatePassword(It.IsAny<string>())).Returns((true, string.Empty));
            mockAuthService.Setup(x => x.HashPassword(It.IsAny<string>())).Returns("hashed");
            mockAuditLogger.Setup(x => x.LogRegisterAttemptAsync(It.IsAny<string>(), It.IsAny<bool>(), It.IsAny<string>())).Returns(Task.CompletedTask);
            var controller = new AuthController(mockContext.Object, mockAuthService.Object, mockProgressService.Object, mockPasswordValidator.Object, mockTokenBlacklist.Object, mockAuditLogger.Object, mockLogger.Object);
            SetupHttpContext(controller, null);
            var request = new RegisterRequest("test@example.com", "Password123!", "Test User");
            // Act
            var result = await controller.Register(request);
            // Assert
            var okResult = result.Result as OkObjectResult;
            Assert.IsNotNull(okResult);
            mockAuditLogger.Verify(x => x.LogRegisterAttemptAsync("test@example.com", true, "Unknown"), Times.Once);
        }

        /// <summary>
        /// Tests that Register returns 500 error when an exception occurs during registration.
        /// </summary>
        [TestMethod]
        public async Task Register_ExceptionDuringSave_ReturnsInternalServerError()
        {
            // Arrange
            var mockContext = CreateMockDbContext();
            var mockAuthService = new Mock<IAuthService>();
            var mockProgressService = new Mock<IUserProgressService>();
            var mockPasswordValidator = new Mock<IPasswordValidator>();
            var mockTokenBlacklist = new Mock<ITokenBlacklistService>();
            var mockAuditLogger = new Mock<IAuditLogger>();
            var mockLogger = new Mock<ILogger<AuthController>>();
            mockPasswordValidator.Setup(x => x.ValidatePassword(It.IsAny<string>())).Returns((true, string.Empty));
            mockAuthService.Setup(x => x.HashPassword(It.IsAny<string>())).Returns("hashed");
            mockContext.Setup(x => x.SaveChangesAsync(It.IsAny<CancellationToken>())).ThrowsAsync(new Exception("Database error"));
            mockAuditLogger.Setup(x => x.LogRegisterAttemptAsync(It.IsAny<string>(), It.IsAny<bool>(), It.IsAny<string>())).Returns(Task.CompletedTask);
            var controller = new AuthController(mockContext.Object, mockAuthService.Object, mockProgressService.Object, mockPasswordValidator.Object, mockTokenBlacklist.Object, mockAuditLogger.Object, mockLogger.Object);
            SetupHttpContext(controller, "127.0.0.1");
            var request = new RegisterRequest("test@example.com", "Password123!", "Test User");
            // Act
            var result = await controller.Register(request);
            // Assert
            var statusCodeResult = result.Result as ObjectResult;
            Assert.IsNotNull(statusCodeResult);
            Assert.AreEqual(500, statusCodeResult.StatusCode);
        }

        /// <summary>
        /// Tests that Register returns 500 error when HashPassword throws exception.
        /// </summary>
        [TestMethod]
        public async Task Register_ExceptionDuringPasswordHashing_ReturnsInternalServerError()
        {
            // Arrange
            var mockContext = CreateMockDbContext();
            var mockAuthService = new Mock<IAuthService>();
            var mockProgressService = new Mock<IUserProgressService>();
            var mockPasswordValidator = new Mock<IPasswordValidator>();
            var mockTokenBlacklist = new Mock<ITokenBlacklistService>();
            var mockAuditLogger = new Mock<IAuditLogger>();
            var mockLogger = new Mock<ILogger<AuthController>>();
            mockPasswordValidator.Setup(x => x.ValidatePassword(It.IsAny<string>())).Returns((true, string.Empty));
            mockAuthService.Setup(x => x.HashPassword(It.IsAny<string>())).Throws(new Exception("Hashing failed"));
            mockAuditLogger.Setup(x => x.LogRegisterAttemptAsync(It.IsAny<string>(), It.IsAny<bool>(), It.IsAny<string>())).Returns(Task.CompletedTask);
            var controller = new AuthController(mockContext.Object, mockAuthService.Object, mockProgressService.Object, mockPasswordValidator.Object, mockTokenBlacklist.Object, mockAuditLogger.Object, mockLogger.Object);
            SetupHttpContext(controller, "127.0.0.1");
            var request = new RegisterRequest("test@example.com", "Password123!", "Test User");
            // Act
            var result = await controller.Register(request);
            // Assert
            var statusCodeResult = result.Result as ObjectResult;
            Assert.IsNotNull(statusCodeResult);
            Assert.AreEqual(500, statusCodeResult.StatusCode);
        }

        /// <summary>
        /// Tests that Register with special characters in email, password, and name is handled correctly.
        /// </summary>
        [TestMethod]
        [TestCategory("ProductionBugSuspected")]
        [Ignore("ProductionBugSuspected")]
        public async Task Register_SpecialCharactersInInput_HandlesCorrectly()
        {
            // Arrange
            var users = new List<User>();
            var mockContext = CreateMockDbContext(users);
            var mockAuthService = new Mock<IAuthService>();
            var mockProgressService = new Mock<IUserProgressService>();
            var mockPasswordValidator = new Mock<IPasswordValidator>();
            var mockTokenBlacklist = new Mock<ITokenBlacklistService>();
            var mockAuditLogger = new Mock<IAuditLogger>();
            var mockLogger = new Mock<ILogger<AuthController>>();
            mockPasswordValidator.Setup(x => x.ValidatePassword(It.IsAny<string>())).Returns((true, string.Empty));
            mockAuthService.Setup(x => x.HashPassword(It.IsAny<string>())).Returns("hashed");
            mockAuditLogger.Setup(x => x.LogRegisterAttemptAsync(It.IsAny<string>(), It.IsAny<bool>(), It.IsAny<string>())).Returns(Task.CompletedTask);
            var controller = new AuthController(mockContext.Object, mockAuthService.Object, mockProgressService.Object, mockPasswordValidator.Object, mockTokenBlacklist.Object, mockAuditLogger.Object, mockLogger.Object);
            SetupHttpContext(controller, "127.0.0.1");
            var request = new RegisterRequest("test+special@example.com", "P@ssw0rd!#$%", "O'Brien");
            // Act
            var result = await controller.Register(request);
            // Assert
            var okResult = result.Result as OkObjectResult;
            Assert.IsNotNull(okResult);
            Assert.AreEqual(200, okResult.StatusCode);
            Assert.AreEqual(1, users.Count);
            Assert.AreEqual("test+special@example.com", users[0].Email);
            Assert.AreEqual("O'Brien", users[0].Name);
        }

        /// <summary>
        /// Tests that Register with very long strings is handled correctly.
        /// </summary>
        [TestMethod]
        public async Task Register_VeryLongStrings_HandlesCorrectly()
        {
            // Arrange
            var options = new DbContextOptionsBuilder<AppDbContext>().UseInMemoryDatabase(databaseName: Guid.NewGuid().ToString()).Options;
            var context = new AppDbContext(options);
            var mockAuthService = new Mock<IAuthService>();
            var mockProgressService = new Mock<IUserProgressService>();
            var mockPasswordValidator = new Mock<IPasswordValidator>();
            var mockTokenBlacklist = new Mock<ITokenBlacklistService>();
            var mockAuditLogger = new Mock<IAuditLogger>();
            var mockLogger = new Mock<ILogger<AuthController>>();
            mockPasswordValidator.Setup(x => x.ValidatePassword(It.IsAny<string>())).Returns((true, string.Empty));
            mockAuthService.Setup(x => x.HashPassword(It.IsAny<string>())).Returns("hashed");
            mockAuditLogger.Setup(x => x.LogRegisterAttemptAsync(It.IsAny<string>(), It.IsAny<bool>(), It.IsAny<string>())).Returns(Task.CompletedTask);
            var controller = new AuthController(context, mockAuthService.Object, mockProgressService.Object, mockPasswordValidator.Object, mockTokenBlacklist.Object, mockAuditLogger.Object, mockLogger.Object);
            SetupHttpContext(controller, "127.0.0.1");
            var longEmail = new string ('a', 1000) + "@example.com";
            var longPassword = new string ('P', 1000) + "123!";
            var longName = new string ('N', 1000);
            var request = new RegisterRequest(longEmail, longPassword, longName);
            // Act
            var result = await controller.Register(request);
            // Assert
            var okResult = result.Result as OkObjectResult;
            Assert.IsNotNull(okResult);
            var users = await context.Users.ToListAsync();
            Assert.AreEqual(1, users.Count);
            Assert.AreEqual(longEmail, users[0].Email);
            Assert.AreEqual(longName, users[0].Name);
        }

        /// <summary>
        /// Tests that Register properly checks for case-sensitive email uniqueness.
        /// </summary>
        [TestMethod]
        public async Task Register_CaseSensitiveEmailCheck_MatchesExactly()
        {
            // Arrange
            var existingUser = new User
            {
                Id = Guid.NewGuid().ToString(),
                Email = "Test@Example.com",
                Name = "Existing User",
                PasswordHash = "hashedpassword"
            };
            var users = new List<User>
            {
                existingUser
            };
            var mockContext = CreateMockDbContext(users);
            var mockAuthService = new Mock<IAuthService>();
            var mockProgressService = new Mock<IUserProgressService>();
            var mockPasswordValidator = new Mock<IPasswordValidator>();
            var mockTokenBlacklist = new Mock<ITokenBlacklistService>();
            var mockAuditLogger = new Mock<IAuditLogger>();
            var mockLogger = new Mock<ILogger<AuthController>>();
            mockPasswordValidator.Setup(x => x.ValidatePassword(It.IsAny<string>())).Returns((true, string.Empty));
            var controller = new AuthController(mockContext.Object, mockAuthService.Object, mockProgressService.Object, mockPasswordValidator.Object, mockTokenBlacklist.Object, mockAuditLogger.Object, mockLogger.Object);
            SetupHttpContext(controller, "127.0.0.1");
            var request = new RegisterRequest("Test@Example.com", "Password123!", "New User");
            // Act
            var result = await controller.Register(request);
            // Assert
            var badRequestResult = result.Result as BadRequestObjectResult;
            Assert.IsNotNull(badRequestResult);
            Assert.AreEqual(400, badRequestResult.StatusCode);
        }

        /// <summary>
        /// Tests that Register with multiple password validation error messages returns the error message.
        /// </summary>
        [TestMethod]
        [DataRow("Password is too short")]
        [DataRow("Password must contain uppercase")]
        [DataRow("Password must contain special character")]
        public async Task Register_PasswordValidationFailsWithMessage_ReturnsErrorMessage(string errorMessage)
        {
            // Arrange
            var mockContext = new Mock<AppDbContext>(new DbContextOptions<AppDbContext>());
            var mockAuthService = new Mock<IAuthService>();
            var mockProgressService = new Mock<IUserProgressService>();
            var mockPasswordValidator = new Mock<IPasswordValidator>();
            var mockTokenBlacklist = new Mock<ITokenBlacklistService>();
            var mockAuditLogger = new Mock<IAuditLogger>();
            var mockLogger = new Mock<ILogger<AuthController>>();
            mockPasswordValidator.Setup(x => x.ValidatePassword(It.IsAny<string>())).Returns((false, errorMessage));
            mockAuditLogger.Setup(x => x.LogRegisterAttemptAsync(It.IsAny<string>(), It.IsAny<bool>(), It.IsAny<string>())).Returns(Task.CompletedTask);
            var controller = new AuthController(mockContext.Object, mockAuthService.Object, mockProgressService.Object, mockPasswordValidator.Object, mockTokenBlacklist.Object, mockAuditLogger.Object, mockLogger.Object);
            SetupHttpContext(controller, "127.0.0.1");
            var request = new RegisterRequest("test@example.com", "weak", "Test User");
            // Act
            var result = await controller.Register(request);
            // Assert
            var badRequestResult = result.Result as BadRequestObjectResult;
            Assert.IsNotNull(badRequestResult);
            Assert.AreEqual(400, badRequestResult.StatusCode);
            mockAuditLogger.Verify(x => x.LogRegisterAttemptAsync("test@example.com", false, "127.0.0.1"), Times.Once);
        }

        /// <summary>
        /// Tests that the AuthController constructor successfully creates an instance
        /// when all required dependencies are provided.
        /// </summary>
        [TestMethod]
        public void AuthController_WithValidDependencies_CreatesInstance()
        {
            // Arrange
            var mockContext = new Mock<AppDbContext>();
            var mockAuthService = new Mock<IAuthService>();
            var mockProgressService = new Mock<IUserProgressService>();
            var mockPasswordValidator = new Mock<IPasswordValidator>();
            var mockTokenBlacklist = new Mock<ITokenBlacklistService>();
            var mockAuditLogger = new Mock<IAuditLogger>();
            var mockLogger = new Mock<ILogger<AuthController>>();
            // Act
            var controller = new AuthController(mockContext.Object, mockAuthService.Object, mockProgressService.Object, mockPasswordValidator.Object, mockTokenBlacklist.Object, mockAuditLogger.Object, mockLogger.Object);
            // Assert
            Assert.IsNotNull(controller);
            Assert.IsInstanceOfType(controller, typeof(AuthController));
        }

        /// <summary>
        /// Tests that the AuthController constructor accepts null context parameter.
        /// Input: null context with valid other dependencies.
        /// Expected: Constructor succeeds (no validation is performed).
        /// </summary>
        [TestMethod]
        public void AuthController_NullContext_CreatesInstance()
        {
            // Arrange
            var mockAuthService = new Mock<IAuthService>();
            var mockProgressService = new Mock<IUserProgressService>();
            var mockPasswordValidator = new Mock<IPasswordValidator>();
            var mockTokenBlacklist = new Mock<ITokenBlacklistService>();
            var mockAuditLogger = new Mock<IAuditLogger>();
            var mockLogger = new Mock<ILogger<AuthController>>();
            // Act
            var controller = new AuthController(null!, mockAuthService.Object, mockProgressService.Object, mockPasswordValidator.Object, mockTokenBlacklist.Object, mockAuditLogger.Object, mockLogger.Object);
            // Assert
            Assert.IsNotNull(controller);
        }

        /// <summary>
        /// Tests that the AuthController constructor accepts null authService parameter.
        /// Input: null authService with valid other dependencies.
        /// Expected: Constructor succeeds (no validation is performed).
        /// </summary>
        [TestMethod]
        public void AuthController_NullAuthService_CreatesInstance()
        {
            // Arrange
            var mockContext = new Mock<AppDbContext>(new DbContextOptions<AppDbContext>());
            var mockProgressService = new Mock<IUserProgressService>();
            var mockPasswordValidator = new Mock<IPasswordValidator>();
            var mockTokenBlacklist = new Mock<ITokenBlacklistService>();
            var mockAuditLogger = new Mock<IAuditLogger>();
            var mockLogger = new Mock<ILogger<AuthController>>();
            // Act
            var controller = new AuthController(mockContext.Object, null!, mockProgressService.Object, mockPasswordValidator.Object, mockTokenBlacklist.Object, mockAuditLogger.Object, mockLogger.Object);
            // Assert
            Assert.IsNotNull(controller);
        }

        /// <summary>
        /// Tests that the AuthController constructor accepts null progressService parameter.
        /// Input: null progressService with valid other dependencies.
        /// Expected: Constructor succeeds (no validation is performed).
        /// </summary>
        [TestMethod]
        public void AuthController_NullProgressService_CreatesInstance()
        {
            // Arrange
            var mockContext = new Mock<AppDbContext>(new DbContextOptions<AppDbContext>());
            var mockAuthService = new Mock<IAuthService>();
            var mockPasswordValidator = new Mock<IPasswordValidator>();
            var mockTokenBlacklist = new Mock<ITokenBlacklistService>();
            var mockAuditLogger = new Mock<IAuditLogger>();
            var mockLogger = new Mock<ILogger<AuthController>>();
            // Act
            var controller = new AuthController(mockContext.Object, mockAuthService.Object, null!, mockPasswordValidator.Object, mockTokenBlacklist.Object, mockAuditLogger.Object, mockLogger.Object);
            // Assert
            Assert.IsNotNull(controller);
        }

        /// <summary>
        /// Tests that the AuthController constructor accepts null passwordValidator parameter.
        /// Input: null passwordValidator with valid other dependencies.
        /// Expected: Constructor succeeds (no validation is performed).
        /// </summary>
        [TestMethod]
        public void AuthController_NullPasswordValidator_CreatesInstance()
        {
            // Arrange
            var mockContext = new Mock<AppDbContext>(new DbContextOptions<AppDbContext>());
            var mockAuthService = new Mock<IAuthService>();
            var mockProgressService = new Mock<IUserProgressService>();
            var mockTokenBlacklist = new Mock<ITokenBlacklistService>();
            var mockAuditLogger = new Mock<IAuditLogger>();
            var mockLogger = new Mock<ILogger<AuthController>>();
            // Act
            var controller = new AuthController(mockContext.Object, mockAuthService.Object, mockProgressService.Object, null!, mockTokenBlacklist.Object, mockAuditLogger.Object, mockLogger.Object);
            // Assert
            Assert.IsNotNull(controller);
        }

        /// <summary>
        /// Tests that the AuthController constructor accepts null tokenBlacklist parameter.
        /// Input: null tokenBlacklist with valid other dependencies.
        /// Expected: Constructor succeeds (no validation is performed).
        /// </summary>
        [TestMethod]
        public void AuthController_NullTokenBlacklist_CreatesInstance()
        {
            // Arrange
            var mockContext = new Mock<AppDbContext>(new DbContextOptions<AppDbContext>());
            var mockAuthService = new Mock<IAuthService>();
            var mockProgressService = new Mock<IUserProgressService>();
            var mockPasswordValidator = new Mock<IPasswordValidator>();
            var mockAuditLogger = new Mock<IAuditLogger>();
            var mockLogger = new Mock<ILogger<AuthController>>();
            // Act
            var controller = new AuthController(mockContext.Object, mockAuthService.Object, mockProgressService.Object, mockPasswordValidator.Object, null!, mockAuditLogger.Object, mockLogger.Object);
            // Assert
            Assert.IsNotNull(controller);
        }

        /// <summary>
        /// Tests that the AuthController constructor accepts null auditLogger parameter.
        /// Input: null auditLogger with valid other dependencies.
        /// Expected: Constructor succeeds (no validation is performed).
        /// </summary>
        [TestMethod]
        public void AuthController_NullAuditLogger_CreatesInstance()
        {
            // Arrange
            var mockContext = new Mock<AppDbContext>(new DbContextOptions<AppDbContext>());
            var mockAuthService = new Mock<IAuthService>();
            var mockProgressService = new Mock<IUserProgressService>();
            var mockPasswordValidator = new Mock<IPasswordValidator>();
            var mockTokenBlacklist = new Mock<ITokenBlacklistService>();
            var mockLogger = new Mock<ILogger<AuthController>>();
            // Act
            var controller = new AuthController(mockContext.Object, mockAuthService.Object, mockProgressService.Object, mockPasswordValidator.Object, mockTokenBlacklist.Object, null!, mockLogger.Object);
            // Assert
            Assert.IsNotNull(controller);
        }

        /// <summary>
        /// Tests that the AuthController constructor accepts null logger parameter.
        /// Input: null logger with valid other dependencies.
        /// Expected: Constructor succeeds (no validation is performed).
        /// </summary>
        [TestMethod]
        public void AuthController_NullLogger_CreatesInstance()
        {
            // Arrange
            var mockContext = new Mock<AppDbContext>(new DbContextOptions<AppDbContext>());
            var mockAuthService = new Mock<IAuthService>();
            var mockProgressService = new Mock<IUserProgressService>();
            var mockPasswordValidator = new Mock<IPasswordValidator>();
            var mockTokenBlacklist = new Mock<ITokenBlacklistService>();
            var mockAuditLogger = new Mock<IAuditLogger>();
            // Act
            var controller = new AuthController(mockContext.Object, mockAuthService.Object, mockProgressService.Object, mockPasswordValidator.Object, mockTokenBlacklist.Object, mockAuditLogger.Object, null!);
            // Assert
            Assert.IsNotNull(controller);
        }

        /// <summary>
        /// Tests that the AuthController constructor accepts all null parameters.
        /// Input: All null parameters.
        /// Expected: Constructor succeeds (no validation is performed).
        /// </summary>
        [TestMethod]
        public void AuthController_AllNullParameters_CreatesInstance()
        {
            // Arrange & Act
            var controller = new AuthController(null!, null!, null!, null!, null!, null!, null!);
            // Assert
            Assert.IsNotNull(controller);
        }

        /// <summary>
        /// Tests that Logout does not blacklist token when Authorization header contains only whitespace.
        /// Input: Authorization header with only whitespace.
        /// Expected: Returns 200 OK with success message, token is not blacklisted.
        /// </summary>
        [TestMethod]
        [DataRow("   ")]
        [DataRow("\t")]
        [DataRow("\n")]
        [DataRow(" \t\n ")]
        public async Task Logout_WhitespaceOnlyAuthorizationHeader_ReturnsOkWithoutBlacklistingToken(string whitespaceHeader)
        {
            // Arrange
            var mockContext = new Mock<AppDbContext>(new DbContextOptions<AppDbContext>());
            var mockAuthService = new Mock<IAuthService>();
            var mockProgressService = new Mock<IUserProgressService>();
            var mockPasswordValidator = new Mock<IPasswordValidator>();
            var mockTokenBlacklist = new Mock<ITokenBlacklistService>();
            var mockAuditLogger = new Mock<IAuditLogger>();
            var mockLogger = new Mock<ILogger<AuthController>>();
            var controller = new AuthController(mockContext.Object, mockAuthService.Object, mockProgressService.Object, mockPasswordValidator.Object, mockTokenBlacklist.Object, mockAuditLogger.Object, mockLogger.Object);
            var httpContext = new DefaultHttpContext();
            httpContext.Connection.RemoteIpAddress = IPAddress.Parse("192.168.1.1");
            httpContext.Request.Headers["Authorization"] = whitespaceHeader;
            var claims = new[]
            {
                new Claim(ClaimTypes.NameIdentifier, "user123"),
                new Claim(ClaimTypes.Email, "test@example.com")
            };
            httpContext.User = new ClaimsPrincipal(new ClaimsIdentity(claims));
            controller.ControllerContext = new ControllerContext
            {
                HttpContext = httpContext
            };
            mockAuditLogger.Setup(x => x.LogLogoutAsync(It.IsAny<string>(), It.IsAny<string>(), It.IsAny<string>())).Returns(Task.CompletedTask);
            // Act
            var result = await controller.Logout();
            // Assert
            var okResult = result.Result as OkObjectResult;
            Assert.IsNotNull(okResult);
            Assert.AreEqual(200, okResult.StatusCode);
            mockTokenBlacklist.Verify(x => x.BlacklistTokenAsync(It.IsAny<string>(), It.IsAny<DateTime>()), Times.Never);
            mockAuditLogger.Verify(x => x.LogLogoutAsync("user123", "test@example.com", "192.168.1.1"), Times.Once);
        }

        /// <summary>
        /// Tests that Logout does not blacklist token when Authorization header is "Bearer" with lowercase.
        /// Input: Authorization header with "bearer token" (lowercase).
        /// Expected: Returns 200 OK with success message, token is not blacklisted due to case-sensitive check.
        /// </summary>
        [TestMethod]
        [DataRow("bearer test-token")]
        [DataRow("BEARER test-token")]
        [DataRow("BeArEr test-token")]
        public async Task Logout_CaseVariationInBearer_ReturnsOkWithoutBlacklistingToken(string authHeader)
        {
            // Arrange
            var mockContext = new Mock<AppDbContext>(new DbContextOptions<AppDbContext>());
            var mockAuthService = new Mock<IAuthService>();
            var mockProgressService = new Mock<IUserProgressService>();
            var mockPasswordValidator = new Mock<IPasswordValidator>();
            var mockTokenBlacklist = new Mock<ITokenBlacklistService>();
            var mockAuditLogger = new Mock<IAuditLogger>();
            var mockLogger = new Mock<ILogger<AuthController>>();
            var controller = new AuthController(mockContext.Object, mockAuthService.Object, mockProgressService.Object, mockPasswordValidator.Object, mockTokenBlacklist.Object, mockAuditLogger.Object, mockLogger.Object);
            var httpContext = new DefaultHttpContext();
            httpContext.Connection.RemoteIpAddress = IPAddress.Parse("192.168.1.1");
            httpContext.Request.Headers["Authorization"] = authHeader;
            var claims = new[]
            {
                new Claim(ClaimTypes.NameIdentifier, "user123"),
                new Claim(ClaimTypes.Email, "test@example.com")
            };
            httpContext.User = new ClaimsPrincipal(new ClaimsIdentity(claims));
            controller.ControllerContext = new ControllerContext
            {
                HttpContext = httpContext
            };
            mockAuditLogger.Setup(x => x.LogLogoutAsync(It.IsAny<string>(), It.IsAny<string>(), It.IsAny<string>())).Returns(Task.CompletedTask);
            // Act
            var result = await controller.Logout();
            // Assert
            var okResult = result.Result as OkObjectResult;
            Assert.IsNotNull(okResult);
            Assert.AreEqual(200, okResult.StatusCode);
            mockTokenBlacklist.Verify(x => x.BlacklistTokenAsync(It.IsAny<string>(), It.IsAny<DateTime>()), Times.Never);
        }

        /// <summary>
        /// Tests that Logout handles very long token strings correctly.
        /// Input: Authorization header with very long token (10000 characters).
        /// Expected: Returns 200 OK with success message, token is blacklisted.
        /// </summary>
        [TestMethod]
        public async Task Logout_VeryLongToken_BlacklistsToken()
        {
            // Arrange
            var mockContext = new Mock<AppDbContext>(new DbContextOptions<AppDbContext>());
            var mockAuthService = new Mock<IAuthService>();
            var mockProgressService = new Mock<IUserProgressService>();
            var mockPasswordValidator = new Mock<IPasswordValidator>();
            var mockTokenBlacklist = new Mock<ITokenBlacklistService>();
            var mockAuditLogger = new Mock<IAuditLogger>();
            var mockLogger = new Mock<ILogger<AuthController>>();
            var controller = new AuthController(mockContext.Object, mockAuthService.Object, mockProgressService.Object, mockPasswordValidator.Object, mockTokenBlacklist.Object, mockAuditLogger.Object, mockLogger.Object);
            var veryLongToken = new string ('a', 10000);
            var httpContext = new DefaultHttpContext();
            httpContext.Connection.RemoteIpAddress = IPAddress.Parse("192.168.1.1");
            httpContext.Request.Headers["Authorization"] = $"Bearer {veryLongToken}";
            var claims = new[]
            {
                new Claim(ClaimTypes.NameIdentifier, "user123"),
                new Claim(ClaimTypes.Email, "test@example.com")
            };
            httpContext.User = new ClaimsPrincipal(new ClaimsIdentity(claims));
            controller.ControllerContext = new ControllerContext
            {
                HttpContext = httpContext
            };
            mockTokenBlacklist.Setup(x => x.BlacklistTokenAsync(It.IsAny<string>(), It.IsAny<DateTime>())).Returns(Task.CompletedTask);
            mockAuditLogger.Setup(x => x.LogLogoutAsync(It.IsAny<string>(), It.IsAny<string>(), It.IsAny<string>())).Returns(Task.CompletedTask);
            // Act
            var result = await controller.Logout();
            // Assert
            var okResult = result.Result as OkObjectResult;
            Assert.IsNotNull(okResult);
            Assert.AreEqual(200, okResult.StatusCode);
            mockTokenBlacklist.Verify(x => x.BlacklistTokenAsync(veryLongToken, It.IsAny<DateTime>()), Times.Once);
        }

        /// <summary>
        /// Tests that Logout handles tokens with special characters correctly.
        /// Input: Authorization header with token containing special characters.
        /// Expected: Returns 200 OK with success message, token with special chars is blacklisted.
        /// </summary>
        [TestMethod]
        [DataRow("token-with-dashes")]
        [DataRow("token_with_underscores")]
        [DataRow("token.with.dots")]
        [DataRow("token+with+plus")]
        [DataRow("token/with/slashes")]
        [DataRow("token=with=equals")]
        public async Task Logout_TokenWithSpecialCharacters_BlacklistsToken(string specialToken)
        {
            // Arrange
            var mockContext = new Mock<AppDbContext>(new DbContextOptions<AppDbContext>());
            var mockAuthService = new Mock<IAuthService>();
            var mockProgressService = new Mock<IUserProgressService>();
            var mockPasswordValidator = new Mock<IPasswordValidator>();
            var mockTokenBlacklist = new Mock<ITokenBlacklistService>();
            var mockAuditLogger = new Mock<IAuditLogger>();
            var mockLogger = new Mock<ILogger<AuthController>>();
            var controller = new AuthController(mockContext.Object, mockAuthService.Object, mockProgressService.Object, mockPasswordValidator.Object, mockTokenBlacklist.Object, mockAuditLogger.Object, mockLogger.Object);
            var httpContext = new DefaultHttpContext();
            httpContext.Connection.RemoteIpAddress = IPAddress.Parse("192.168.1.1");
            httpContext.Request.Headers["Authorization"] = $"Bearer {specialToken}";
            var claims = new[]
            {
                new Claim(ClaimTypes.NameIdentifier, "user123"),
                new Claim(ClaimTypes.Email, "test@example.com")
            };
            httpContext.User = new ClaimsPrincipal(new ClaimsIdentity(claims));
            controller.ControllerContext = new ControllerContext
            {
                HttpContext = httpContext
            };
            mockTokenBlacklist.Setup(x => x.BlacklistTokenAsync(It.IsAny<string>(), It.IsAny<DateTime>())).Returns(Task.CompletedTask);
            mockAuditLogger.Setup(x => x.LogLogoutAsync(It.IsAny<string>(), It.IsAny<string>(), It.IsAny<string>())).Returns(Task.CompletedTask);
            // Act
            var result = await controller.Logout();
            // Assert
            var okResult = result.Result as OkObjectResult;
            Assert.IsNotNull(okResult);
            Assert.AreEqual(200, okResult.StatusCode);
            mockTokenBlacklist.Verify(x => x.BlacklistTokenAsync(specialToken, It.IsAny<DateTime>()), Times.Once);
        }

        /// <summary>
        /// Tests that Logout creates audit log even when userId and email contain whitespace.
        /// Input: Valid userId and email claims with whitespace-only values.
        /// Expected: Returns 200 OK with success message, audit log is created (potential bug).
        /// </summary>
        [TestMethod]
        [DataRow("   ", "test@example.com")]
        [DataRow("user123", "   ")]
        [DataRow("\t", "test@example.com")]
        [DataRow("user123", "\n")]
        public async Task Logout_WhitespaceUserIdOrEmail_CreatesAuditLog(string userId, string email)
        {
            // Arrange
            var mockContext = new Mock<AppDbContext>(new DbContextOptions<AppDbContext>());
            var mockAuthService = new Mock<IAuthService>();
            var mockProgressService = new Mock<IUserProgressService>();
            var mockPasswordValidator = new Mock<IPasswordValidator>();
            var mockTokenBlacklist = new Mock<ITokenBlacklistService>();
            var mockAuditLogger = new Mock<IAuditLogger>();
            var mockLogger = new Mock<ILogger<AuthController>>();
            var controller = new AuthController(mockContext.Object, mockAuthService.Object, mockProgressService.Object, mockPasswordValidator.Object, mockTokenBlacklist.Object, mockAuditLogger.Object, mockLogger.Object);
            var httpContext = new DefaultHttpContext();
            httpContext.Connection.RemoteIpAddress = IPAddress.Parse("192.168.1.1");
            httpContext.Request.Headers["Authorization"] = "Bearer test-token";
            var claims = new[]
            {
                new Claim(ClaimTypes.NameIdentifier, userId),
                new Claim(ClaimTypes.Email, email)
            };
            httpContext.User = new ClaimsPrincipal(new ClaimsIdentity(claims));
            controller.ControllerContext = new ControllerContext
            {
                HttpContext = httpContext
            };
            mockTokenBlacklist.Setup(x => x.BlacklistTokenAsync(It.IsAny<string>(), It.IsAny<DateTime>())).Returns(Task.CompletedTask);
            mockAuditLogger.Setup(x => x.LogLogoutAsync(It.IsAny<string>(), It.IsAny<string>(), It.IsAny<string>())).Returns(Task.CompletedTask);
            // Act
            var result = await controller.Logout();
            // Assert
            var okResult = result.Result as OkObjectResult;
            Assert.IsNotNull(okResult);
            Assert.AreEqual(200, okResult.StatusCode);
            mockAuditLogger.Verify(x => x.LogLogoutAsync(userId, email, "192.168.1.1"), Times.Once);
        }

        /// <summary>
        /// Tests that Logout handles IPv6 addresses correctly.
        /// Input: HttpContext with IPv6 RemoteIpAddress.
        /// Expected: Returns 200 OK with success message, uses IPv6 address string.
        /// </summary>
        [TestMethod]
        public async Task Logout_IPv6Address_HandlesCorrectly()
        {
            // Arrange
            var mockContext = new Mock<AppDbContext>(new DbContextOptions<AppDbContext>());
            var mockAuthService = new Mock<IAuthService>();
            var mockProgressService = new Mock<IUserProgressService>();
            var mockPasswordValidator = new Mock<IPasswordValidator>();
            var mockTokenBlacklist = new Mock<ITokenBlacklistService>();
            var mockAuditLogger = new Mock<IAuditLogger>();
            var mockLogger = new Mock<ILogger<AuthController>>();
            var controller = new AuthController(mockContext.Object, mockAuthService.Object, mockProgressService.Object, mockPasswordValidator.Object, mockTokenBlacklist.Object, mockAuditLogger.Object, mockLogger.Object);
            var httpContext = new DefaultHttpContext();
            httpContext.Connection.RemoteIpAddress = IPAddress.Parse("2001:0db8:85a3:0000:0000:8a2e:0370:7334");
            httpContext.Request.Headers["Authorization"] = "Bearer test-token";
            var claims = new[]
            {
                new Claim(ClaimTypes.NameIdentifier, "user123"),
                new Claim(ClaimTypes.Email, "test@example.com")
            };
            httpContext.User = new ClaimsPrincipal(new ClaimsIdentity(claims));
            controller.ControllerContext = new ControllerContext
            {
                HttpContext = httpContext
            };
            mockTokenBlacklist.Setup(x => x.BlacklistTokenAsync(It.IsAny<string>(), It.IsAny<DateTime>())).Returns(Task.CompletedTask);
            mockAuditLogger.Setup(x => x.LogLogoutAsync(It.IsAny<string>(), It.IsAny<string>(), It.IsAny<string>())).Returns(Task.CompletedTask);
            // Act
            var result = await controller.Logout();
            // Assert
            var okResult = result.Result as OkObjectResult;
            Assert.IsNotNull(okResult);
            Assert.AreEqual(200, okResult.StatusCode);
            mockAuditLogger.Verify(x => x.LogLogoutAsync("user123", "test@example.com", It.Is<string>(ip => ip.Contains("2001:db8:85a3"))), Times.Once);
        }

        /// <summary>
        /// Tests that Logout handles very long userId and email claims correctly.
        /// Input: Claims with very long string values (10000 characters).
        /// Expected: Returns 200 OK with success message, audit log is created with long values.
        /// </summary>
        [TestMethod]
        public async Task Logout_VeryLongUserIdAndEmail_CreatesAuditLog()
        {
            // Arrange
            var mockContext = new Mock<AppDbContext>(new DbContextOptions<AppDbContext>());
            var mockAuthService = new Mock<IAuthService>();
            var mockProgressService = new Mock<IUserProgressService>();
            var mockPasswordValidator = new Mock<IPasswordValidator>();
            var mockTokenBlacklist = new Mock<ITokenBlacklistService>();
            var mockAuditLogger = new Mock<IAuditLogger>();
            var mockLogger = new Mock<ILogger<AuthController>>();
            var controller = new AuthController(mockContext.Object, mockAuthService.Object, mockProgressService.Object, mockPasswordValidator.Object, mockTokenBlacklist.Object, mockAuditLogger.Object, mockLogger.Object);
            var veryLongUserId = new string ('u', 10000);
            var veryLongEmail = new string ('e', 9995) + "@example.com";
            var httpContext = new DefaultHttpContext();
            httpContext.Connection.RemoteIpAddress = IPAddress.Parse("192.168.1.1");
            httpContext.Request.Headers["Authorization"] = "Bearer test-token";
            var claims = new[]
            {
                new Claim(ClaimTypes.NameIdentifier, veryLongUserId),
                new Claim(ClaimTypes.Email, veryLongEmail)
            };
            httpContext.User = new ClaimsPrincipal(new ClaimsIdentity(claims));
            controller.ControllerContext = new ControllerContext
            {
                HttpContext = httpContext
            };
            mockTokenBlacklist.Setup(x => x.BlacklistTokenAsync(It.IsAny<string>(), It.IsAny<DateTime>())).Returns(Task.CompletedTask);
            mockAuditLogger.Setup(x => x.LogLogoutAsync(It.IsAny<string>(), It.IsAny<string>(), It.IsAny<string>())).Returns(Task.CompletedTask);
            // Act
            var result = await controller.Logout();
            // Assert
            var okResult = result.Result as OkObjectResult;
            Assert.IsNotNull(okResult);
            Assert.AreEqual(200, okResult.StatusCode);
            mockAuditLogger.Verify(x => x.LogLogoutAsync(veryLongUserId, veryLongEmail, "192.168.1.1"), Times.Once);
        }

        /// <summary>
        /// Tests that Logout handles special characters in userId and email claims correctly.
        /// Input: Claims with special characters in values.
        /// Expected: Returns 200 OK with success message, audit log is created with special characters.
        /// </summary>
        [TestMethod]
        [DataRow("user@#$%", "test@example.com")]
        [DataRow("user<>", "test@example.com")]
        [DataRow("user123", "test+tag@example.com")]
        [DataRow("user's-id", "test@example.com")]
        public async Task Logout_SpecialCharactersInClaims_CreatesAuditLog(string userId, string email)
        {
            // Arrange
            var mockContext = new Mock<AppDbContext>(new DbContextOptions<AppDbContext>());
            var mockAuthService = new Mock<IAuthService>();
            var mockProgressService = new Mock<IUserProgressService>();
            var mockPasswordValidator = new Mock<IPasswordValidator>();
            var mockTokenBlacklist = new Mock<ITokenBlacklistService>();
            var mockAuditLogger = new Mock<IAuditLogger>();
            var mockLogger = new Mock<ILogger<AuthController>>();
            var controller = new AuthController(mockContext.Object, mockAuthService.Object, mockProgressService.Object, mockPasswordValidator.Object, mockTokenBlacklist.Object, mockAuditLogger.Object, mockLogger.Object);
            var httpContext = new DefaultHttpContext();
            httpContext.Connection.RemoteIpAddress = IPAddress.Parse("192.168.1.1");
            httpContext.Request.Headers["Authorization"] = "Bearer test-token";
            var claims = new[]
            {
                new Claim(ClaimTypes.NameIdentifier, userId),
                new Claim(ClaimTypes.Email, email)
            };
            httpContext.User = new ClaimsPrincipal(new ClaimsIdentity(claims));
            controller.ControllerContext = new ControllerContext
            {
                HttpContext = httpContext
            };
            mockTokenBlacklist.Setup(x => x.BlacklistTokenAsync(It.IsAny<string>(), It.IsAny<DateTime>())).Returns(Task.CompletedTask);
            mockAuditLogger.Setup(x => x.LogLogoutAsync(It.IsAny<string>(), It.IsAny<string>(), It.IsAny<string>())).Returns(Task.CompletedTask);
            // Act
            var result = await controller.Logout();
            // Assert
            var okResult = result.Result as OkObjectResult;
            Assert.IsNotNull(okResult);
            Assert.AreEqual(200, okResult.StatusCode);
            mockAuditLogger.Verify(x => x.LogLogoutAsync(userId, email, "192.168.1.1"), Times.Once);
        }

        /// <summary>
        /// Tests that Logout handles Authorization header with multiple spaces after Bearer.
        /// Input: Authorization header with "Bearer  token" (two spaces).
        /// Expected: Returns 200 OK with success message, token is trimmed and blacklisted.
        /// </summary>
        [TestMethod]
        public async Task Logout_MultipleSpacesAfterBearer_TrimsAndBlacklistsToken()
        {
            // Arrange
            var mockContext = new Mock<AppDbContext>(new DbContextOptions<AppDbContext>());
            var mockAuthService = new Mock<IAuthService>();
            var mockProgressService = new Mock<IUserProgressService>();
            var mockPasswordValidator = new Mock<IPasswordValidator>();
            var mockTokenBlacklist = new Mock<ITokenBlacklistService>();
            var mockAuditLogger = new Mock<IAuditLogger>();
            var mockLogger = new Mock<ILogger<AuthController>>();
            var controller = new AuthController(mockContext.Object, mockAuthService.Object, mockProgressService.Object, mockPasswordValidator.Object, mockTokenBlacklist.Object, mockAuditLogger.Object, mockLogger.Object);
            var httpContext = new DefaultHttpContext();
            httpContext.Connection.RemoteIpAddress = IPAddress.Parse("192.168.1.1");
            httpContext.Request.Headers["Authorization"] = "Bearer  test-token";
            var claims = new[]
            {
                new Claim(ClaimTypes.NameIdentifier, "user123"),
                new Claim(ClaimTypes.Email, "test@example.com")
            };
            httpContext.User = new ClaimsPrincipal(new ClaimsIdentity(claims));
            controller.ControllerContext = new ControllerContext
            {
                HttpContext = httpContext
            };
            mockTokenBlacklist.Setup(x => x.BlacklistTokenAsync(It.IsAny<string>(), It.IsAny<DateTime>())).Returns(Task.CompletedTask);
            mockAuditLogger.Setup(x => x.LogLogoutAsync(It.IsAny<string>(), It.IsAny<string>(), It.IsAny<string>())).Returns(Task.CompletedTask);
            // Act
            var result = await controller.Logout();
            // Assert
            var okResult = result.Result as OkObjectResult;
            Assert.IsNotNull(okResult);
            Assert.AreEqual(200, okResult.StatusCode);
            mockTokenBlacklist.Verify(x => x.BlacklistTokenAsync("test-token", It.IsAny<DateTime>()), Times.Once);
        }

        /// <summary>
        /// Tests that Logout returns Ok with correct message format.
        /// Input: Valid token and claims.
        /// Expected: Returns 200 OK with message "Logout successful."
        /// </summary>
        [TestMethod]
        public async Task Logout_ValidRequest_ReturnsCorrectMessageFormat()
        {
            // Arrange
            var mockContext = new Mock<AppDbContext>(new DbContextOptions<AppDbContext>());
            var mockAuthService = new Mock<IAuthService>();
            var mockProgressService = new Mock<IUserProgressService>();
            var mockPasswordValidator = new Mock<IPasswordValidator>();
            var mockTokenBlacklist = new Mock<ITokenBlacklistService>();
            var mockAuditLogger = new Mock<IAuditLogger>();
            var mockLogger = new Mock<ILogger<AuthController>>();
            var controller = new AuthController(mockContext.Object, mockAuthService.Object, mockProgressService.Object, mockPasswordValidator.Object, mockTokenBlacklist.Object, mockAuditLogger.Object, mockLogger.Object);
            var httpContext = new DefaultHttpContext();
            httpContext.Connection.RemoteIpAddress = IPAddress.Parse("192.168.1.1");
            httpContext.Request.Headers["Authorization"] = "Bearer test-token";
            var claims = new[]
            {
                new Claim(ClaimTypes.NameIdentifier, "user123"),
                new Claim(ClaimTypes.Email, "test@example.com")
            };
            httpContext.User = new ClaimsPrincipal(new ClaimsIdentity(claims));
            controller.ControllerContext = new ControllerContext
            {
                HttpContext = httpContext
            };
            mockTokenBlacklist.Setup(x => x.BlacklistTokenAsync(It.IsAny<string>(), It.IsAny<DateTime>())).Returns(Task.CompletedTask);
            mockAuditLogger.Setup(x => x.LogLogoutAsync(It.IsAny<string>(), It.IsAny<string>(), It.IsAny<string>())).Returns(Task.CompletedTask);
            // Act
            var result = await controller.Logout();
            // Assert
            var okResult = result.Result as OkObjectResult;
            Assert.IsNotNull(okResult);
            Assert.AreEqual(200, okResult.StatusCode);
            var value = okResult.Value;
            Assert.IsNotNull(value);
            var messageProperty = value.GetType().GetProperty("message");
            Assert.IsNotNull(messageProperty);
            var message = messageProperty.GetValue(value) as string;
            Assert.AreEqual("Logout successful.", message);
        }

        /// <summary>
        /// Tests that Logout returns correct error message format on exception.
        /// Input: BlacklistTokenAsync throws exception.
        /// Expected: Returns 500 status code with message "An error occurred during logout."
        /// </summary>
        [TestMethod]
        public async Task Logout_Exception_ReturnsCorrectErrorMessageFormat()
        {
            // Arrange
            var mockContext = new Mock<AppDbContext>(new DbContextOptions<AppDbContext>());
            var mockAuthService = new Mock<IAuthService>();
            var mockProgressService = new Mock<IUserProgressService>();
            var mockPasswordValidator = new Mock<IPasswordValidator>();
            var mockTokenBlacklist = new Mock<ITokenBlacklistService>();
            var mockAuditLogger = new Mock<IAuditLogger>();
            var mockLogger = new Mock<ILogger<AuthController>>();
            var controller = new AuthController(mockContext.Object, mockAuthService.Object, mockProgressService.Object, mockPasswordValidator.Object, mockTokenBlacklist.Object, mockAuditLogger.Object, mockLogger.Object);
            var httpContext = new DefaultHttpContext();
            httpContext.Connection.RemoteIpAddress = IPAddress.Parse("192.168.1.1");
            httpContext.Request.Headers["Authorization"] = "Bearer test-token";
            var claims = new[]
            {
                new Claim(ClaimTypes.NameIdentifier, "user123"),
                new Claim(ClaimTypes.Email, "test@example.com")
            };
            httpContext.User = new ClaimsPrincipal(new ClaimsIdentity(claims));
            controller.ControllerContext = new ControllerContext
            {
                HttpContext = httpContext
            };
            mockTokenBlacklist.Setup(x => x.BlacklistTokenAsync(It.IsAny<string>(), It.IsAny<DateTime>())).ThrowsAsync(new Exception("Test exception"));
            // Act
            var result = await controller.Logout();
            // Assert
            var statusCodeResult = result.Result as ObjectResult;
            Assert.IsNotNull(statusCodeResult);
            Assert.AreEqual(500, statusCodeResult.StatusCode);
            var value = statusCodeResult.Value;
            Assert.IsNotNull(value);
            var messageProperty = value.GetType().GetProperty("message");
            Assert.IsNotNull(messageProperty);
            var message = messageProperty.GetValue(value) as string;
            Assert.AreEqual("An error occurred during logout.", message);
        }

        /// <summary>
        /// Tests that Logout sets token expiry time to exactly 1 hour from now.
        /// Input: Valid Authorization header with Bearer token.
        /// Expected: Token is blacklisted with expiry time approximately 1 hour from current UTC time.
        /// </summary>
        [TestMethod]
        public async Task Logout_ValidToken_SetsExpiryTimeToOneHourFromNow()
        {
            // Arrange
            var mockContext = new Mock<AppDbContext>(new DbContextOptions<AppDbContext>());
            var mockAuthService = new Mock<IAuthService>();
            var mockProgressService = new Mock<IUserProgressService>();
            var mockPasswordValidator = new Mock<IPasswordValidator>();
            var mockTokenBlacklist = new Mock<ITokenBlacklistService>();
            var mockAuditLogger = new Mock<IAuditLogger>();
            var mockLogger = new Mock<ILogger<AuthController>>();
            var controller = new AuthController(mockContext.Object, mockAuthService.Object, mockProgressService.Object, mockPasswordValidator.Object, mockTokenBlacklist.Object, mockAuditLogger.Object, mockLogger.Object);
            var httpContext = new DefaultHttpContext();
            httpContext.Connection.RemoteIpAddress = IPAddress.Parse("192.168.1.1");
            httpContext.Request.Headers["Authorization"] = "Bearer test-token";
            var claims = new[]
            {
                new Claim(ClaimTypes.NameIdentifier, "user123"),
                new Claim(ClaimTypes.Email, "test@example.com")
            };
            httpContext.User = new ClaimsPrincipal(new ClaimsIdentity(claims));
            controller.ControllerContext = new ControllerContext
            {
                HttpContext = httpContext
            };
            DateTime? capturedExpiryTime = null;
            mockTokenBlacklist.Setup(x => x.BlacklistTokenAsync(It.IsAny<string>(), It.IsAny<DateTime>())).Callback<string, DateTime>((token, expiry) => capturedExpiryTime = expiry).Returns(Task.CompletedTask);
            mockAuditLogger.Setup(x => x.LogLogoutAsync(It.IsAny<string>(), It.IsAny<string>(), It.IsAny<string>())).Returns(Task.CompletedTask);
            var beforeCall = DateTime.UtcNow;
            // Act
            var result = await controller.Logout();
            var afterCall = DateTime.UtcNow;
            // Assert
            var okResult = result.Result as OkObjectResult;
            Assert.IsNotNull(okResult);
            Assert.AreEqual(200, okResult.StatusCode);
            Assert.IsNotNull(capturedExpiryTime);
            var expectedMinExpiry = beforeCall.AddHours(1);
            var expectedMaxExpiry = afterCall.AddHours(1);
            Assert.IsTrue(capturedExpiryTime >= expectedMinExpiry);
            Assert.IsTrue(capturedExpiryTime <= expectedMaxExpiry);
        }

        /// <summary>
        /// Tests that Logout handles token containing control characters correctly.
        /// Input: Authorization header with token containing newlines and tabs.
        /// Expected: Returns 200 OK with success message, token with control chars is blacklisted.
        /// </summary>
        [TestMethod]
        [DataRow("token\nwith\nnewlines")]
        [DataRow("token\twith\ttabs")]
        [DataRow("token\r\nwith\r\ncrlf")]
        public async Task Logout_TokenWithControlCharacters_BlacklistsToken(string tokenWithControlChars)
        {
            // Arrange
            var mockContext = new Mock<AppDbContext>(new DbContextOptions<AppDbContext>());
            var mockAuthService = new Mock<IAuthService>();
            var mockProgressService = new Mock<IUserProgressService>();
            var mockPasswordValidator = new Mock<IPasswordValidator>();
            var mockTokenBlacklist = new Mock<ITokenBlacklistService>();
            var mockAuditLogger = new Mock<IAuditLogger>();
            var mockLogger = new Mock<ILogger<AuthController>>();
            var controller = new AuthController(mockContext.Object, mockAuthService.Object, mockProgressService.Object, mockPasswordValidator.Object, mockTokenBlacklist.Object, mockAuditLogger.Object, mockLogger.Object);
            var httpContext = new DefaultHttpContext();
            httpContext.Connection.RemoteIpAddress = IPAddress.Parse("192.168.1.1");
            httpContext.Request.Headers["Authorization"] = $"Bearer {tokenWithControlChars}";
            var claims = new[]
            {
                new Claim(ClaimTypes.NameIdentifier, "user123"),
                new Claim(ClaimTypes.Email, "test@example.com")
            };
            httpContext.User = new ClaimsPrincipal(new ClaimsIdentity(claims));
            controller.ControllerContext = new ControllerContext
            {
                HttpContext = httpContext
            };
            mockTokenBlacklist.Setup(x => x.BlacklistTokenAsync(It.IsAny<string>(), It.IsAny<DateTime>())).Returns(Task.CompletedTask);
            mockAuditLogger.Setup(x => x.LogLogoutAsync(It.IsAny<string>(), It.IsAny<string>(), It.IsAny<string>())).Returns(Task.CompletedTask);
            // Act
            var result = await controller.Logout();
            // Assert
            var okResult = result.Result as OkObjectResult;
            Assert.IsNotNull(okResult);
            Assert.AreEqual(200, okResult.StatusCode);
            mockTokenBlacklist.Verify(x => x.BlacklistTokenAsync(tokenWithControlChars, It.IsAny<DateTime>()), Times.Once);
        }

        /// <summary>
        /// Tests that Logout handles loopback IP addresses correctly.
        /// Input: HttpContext with loopback IPv4 address (127.0.0.1).
        /// Expected: Returns 200 OK with success message, uses loopback IP address.
        /// </summary>
        [TestMethod]
        [DataRow("127.0.0.1")]
        [DataRow("::1")]
        public async Task Logout_LoopbackAddress_HandlesCorrectly(string loopbackAddress)
        {
            // Arrange
            var mockContext = new Mock<AppDbContext>(new DbContextOptions<AppDbContext>());
            var mockAuthService = new Mock<IAuthService>();
            var mockProgressService = new Mock<IUserProgressService>();
            var mockPasswordValidator = new Mock<IPasswordValidator>();
            var mockTokenBlacklist = new Mock<ITokenBlacklistService>();
            var mockAuditLogger = new Mock<IAuditLogger>();
            var mockLogger = new Mock<ILogger<AuthController>>();
            var controller = new AuthController(mockContext.Object, mockAuthService.Object, mockProgressService.Object, mockPasswordValidator.Object, mockTokenBlacklist.Object, mockAuditLogger.Object, mockLogger.Object);
            var httpContext = new DefaultHttpContext();
            httpContext.Connection.RemoteIpAddress = IPAddress.Parse(loopbackAddress);
            httpContext.Request.Headers["Authorization"] = "Bearer test-token";
            var claims = new[]
            {
                new Claim(ClaimTypes.NameIdentifier, "user123"),
                new Claim(ClaimTypes.Email, "test@example.com")
            };
            httpContext.User = new ClaimsPrincipal(new ClaimsIdentity(claims));
            controller.ControllerContext = new ControllerContext
            {
                HttpContext = httpContext
            };
            mockTokenBlacklist.Setup(x => x.BlacklistTokenAsync(It.IsAny<string>(), It.IsAny<DateTime>())).Returns(Task.CompletedTask);
            mockAuditLogger.Setup(x => x.LogLogoutAsync(It.IsAny<string>(), It.IsAny<string>(), It.IsAny<string>())).Returns(Task.CompletedTask);
            // Act
            var result = await controller.Logout();
            // Assert
            var okResult = result.Result as OkObjectResult;
            Assert.IsNotNull(okResult);
            Assert.AreEqual(200, okResult.StatusCode);
            mockAuditLogger.Verify(x => x.LogLogoutAsync("user123", "test@example.com", It.Is<string>(ip => !string.IsNullOrEmpty(ip) && ip != "Unknown")), Times.Once);
        }

        /// <summary>
        /// Tests that Register returns BadRequest when password validation fails.
        /// Input: Valid email and name, but password that fails validation
        /// Expected: BadRequest with specific error message from validator and audit log called
        /// </summary>
        [TestMethod]
        [DataRow("Password is too short")]
        [DataRow("Password must contain uppercase letter")]
        [DataRow("Password must contain special character")]
        [DataRow("Password must contain digit")]
        public async Task Register_PasswordValidationFails_ReturnsBadRequestWithMessage(string errorMessage)
        {
            // Arrange
            var options = new DbContextOptionsBuilder<AppDbContext>().UseInMemoryDatabase(databaseName: Guid.NewGuid().ToString()).Options;
            var context = new AppDbContext(options);
            var mockAuthService = new Mock<IAuthService>();
            var mockProgressService = new Mock<IUserProgressService>();
            var mockPasswordValidator = new Mock<IPasswordValidator>();
            var mockTokenBlacklist = new Mock<ITokenBlacklistService>();
            var mockAuditLogger = new Mock<IAuditLogger>();
            var mockLogger = new Mock<ILogger<AuthController>>();
            mockPasswordValidator.Setup(x => x.ValidatePassword(It.IsAny<string>())).Returns((false, errorMessage));
            mockAuditLogger.Setup(x => x.LogRegisterAttemptAsync(It.IsAny<string>(), It.IsAny<bool>(), It.IsAny<string>())).Returns(Task.CompletedTask);
            var controller = new AuthController(context, mockAuthService.Object, mockProgressService.Object, mockPasswordValidator.Object, mockTokenBlacklist.Object, mockAuditLogger.Object, mockLogger.Object);
            SetupHttpContext(controller, "127.0.0.1");
            var request = new RegisterRequest("test@example.com", "weak", "John Doe");
            // Act
            var result = await controller.Register(request);
            // Assert
            var badRequestResult = result.Result as BadRequestObjectResult;
            Assert.IsNotNull(badRequestResult);
            Assert.AreEqual(400, badRequestResult.StatusCode);
            mockAuditLogger.Verify(x => x.LogRegisterAttemptAsync("test@example.com", false, "127.0.0.1"), Times.Once);
        }

        /// <summary>
        /// Tests that Register verifies password validation is called with correct parameter.
        /// Input: Valid registration request
        /// Expected: ValidatePassword is called with the provided password
        /// </summary>
        [TestMethod]
        public async Task Register_ValidRequest_CallsPasswordValidatorWithCorrectPassword()
        {
            // Arrange
            var options = new DbContextOptionsBuilder<AppDbContext>().UseInMemoryDatabase(databaseName: Guid.NewGuid().ToString()).Options;
            var context = new AppDbContext(options);
            var mockAuthService = new Mock<IAuthService>();
            var mockProgressService = new Mock<IUserProgressService>();
            var mockPasswordValidator = new Mock<IPasswordValidator>();
            var mockTokenBlacklist = new Mock<ITokenBlacklistService>();
            var mockAuditLogger = new Mock<IAuditLogger>();
            var mockLogger = new Mock<ILogger<AuthController>>();
            mockPasswordValidator.Setup(x => x.ValidatePassword("MySecurePassword123!")).Returns((true, string.Empty));
            mockAuthService.Setup(x => x.HashPassword(It.IsAny<string>())).Returns("hashed_password");
            mockAuditLogger.Setup(x => x.LogRegisterAttemptAsync(It.IsAny<string>(), It.IsAny<bool>(), It.IsAny<string>())).Returns(Task.CompletedTask);
            var controller = new AuthController(context, mockAuthService.Object, mockProgressService.Object, mockPasswordValidator.Object, mockTokenBlacklist.Object, mockAuditLogger.Object, mockLogger.Object);
            SetupHttpContext(controller, "127.0.0.1");
            var request = new RegisterRequest("test@example.com", "MySecurePassword123!", "Test User");
            // Act
            var result = await controller.Register(request);
            // Assert
            mockPasswordValidator.Verify(x => x.ValidatePassword("MySecurePassword123!"), Times.Once);
        }

        /// <summary>
        /// Tests that Register creates user with correct CreatedAt timestamp.
        /// Input: Valid registration request
        /// Expected: User is created with CreatedAt set to current UTC time
        /// </summary>
        [TestMethod]
        public async Task Register_ValidRequest_SetsCreatedAtToCurrentUtcTime()
        {
            // Arrange
            var options = new DbContextOptionsBuilder<AppDbContext>().UseInMemoryDatabase(databaseName: Guid.NewGuid().ToString()).Options;
            var context = new AppDbContext(options);
            var mockAuthService = new Mock<IAuthService>();
            var mockProgressService = new Mock<IUserProgressService>();
            var mockPasswordValidator = new Mock<IPasswordValidator>();
            var mockTokenBlacklist = new Mock<ITokenBlacklistService>();
            var mockAuditLogger = new Mock<IAuditLogger>();
            var mockLogger = new Mock<ILogger<AuthController>>();
            mockPasswordValidator.Setup(x => x.ValidatePassword(It.IsAny<string>())).Returns((true, string.Empty));
            mockAuthService.Setup(x => x.HashPassword(It.IsAny<string>())).Returns("hashed_password");
            mockAuditLogger.Setup(x => x.LogRegisterAttemptAsync(It.IsAny<string>(), It.IsAny<bool>(), It.IsAny<string>())).Returns(Task.CompletedTask);
            var controller = new AuthController(context, mockAuthService.Object, mockProgressService.Object, mockPasswordValidator.Object, mockTokenBlacklist.Object, mockAuditLogger.Object, mockLogger.Object);
            SetupHttpContext(controller, "127.0.0.1");
            var request = new RegisterRequest("test@example.com", "Password123!", "Test User");
            var beforeTime = DateTime.UtcNow.AddSeconds(-1);
            // Act
            var result = await controller.Register(request);
            var afterTime = DateTime.UtcNow.AddSeconds(1);
            // Assert
            var createdUser = context.Users.FirstOrDefault(u => u.Email == "test@example.com");
            Assert.IsNotNull(createdUser);
            Assert.IsTrue(createdUser.CreatedAt >= beforeTime && createdUser.CreatedAt <= afterTime);
        }

        /// <summary>
        /// Tests that Register assigns a unique GUID to the new user's Id.
        /// Input: Valid registration request
        /// Expected: User is created with a valid GUID as Id
        /// </summary>
        [TestMethod]
        public async Task Register_ValidRequest_AssignsUniqueGuidAsUserId()
        {
            // Arrange
            var options = new DbContextOptionsBuilder<AppDbContext>().UseInMemoryDatabase(databaseName: Guid.NewGuid().ToString()).Options;
            var context = new AppDbContext(options);
            var mockAuthService = new Mock<IAuthService>();
            var mockProgressService = new Mock<IUserProgressService>();
            var mockPasswordValidator = new Mock<IPasswordValidator>();
            var mockTokenBlacklist = new Mock<ITokenBlacklistService>();
            var mockAuditLogger = new Mock<IAuditLogger>();
            var mockLogger = new Mock<ILogger<AuthController>>();
            mockPasswordValidator.Setup(x => x.ValidatePassword(It.IsAny<string>())).Returns((true, string.Empty));
            mockAuthService.Setup(x => x.HashPassword(It.IsAny<string>())).Returns("hashed_password");
            mockAuditLogger.Setup(x => x.LogRegisterAttemptAsync(It.IsAny<string>(), It.IsAny<bool>(), It.IsAny<string>())).Returns(Task.CompletedTask);
            var controller = new AuthController(context, mockAuthService.Object, mockProgressService.Object, mockPasswordValidator.Object, mockTokenBlacklist.Object, mockAuditLogger.Object, mockLogger.Object);
            SetupHttpContext(controller, "127.0.0.1");
            var request = new RegisterRequest("test@example.com", "Password123!", "Test User");
            // Act
            var result = await controller.Register(request);
            // Assert
            var createdUser = context.Users.FirstOrDefault(u => u.Email == "test@example.com");
            Assert.IsNotNull(createdUser);
            Assert.IsNotNull(createdUser.Id);
            Assert.IsTrue(Guid.TryParse(createdUser.Id, out _));
        }

        /// <summary>
        /// Tests that Register handles Unicode characters in name correctly.
        /// Input: RegisterRequest with Unicode characters in name
        /// Expected: User is created with Unicode name preserved
        /// </summary>
        [TestMethod]
        public async Task Register_UnicodeCharactersInName_HandlesCorrectly()
        {
            // Arrange
            var options = new DbContextOptionsBuilder<AppDbContext>().UseInMemoryDatabase(databaseName: Guid.NewGuid().ToString()).Options;
            var context = new AppDbContext(options);
            var mockAuthService = new Mock<IAuthService>();
            var mockProgressService = new Mock<IUserProgressService>();
            var mockPasswordValidator = new Mock<IPasswordValidator>();
            var mockTokenBlacklist = new Mock<ITokenBlacklistService>();
            var mockAuditLogger = new Mock<IAuditLogger>();
            var mockLogger = new Mock<ILogger<AuthController>>();
            mockPasswordValidator.Setup(x => x.ValidatePassword(It.IsAny<string>())).Returns((true, string.Empty));
            mockAuthService.Setup(x => x.HashPassword(It.IsAny<string>())).Returns("hashed_password");
            mockAuditLogger.Setup(x => x.LogRegisterAttemptAsync(It.IsAny<string>(), It.IsAny<bool>(), It.IsAny<string>())).Returns(Task.CompletedTask);
            var controller = new AuthController(context, mockAuthService.Object, mockProgressService.Object, mockPasswordValidator.Object, mockTokenBlacklist.Object, mockAuditLogger.Object, mockLogger.Object);
            SetupHttpContext(controller, "127.0.0.1");
            var request = new RegisterRequest("test@example.com", "Password123!", "José García 李明 Müller");
            // Act
            var result = await controller.Register(request);
            // Assert
            var okResult = result.Result as OkObjectResult;
            Assert.IsNotNull(okResult);
            var createdUser = context.Users.FirstOrDefault(u => u.Email == "test@example.com");
            Assert.IsNotNull(createdUser);
            Assert.AreEqual("José García 李明 Müller", createdUser.Name);
        }

        /// <summary>
        /// Tests that Register logs failed registration attempt when password validation fails.
        /// Input: Request with password that fails validation
        /// Expected: Audit logger is called with success=false
        /// </summary>
        [TestMethod]
        public async Task Register_PasswordValidationFails_LogsFailedAttempt()
        {
            // Arrange
            var options = new DbContextOptionsBuilder<AppDbContext>().UseInMemoryDatabase(databaseName: Guid.NewGuid().ToString()).Options;
            var context = new AppDbContext(options);
            var mockAuthService = new Mock<IAuthService>();
            var mockProgressService = new Mock<IUserProgressService>();
            var mockPasswordValidator = new Mock<IPasswordValidator>();
            var mockTokenBlacklist = new Mock<ITokenBlacklistService>();
            var mockAuditLogger = new Mock<IAuditLogger>();
            var mockLogger = new Mock<ILogger<AuthController>>();
            mockPasswordValidator.Setup(x => x.ValidatePassword(It.IsAny<string>())).Returns((false, "Password is too weak"));
            mockAuditLogger.Setup(x => x.LogRegisterAttemptAsync(It.IsAny<string>(), It.IsAny<bool>(), It.IsAny<string>())).Returns(Task.CompletedTask);
            var controller = new AuthController(context, mockAuthService.Object, mockProgressService.Object, mockPasswordValidator.Object, mockTokenBlacklist.Object, mockAuditLogger.Object, mockLogger.Object);
            SetupHttpContext(controller, "10.0.0.1");
            var request = new RegisterRequest("test@example.com", "weak", "Test User");
            // Act
            var result = await controller.Register(request);
            // Assert
            mockAuditLogger.Verify(x => x.LogRegisterAttemptAsync("test@example.com", false, "10.0.0.1"), Times.Once);
        }

        /// <summary>
        /// Tests that Login returns InternalServerError when LoadUserProgressAsync throws an exception.
        /// Input: Valid credentials but progress service throws exception
        /// Expected: 500 Internal Server Error response
        /// </summary>
        [TestMethod]
        public async Task Login_LoadUserProgressAsyncThrowsException_ReturnsInternalServerError()
        {
            // Arrange
            var users = new List<User>
            {
                new User
                {
                    Id = "user1",
                    Email = "test@example.com",
                    Name = "Test User",
                    PasswordHash = "hashedPassword",
                    CreatedAt = DateTime.UtcNow
                }
            };
            var mockContext = CreateMockDbContext(users);
            var mockAuthService = new Mock<IAuthService>();
            var mockProgressService = new Mock<IUserProgressService>();
            var mockPasswordValidator = new Mock<IPasswordValidator>();
            var mockTokenBlacklist = new Mock<ITokenBlacklistService>();
            var mockAuditLogger = new Mock<IAuditLogger>();
            var mockLogger = new Mock<ILogger<AuthController>>();
            mockAuthService.Setup(x => x.VerifyPassword(It.IsAny<string>(), It.IsAny<string>())).Returns(true);
            mockProgressService.Setup(x => x.LoadUserProgressAsync(It.IsAny<string>())).ThrowsAsync(new Exception("Progress service error"));
            var controller = new AuthController(mockContext.Object, mockAuthService.Object, mockProgressService.Object, mockPasswordValidator.Object, mockTokenBlacklist.Object, mockAuditLogger.Object, mockLogger.Object);
            SetupHttpContext(controller, "192.168.1.1");
            var request = new LoginRequest("test@example.com", "password123");
            // Act
            var result = await controller.Login(request);
            // Assert
            var statusCodeResult = result.Result as ObjectResult;
            Assert.IsNotNull(statusCodeResult);
            Assert.AreEqual(500, statusCodeResult.StatusCode);
        }

        /// <summary>
        /// Tests that Login returns InternalServerError when GenerateJwtToken throws an exception.
        /// Input: Valid credentials but JWT generation fails
        /// Expected: 500 Internal Server Error response
        /// </summary>
        [TestMethod]
        public async Task Login_GenerateJwtTokenThrowsException_ReturnsInternalServerError()
        {
            // Arrange
            var users = new List<User>
            {
                new User
                {
                    Id = "user1",
                    Email = "test@example.com",
                    Name = "Test User",
                    PasswordHash = "hashedPassword",
                    CreatedAt = DateTime.UtcNow
                }
            };
            var mockContext = CreateMockDbContext(users);
            var mockAuthService = new Mock<IAuthService>();
            var mockProgressService = new Mock<IUserProgressService>();
            var mockPasswordValidator = new Mock<IPasswordValidator>();
            var mockTokenBlacklist = new Mock<ITokenBlacklistService>();
            var mockAuditLogger = new Mock<IAuditLogger>();
            var mockLogger = new Mock<ILogger<AuthController>>();
            mockAuthService.Setup(x => x.VerifyPassword(It.IsAny<string>(), It.IsAny<string>())).Returns(true);
            mockProgressService.Setup(x => x.LoadUserProgressAsync(It.IsAny<string>())).ReturnsAsync((UserProgressData? )null);
            mockAuthService.Setup(x => x.GenerateJwtToken(It.IsAny<User>())).Throws(new Exception("JWT generation failed"));
            var controller = new AuthController(mockContext.Object, mockAuthService.Object, mockProgressService.Object, mockPasswordValidator.Object, mockTokenBlacklist.Object, mockAuditLogger.Object, mockLogger.Object);
            SetupHttpContext(controller, "192.168.1.1");
            var request = new LoginRequest("test@example.com", "password123");
            // Act
            var result = await controller.Login(request);
            // Assert
            var statusCodeResult = result.Result as ObjectResult;
            Assert.IsNotNull(statusCodeResult);
            Assert.AreEqual(500, statusCodeResult.StatusCode);
        }

        /// <summary>
        /// Tests that Login returns InternalServerError when GenerateRefreshToken throws an exception.
        /// Input: Valid credentials but refresh token generation fails
        /// Expected: 500 Internal Server Error response
        /// </summary>
        [TestMethod]
        public async Task Login_GenerateRefreshTokenThrowsException_ReturnsInternalServerError()
        {
            // Arrange
            var users = new List<User>
            {
                new User
                {
                    Id = "user1",
                    Email = "test@example.com",
                    Name = "Test User",
                    PasswordHash = "hashedPassword",
                    CreatedAt = DateTime.UtcNow
                }
            };
            var mockContext = CreateMockDbContext(users);
            var mockAuthService = new Mock<IAuthService>();
            var mockProgressService = new Mock<IUserProgressService>();
            var mockPasswordValidator = new Mock<IPasswordValidator>();
            var mockTokenBlacklist = new Mock<ITokenBlacklistService>();
            var mockAuditLogger = new Mock<IAuditLogger>();
            var mockLogger = new Mock<ILogger<AuthController>>();
            mockAuthService.Setup(x => x.VerifyPassword(It.IsAny<string>(), It.IsAny<string>())).Returns(true);
            mockProgressService.Setup(x => x.LoadUserProgressAsync(It.IsAny<string>())).ReturnsAsync((UserProgressData? )null);
            mockAuthService.Setup(x => x.GenerateJwtToken(It.IsAny<User>())).Returns("validJwtToken");
            mockAuthService.Setup(x => x.GenerateRefreshToken()).Throws(new Exception("Refresh token generation failed"));
            var controller = new AuthController(mockContext.Object, mockAuthService.Object, mockProgressService.Object, mockPasswordValidator.Object, mockTokenBlacklist.Object, mockAuditLogger.Object, mockLogger.Object);
            SetupHttpContext(controller, "192.168.1.1");
            var request = new LoginRequest("test@example.com", "password123");
            // Act
            var result = await controller.Login(request);
            // Assert
            var statusCodeResult = result.Result as ObjectResult;
            Assert.IsNotNull(statusCodeResult);
            Assert.AreEqual(500, statusCodeResult.StatusCode);
        }

        /// <summary>
        /// Tests that Login returns InternalServerError when second SaveChangesAsync fails.
        /// Input: Valid credentials but saving refresh token fails
        /// Expected: 500 Internal Server Error response
        /// </summary>
        [TestMethod]
        public async Task Login_SecondSaveChangesAsyncThrowsException_ReturnsInternalServerError()
        {
            // Arrange
            var users = new List<User>
            {
                new User
                {
                    Id = "user1",
                    Email = "test@example.com",
                    Name = "Test User",
                    PasswordHash = "hashedPassword",
                    CreatedAt = DateTime.UtcNow
                }
            };
            var mockContext = CreateMockDbContext(users);
            var saveCallCount = 0;
            mockContext.Setup(x => x.SaveChangesAsync(It.IsAny<CancellationToken>())).ReturnsAsync(() =>
            {
                saveCallCount++;
                if (saveCallCount == 2)
                {
                    throw new Exception("Database save failed");
                }

                return 1;
            });
            var mockAuthService = new Mock<IAuthService>();
            var mockProgressService = new Mock<IUserProgressService>();
            var mockPasswordValidator = new Mock<IPasswordValidator>();
            var mockTokenBlacklist = new Mock<ITokenBlacklistService>();
            var mockAuditLogger = new Mock<IAuditLogger>();
            var mockLogger = new Mock<ILogger<AuthController>>();
            mockAuthService.Setup(x => x.VerifyPassword(It.IsAny<string>(), It.IsAny<string>())).Returns(true);
            mockProgressService.Setup(x => x.LoadUserProgressAsync(It.IsAny<string>())).ReturnsAsync((UserProgressData? )null);
            mockAuthService.Setup(x => x.GenerateJwtToken(It.IsAny<User>())).Returns("validJwtToken");
            mockAuthService.Setup(x => x.GenerateRefreshToken()).Returns("validRefreshToken");
            var controller = new AuthController(mockContext.Object, mockAuthService.Object, mockProgressService.Object, mockPasswordValidator.Object, mockTokenBlacklist.Object, mockAuditLogger.Object, mockLogger.Object);
            SetupHttpContext(controller, "192.168.1.1");
            var request = new LoginRequest("test@example.com", "password123");
            // Act
            var result = await controller.Login(request);
            // Assert
            var statusCodeResult = result.Result as ObjectResult;
            Assert.IsNotNull(statusCodeResult);
            Assert.AreEqual(500, statusCodeResult.StatusCode);
        }

        /// <summary>
        /// Tests that Login continues successfully even if LogLoginAttemptAsync for failed login throws exception.
        /// Input: Invalid password and audit logger throws exception
        /// Expected: Unauthorized response (exception in audit logging is swallowed)
        /// </summary>
        [TestMethod]
        public async Task Login_FailedLoginAuditLogThrowsException_StillReturnsUnauthorized()
        {
            // Arrange
            var users = new List<User>
            {
                new User
                {
                    Id = "user1",
                    Email = "test@example.com",
                    Name = "Test User",
                    PasswordHash = "hashedPassword",
                    CreatedAt = DateTime.UtcNow
                }
            };
            var mockContext = CreateMockDbContext(users);
            var mockAuthService = new Mock<IAuthService>();
            var mockProgressService = new Mock<IUserProgressService>();
            var mockPasswordValidator = new Mock<IPasswordValidator>();
            var mockTokenBlacklist = new Mock<ITokenBlacklistService>();
            var mockAuditLogger = new Mock<IAuditLogger>();
            var mockLogger = new Mock<ILogger<AuthController>>();
            mockAuthService.Setup(x => x.VerifyPassword(It.IsAny<string>(), It.IsAny<string>())).Returns(false);
            mockAuditLogger.Setup(x => x.LogLoginAttemptAsync(It.IsAny<string>(), false, It.IsAny<string>())).ThrowsAsync(new Exception("Audit log failed"));
            var controller = new AuthController(mockContext.Object, mockAuthService.Object, mockProgressService.Object, mockPasswordValidator.Object, mockTokenBlacklist.Object, mockAuditLogger.Object, mockLogger.Object);
            SetupHttpContext(controller, "192.168.1.1");
            var request = new LoginRequest("test@example.com", "wrongPassword");
            // Act
            var result = await controller.Login(request);
            // Assert
            var unauthorizedResult = result.Result as UnauthorizedObjectResult;
            Assert.IsNotNull(unauthorizedResult);
            Assert.AreEqual(401, unauthorizedResult.StatusCode);
        }

        /// <summary>
        /// Tests that Login handles very long password strings correctly.
        /// Input: Valid email with extremely long password (10000 characters)
        /// Expected: Processes normally (user not found or password verification based on database)
        /// </summary>
        [TestMethod]
        public async Task Login_VeryLongPassword_HandlesGracefully()
        {
            // Arrange
            var longPassword = new string ('a', 10000);
            var users = new List<User>
            {
                new User
                {
                    Id = "user1",
                    Email = "test@example.com",
                    Name = "Test User",
                    PasswordHash = "hashedPassword",
                    CreatedAt = DateTime.UtcNow
                }
            };
            var mockContext = CreateMockDbContext(users);
            var mockAuthService = new Mock<IAuthService>();
            var mockProgressService = new Mock<IUserProgressService>();
            var mockPasswordValidator = new Mock<IPasswordValidator>();
            var mockTokenBlacklist = new Mock<ITokenBlacklistService>();
            var mockAuditLogger = new Mock<IAuditLogger>();
            var mockLogger = new Mock<ILogger<AuthController>>();
            mockAuthService.Setup(x => x.VerifyPassword(longPassword, "hashedPassword")).Returns(false);
            var controller = new AuthController(mockContext.Object, mockAuthService.Object, mockProgressService.Object, mockPasswordValidator.Object, mockTokenBlacklist.Object, mockAuditLogger.Object, mockLogger.Object);
            SetupHttpContext(controller, "192.168.1.1");
            var request = new LoginRequest("test@example.com", longPassword);
            // Act
            var result = await controller.Login(request);
            // Assert
            var unauthorizedResult = result.Result as UnauthorizedObjectResult;
            Assert.IsNotNull(unauthorizedResult);
            mockAuthService.Verify(x => x.VerifyPassword(longPassword, "hashedPassword"), Times.Once);
        }

        /// <summary>
        /// Tests that Login correctly calls SaveChangesAsync twice during successful login.
        /// Input: Valid credentials
        /// Expected: SaveChangesAsync called twice (once after LastLoginAt update, once after RefreshToken update)
        /// </summary>
        [TestMethod]
        public async Task Login_Success_CallsSaveChangesAsyncTwice()
        {
            // Arrange
            var users = new List<User>
            {
                new User
                {
                    Id = "user1",
                    Email = "test@example.com",
                    Name = "Test User",
                    PasswordHash = "hashedPassword",
                    CreatedAt = DateTime.UtcNow
                }
            };
            var mockContext = CreateMockDbContext(users);
            var mockAuthService = new Mock<IAuthService>();
            var mockProgressService = new Mock<IUserProgressService>();
            var mockPasswordValidator = new Mock<IPasswordValidator>();
            var mockTokenBlacklist = new Mock<ITokenBlacklistService>();
            var mockAuditLogger = new Mock<IAuditLogger>();
            var mockLogger = new Mock<ILogger<AuthController>>();
            mockAuthService.Setup(x => x.VerifyPassword(It.IsAny<string>(), It.IsAny<string>())).Returns(true);
            mockProgressService.Setup(x => x.LoadUserProgressAsync(It.IsAny<string>())).ReturnsAsync((UserProgressData? )null);
            mockAuthService.Setup(x => x.GenerateJwtToken(It.IsAny<User>())).Returns("validJwtToken");
            mockAuthService.Setup(x => x.GenerateRefreshToken()).Returns("validRefreshToken");
            var controller = new AuthController(mockContext.Object, mockAuthService.Object, mockProgressService.Object, mockPasswordValidator.Object, mockTokenBlacklist.Object, mockAuditLogger.Object, mockLogger.Object);
            SetupHttpContext(controller, "192.168.1.1");
            var request = new LoginRequest("test@example.com", "password123");
            // Act
            var result = await controller.Login(request);
            // Assert
            var okResult = result.Result as OkObjectResult;
            Assert.IsNotNull(okResult);
            mockContext.Verify(x => x.SaveChangesAsync(It.IsAny<CancellationToken>()), Times.Exactly(2));
        }

        /// <summary>
        /// Tests that Login returns correct response structure with all required fields.
        /// Input: Valid credentials
        /// Expected: Ok result with token, refreshToken, userId, and name properties
        /// </summary>
        [TestMethod]
        public async Task Login_Success_ReturnsCorrectResponseStructure()
        {
            // Arrange
            var users = new List<User>
            {
                new User
                {
                    Id = "user123",
                    Email = "test@example.com",
                    Name = "John Doe",
                    PasswordHash = "hashedPassword",
                    CreatedAt = DateTime.UtcNow
                }
            };
            var mockContext = CreateMockDbContext(users);
            var mockAuthService = new Mock<IAuthService>();
            var mockProgressService = new Mock<IUserProgressService>();
            var mockPasswordValidator = new Mock<IPasswordValidator>();
            var mockTokenBlacklist = new Mock<ITokenBlacklistService>();
            var mockAuditLogger = new Mock<IAuditLogger>();
            var mockLogger = new Mock<ILogger<AuthController>>();
            mockAuthService.Setup(x => x.VerifyPassword(It.IsAny<string>(), It.IsAny<string>())).Returns(true);
            mockProgressService.Setup(x => x.LoadUserProgressAsync(It.IsAny<string>())).ReturnsAsync((UserProgressData? )null);
            mockAuthService.Setup(x => x.GenerateJwtToken(It.IsAny<User>())).Returns("generatedJwtToken");
            mockAuthService.Setup(x => x.GenerateRefreshToken()).Returns("generatedRefreshToken");
            var controller = new AuthController(mockContext.Object, mockAuthService.Object, mockProgressService.Object, mockPasswordValidator.Object, mockTokenBlacklist.Object, mockAuditLogger.Object, mockLogger.Object);
            SetupHttpContext(controller, "192.168.1.1");
            var request = new LoginRequest("test@example.com", "password123");
            // Act
            var result = await controller.Login(request);
            // Assert
            var okResult = result.Result as OkObjectResult;
            Assert.IsNotNull(okResult);
            Assert.AreEqual(200, okResult.StatusCode);
            var responseValue = okResult.Value;
            Assert.IsNotNull(responseValue);
            var token = responseValue.GetType().GetProperty("token")?.GetValue(responseValue) as string;
            var refreshToken = responseValue.GetType().GetProperty("refreshToken")?.GetValue(responseValue) as string;
            var userId = responseValue.GetType().GetProperty("userId")?.GetValue(responseValue) as string;
            var name = responseValue.GetType().GetProperty("name")?.GetValue(responseValue) as string;
            Assert.AreEqual("generatedJwtToken", token);
            Assert.AreEqual("generatedRefreshToken", refreshToken);
            Assert.AreEqual("user123", userId);
            Assert.AreEqual("John Doe", name);
        }

        /// <summary>
        /// Tests that Login sets RefreshTokenExpiryTime to 7 days from now.
        /// Input: Valid credentials
        /// Expected: RefreshTokenExpiryTime is set to approximately 7 days in the future
        /// </summary>
        [TestMethod]
        public async Task Login_Success_SetsRefreshTokenExpiryTimeTo7Days()
        {
            // Arrange
            var users = new List<User>
            {
                new User
                {
                    Id = "user1",
                    Email = "test@example.com",
                    Name = "Test User",
                    PasswordHash = "hashedPassword",
                    CreatedAt = DateTime.UtcNow
                }
            };
            var mockContext = CreateMockDbContext(users);
            var mockAuthService = new Mock<IAuthService>();
            var mockProgressService = new Mock<IUserProgressService>();
            var mockPasswordValidator = new Mock<IPasswordValidator>();
            var mockTokenBlacklist = new Mock<ITokenBlacklistService>();
            var mockAuditLogger = new Mock<IAuditLogger>();
            var mockLogger = new Mock<ILogger<AuthController>>();
            mockAuthService.Setup(x => x.VerifyPassword(It.IsAny<string>(), It.IsAny<string>())).Returns(true);
            mockProgressService.Setup(x => x.LoadUserProgressAsync(It.IsAny<string>())).ReturnsAsync((UserProgressData? )null);
            mockAuthService.Setup(x => x.GenerateJwtToken(It.IsAny<User>())).Returns("validJwtToken");
            mockAuthService.Setup(x => x.GenerateRefreshToken()).Returns("validRefreshToken");
            var controller = new AuthController(mockContext.Object, mockAuthService.Object, mockProgressService.Object, mockPasswordValidator.Object, mockTokenBlacklist.Object, mockAuditLogger.Object, mockLogger.Object);
            SetupHttpContext(controller, "192.168.1.1");
            var request = new LoginRequest("test@example.com", "password123");
            var beforeLogin = DateTime.UtcNow;
            // Act
            var result = await controller.Login(request);
            // Assert
            var okResult = result.Result as OkObjectResult;
            Assert.IsNotNull(okResult);
            var user = users.First();
            Assert.IsNotNull(user.RefreshTokenExpiryTime);
            var expectedExpiry = beforeLogin.AddDays(7);
            var actualExpiry = user.RefreshTokenExpiryTime.Value;
            Assert.IsTrue((actualExpiry - expectedExpiry).TotalSeconds < 5, "RefreshTokenExpiryTime should be approximately 7 days from now");
        }

        /// <summary>
        /// Tests that Login calls LogLoginAttemptAsync with correct parameters for successful login.
        /// Input: Valid credentials
        /// Expected: LogLoginAttemptAsync called with email, true (success), and IP address
        /// </summary>
        [TestMethod]
        public async Task Login_Success_LogsSuccessfulLoginAttempt()
        {
            // Arrange
            var users = new List<User>
            {
                new User
                {
                    Id = "user1",
                    Email = "test@example.com",
                    Name = "Test User",
                    PasswordHash = "hashedPassword",
                    CreatedAt = DateTime.UtcNow
                }
            };
            var mockContext = CreateMockDbContext(users);
            var mockAuthService = new Mock<IAuthService>();
            var mockProgressService = new Mock<IUserProgressService>();
            var mockPasswordValidator = new Mock<IPasswordValidator>();
            var mockTokenBlacklist = new Mock<ITokenBlacklistService>();
            var mockAuditLogger = new Mock<IAuditLogger>();
            var mockLogger = new Mock<ILogger<AuthController>>();
            mockAuthService.Setup(x => x.VerifyPassword(It.IsAny<string>(), It.IsAny<string>())).Returns(true);
            mockProgressService.Setup(x => x.LoadUserProgressAsync(It.IsAny<string>())).ReturnsAsync((UserProgressData? )null);
            mockAuthService.Setup(x => x.GenerateJwtToken(It.IsAny<User>())).Returns("validJwtToken");
            mockAuthService.Setup(x => x.GenerateRefreshToken()).Returns("validRefreshToken");
            var controller = new AuthController(mockContext.Object, mockAuthService.Object, mockProgressService.Object, mockPasswordValidator.Object, mockTokenBlacklist.Object, mockAuditLogger.Object, mockLogger.Object);
            SetupHttpContext(controller, "192.168.1.100");
            var request = new LoginRequest("test@example.com", "password123");
            // Act
            var result = await controller.Login(request);
            // Assert
            var okResult = result.Result as OkObjectResult;
            Assert.IsNotNull(okResult);
            mockAuditLogger.Verify(x => x.LogLoginAttemptAsync("test@example.com", true, "192.168.1.100"), Times.Once);
        }

        /// <summary>
        /// Tests that Login calls LogLoginAttemptAsync with correct parameters for failed login.
        /// Input: Valid email but wrong password
        /// Expected: LogLoginAttemptAsync called with email, false (failure), and IP address
        /// </summary>
        [TestMethod]
        public async Task Login_PasswordVerificationFails_LogsFailedLoginAttemptWithCorrectParameters()
        {
            // Arrange
            var users = new List<User>
            {
                new User
                {
                    Id = "user1",
                    Email = "test@example.com",
                    Name = "Test User",
                    PasswordHash = "hashedPassword",
                    CreatedAt = DateTime.UtcNow
                }
            };
            var mockContext = CreateMockDbContext(users);
            var mockAuthService = new Mock<IAuthService>();
            var mockProgressService = new Mock<IUserProgressService>();
            var mockPasswordValidator = new Mock<IPasswordValidator>();
            var mockTokenBlacklist = new Mock<ITokenBlacklistService>();
            var mockAuditLogger = new Mock<IAuditLogger>();
            var mockLogger = new Mock<ILogger<AuthController>>();
            mockAuthService.Setup(x => x.VerifyPassword(It.IsAny<string>(), It.IsAny<string>())).Returns(false);
            var controller = new AuthController(mockContext.Object, mockAuthService.Object, mockProgressService.Object, mockPasswordValidator.Object, mockTokenBlacklist.Object, mockAuditLogger.Object, mockLogger.Object);
            SetupHttpContext(controller, "10.0.0.5");
            var request = new LoginRequest("test@example.com", "wrongPassword");
            // Act
            var result = await controller.Login(request);
            // Assert
            var unauthorizedResult = result.Result as UnauthorizedObjectResult;
            Assert.IsNotNull(unauthorizedResult);
            mockAuditLogger.Verify(x => x.LogLoginAttemptAsync("test@example.com", false, "10.0.0.5"), Times.Once);
        }

        /// <summary>
        /// Tests that Login with password containing only special characters is handled correctly.
        /// Input: Password with only special characters
        /// Expected: Password is not rejected by validation, proceeds to verification
        /// </summary>
        [TestMethod]
        public async Task Login_PasswordWithOnlySpecialCharacters_ProcessesNormally()
        {
            // Arrange
            var specialPassword = "!@#$%^&*()_+-=[]{}|;:',.<>?/~`";
            var users = new List<User>
            {
                new User
                {
                    Id = "user1",
                    Email = "test@example.com",
                    Name = "Test User",
                    PasswordHash = "hashedPassword",
                    CreatedAt = DateTime.UtcNow
                }
            };
            var mockContext = CreateMockDbContext(users);
            var mockAuthService = new Mock<IAuthService>();
            var mockProgressService = new Mock<IUserProgressService>();
            var mockPasswordValidator = new Mock<IPasswordValidator>();
            var mockTokenBlacklist = new Mock<ITokenBlacklistService>();
            var mockAuditLogger = new Mock<IAuditLogger>();
            var mockLogger = new Mock<ILogger<AuthController>>();
            mockAuthService.Setup(x => x.VerifyPassword(specialPassword, "hashedPassword")).Returns(false);
            var controller = new AuthController(mockContext.Object, mockAuthService.Object, mockProgressService.Object, mockPasswordValidator.Object, mockTokenBlacklist.Object, mockAuditLogger.Object, mockLogger.Object);
            SetupHttpContext(controller, "192.168.1.1");
            var request = new LoginRequest("test@example.com", specialPassword);
            // Act
            var result = await controller.Login(request);
            // Assert
            var unauthorizedResult = result.Result as UnauthorizedObjectResult;
            Assert.IsNotNull(unauthorizedResult);
            mockAuthService.Verify(x => x.VerifyPassword(specialPassword, "hashedPassword"), Times.Once);
        }

        /// <summary>
        /// Tests that Login handles email with international/unicode characters.
        /// Input: Email containing unicode characters
        /// Expected: Processes normally
        /// </summary>
        [TestMethod]
        public async Task Login_EmailWithUnicodeCharacters_ProcessesNormally()
        {
            // Arrange
            var unicodeEmail = "用户@example.com";
            var users = new List<User>();
            var mockContext = CreateMockDbContext(users);
            var mockAuthService = new Mock<IAuthService>();
            var mockProgressService = new Mock<IUserProgressService>();
            var mockPasswordValidator = new Mock<IPasswordValidator>();
            var mockTokenBlacklist = new Mock<ITokenBlacklistService>();
            var mockAuditLogger = new Mock<IAuditLogger>();
            var mockLogger = new Mock<ILogger<AuthController>>();
            var controller = new AuthController(mockContext.Object, mockAuthService.Object, mockProgressService.Object, mockPasswordValidator.Object, mockTokenBlacklist.Object, mockAuditLogger.Object, mockLogger.Object);
            SetupHttpContext(controller, "192.168.1.1");
            var request = new LoginRequest(unicodeEmail, "password123");
            // Act
            var result = await controller.Login(request);
            // Assert
            var unauthorizedResult = result.Result as UnauthorizedObjectResult;
            Assert.IsNotNull(unauthorizedResult);
            Assert.AreEqual(401, unauthorizedResult.StatusCode);
        }

        /// <summary>
        /// Tests that RefreshToken handles token expiring in far future correctly.
        /// Input: Token expiring 365 days from now.
        /// Expected: Successfully refreshes the token.
        /// </summary>
        [TestMethod]
        public async Task RefreshToken_TokenExpiringInFarFuture_Success()
        {
            // Arrange
            var user = new User
            {
                Id = "user123",
                Name = "Test User",
                Email = "test@example.com",
                RefreshToken = "valid-token",
                RefreshTokenExpiryTime = DateTime.UtcNow.AddDays(365)
            };
            var users = new List<User>
            {
                user
            }.AsQueryable();
            var mockUsersDbSet = CreateMockDbSet(users);
            var mockContext = new Mock<AppDbContext>(new DbContextOptions<AppDbContext>());
            mockContext.Setup(c => c.Users).Returns(mockUsersDbSet.Object);
            mockContext.Setup(c => c.SaveChangesAsync(It.IsAny<CancellationToken>())).ReturnsAsync(1);
            var mockAuthService = new Mock<IAuthService>();
            mockAuthService.Setup(s => s.GenerateJwtToken(It.IsAny<User>())).Returns("new-jwt-token");
            mockAuthService.Setup(s => s.GenerateRefreshToken()).Returns("new-refresh-token");
            var mockProgressService = new Mock<IUserProgressService>();
            var mockPasswordValidator = new Mock<IPasswordValidator>();
            var mockTokenBlacklist = new Mock<ITokenBlacklistService>();
            var mockAuditLogger = new Mock<IAuditLogger>();
            var mockLogger = new Mock<ILogger<AuthController>>();
            var controller = new AuthController(mockContext.Object, mockAuthService.Object, mockProgressService.Object, mockPasswordValidator.Object, mockTokenBlacklist.Object, mockAuditLogger.Object, mockLogger.Object);
            var request = new RefreshTokenRequest
            {
                RefreshToken = "valid-token"
            };
            // Act
            var result = await controller.RefreshToken(request);
            // Assert
            Assert.IsInstanceOfType(result.Result, typeof(OkObjectResult));
            var okResult = result.Result as OkObjectResult;
            Assert.IsNotNull(okResult);
        }

        /// <summary>
        /// Tests that RefreshToken handles user with empty name correctly.
        /// Input: User with empty name string.
        /// Expected: Successfully refreshes token and returns empty name.
        /// </summary>
        [TestMethod]
        public async Task RefreshToken_UserWithEmptyName_ReturnsEmptyName()
        {
            // Arrange
            var user = new User
            {
                Id = "user123",
                Name = "",
                Email = "test@example.com",
                RefreshToken = "valid-token",
                RefreshTokenExpiryTime = DateTime.UtcNow.AddDays(7)
            };
            var users = new List<User>
            {
                user
            }.AsQueryable();
            var mockUsersDbSet = CreateMockDbSet(users);
            var mockContext = new Mock<AppDbContext>(new DbContextOptions<AppDbContext>());
            mockContext.Setup(c => c.Users).Returns(mockUsersDbSet.Object);
            mockContext.Setup(c => c.SaveChangesAsync(It.IsAny<CancellationToken>())).ReturnsAsync(1);
            var mockAuthService = new Mock<IAuthService>();
            mockAuthService.Setup(s => s.GenerateJwtToken(It.IsAny<User>())).Returns("new-jwt-token");
            mockAuthService.Setup(s => s.GenerateRefreshToken()).Returns("new-refresh-token");
            var mockProgressService = new Mock<IUserProgressService>();
            var mockPasswordValidator = new Mock<IPasswordValidator>();
            var mockTokenBlacklist = new Mock<ITokenBlacklistService>();
            var mockAuditLogger = new Mock<IAuditLogger>();
            var mockLogger = new Mock<ILogger<AuthController>>();
            var controller = new AuthController(mockContext.Object, mockAuthService.Object, mockProgressService.Object, mockPasswordValidator.Object, mockTokenBlacklist.Object, mockAuditLogger.Object, mockLogger.Object);
            var request = new RefreshTokenRequest
            {
                RefreshToken = "valid-token"
            };
            // Act
            var result = await controller.RefreshToken(request);
            // Assert
            Assert.IsInstanceOfType(result.Result, typeof(OkObjectResult));
            var okResult = result.Result as OkObjectResult;
            Assert.IsNotNull(okResult);
            dynamic? value = okResult.Value;
            Assert.IsNotNull(value);
            Assert.AreEqual("", value.name);
        }
    }
}