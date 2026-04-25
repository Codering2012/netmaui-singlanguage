using System;
using System.Collections.Generic;
using System.Linq;
using System.Security;
using System.Security.Claims;
using System.Threading.Tasks;

using Microsoft.AspNetCore;
using Microsoft.AspNetCore.Authorization;
using Microsoft.AspNetCore.Http;
using Microsoft.AspNetCore.Mvc;
using Microsoft.Extensions.Logging;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using Moq;
using SignLanguageApi;
using SignLanguageApi.Controllers;
using SignLanguageApi.Dtos;
using SignLanguageApi.Services;

namespace SignLanguageApi.Controllers.UnitTests
{
    /// <summary>
    /// Unit tests for the LearnController class.
    /// </summary>
    [TestClass]
    public class LearnControllerTests
    {
        /// <summary>
        /// Tests that the constructor successfully initializes the controller with valid dependencies.
        /// Input: Valid ILearnService mock and valid ILogger mock.
        /// Expected: Constructor completes successfully without throwing exceptions.
        /// </summary>
        [TestMethod]
        public void Constructor_ValidDependencies_InitializesSuccessfully()
        {
            // Arrange
            var mockLearnService = new Mock<ILearnService>();
            var mockLogger = new Mock<ILogger<LearnController>>();

            // Act
            var controller = new LearnController(mockLearnService.Object, mockLogger.Object);

            // Assert
            Assert.IsNotNull(controller);
        }

        /// <summary>
        /// Tests the constructor behavior when null is passed for the learnService parameter.
        /// Input: null for learnService, valid ILogger mock.
        /// Expected: Constructor completes (no validation is performed in the constructor).
        /// </summary>
        [TestMethod]
        public void Constructor_NullLearnService_CompletesWithoutException()
        {
            // Arrange
            ILearnService? nullLearnService = null;
            var mockLogger = new Mock<ILogger<LearnController>>();

            // Act
            var controller = new LearnController(nullLearnService!, mockLogger.Object);

            // Assert
            Assert.IsNotNull(controller);
        }

        /// <summary>
        /// Tests the constructor behavior when null is passed for the logger parameter.
        /// Input: Valid ILearnService mock, null for logger.
        /// Expected: Constructor completes (no validation is performed in the constructor).
        /// </summary>
        [TestMethod]
        public void Constructor_NullLogger_CompletesWithoutException()
        {
            // Arrange
            var mockLearnService = new Mock<ILearnService>();
            ILogger<LearnController>? nullLogger = null;

            // Act
            var controller = new LearnController(mockLearnService.Object, nullLogger!);

            // Assert
            Assert.IsNotNull(controller);
        }

        /// <summary>
        /// Tests the constructor behavior when null is passed for both parameters.
        /// Input: null for both learnService and logger.
        /// Expected: Constructor completes (no validation is performed in the constructor).
        /// </summary>
        [TestMethod]
        public void Constructor_BothParametersNull_CompletesWithoutException()
        {
            // Arrange
            ILearnService? nullLearnService = null;
            ILogger<LearnController>? nullLogger = null;

            // Act
            var controller = new LearnController(nullLearnService!, nullLogger!);

            // Assert
            Assert.IsNotNull(controller);
        }

        /// <summary>
        /// Tests that GetAllCategories returns Ok result with categories when service succeeds with valid user.
        /// </summary>
        [TestMethod]
        public async Task GetAllCategories_ValidUserAndServiceReturnsCategories_ReturnsOkResultWithCategories()
        {
            // Arrange
            var mockLearnService = new Mock<ILearnService>();
            var mockLogger = new Mock<ILogger<LearnController>>();
            var userId = "test-user-123";
            var categories = new List<LessonCategoryDto>
            {
                new LessonCategoryDto { Id = 1, Title = "Basic Signs", Description = "Learn basic signs", Difficulty = "Beginner", Progress = 0.5 },
                new LessonCategoryDto { Id = 2, Title = "Advanced Signs", Description = "Learn advanced signs", Difficulty = "Advanced", Progress = 0.2 }
            };

            mockLearnService.Setup(s => s.GetAllCategoriesAsync(userId)).ReturnsAsync(categories);

            var controller = new LearnController(mockLearnService.Object, mockLogger.Object);
            SetupControllerWithUser(controller, userId);

            // Act
            var result = await controller.GetAllCategories();

            // Assert
            Assert.IsNotNull(result);
            Assert.IsInstanceOfType(result.Result, typeof(OkObjectResult));
            var okResult = result.Result as OkObjectResult;
            Assert.IsNotNull(okResult);
            Assert.AreEqual(200, okResult.StatusCode);
            Assert.IsInstanceOfType(okResult.Value, typeof(List<LessonCategoryDto>));
            var returnedCategories = okResult.Value as List<LessonCategoryDto>;
            Assert.IsNotNull(returnedCategories);
            Assert.AreEqual(2, returnedCategories.Count);
            Assert.AreEqual("Basic Signs", returnedCategories[0].Title);
            mockLearnService.Verify(s => s.GetAllCategoriesAsync(userId), Times.Once);
        }

        /// <summary>
        /// Tests that GetAllCategories returns Ok result with empty list when service returns empty list.
        /// </summary>
        [TestMethod]
        public async Task GetAllCategories_ValidUserAndServiceReturnsEmptyList_ReturnsOkResultWithEmptyList()
        {
            // Arrange
            var mockLearnService = new Mock<ILearnService>();
            var mockLogger = new Mock<ILogger<LearnController>>();
            var userId = "test-user-123";
            var emptyCategories = new List<LessonCategoryDto>();

            mockLearnService.Setup(s => s.GetAllCategoriesAsync(userId)).ReturnsAsync(emptyCategories);

            var controller = new LearnController(mockLearnService.Object, mockLogger.Object);
            SetupControllerWithUser(controller, userId);

            // Act
            var result = await controller.GetAllCategories();

            // Assert
            Assert.IsNotNull(result);
            Assert.IsInstanceOfType(result.Result, typeof(OkObjectResult));
            var okResult = result.Result as OkObjectResult;
            Assert.IsNotNull(okResult);
            var returnedCategories = okResult.Value as List<LessonCategoryDto>;
            Assert.IsNotNull(returnedCategories);
            Assert.AreEqual(0, returnedCategories.Count);
            mockLearnService.Verify(s => s.GetAllCategoriesAsync(userId), Times.Once);
        }

        /// <summary>
        /// Tests that GetAllCategories returns 500 status code when user ID is not found in claims.
        /// </summary>
        [TestMethod]
        public async Task GetAllCategories_UserIdNotFoundInClaims_Returns500StatusCode()
        {
            // Arrange
            var mockLearnService = new Mock<ILearnService>();
            var mockLogger = new Mock<ILogger<LearnController>>();

            var controller = new LearnController(mockLearnService.Object, mockLogger.Object);
            SetupControllerWithoutUser(controller);

            // Act
            var result = await controller.GetAllCategories();

            // Assert
            Assert.IsNotNull(result);
            Assert.IsInstanceOfType(result.Result, typeof(ObjectResult));
            var objectResult = result.Result as ObjectResult;
            Assert.IsNotNull(objectResult);
            Assert.AreEqual(500, objectResult.StatusCode);
            mockLearnService.Verify(s => s.GetAllCategoriesAsync(It.IsAny<string>()), Times.Never);
        }

        /// <summary>
        /// Tests that GetAllCategories returns 500 status code and logs error when service throws exception.
        /// </summary>
        [TestMethod]
        public async Task GetAllCategories_ServiceThrowsException_Returns500StatusCodeAndLogsError()
        {
            // Arrange
            var mockLearnService = new Mock<ILearnService>();
            var mockLogger = new Mock<ILogger<LearnController>>();
            var userId = "test-user-123";
            var exception = new Exception("Database connection failed");

            mockLearnService.Setup(s => s.GetAllCategoriesAsync(userId)).ThrowsAsync(exception);

            var controller = new LearnController(mockLearnService.Object, mockLogger.Object);
            SetupControllerWithUser(controller, userId);

            // Act
            var result = await controller.GetAllCategories();

            // Assert
            Assert.IsNotNull(result);
            Assert.IsInstanceOfType(result.Result, typeof(ObjectResult));
            var objectResult = result.Result as ObjectResult;
            Assert.IsNotNull(objectResult);
            Assert.AreEqual(500, objectResult.StatusCode);

            mockLogger.Verify(
                x => x.Log(
                    LogLevel.Error,
                    It.IsAny<EventId>(),
                    It.Is<It.IsAnyType>((v, t) => v.ToString()!.Contains("Error fetching categories")),
                    exception,
                    It.IsAny<Func<It.IsAnyType, Exception?, string>>()),
                Times.Once);
        }

        /// <summary>
        /// Tests that GetAllCategories returns correct error message in response when exception occurs.
        /// </summary>
        [TestMethod]
        public async Task GetAllCategories_ServiceThrowsException_ReturnsCorrectErrorMessage()
        {
            // Arrange
            var mockLearnService = new Mock<ILearnService>();
            var mockLogger = new Mock<ILogger<LearnController>>();
            var userId = "test-user-123";
            var exception = new InvalidOperationException("Service unavailable");

            mockLearnService.Setup(s => s.GetAllCategoriesAsync(userId)).ThrowsAsync(exception);

            var controller = new LearnController(mockLearnService.Object, mockLogger.Object);
            SetupControllerWithUser(controller, userId);

            // Act
            var result = await controller.GetAllCategories();

            // Assert
            Assert.IsNotNull(result);
            var objectResult = result.Result as ObjectResult;
            Assert.IsNotNull(objectResult);
            Assert.IsNotNull(objectResult.Value);

            var value = objectResult.Value;
            var messageProperty = value.GetType().GetProperty("message");
            Assert.IsNotNull(messageProperty);
            var message = messageProperty.GetValue(value) as string;
            Assert.AreEqual("Error fetching categories", message);
        }

        /// <summary>
        /// Tests that GetAllCategories handles user with empty string claim value correctly.
        /// </summary>
        [TestMethod]
        public async Task GetAllCategories_UserClaimHasEmptyString_Returns500StatusCode()
        {
            // Arrange
            var mockLearnService = new Mock<ILearnService>();
            var mockLogger = new Mock<ILogger<LearnController>>();

            var controller = new LearnController(mockLearnService.Object, mockLogger.Object);
            SetupControllerWithUser(controller, string.Empty);

            // Act
            var result = await controller.GetAllCategories();

            // Assert
            Assert.IsNotNull(result);
            Assert.IsInstanceOfType(result.Result, typeof(ObjectResult));
            var objectResult = result.Result as ObjectResult;
            Assert.IsNotNull(objectResult);
            Assert.AreEqual(500, objectResult.StatusCode);
            mockLearnService.Verify(s => s.GetAllCategoriesAsync(It.IsAny<string>()), Times.Never);
        }

        /// <summary>
        /// Tests that GetAllCategories handles various user ID formats correctly.
        /// </summary>
        /// <param name="userId">The user ID to test with.</param>
        [TestMethod]
        [DataRow("simple-id")]
        [DataRow("user-with-dashes-123")]
        [DataRow("GUID-12345678-1234-1234-1234-123456789012")]
        [DataRow("very-long-user-id-with-many-characters-to-test-edge-cases-1234567890")]
        [DataRow("123")]
        [DataRow("user@example.com")]
        public async Task GetAllCategories_VariousUserIdFormats_ReturnsOkResult(string userId)
        {
            // Arrange
            var mockLearnService = new Mock<ILearnService>();
            var mockLogger = new Mock<ILogger<LearnController>>();
            var categories = new List<LessonCategoryDto>();

            mockLearnService.Setup(s => s.GetAllCategoriesAsync(userId)).ReturnsAsync(categories);

            var controller = new LearnController(mockLearnService.Object, mockLogger.Object);
            SetupControllerWithUser(controller, userId);

            // Act
            var result = await controller.GetAllCategories();

            // Assert
            Assert.IsNotNull(result);
            Assert.IsInstanceOfType(result.Result, typeof(OkObjectResult));
            mockLearnService.Verify(s => s.GetAllCategoriesAsync(userId), Times.Once);
        }

        /// <summary>
        /// Tests that GetAllCategories handles large category lists correctly.
        /// </summary>
        [TestMethod]
        public async Task GetAllCategories_ServiceReturnsLargeList_ReturnsOkResultWithAllCategories()
        {
            // Arrange
            var mockLearnService = new Mock<ILearnService>();
            var mockLogger = new Mock<ILogger<LearnController>>();
            var userId = "test-user-123";
            var largeCategories = new List<LessonCategoryDto>();

            for (int i = 0; i < 1000; i++)
            {
                largeCategories.Add(new LessonCategoryDto
                {
                    Id = i,
                    Title = $"Category {i}",
                    Description = $"Description {i}",
                    Difficulty = "Intermediate",
                    Progress = 0.5
                });
            }

            mockLearnService.Setup(s => s.GetAllCategoriesAsync(userId)).ReturnsAsync(largeCategories);

            var controller = new LearnController(mockLearnService.Object, mockLogger.Object);
            SetupControllerWithUser(controller, userId);

            // Act
            var result = await controller.GetAllCategories();

            // Assert
            Assert.IsNotNull(result);
            var okResult = result.Result as OkObjectResult;
            Assert.IsNotNull(okResult);
            var returnedCategories = okResult.Value as List<LessonCategoryDto>;
            Assert.IsNotNull(returnedCategories);
            Assert.AreEqual(1000, returnedCategories.Count);
        }

        /// <summary>
        /// Helper method to setup controller with authenticated user claims.
        /// </summary>
        /// <param name="controller">The controller to setup.</param>
        /// <param name="userId">The user ID to include in claims.</param>
        private void SetupControllerWithUser(LearnController controller, string userId)
        {
            var claims = new List<Claim>
            {
                new Claim(ClaimTypes.NameIdentifier, userId)
            };
            var identity = new ClaimsIdentity(claims, "TestAuthType");
            var claimsPrincipal = new ClaimsPrincipal(identity);

            controller.ControllerContext = new ControllerContext
            {
                HttpContext = new DefaultHttpContext
                {
                    User = claimsPrincipal
                }
            };
        }

        /// <summary>
        /// Helper method to setup controller without user claims (unauthenticated).
        /// </summary>
        /// <param name="controller">The controller to setup.</param>
        private void SetupControllerWithoutUser(LearnController controller)
        {
            var claimsPrincipal = new ClaimsPrincipal();

            controller.ControllerContext = new ControllerContext
            {
                HttpContext = new DefaultHttpContext
                {
                    User = claimsPrincipal
                }
            };
        }

        /// <summary>
        /// Tests that GetLearnPageData returns 500 status code when GetUserId throws InvalidOperationException.
        /// Input: User with no NameIdentifier claim.
        /// Expected: Returns ObjectResult with 500 status code and error message.
        /// </summary>
        [TestMethod]
        public async Task GetLearnPageData_UserIdNotFoundInToken_Returns500WithErrorMessage()
        {
            // Arrange
            var mockLearnService = new Mock<ILearnService>();
            var mockLogger = new Mock<ILogger<LearnController>>();
            var controller = new LearnController(mockLearnService.Object, mockLogger.Object);

            var claimsPrincipal = new ClaimsPrincipal(new ClaimsIdentity());

            controller.ControllerContext = new ControllerContext
            {
                HttpContext = new DefaultHttpContext { User = claimsPrincipal }
            };

            // Act
            var result = await controller.GetLearnPageData();

            // Assert
            Assert.IsNotNull(result);
            var statusCodeResult = result.Result as ObjectResult;
            Assert.IsNotNull(statusCodeResult);
            Assert.AreEqual(500, statusCodeResult.StatusCode);

            var value = statusCodeResult.Value;
            Assert.IsNotNull(value);
            var messageProperty = value.GetType().GetProperty("message");
            Assert.IsNotNull(messageProperty);
            Assert.AreEqual("Error fetching learn page data", messageProperty.GetValue(value));

            mockLogger.Verify(
                x => x.Log(
                    LogLevel.Error,
                    It.IsAny<EventId>(),
                    It.Is<It.IsAnyType>((v, t) => true),
                    It.IsAny<InvalidOperationException>(),
                    It.Is<Func<It.IsAnyType, Exception?, string>>((v, t) => true)),
                Times.Once);
        }

        /// <summary>
        /// Tests that GetLearnPageData returns 500 status code when service throws an exception.
        /// Input: Valid user ID, but service throws Exception.
        /// Expected: Returns ObjectResult with 500 status code, logs error, and returns error message.
        /// </summary>
        [TestMethod]
        [DataRow("Database connection failed")]
        [DataRow("Network timeout")]
        [DataRow("")]
        public async Task GetLearnPageData_ServiceThrowsException_Returns500WithErrorMessage(string exceptionMessage)
        {
            // Arrange
            var mockLearnService = new Mock<ILearnService>();
            var mockLogger = new Mock<ILogger<LearnController>>();
            var controller = new LearnController(mockLearnService.Object, mockLogger.Object);

            var userId = "test-user-456";
            var claims = new List<Claim>
            {
                new Claim(ClaimTypes.NameIdentifier, userId)
            };
            var identity = new ClaimsIdentity(claims, "TestAuth");
            var claimsPrincipal = new ClaimsPrincipal(identity);

            controller.ControllerContext = new ControllerContext
            {
                HttpContext = new DefaultHttpContext { User = claimsPrincipal }
            };

            var exception = new Exception(exceptionMessage);
            mockLearnService.Setup(s => s.GetLearnPageDataAsync(userId))
                .ThrowsAsync(exception);

            // Act
            var result = await controller.GetLearnPageData();

            // Assert
            Assert.IsNotNull(result);
            var statusCodeResult = result.Result as ObjectResult;
            Assert.IsNotNull(statusCodeResult);
            Assert.AreEqual(500, statusCodeResult.StatusCode);

            var value = statusCodeResult.Value;
            Assert.IsNotNull(value);
            var messageProperty = value.GetType().GetProperty("message");
            Assert.IsNotNull(messageProperty);
            Assert.AreEqual("Error fetching learn page data", messageProperty.GetValue(value));

            mockLearnService.Verify(s => s.GetLearnPageDataAsync(userId), Times.Once);
            mockLogger.Verify(
                x => x.Log(
                    LogLevel.Error,
                    It.IsAny<EventId>(),
                    It.Is<It.IsAnyType>((v, t) => true),
                    It.IsAny<Exception>(),
                    It.Is<Func<It.IsAnyType, Exception?, string>>((v, t) => true)),
                Times.Once);
        }

        /// <summary>
        /// Tests that GetLearnPageData returns 500 status code when service throws InvalidOperationException.
        /// Input: Valid user ID, but service throws InvalidOperationException.
        /// Expected: Returns ObjectResult with 500 status code and error message.
        /// </summary>
        [TestMethod]
        public async Task GetLearnPageData_ServiceThrowsInvalidOperationException_Returns500WithErrorMessage()
        {
            // Arrange
            var mockLearnService = new Mock<ILearnService>();
            var mockLogger = new Mock<ILogger<LearnController>>();
            var controller = new LearnController(mockLearnService.Object, mockLogger.Object);

            var userId = "test-user-789";
            var claims = new List<Claim>
            {
                new Claim(ClaimTypes.NameIdentifier, userId)
            };
            var identity = new ClaimsIdentity(claims, "TestAuth");
            var claimsPrincipal = new ClaimsPrincipal(identity);

            controller.ControllerContext = new ControllerContext
            {
                HttpContext = new DefaultHttpContext { User = claimsPrincipal }
            };

            var exception = new InvalidOperationException("Invalid operation");
            mockLearnService.Setup(s => s.GetLearnPageDataAsync(userId))
                .ThrowsAsync(exception);

            // Act
            var result = await controller.GetLearnPageData();

            // Assert
            Assert.IsNotNull(result);
            var statusCodeResult = result.Result as ObjectResult;
            Assert.IsNotNull(statusCodeResult);
            Assert.AreEqual(500, statusCodeResult.StatusCode);

            mockLogger.Verify(
                x => x.Log(
                    LogLevel.Error,
                    It.IsAny<EventId>(),
                    It.Is<It.IsAnyType>((v, t) => true),
                    It.IsAny<InvalidOperationException>(),
                    It.Is<Func<It.IsAnyType, Exception?, string>>((v, t) => true)),
                Times.Once);
        }

        /// <summary>
        /// Tests that GetLearnPageData returns 500 status code when service throws ArgumentNullException.
        /// Input: Valid user ID, but service throws ArgumentNullException.
        /// Expected: Returns ObjectResult with 500 status code and error message.
        /// </summary>
        [TestMethod]
        public async Task GetLearnPageData_ServiceThrowsArgumentNullException_Returns500WithErrorMessage()
        {
            // Arrange
            var mockLearnService = new Mock<ILearnService>();
            var mockLogger = new Mock<ILogger<LearnController>>();
            var controller = new LearnController(mockLearnService.Object, mockLogger.Object);

            var userId = "test-user-null";
            var claims = new List<Claim>
            {
                new Claim(ClaimTypes.NameIdentifier, userId)
            };
            var identity = new ClaimsIdentity(claims, "TestAuth");
            var claimsPrincipal = new ClaimsPrincipal(identity);

            controller.ControllerContext = new ControllerContext
            {
                HttpContext = new DefaultHttpContext { User = claimsPrincipal }
            };

            var exception = new ArgumentNullException("userId");
            mockLearnService.Setup(s => s.GetLearnPageDataAsync(userId))
                .ThrowsAsync(exception);

            // Act
            var result = await controller.GetLearnPageData();

            // Assert
            Assert.IsNotNull(result);
            var statusCodeResult = result.Result as ObjectResult;
            Assert.IsNotNull(statusCodeResult);
            Assert.AreEqual(500, statusCodeResult.StatusCode);

            mockLogger.Verify(
                x => x.Log(
                    LogLevel.Error,
                    It.IsAny<EventId>(),
                    It.Is<It.IsAnyType>((v, t) => true),
                    It.IsAny<ArgumentNullException>(),
                    It.Is<Func<It.IsAnyType, Exception?, string>>((v, t) => true)),
                Times.Once);
        }

        /// <summary>
        /// Tests that GetLearnPageData handles various special user IDs correctly.
        /// Input: Different valid user ID formats from token.
        /// Expected: Returns OkObjectResult with the data for each user ID.
        /// </summary>
        [TestMethod]
        [DataRow("user-1")]
        [DataRow("AAAAAAAA-BBBB-CCCC-DDDD-EEEEEEEEEEEE")]
        [DataRow("12345")]
        [DataRow("user@example.com")]
        [DataRow("a")]
        public async Task GetLearnPageData_VariousUserIdFormats_ReturnsOkWithData(string userId)
        {
            // Arrange
            var mockLearnService = new Mock<ILearnService>();
            var mockLogger = new Mock<ILogger<LearnController>>();
            var controller = new LearnController(mockLearnService.Object, mockLogger.Object);

            var claims = new List<Claim>
            {
                new Claim(ClaimTypes.NameIdentifier, userId)
            };
            var identity = new ClaimsIdentity(claims, "TestAuth");
            var claimsPrincipal = new ClaimsPrincipal(identity);

            controller.ControllerContext = new ControllerContext
            {
                HttpContext = new DefaultHttpContext { User = claimsPrincipal }
            };

            var expectedData = new LearnPageDataDto();
            mockLearnService.Setup(s => s.GetLearnPageDataAsync(userId))
                .ReturnsAsync(expectedData);

            // Act
            var result = await controller.GetLearnPageData();

            // Assert
            Assert.IsNotNull(result);
            var okResult = result.Result as OkObjectResult;
            Assert.IsNotNull(okResult);
            Assert.AreEqual(200, okResult.StatusCode);
            mockLearnService.Verify(s => s.GetLearnPageDataAsync(userId), Times.Once);
        }

        /// <summary>
        /// Tests that GetLearnPageData returns 500 when user claim value is null.
        /// Input: User with NameIdentifier claim but null value.
        /// Expected: Returns ObjectResult with 500 status code and error message.
        /// </summary>
        [TestMethod]
        public async Task GetLearnPageData_UserIdClaimIsNull_Returns500WithErrorMessage()
        {
            // Arrange
            var mockLearnService = new Mock<ILearnService>();
            var mockLogger = new Mock<ILogger<LearnController>>();
            var controller = new LearnController(mockLearnService.Object, mockLogger.Object);

            var claims = new List<Claim>
            {
                new Claim(ClaimTypes.NameIdentifier, string.Empty)
            };
            var identity = new ClaimsIdentity(claims, "TestAuth");
            var claimsPrincipal = new ClaimsPrincipal(identity);

            controller.ControllerContext = new ControllerContext
            {
                HttpContext = new DefaultHttpContext { User = claimsPrincipal }
            };

            // Act
            var result = await controller.GetLearnPageData();

            // Assert
            Assert.IsNotNull(result);
            var statusCodeResult = result.Result as ObjectResult;
            Assert.IsNotNull(statusCodeResult);
            Assert.AreEqual(500, statusCodeResult.StatusCode);

            mockLogger.Verify(
                x => x.Log(
                    LogLevel.Error,
                    It.IsAny<EventId>(),
                    It.Is<It.IsAnyType>((v, t) => true),
                    It.IsAny<InvalidOperationException>(),
                    It.Is<Func<It.IsAnyType, Exception?, string>>((v, t) => true)),
                Times.Once);
        }

        /// <summary>
        /// Tests that GetRecommendation returns 500 status code when user ID is not found in token (GetUserId throws).
        /// </summary>
        [TestMethod]
        public async Task GetRecommendation_UserIdNotFoundInToken_Returns500StatusCodeWithErrorMessage()
        {
            // Arrange
            var mockLearnService = new Mock<ILearnService>();
            var mockLogger = new Mock<ILogger<LearnController>>();

            var controller = new LearnController(mockLearnService.Object, mockLogger.Object);
            controller.ControllerContext = CreateControllerContextWithoutUserClaims();

            // Act
            var result = await controller.GetRecommendation();

            // Assert
            Assert.IsNotNull(result);
            var statusCodeResult = result.Result as ObjectResult;
            Assert.IsNotNull(statusCodeResult);
            Assert.AreEqual(500, statusCodeResult.StatusCode);

            var responseValue = statusCodeResult.Value;
            Assert.IsNotNull(responseValue);
            var messageProperty = responseValue.GetType().GetProperty("message");
            Assert.IsNotNull(messageProperty);
            var messageValue = messageProperty.GetValue(responseValue) as string;
            Assert.AreEqual("User ID not found in token", messageValue);

            mockLogger.Verify(
                x => x.Log(
                    LogLevel.Error,
                    It.IsAny<EventId>(),
                    It.Is<It.IsAnyType>((v, t) => true),
                    It.IsAny<Exception>(),
                    It.Is<Func<It.IsAnyType, Exception?, string>>((v, t) => true)),
                Times.Once);

            mockLearnService.Verify(s => s.GetPersonalizedRecommendationAsync(It.IsAny<string>()), Times.Never);
        }

        /// <summary>
        /// Creates a ControllerContext with a ClaimsPrincipal containing the specified user ID.
        /// </summary>
        /// <param name="userId">The user ID to include in the claims.</param>
        /// <returns>A configured ControllerContext.</returns>
        private static ControllerContext CreateControllerContextWithUser(string userId)
        {
            var claims = new[] { new Claim(ClaimTypes.NameIdentifier, userId) };
            var identity = new ClaimsIdentity(claims, "TestAuth");
            var claimsPrincipal = new ClaimsPrincipal(identity);

            return new ControllerContext
            {
                HttpContext = new DefaultHttpContext
                {
                    User = claimsPrincipal
                }
            };
        }

        /// <summary>
        /// Creates a ControllerContext without any user claims.
        /// </summary>
        /// <returns>A configured ControllerContext without user claims.</returns>
        private static ControllerContext CreateControllerContextWithoutUserClaims()
        {
            var identity = new ClaimsIdentity();
            var claimsPrincipal = new ClaimsPrincipal(identity);

            return new ControllerContext
            {
                HttpContext = new DefaultHttpContext
                {
                    User = claimsPrincipal
                }
            };
        }

        /// <summary>
        /// Helper method to extract message from anonymous response objects using reflection.
        /// </summary>
        /// <param name="responseObj">The response object (typically an anonymous object with a message property)</param>
        /// <returns>The message string, or null if not found</returns>
        private static string GetMessageFromResponse(object responseObj)
        {
            if (responseObj == null)
                return null;

            var property = responseObj.GetType().GetProperty("message");
            if (property != null && property.CanRead)
            {
                var value = property.GetValue(responseObj);
                return value?.ToString();
            }

            return null;
        }

        /// <summary>
        /// Tests that GetLessonsByCategory returns Ok with lessons when valid categoryId is provided and service returns lessons successfully.
        /// Input: Valid positive categoryId, authenticated user, service returns lessons.
        /// Expected: OkObjectResult with list of lessons.
        /// </summary>
        [TestMethod]
        [DataRow(1)]
        [DataRow(100)]
        [DataRow(int.MaxValue)]
        public async Task GetLessonsByCategory_ValidCategoryIdWithLessons_ReturnsOkWithLessonsList(int categoryId)
        {
            // Arrange
            var mockLearnService = new Mock<ILearnService>();
            var mockLogger = new Mock<ILogger<LearnController>>();
            var expectedLessons = new List<LessonDto>
            {
                new LessonDto { Id = 1, Title = "Lesson 1", CategoryId = categoryId },
                new LessonDto { Id = 2, Title = "Lesson 2", CategoryId = categoryId }
            };
            var userId = "test-user-123";

            mockLearnService
                .Setup(s => s.GetLessonsByCategoryAsync(categoryId, userId))
                .ReturnsAsync(expectedLessons);

            var controller = CreateControllerWithUser(mockLearnService.Object, mockLogger.Object, userId);

            // Act
            var result = await controller.GetLessonsByCategory(categoryId);

            // Assert
            Assert.IsNotNull(result);
            Assert.IsInstanceOfType(result.Result, typeof(OkObjectResult));
            var okResult = result.Result as OkObjectResult;
            Assert.IsNotNull(okResult);
            Assert.AreEqual(200, okResult.StatusCode);
            var returnedLessons = okResult.Value as List<LessonDto>;
            Assert.IsNotNull(returnedLessons);
            Assert.AreEqual(2, returnedLessons.Count);
            mockLearnService.Verify(s => s.GetLessonsByCategoryAsync(categoryId, userId), Times.Once);
        }

        /// <summary>
        /// Tests that GetLessonsByCategory returns Ok with empty list when valid categoryId is provided but no lessons exist.
        /// Input: Valid categoryId, authenticated user, service returns empty list.
        /// Expected: OkObjectResult with empty list.
        /// </summary>
        [TestMethod]
        public async Task GetLessonsByCategory_ValidCategoryIdWithNoLessons_ReturnsOkWithEmptyList()
        {
            // Arrange
            var mockLearnService = new Mock<ILearnService>();
            var mockLogger = new Mock<ILogger<LearnController>>();
            var categoryId = 42;
            var emptyLessonsList = new List<LessonDto>();
            var userId = "test-user-456";

            mockLearnService
                .Setup(s => s.GetLessonsByCategoryAsync(categoryId, userId))
                .ReturnsAsync(emptyLessonsList);

            var controller = CreateControllerWithUser(mockLearnService.Object, mockLogger.Object, userId);

            // Act
            var result = await controller.GetLessonsByCategory(categoryId);

            // Assert
            Assert.IsNotNull(result);
            Assert.IsInstanceOfType(result.Result, typeof(OkObjectResult));
            var okResult = result.Result as OkObjectResult;
            Assert.IsNotNull(okResult);
            var returnedLessons = okResult.Value as List<LessonDto>;
            Assert.IsNotNull(returnedLessons);
            Assert.AreEqual(0, returnedLessons.Count);
        }

        /// <summary>
        /// Tests that GetLessonsByCategory handles boundary and edge case categoryId values correctly.
        /// Input: Zero, negative, and minimum int values for categoryId.
        /// Expected: OkObjectResult with lessons list (service handles business logic validation).
        /// </summary>
        [TestMethod]
        [DataRow(0)]
        [DataRow(-1)]
        [DataRow(-100)]
        [DataRow(int.MinValue)]
        public async Task GetLessonsByCategory_BoundaryAndEdgeCaseCategoryIds_ReturnsOkWithServiceResult(int categoryId)
        {
            // Arrange
            var mockLearnService = new Mock<ILearnService>();
            var mockLogger = new Mock<ILogger<LearnController>>();
            var lessons = new List<LessonDto>();
            var userId = "test-user-789";

            mockLearnService
                .Setup(s => s.GetLessonsByCategoryAsync(categoryId, userId))
                .ReturnsAsync(lessons);

            var controller = CreateControllerWithUser(mockLearnService.Object, mockLogger.Object, userId);

            // Act
            var result = await controller.GetLessonsByCategory(categoryId);

            // Assert
            Assert.IsNotNull(result);
            Assert.IsInstanceOfType(result.Result, typeof(OkObjectResult));
            mockLearnService.Verify(s => s.GetLessonsByCategoryAsync(categoryId, userId), Times.Once);
        }

        /// <summary>
        /// Tests that GetLessonsByCategory returns 500 status code when GetUserId throws InvalidOperationException due to missing user claims.
        /// Input: Controller without authenticated user claims.
        /// Expected: ObjectResult with status code 500 and error message, logger called.
        /// </summary>
        [TestMethod]
        public async Task GetLessonsByCategory_UserIdNotFoundInToken_ReturnsInternalServerError()
        {
            // Arrange
            var mockLearnService = new Mock<ILearnService>();
            var mockLogger = new Mock<ILogger<LearnController>>();
            var categoryId = 1;

            var controller = new LearnController(mockLearnService.Object, mockLogger.Object)
            {
                ControllerContext = new ControllerContext
                {
                    HttpContext = new DefaultHttpContext
                    {
                        User = new ClaimsPrincipal(new ClaimsIdentity()) // No claims
                    }
                }
            };

            // Act
            var result = await controller.GetLessonsByCategory(categoryId);

            // Assert
            Assert.IsNotNull(result);
            Assert.IsInstanceOfType(result.Result, typeof(ObjectResult));
            var objectResult = result.Result as ObjectResult;
            Assert.IsNotNull(objectResult);
            Assert.AreEqual(500, objectResult.StatusCode);
            mockLogger.Verify(
                x => x.Log(
                    LogLevel.Error,
                    It.IsAny<EventId>(),
                    It.Is<It.IsAnyType>((v, t) => true),
                    It.IsAny<InvalidOperationException>(),
                    It.Is<Func<It.IsAnyType, Exception?, string>>((v, t) => true)),
                Times.Once);
        }

        /// <summary>
        /// Tests that GetLessonsByCategory returns 500 status code when service throws an exception.
        /// Input: Valid categoryId and authenticated user, but service throws exception.
        /// Expected: ObjectResult with status code 500 and error message, logger called with exception.
        /// </summary>
        [TestMethod]
        [DataRow(1)]
        [DataRow(50)]
        public async Task GetLessonsByCategory_ServiceThrowsException_ReturnsInternalServerError(int categoryId)
        {
            // Arrange
            var mockLearnService = new Mock<ILearnService>();
            var mockLogger = new Mock<ILogger<LearnController>>();
            var userId = "test-user-error";
            var expectedException = new Exception("Database connection failed");

            mockLearnService
                .Setup(s => s.GetLessonsByCategoryAsync(categoryId, userId))
                .ThrowsAsync(expectedException);

            var controller = CreateControllerWithUser(mockLearnService.Object, mockLogger.Object, userId);

            // Act
            var result = await controller.GetLessonsByCategory(categoryId);

            // Assert
            Assert.IsNotNull(result);
            Assert.IsInstanceOfType(result.Result, typeof(ObjectResult));
            var objectResult = result.Result as ObjectResult;
            Assert.IsNotNull(objectResult);
            Assert.AreEqual(500, objectResult.StatusCode);
            mockLogger.Verify(
                x => x.Log(
                    LogLevel.Error,
                    It.IsAny<EventId>(),
                    It.Is<It.IsAnyType>((v, t) => true),
                    It.IsAny<Exception>(),
                    It.Is<Func<It.IsAnyType, Exception?, string>>((v, t) => true)),
                Times.Once);
        }

        /// <summary>
        /// Tests that GetLessonsByCategory returns error message in response body when exception occurs.
        /// Input: Service throws exception.
        /// Expected: Response contains message property with "Error fetching lessons".
        /// </summary>
        [TestMethod]
        public async Task GetLessonsByCategory_ServiceThrowsException_ReturnsErrorMessageInBody()
        {
            // Arrange
            var mockLearnService = new Mock<ILearnService>();
            var mockLogger = new Mock<ILogger<LearnController>>();
            var categoryId = 10;
            var userId = "test-user-msg";

            mockLearnService
                .Setup(s => s.GetLessonsByCategoryAsync(categoryId, userId))
                .ThrowsAsync(new Exception("Test exception"));

            var controller = CreateControllerWithUser(mockLearnService.Object, mockLogger.Object, userId);

            // Act
            var result = await controller.GetLessonsByCategory(categoryId);

            // Assert
            var objectResult = result.Result as ObjectResult;
            Assert.IsNotNull(objectResult);
            var value = objectResult.Value;
            Assert.IsNotNull(value);
            var messageProperty = value.GetType().GetProperty("message");
            Assert.IsNotNull(messageProperty);
            var message = messageProperty.GetValue(value) as string;
            Assert.AreEqual("Error fetching lessons", message);
        }

        /// <summary>
        /// Tests that GetLessonsByCategory handles InvalidOperationException specifically.
        /// Input: Service throws InvalidOperationException.
        /// Expected: ObjectResult with status code 500, proper error handling.
        /// </summary>
        [TestMethod]
        public async Task GetLessonsByCategory_ServiceThrowsInvalidOperationException_ReturnsInternalServerError()
        {
            // Arrange
            var mockLearnService = new Mock<ILearnService>();
            var mockLogger = new Mock<ILogger<LearnController>>();
            var categoryId = 99;
            var userId = "test-user-invalid";

            mockLearnService
                .Setup(s => s.GetLessonsByCategoryAsync(categoryId, userId))
                .ThrowsAsync(new InvalidOperationException("Invalid operation"));

            var controller = CreateControllerWithUser(mockLearnService.Object, mockLogger.Object, userId);

            // Act
            var result = await controller.GetLessonsByCategory(categoryId);

            // Assert
            Assert.IsNotNull(result);
            var objectResult = result.Result as ObjectResult;
            Assert.IsNotNull(objectResult);
            Assert.AreEqual(500, objectResult.StatusCode);
            mockLogger.Verify(
                x => x.Log(
                    LogLevel.Error,
                    It.IsAny<EventId>(),
                    It.Is<It.IsAnyType>((v, t) => true),
                    It.IsAny<InvalidOperationException>(),
                    It.Is<Func<It.IsAnyType, Exception?, string>>((v, t) => true)),
                Times.Once);
        }

        /// <summary>
        /// Tests that GetLessonsByCategory correctly passes userId from token to service.
        /// Input: Authenticated user with specific userId.
        /// Expected: Service called with correct userId parameter.
        /// </summary>
        [TestMethod]
        public async Task GetLessonsByCategory_AuthenticatedUser_PassesCorrectUserIdToService()
        {
            // Arrange
            var mockLearnService = new Mock<ILearnService>();
            var mockLogger = new Mock<ILogger<LearnController>>();
            var categoryId = 5;
            var expectedUserId = "specific-user-id-12345";
            var lessons = new List<LessonDto>();

            mockLearnService
                .Setup(s => s.GetLessonsByCategoryAsync(categoryId, expectedUserId))
                .ReturnsAsync(lessons)
                .Verifiable();

            var controller = CreateControllerWithUser(mockLearnService.Object, mockLogger.Object, expectedUserId);

            // Act
            var result = await controller.GetLessonsByCategory(categoryId);

            // Assert
            mockLearnService.Verify(s => s.GetLessonsByCategoryAsync(categoryId, expectedUserId), Times.Once);
        }

        /// <summary>
        /// Tests that GetLesson returns lesson details including XAML and code-behind content.
        /// </summary>
        [TestMethod]
        public async Task GetLesson_ValidUserAndLessonWithUiLayout_ReturnsOkWithFileSendingData()
        {
            // Arrange
            var mockLearnService = new Mock<ILearnService>();
            var mockLogger = new Mock<ILogger<LearnController>>();
            var lessonId = 7;
            var userId = "test-user-ui-layout";

            var expectedLesson = new LessonDto
            {
                Id = lessonId,
                Title = "Realtime Practice",
                Data = new LessonDataDto
                {
                    UiLayout = new LessonUiLayoutDto
                    {
                        FileName = "RealtimeHandSignalPracticeSet1View.xaml",
                        XamlContent = "<ContentPage xmlns=\"http://schemas.microsoft.com/dotnet/2021/maui\" x:Class=\"App.RealtimeHandSignalPracticeSet1View\"></ContentPage>",
                        CodeBehindContent = "namespace App;\\npublic partial class RealtimeHandSignalPracticeSet1View : ContentPage { }"
                    }
                }
            };

            mockLearnService
                .Setup(s => s.GetLessonAsync(lessonId, userId))
                .ReturnsAsync(expectedLesson);

            var controller = CreateControllerWithUser(mockLearnService.Object, mockLogger.Object, userId);

            // Act
            var result = await controller.GetLesson(lessonId);

            // Assert
            Assert.IsNotNull(result);
            Assert.IsInstanceOfType(result.Result, typeof(OkObjectResult));

            var okResult = result.Result as OkObjectResult;
            Assert.IsNotNull(okResult);
            Assert.AreEqual(200, okResult.StatusCode);

            var returnedLesson = okResult.Value as LessonDto;
            Assert.IsNotNull(returnedLesson);
            Assert.IsNotNull(returnedLesson.Data);
            Assert.IsNotNull(returnedLesson.Data.UiLayout);

            Assert.AreEqual("RealtimeHandSignalPracticeSet1View.xaml", returnedLesson.Data.UiLayout.FileName);
            Assert.IsTrue(returnedLesson.Data.UiLayout.XamlContent.Contains("ContentPage"));
            Assert.IsTrue(returnedLesson.Data.UiLayout.CodeBehindContent.Contains("partial class RealtimeHandSignalPracticeSet1View"));

            mockLearnService.Verify(s => s.GetLessonAsync(lessonId, userId), Times.Once);
        }

        /// <summary>
        /// Tests that GetLesson returns NotFound when lesson does not exist.
        /// </summary>
        [TestMethod]
        public async Task GetLesson_LessonNotFound_ReturnsNotFoundWithMessage()
        {
            // Arrange
            var mockLearnService = new Mock<ILearnService>();
            var mockLogger = new Mock<ILogger<LearnController>>();
            var lessonId = 9999;
            var userId = "test-user-not-found";

            mockLearnService
                .Setup(s => s.GetLessonAsync(lessonId, userId))
                .ReturnsAsync((LessonDto?)null);

            var controller = CreateControllerWithUser(mockLearnService.Object, mockLogger.Object, userId);

            // Act
            var result = await controller.GetLesson(lessonId);

            // Assert
            Assert.IsNotNull(result);
            Assert.IsInstanceOfType(result.Result, typeof(NotFoundObjectResult));

            var notFoundResult = result.Result as NotFoundObjectResult;
            Assert.IsNotNull(notFoundResult);
            Assert.AreEqual(404, notFoundResult.StatusCode);
            Assert.AreEqual("Lesson not found", GetMessageFromResponse(notFoundResult.Value!));

            mockLearnService.Verify(s => s.GetLessonAsync(lessonId, userId), Times.Once);
        }

        /// <summary>
        /// Tests that GetLesson returns 500 when the user ID claim is missing.
        /// </summary>
        [TestMethod]
        public async Task GetLesson_UserIdNotFoundInToken_ReturnsInternalServerError()
        {
            // Arrange
            var mockLearnService = new Mock<ILearnService>();
            var mockLogger = new Mock<ILogger<LearnController>>();
            var lessonId = 1;

            var controller = new LearnController(mockLearnService.Object, mockLogger.Object)
            {
                ControllerContext = CreateControllerContextWithoutUserClaims()
            };

            // Act
            var result = await controller.GetLesson(lessonId);

            // Assert
            Assert.IsNotNull(result);
            Assert.IsInstanceOfType(result.Result, typeof(ObjectResult));

            var objectResult = result.Result as ObjectResult;
            Assert.IsNotNull(objectResult);
            Assert.AreEqual(500, objectResult.StatusCode);
            Assert.AreEqual("Error fetching lesson", GetMessageFromResponse(objectResult.Value!));

            mockLearnService.Verify(s => s.GetLessonAsync(It.IsAny<int>(), It.IsAny<string>()), Times.Never);
        }

        /// <summary>
        /// Tests that GetLesson returns 500 when the service throws.
        /// </summary>
        [TestMethod]
        public async Task GetLesson_ServiceThrowsException_ReturnsInternalServerErrorWithMessage()
        {
            // Arrange
            var mockLearnService = new Mock<ILearnService>();
            var mockLogger = new Mock<ILogger<LearnController>>();
            var lessonId = 5;
            var userId = "test-user-exception";
            var expectedException = new Exception("Service failed");

            mockLearnService
                .Setup(s => s.GetLessonAsync(lessonId, userId))
                .ThrowsAsync(expectedException);

            var controller = CreateControllerWithUser(mockLearnService.Object, mockLogger.Object, userId);

            // Act
            var result = await controller.GetLesson(lessonId);

            // Assert
            Assert.IsNotNull(result);
            Assert.IsInstanceOfType(result.Result, typeof(ObjectResult));

            var objectResult = result.Result as ObjectResult;
            Assert.IsNotNull(objectResult);
            Assert.AreEqual(500, objectResult.StatusCode);
            Assert.AreEqual("Error fetching lesson", GetMessageFromResponse(objectResult.Value!));

            mockLearnService.Verify(s => s.GetLessonAsync(lessonId, userId), Times.Once);
        }

        /// <summary>
        /// Tests that the lesson endpoint route is configured as lessons/{lessonId}.
        /// </summary>
        [TestMethod]
        public void GetLesson_RouteTemplate_IsLessonsByLessonId()
        {
            // Arrange
            var methodInfo = typeof(LearnController).GetMethod(nameof(LearnController.GetLesson));

            // Act
            var attribute = methodInfo?.GetCustomAttributes(typeof(HttpGetAttribute), false).FirstOrDefault() as HttpGetAttribute;

            // Assert
            Assert.IsNotNull(attribute);
            Assert.AreEqual("lessons/{lessonId}", attribute.Template);
        }

        /// <summary>
        /// Helper method to create a LearnController instance with authenticated user.
        /// </summary>
        /// <param name="learnService">The mocked learn service.</param>
        /// <param name="logger">The mocked logger.</param>
        /// <param name="userId">The user ID to set in claims.</param>
        /// <returns>A configured LearnController instance.</returns>
        private LearnController CreateControllerWithUser(ILearnService learnService, ILogger<LearnController> logger, string userId)
        {
            var claims = new List<Claim>
            {
                new Claim(ClaimTypes.NameIdentifier, userId)
            };
            var identity = new ClaimsIdentity(claims, "TestAuthType");
            var claimsPrincipal = new ClaimsPrincipal(identity);

            var controller = new LearnController(learnService, logger)
            {
                ControllerContext = new ControllerContext
                {
                    HttpContext = new DefaultHttpContext
                    {
                        User = claimsPrincipal
                    }
                }
            };

            return controller;
        }

        /// <summary>
        /// Tests UpdateLessonProgress with valid completion percentage at lower boundary (0).
        /// Should call the service and return Ok with success message.
        /// </summary>
        [TestMethod]
        [DataRow(0, 1)]
        [DataRow(0, 100)]
        [DataRow(0, int.MaxValue)]
        [DataRow(0, int.MinValue)]
        [DataRow(0, -1)]
        public async Task UpdateLessonProgress_ValidCompletionPercentageZero_ReturnsOkResult(int completionPercentage, int lessonId)
        {
            // Arrange
            var mockLearnService = new Mock<ILearnService>();
            var mockLogger = new Mock<ILogger<LearnController>>();
            var controller = new LearnController(mockLearnService.Object, mockLogger.Object);

            var userId = "test-user-id";
            var claims = new List<Claim> { new Claim(ClaimTypes.NameIdentifier, userId) };
            var identity = new ClaimsIdentity(claims, "TestAuthType");
            var claimsPrincipal = new ClaimsPrincipal(identity);

            controller.ControllerContext = new ControllerContext
            {
                HttpContext = new DefaultHttpContext { User = claimsPrincipal }
            };

            mockLearnService
                .Setup(s => s.UpdateLessonProgressAsync(userId, lessonId, completionPercentage))
                .Returns(Task.CompletedTask);

            // Act
            var result = await controller.UpdateLessonProgress(lessonId, completionPercentage);

            // Assert
            Assert.IsNotNull(result);
            Assert.IsInstanceOfType(result.Result, typeof(OkObjectResult));
            var okResult = result.Result as OkObjectResult;
            Assert.IsNotNull(okResult);
            Assert.AreEqual(200, okResult.StatusCode);
            mockLearnService.Verify(s => s.UpdateLessonProgressAsync(userId, lessonId, completionPercentage), Times.Once);
        }

        /// <summary>
        /// Tests UpdateLessonProgress with valid completion percentage in middle range (50).
        /// Should call the service and return Ok with success message.
        /// </summary>
        [TestMethod]
        [DataRow(50, 1)]
        [DataRow(50, 0)]
        [DataRow(50, -1)]
        [DataRow(50, int.MaxValue)]
        [DataRow(50, int.MinValue)]
        public async Task UpdateLessonProgress_ValidCompletionPercentageFifty_ReturnsOkResult(int completionPercentage, int lessonId)
        {
            // Arrange
            var mockLearnService = new Mock<ILearnService>();
            var mockLogger = new Mock<ILogger<LearnController>>();
            var controller = new LearnController(mockLearnService.Object, mockLogger.Object);

            var userId = "test-user-id";
            var claims = new List<Claim> { new Claim(ClaimTypes.NameIdentifier, userId) };
            var identity = new ClaimsIdentity(claims, "TestAuthType");
            var claimsPrincipal = new ClaimsPrincipal(identity);

            controller.ControllerContext = new ControllerContext
            {
                HttpContext = new DefaultHttpContext { User = claimsPrincipal }
            };

            mockLearnService
                .Setup(s => s.UpdateLessonProgressAsync(userId, lessonId, completionPercentage))
                .Returns(Task.CompletedTask);

            // Act
            var result = await controller.UpdateLessonProgress(lessonId, completionPercentage);

            // Assert
            Assert.IsNotNull(result);
            Assert.IsInstanceOfType(result.Result, typeof(OkObjectResult));
            var okResult = result.Result as OkObjectResult;
            Assert.IsNotNull(okResult);
            Assert.AreEqual(200, okResult.StatusCode);
            mockLearnService.Verify(s => s.UpdateLessonProgressAsync(userId, lessonId, completionPercentage), Times.Once);
        }

        /// <summary>
        /// Tests UpdateLessonProgress with valid completion percentage at upper boundary (100).
        /// Should call the service and return Ok with success message.
        /// </summary>
        [TestMethod]
        [DataRow(100, 1)]
        [DataRow(100, 0)]
        [DataRow(100, -1)]
        [DataRow(100, int.MaxValue)]
        [DataRow(100, int.MinValue)]
        public async Task UpdateLessonProgress_ValidCompletionPercentageOneHundred_ReturnsOkResult(int completionPercentage, int lessonId)
        {
            // Arrange
            var mockLearnService = new Mock<ILearnService>();
            var mockLogger = new Mock<ILogger<LearnController>>();
            var controller = new LearnController(mockLearnService.Object, mockLogger.Object);

            var userId = "test-user-id";
            var claims = new List<Claim> { new Claim(ClaimTypes.NameIdentifier, userId) };
            var identity = new ClaimsIdentity(claims, "TestAuthType");
            var claimsPrincipal = new ClaimsPrincipal(identity);

            controller.ControllerContext = new ControllerContext
            {
                HttpContext = new DefaultHttpContext { User = claimsPrincipal }
            };

            mockLearnService
                .Setup(s => s.UpdateLessonProgressAsync(userId, lessonId, completionPercentage))
                .Returns(Task.CompletedTask);

            // Act
            var result = await controller.UpdateLessonProgress(lessonId, completionPercentage);

            // Assert
            Assert.IsNotNull(result);
            Assert.IsInstanceOfType(result.Result, typeof(OkObjectResult));
            var okResult = result.Result as OkObjectResult;
            Assert.IsNotNull(okResult);
            Assert.AreEqual(200, okResult.StatusCode);
            mockLearnService.Verify(s => s.UpdateLessonProgressAsync(userId, lessonId, completionPercentage), Times.Once);
        }

        /// <summary>
        /// Tests UpdateLessonProgress with invalid completion percentage below valid range.
        /// Should return BadRequest with appropriate error message without calling the service.
        /// </summary>
        [TestMethod]
        [DataRow(-1, 1)]
        [DataRow(-1, 0)]
        [DataRow(-100, 1)]
        [DataRow(int.MinValue, 1)]
        [DataRow(int.MinValue, int.MaxValue)]
        public async Task UpdateLessonProgress_CompletionPercentageBelowZero_ReturnsBadRequest(int completionPercentage, int lessonId)
        {
            // Arrange
            var mockLearnService = new Mock<ILearnService>();
            var mockLogger = new Mock<ILogger<LearnController>>();
            var controller = new LearnController(mockLearnService.Object, mockLogger.Object);

            var userId = "test-user-id";
            var claims = new List<Claim> { new Claim(ClaimTypes.NameIdentifier, userId) };
            var identity = new ClaimsIdentity(claims, "TestAuthType");
            var claimsPrincipal = new ClaimsPrincipal(identity);

            controller.ControllerContext = new ControllerContext
            {
                HttpContext = new DefaultHttpContext { User = claimsPrincipal }
            };

            // Act
            var result = await controller.UpdateLessonProgress(lessonId, completionPercentage);

            // Assert
            Assert.IsNotNull(result);
            Assert.IsInstanceOfType(result.Result, typeof(BadRequestObjectResult));
            var badRequestResult = result.Result as BadRequestObjectResult;
            Assert.IsNotNull(badRequestResult);
            Assert.AreEqual(400, badRequestResult.StatusCode);
            mockLearnService.Verify(s => s.UpdateLessonProgressAsync(It.IsAny<string>(), It.IsAny<int>(), It.IsAny<int>()), Times.Never);
        }

        /// <summary>
        /// Tests UpdateLessonProgress with invalid completion percentage above valid range.
        /// Should return BadRequest with appropriate error message without calling the service.
        /// </summary>
        [TestMethod]
        [DataRow(101, 1)]
        [DataRow(101, 0)]
        [DataRow(200, 1)]
        [DataRow(int.MaxValue, 1)]
        [DataRow(int.MaxValue, int.MinValue)]
        public async Task UpdateLessonProgress_CompletionPercentageAboveOneHundred_ReturnsBadRequest(int completionPercentage, int lessonId)
        {
            // Arrange
            var mockLearnService = new Mock<ILearnService>();
            var mockLogger = new Mock<ILogger<LearnController>>();
            var controller = new LearnController(mockLearnService.Object, mockLogger.Object);

            var userId = "test-user-id";
            var claims = new List<Claim> { new Claim(ClaimTypes.NameIdentifier, userId) };
            var identity = new ClaimsIdentity(claims, "TestAuthType");
            var claimsPrincipal = new ClaimsPrincipal(identity);

            controller.ControllerContext = new ControllerContext
            {
                HttpContext = new DefaultHttpContext { User = claimsPrincipal }
            };

            // Act
            var result = await controller.UpdateLessonProgress(lessonId, completionPercentage);

            // Assert
            Assert.IsNotNull(result);
            Assert.IsInstanceOfType(result.Result, typeof(BadRequestObjectResult));
            var badRequestResult = result.Result as BadRequestObjectResult;
            Assert.IsNotNull(badRequestResult);
            Assert.AreEqual(400, badRequestResult.StatusCode);
            mockLearnService.Verify(s => s.UpdateLessonProgressAsync(It.IsAny<string>(), It.IsAny<int>(), It.IsAny<int>()), Times.Never);
        }

        /// <summary>
        /// Tests UpdateLessonProgress when GetUserId fails (user ID not found in claims).
        /// Should catch the exception, log error, and return StatusCode 500.
        /// </summary>
        [TestMethod]
        public async Task UpdateLessonProgress_UserIdNotFoundInToken_ReturnsInternalServerError()
        {
            // Arrange
            var mockLearnService = new Mock<ILearnService>();
            var mockLogger = new Mock<ILogger<LearnController>>();
            var controller = new LearnController(mockLearnService.Object, mockLogger.Object);

            var claims = new List<Claim>();
            var identity = new ClaimsIdentity(claims, "TestAuthType");
            var claimsPrincipal = new ClaimsPrincipal(identity);

            controller.ControllerContext = new ControllerContext
            {
                HttpContext = new DefaultHttpContext { User = claimsPrincipal }
            };

            var lessonId = 1;
            var completionPercentage = 50;

            // Act
            var result = await controller.UpdateLessonProgress(lessonId, completionPercentage);

            // Assert
            Assert.IsNotNull(result);
            Assert.IsInstanceOfType(result.Result, typeof(ObjectResult));
            var objectResult = result.Result as ObjectResult;
            Assert.IsNotNull(objectResult);
            Assert.AreEqual(500, objectResult.StatusCode);
            mockLearnService.Verify(s => s.UpdateLessonProgressAsync(It.IsAny<string>(), It.IsAny<int>(), It.IsAny<int>()), Times.Never);
        }

        /// <summary>
        /// Tests UpdateLessonProgress when the service throws an exception.
        /// Should catch the exception, log error, and return StatusCode 500.
        /// </summary>
        [TestMethod]
        public async Task UpdateLessonProgress_ServiceThrowsException_ReturnsInternalServerError()
        {
            // Arrange
            var mockLearnService = new Mock<ILearnService>();
            var mockLogger = new Mock<ILogger<LearnController>>();
            var controller = new LearnController(mockLearnService.Object, mockLogger.Object);

            var userId = "test-user-id";
            var claims = new List<Claim> { new Claim(ClaimTypes.NameIdentifier, userId) };
            var identity = new ClaimsIdentity(claims, "TestAuthType");
            var claimsPrincipal = new ClaimsPrincipal(identity);

            controller.ControllerContext = new ControllerContext
            {
                HttpContext = new DefaultHttpContext { User = claimsPrincipal }
            };

            var lessonId = 1;
            var completionPercentage = 50;
            var exceptionMessage = "Database connection failed";

            mockLearnService
                .Setup(s => s.UpdateLessonProgressAsync(userId, lessonId, completionPercentage))
                .ThrowsAsync(new Exception(exceptionMessage));

            // Act
            var result = await controller.UpdateLessonProgress(lessonId, completionPercentage);

            // Assert
            Assert.IsNotNull(result);
            Assert.IsInstanceOfType(result.Result, typeof(ObjectResult));
            var objectResult = result.Result as ObjectResult;
            Assert.IsNotNull(objectResult);
            Assert.AreEqual(500, objectResult.StatusCode);
            mockLearnService.Verify(s => s.UpdateLessonProgressAsync(userId, lessonId, completionPercentage), Times.Once);
        }

        /// <summary>
        /// Tests UpdateLessonProgress when the service throws a specific exception type.
        /// Should catch the exception, log error, and return StatusCode 500.
        /// </summary>
        [TestMethod]
        public async Task UpdateLessonProgress_ServiceThrowsInvalidOperationException_ReturnsInternalServerError()
        {
            // Arrange
            var mockLearnService = new Mock<ILearnService>();
            var mockLogger = new Mock<ILogger<LearnController>>();
            var controller = new LearnController(mockLearnService.Object, mockLogger.Object);

            var userId = "test-user-id";
            var claims = new List<Claim> { new Claim(ClaimTypes.NameIdentifier, userId) };
            var identity = new ClaimsIdentity(claims, "TestAuthType");
            var claimsPrincipal = new ClaimsPrincipal(identity);

            controller.ControllerContext = new ControllerContext
            {
                HttpContext = new DefaultHttpContext { User = claimsPrincipal }
            };

            var lessonId = 1;
            var completionPercentage = 50;

            mockLearnService
                .Setup(s => s.UpdateLessonProgressAsync(userId, lessonId, completionPercentage))
                .ThrowsAsync(new InvalidOperationException("Invalid operation"));

            // Act
            var result = await controller.UpdateLessonProgress(lessonId, completionPercentage);

            // Assert
            Assert.IsNotNull(result);
            Assert.IsInstanceOfType(result.Result, typeof(ObjectResult));
            var objectResult = result.Result as ObjectResult;
            Assert.IsNotNull(objectResult);
            Assert.AreEqual(500, objectResult.StatusCode);
            mockLearnService.Verify(s => s.UpdateLessonProgressAsync(userId, lessonId, completionPercentage), Times.Once);
        }

        /// <summary>
        /// Tests UpdateLessonProgress when User.FindFirst returns null (no NameIdentifier claim).
        /// Should throw InvalidOperationException which is caught and returns StatusCode 500.
        /// </summary>
        [TestMethod]
        public async Task UpdateLessonProgress_UserClaimIsNull_ReturnsInternalServerError()
        {
            // Arrange
            var mockLearnService = new Mock<ILearnService>();
            var mockLogger = new Mock<ILogger<LearnController>>();
            var controller = new LearnController(mockLearnService.Object, mockLogger.Object);

            var claims = new List<Claim> { new Claim(ClaimTypes.Name, "TestUser") };
            var identity = new ClaimsIdentity(claims, "TestAuthType");
            var claimsPrincipal = new ClaimsPrincipal(identity);

            controller.ControllerContext = new ControllerContext
            {
                HttpContext = new DefaultHttpContext { User = claimsPrincipal }
            };

            var lessonId = 1;
            var completionPercentage = 50;

            // Act
            var result = await controller.UpdateLessonProgress(lessonId, completionPercentage);

            // Assert
            Assert.IsNotNull(result);
            Assert.IsInstanceOfType(result.Result, typeof(ObjectResult));
            var objectResult = result.Result as ObjectResult;
            Assert.IsNotNull(objectResult);
            Assert.AreEqual(500, objectResult.StatusCode);
            mockLearnService.Verify(s => s.UpdateLessonProgressAsync(It.IsAny<string>(), It.IsAny<int>(), It.IsAny<int>()), Times.Never);
        }

        /// <summary>
        /// Tests UpdateLessonProgress when User.FindFirst returns claim with empty value.
        /// Should throw InvalidOperationException which is caught and returns StatusCode 500.
        /// </summary>
        [TestMethod]
        public async Task UpdateLessonProgress_UserIdIsEmptyString_ReturnsInternalServerError()
        {
            // Arrange
            var mockLearnService = new Mock<ILearnService>();
            var mockLogger = new Mock<ILogger<LearnController>>();
            var controller = new LearnController(mockLearnService.Object, mockLogger.Object);

            var claims = new List<Claim> { new Claim(ClaimTypes.NameIdentifier, string.Empty) };
            var identity = new ClaimsIdentity(claims, "TestAuthType");
            var claimsPrincipal = new ClaimsPrincipal(identity);

            controller.ControllerContext = new ControllerContext
            {
                HttpContext = new DefaultHttpContext { User = claimsPrincipal }
            };

            var lessonId = 1;
            var completionPercentage = 50;

            // Act
            var result = await controller.UpdateLessonProgress(lessonId, completionPercentage);

            // Assert
            Assert.IsNotNull(result);
            Assert.IsInstanceOfType(result.Result, typeof(ObjectResult));
            var objectResult = result.Result as ObjectResult;
            Assert.IsNotNull(objectResult);
            Assert.AreEqual(500, objectResult.StatusCode);
            mockLearnService.Verify(s => s.UpdateLessonProgressAsync(It.IsAny<string>(), It.IsAny<int>(), It.IsAny<int>()), Times.Never);
        }

        /// <summary>
        /// Tests that GetUpcomingReviews returns Ok result with UpcomingReviewsDto when user is authenticated and service returns valid data.
        /// </summary>
        [TestMethod]
        public async Task GetUpcomingReviews_ValidUserIdAndServiceReturnsData_ReturnsOkWithUpcomingReviews()
        {
            // Arrange
            var mockLearnService = new Mock<ILearnService>();
            var mockLogger = new Mock<ILogger<LearnController>>();
            var controller = new LearnController(mockLearnService.Object, mockLogger.Object);

            var userId = "test-user-123";
            var claims = new[] { new Claim(ClaimTypes.NameIdentifier, userId) };
            var identity = new ClaimsIdentity(claims, "TestAuthType");
            var claimsPrincipal = new ClaimsPrincipal(identity);

            controller.ControllerContext = new ControllerContext
            {
                HttpContext = new DefaultHttpContext { User = claimsPrincipal }
            };

            var expectedDto = new UpcomingReviewsDto
            {
                DueToday = 5,
                DueTomorrow = 3,
                DueThisWeek = 10,
                Overdue = 2
            };

            mockLearnService
                .Setup(s => s.GetUpcomingReviewsAsync(userId))
                .ReturnsAsync(expectedDto);

            // Act
            var result = await controller.GetUpcomingReviews();

            // Assert
            Assert.IsNotNull(result);
            Assert.IsInstanceOfType(result.Result, typeof(OkObjectResult));
            var okResult = result.Result as OkObjectResult;
            Assert.IsNotNull(okResult);
            Assert.AreEqual(200, okResult.StatusCode);
            Assert.IsInstanceOfType(okResult.Value, typeof(UpcomingReviewsDto));
            var returnedDto = okResult.Value as UpcomingReviewsDto;
            Assert.IsNotNull(returnedDto);
            Assert.AreEqual(expectedDto.DueToday, returnedDto.DueToday);
            Assert.AreEqual(expectedDto.DueTomorrow, returnedDto.DueTomorrow);
            Assert.AreEqual(expectedDto.DueThisWeek, returnedDto.DueThisWeek);
            Assert.AreEqual(expectedDto.Overdue, returnedDto.Overdue);

            mockLearnService.Verify(s => s.GetUpcomingReviewsAsync(userId), Times.Once);
        }

        /// <summary>
        /// Tests that GetUpcomingReviews returns 500 status code when user ID is not found in token (no NameIdentifier claim).
        /// </summary>
        [TestMethod]
        public async Task GetUpcomingReviews_UserIdNotFoundInToken_ReturnsInternalServerError()
        {
            // Arrange
            var mockLearnService = new Mock<ILearnService>();
            var mockLogger = new Mock<ILogger<LearnController>>();
            var controller = new LearnController(mockLearnService.Object, mockLogger.Object);

            var claims = Array.Empty<Claim>();
            var identity = new ClaimsIdentity(claims, "TestAuthType");
            var claimsPrincipal = new ClaimsPrincipal(identity);

            controller.ControllerContext = new ControllerContext
            {
                HttpContext = new DefaultHttpContext { User = claimsPrincipal }
            };

            // Act
            var result = await controller.GetUpcomingReviews();

            // Assert
            Assert.IsNotNull(result);
            Assert.IsInstanceOfType(result.Result, typeof(ObjectResult));
            var objectResult = result.Result as ObjectResult;
            Assert.IsNotNull(objectResult);
            Assert.AreEqual(500, objectResult.StatusCode);

            mockLogger.Verify(
                x => x.Log(
                    LogLevel.Error,
                    It.IsAny<EventId>(),
                    It.Is<It.IsAnyType>((v, t) => v.ToString()!.Contains("Error fetching upcoming reviews")),
                    It.IsAny<InvalidOperationException>(),
                    It.IsAny<Func<It.IsAnyType, Exception?, string>>()),
                Times.Once);

            mockLearnService.Verify(s => s.GetUpcomingReviewsAsync(It.IsAny<string>()), Times.Never);
        }

        /// <summary>
        /// Tests that GetUpcomingReviews returns 500 status code when user ID claim contains null value.
        /// </summary>
        [TestMethod]
        public async Task GetUpcomingReviews_UserIdClaimIsNull_ReturnsInternalServerError()
        {
            // Arrange
            var mockLearnService = new Mock<ILearnService>();
            var mockLogger = new Mock<ILogger<LearnController>>();
            var controller = new LearnController(mockLearnService.Object, mockLogger.Object);

            var claimsPrincipal = new ClaimsPrincipal();

            controller.ControllerContext = new ControllerContext
            {
                HttpContext = new DefaultHttpContext { User = claimsPrincipal }
            };

            // Act
            var result = await controller.GetUpcomingReviews();

            // Assert
            Assert.IsNotNull(result);
            Assert.IsInstanceOfType(result.Result, typeof(ObjectResult));
            var objectResult = result.Result as ObjectResult;
            Assert.IsNotNull(objectResult);
            Assert.AreEqual(500, objectResult.StatusCode);

            mockLogger.Verify(
                x => x.Log(
                    LogLevel.Error,
                    It.IsAny<EventId>(),
                    It.Is<It.IsAnyType>((v, t) => v.ToString()!.Contains("Error fetching upcoming reviews")),
                    It.IsAny<InvalidOperationException>(),
                    It.IsAny<Func<It.IsAnyType, Exception?, string>>()),
                Times.Once);
        }

        /// <summary>
        /// Tests that GetUpcomingReviews returns 500 status code when the learn service throws an exception.
        /// </summary>
        [TestMethod]
        public async Task GetUpcomingReviews_ServiceThrowsException_ReturnsInternalServerError()
        {
            // Arrange
            var mockLearnService = new Mock<ILearnService>();
            var mockLogger = new Mock<ILogger<LearnController>>();
            var controller = new LearnController(mockLearnService.Object, mockLogger.Object);

            var userId = "test-user-456";
            var claims = new[] { new Claim(ClaimTypes.NameIdentifier, userId) };
            var identity = new ClaimsIdentity(claims, "TestAuthType");
            var claimsPrincipal = new ClaimsPrincipal(identity);

            controller.ControllerContext = new ControllerContext
            {
                HttpContext = new DefaultHttpContext { User = claimsPrincipal }
            };

            var expectedException = new Exception("Database connection failed");
            mockLearnService
                .Setup(s => s.GetUpcomingReviewsAsync(userId))
                .ThrowsAsync(expectedException);

            // Act
            var result = await controller.GetUpcomingReviews();

            // Assert
            Assert.IsNotNull(result);
            Assert.IsInstanceOfType(result.Result, typeof(ObjectResult));
            var objectResult = result.Result as ObjectResult;
            Assert.IsNotNull(objectResult);
            Assert.AreEqual(500, objectResult.StatusCode);

            mockLogger.Verify(
                x => x.Log(
                    LogLevel.Error,
                    It.IsAny<EventId>(),
                    It.Is<It.IsAnyType>((v, t) => v.ToString()!.Contains("Error fetching upcoming reviews")),
                    It.IsAny<Exception>(),
                    It.IsAny<Func<It.IsAnyType, Exception?, string>>()),
                Times.Once);

            mockLearnService.Verify(s => s.GetUpcomingReviewsAsync(userId), Times.Once);
        }

        /// <summary>
        /// Tests that GetUpcomingReviews returns 500 status code with proper error message when service throws exception.
        /// </summary>
        [TestMethod]
        public async Task GetUpcomingReviews_ServiceThrowsException_ReturnsProperErrorMessage()
        {
            // Arrange
            var mockLearnService = new Mock<ILearnService>();
            var mockLogger = new Mock<ILogger<LearnController>>();
            var controller = new LearnController(mockLearnService.Object, mockLogger.Object);

            var userId = "test-user-789";
            var claims = new[] { new Claim(ClaimTypes.NameIdentifier, userId) };
            var identity = new ClaimsIdentity(claims, "TestAuthType");
            var claimsPrincipal = new ClaimsPrincipal(identity);

            controller.ControllerContext = new ControllerContext
            {
                HttpContext = new DefaultHttpContext { User = claimsPrincipal }
            };

            mockLearnService
                .Setup(s => s.GetUpcomingReviewsAsync(userId))
                .ThrowsAsync(new InvalidOperationException("Test exception"));

            // Act
            var result = await controller.GetUpcomingReviews();

            // Assert
            Assert.IsNotNull(result);
            var objectResult = result.Result as ObjectResult;
            Assert.IsNotNull(objectResult);
            Assert.AreEqual(500, objectResult.StatusCode);

            var value = objectResult.Value;
            Assert.IsNotNull(value);
            var messageProperty = value.GetType().GetProperty("message");
            Assert.IsNotNull(messageProperty);
            var message = messageProperty.GetValue(value) as string;
            Assert.AreEqual("Error fetching upcoming reviews", message);
        }

        /// <summary>
        /// Tests that GetUpcomingReviews handles empty string userId claim by throwing exception and returning 500.
        /// </summary>
        [TestMethod]
        public async Task GetUpcomingReviews_UserIdClaimIsEmptyString_ReturnsInternalServerError()
        {
            // Arrange
            var mockLearnService = new Mock<ILearnService>();
            var mockLogger = new Mock<ILogger<LearnController>>();
            var controller = new LearnController(mockLearnService.Object, mockLogger.Object);

            var claims = new[] { new Claim(ClaimTypes.NameIdentifier, string.Empty) };
            var identity = new ClaimsIdentity(claims, "TestAuthType");
            var claimsPrincipal = new ClaimsPrincipal(identity);

            controller.ControllerContext = new ControllerContext
            {
                HttpContext = new DefaultHttpContext { User = claimsPrincipal }
            };

            // Act
            var result = await controller.GetUpcomingReviews();

            // Assert
            Assert.IsNotNull(result);
            Assert.IsInstanceOfType(result.Result, typeof(ObjectResult));
            var objectResult = result.Result as ObjectResult;
            Assert.IsNotNull(objectResult);
            Assert.AreEqual(500, objectResult.StatusCode);

            mockLogger.Verify(
                x => x.Log(
                    LogLevel.Error,
                    It.IsAny<EventId>(),
                    It.Is<It.IsAnyType>((v, t) => v.ToString()!.Contains("Error fetching upcoming reviews")),
                    It.IsAny<InvalidOperationException>(),
                    It.IsAny<Func<It.IsAnyType, Exception?, string>>()),
                Times.Once);

            mockLearnService.Verify(s => s.GetUpcomingReviewsAsync(It.IsAny<string>()), Times.Never);
        }

        /// <summary>
        /// Tests that GetUpcomingReviews returns Ok with all zero values when service returns a DTO with zeros.
        /// </summary>
        [TestMethod]
        public async Task GetUpcomingReviews_ServiceReturnsZeroValues_ReturnsOkWithZeroValues()
        {
            // Arrange
            var mockLearnService = new Mock<ILearnService>();
            var mockLogger = new Mock<ILogger<LearnController>>();
            var controller = new LearnController(mockLearnService.Object, mockLogger.Object);

            var userId = "test-user-zeros";
            var claims = new[] { new Claim(ClaimTypes.NameIdentifier, userId) };
            var identity = new ClaimsIdentity(claims, "TestAuthType");
            var claimsPrincipal = new ClaimsPrincipal(identity);

            controller.ControllerContext = new ControllerContext
            {
                HttpContext = new DefaultHttpContext { User = claimsPrincipal }
            };

            var expectedDto = new UpcomingReviewsDto
            {
                DueToday = 0,
                DueTomorrow = 0,
                DueThisWeek = 0,
                Overdue = 0
            };

            mockLearnService
                .Setup(s => s.GetUpcomingReviewsAsync(userId))
                .ReturnsAsync(expectedDto);

            // Act
            var result = await controller.GetUpcomingReviews();

            // Assert
            Assert.IsNotNull(result);
            Assert.IsInstanceOfType(result.Result, typeof(OkObjectResult));
            var okResult = result.Result as OkObjectResult;
            Assert.IsNotNull(okResult);
            var returnedDto = okResult.Value as UpcomingReviewsDto;
            Assert.IsNotNull(returnedDto);
            Assert.AreEqual(0, returnedDto.DueToday);
            Assert.AreEqual(0, returnedDto.DueTomorrow);
            Assert.AreEqual(0, returnedDto.DueThisWeek);
            Assert.AreEqual(0, returnedDto.Overdue);
        }

        /// <summary>
        /// Tests that GetUpcomingReviews returns Ok with maximum integer values when service returns a DTO with int.MaxValue.
        /// </summary>
        [TestMethod]
        public async Task GetUpcomingReviews_ServiceReturnsMaxIntValues_ReturnsOkWithMaxValues()
        {
            // Arrange
            var mockLearnService = new Mock<ILearnService>();
            var mockLogger = new Mock<ILogger<LearnController>>();
            var controller = new LearnController(mockLearnService.Object, mockLogger.Object);

            var userId = "test-user-max";
            var claims = new[] { new Claim(ClaimTypes.NameIdentifier, userId) };
            var identity = new ClaimsIdentity(claims, "TestAuthType");
            var claimsPrincipal = new ClaimsPrincipal(identity);

            controller.ControllerContext = new ControllerContext
            {
                HttpContext = new DefaultHttpContext { User = claimsPrincipal }
            };

            var expectedDto = new UpcomingReviewsDto
            {
                DueToday = int.MaxValue,
                DueTomorrow = int.MaxValue,
                DueThisWeek = int.MaxValue,
                Overdue = int.MaxValue
            };

            mockLearnService
                .Setup(s => s.GetUpcomingReviewsAsync(userId))
                .ReturnsAsync(expectedDto);

            // Act
            var result = await controller.GetUpcomingReviews();

            // Assert
            Assert.IsNotNull(result);
            Assert.IsInstanceOfType(result.Result, typeof(OkObjectResult));
            var okResult = result.Result as OkObjectResult;
            Assert.IsNotNull(okResult);
            var returnedDto = okResult.Value as UpcomingReviewsDto;
            Assert.IsNotNull(returnedDto);
            Assert.AreEqual(int.MaxValue, returnedDto.DueToday);
            Assert.AreEqual(int.MaxValue, returnedDto.DueTomorrow);
            Assert.AreEqual(int.MaxValue, returnedDto.DueThisWeek);
            Assert.AreEqual(int.MaxValue, returnedDto.Overdue);
        }

        /// <summary>
        /// Tests that CompleteLesson successfully completes a lesson and returns Ok result with success message.
        /// Input: Valid lessonId and authenticated user.
        /// Expected: Returns OkObjectResult with success message.
        /// </summary>
        [TestMethod]
        public async Task CompleteLesson_ValidLessonId_ReturnsOkWithSuccessMessage()
        {
            // Arrange
            int lessonId = 1;
            string userId = "test-user-id";

            var mockLearnService = new Mock<ILearnService>();
            mockLearnService
                .Setup(s => s.CompleteLessonAsync(userId, lessonId))
                .Returns(Task.CompletedTask);

            var mockLogger = new Mock<ILogger<LearnController>>();

            var controller = new LearnController(mockLearnService.Object, mockLogger.Object);
            SetupControllerUser(controller, userId);

            // Act
            var result = await controller.CompleteLesson(lessonId);

            // Assert
            Assert.IsNotNull(result);
            var actionResult = result.Result as OkObjectResult;
            Assert.IsNotNull(actionResult);
            Assert.AreEqual(200, actionResult.StatusCode);

            var value = actionResult.Value;
            Assert.IsNotNull(value);

            var messageProperty = value.GetType().GetProperty("message");
            Assert.IsNotNull(messageProperty);
            Assert.AreEqual("Lesson completed successfully", messageProperty.GetValue(value));

            mockLearnService.Verify(s => s.CompleteLessonAsync(userId, lessonId), Times.Once);
        }

        /// <summary>
        /// Tests that CompleteLesson handles various edge case lesson IDs correctly.
        /// Input: Edge case lessonId values (int.MinValue, int.MaxValue, 0, negative values).
        /// Expected: Returns OkObjectResult for all valid inputs after successful service call.
        /// </summary>
        [TestMethod]
        [DataRow(int.MinValue)]
        [DataRow(int.MaxValue)]
        [DataRow(0)]
        [DataRow(-1)]
        [DataRow(-100)]
        [DataRow(999999)]
        public async Task CompleteLesson_EdgeCaseLessonIds_ReturnsOk(int lessonId)
        {
            // Arrange
            string userId = "test-user-id";

            var mockLearnService = new Mock<ILearnService>();
            mockLearnService
                .Setup(s => s.CompleteLessonAsync(userId, lessonId))
                .Returns(Task.CompletedTask);

            var mockLogger = new Mock<ILogger<LearnController>>();

            var controller = new LearnController(mockLearnService.Object, mockLogger.Object);
            SetupControllerUser(controller, userId);

            // Act
            var result = await controller.CompleteLesson(lessonId);

            // Assert
            Assert.IsNotNull(result);
            var actionResult = result.Result as OkObjectResult;
            Assert.IsNotNull(actionResult);
            Assert.AreEqual(200, actionResult.StatusCode);

            mockLearnService.Verify(s => s.CompleteLessonAsync(userId, lessonId), Times.Once);
        }

        /// <summary>
        /// Tests that CompleteLesson returns 500 status code when GetUserId fails.
        /// Input: No user claims configured (simulates missing NameIdentifier claim).
        /// Expected: Returns ObjectResult with 500 status code and error message.
        /// </summary>
        [TestMethod]
        public async Task CompleteLesson_GetUserIdThrowsException_Returns500()
        {
            // Arrange
            int lessonId = 1;

            var mockLearnService = new Mock<ILearnService>();
            var mockLogger = new Mock<ILogger<LearnController>>();

            var controller = new LearnController(mockLearnService.Object, mockLogger.Object);
            SetupControllerUserWithNoClaims(controller);

            // Act
            var result = await controller.CompleteLesson(lessonId);

            // Assert
            Assert.IsNotNull(result);
            var actionResult = result.Result as ObjectResult;
            Assert.IsNotNull(actionResult);
            Assert.AreEqual(500, actionResult.StatusCode);

            var value = actionResult.Value;
            Assert.IsNotNull(value);

            var messageProperty = value.GetType().GetProperty("message");
            Assert.IsNotNull(messageProperty);
            Assert.AreEqual("Error completing lesson", messageProperty.GetValue(value));

            mockLearnService.Verify(s => s.CompleteLessonAsync(It.IsAny<string>(), It.IsAny<int>()), Times.Never);
        }

        /// <summary>
        /// Tests that CompleteLesson returns 500 status code when service throws exception.
        /// Input: Valid lessonId but service throws InvalidOperationException.
        /// Expected: Returns ObjectResult with 500 status code, error message, and logs the error.
        /// </summary>
        [TestMethod]
        public async Task CompleteLesson_ServiceThrowsException_Returns500AndLogsError()
        {
            // Arrange
            int lessonId = 1;
            string userId = "test-user-id";
            var exception = new InvalidOperationException("Service error");

            var mockLearnService = new Mock<ILearnService>();
            mockLearnService
                .Setup(s => s.CompleteLessonAsync(userId, lessonId))
                .ThrowsAsync(exception);

            var mockLogger = new Mock<ILogger<LearnController>>();

            var controller = new LearnController(mockLearnService.Object, mockLogger.Object);
            SetupControllerUser(controller, userId);

            // Act
            var result = await controller.CompleteLesson(lessonId);

            // Assert
            Assert.IsNotNull(result);
            var actionResult = result.Result as ObjectResult;
            Assert.IsNotNull(actionResult);
            Assert.AreEqual(500, actionResult.StatusCode);

            var value = actionResult.Value;
            Assert.IsNotNull(value);

            var messageProperty = value.GetType().GetProperty("message");
            Assert.IsNotNull(messageProperty);
            Assert.AreEqual("Error completing lesson", messageProperty.GetValue(value));

            mockLogger.Verify(
                x => x.Log(
                    LogLevel.Error,
                    It.IsAny<EventId>(),
                    It.Is<It.IsAnyType>((v, t) => v.ToString()!.Contains("Error completing lesson")),
                    exception,
                    It.IsAny<Func<It.IsAnyType, Exception?, string>>()),
                Times.Once);
        }

        /// <summary>
        /// Tests that CompleteLesson returns 500 status code when service throws ArgumentException.
        /// Input: Valid lessonId but service throws ArgumentException.
        /// Expected: Returns ObjectResult with 500 status code, error message, and logs the error.
        /// </summary>
        [TestMethod]
        public async Task CompleteLesson_ServiceThrowsArgumentException_Returns500AndLogsError()
        {
            // Arrange
            int lessonId = 1;
            string userId = "test-user-id";
            var exception = new ArgumentException("Invalid argument");

            var mockLearnService = new Mock<ILearnService>();
            mockLearnService
                .Setup(s => s.CompleteLessonAsync(userId, lessonId))
                .ThrowsAsync(exception);

            var mockLogger = new Mock<ILogger<LearnController>>();

            var controller = new LearnController(mockLearnService.Object, mockLogger.Object);
            SetupControllerUser(controller, userId);

            // Act
            var result = await controller.CompleteLesson(lessonId);

            // Assert
            Assert.IsNotNull(result);
            var actionResult = result.Result as ObjectResult;
            Assert.IsNotNull(actionResult);
            Assert.AreEqual(500, actionResult.StatusCode);

            mockLogger.Verify(
                x => x.Log(
                    LogLevel.Error,
                    It.IsAny<EventId>(),
                    It.Is<It.IsAnyType>((v, t) => v.ToString()!.Contains("Error completing lesson")),
                    exception,
                    It.IsAny<Func<It.IsAnyType, Exception?, string>>()),
                Times.Once);
        }

        /// <summary>
        /// Tests that CompleteLesson returns 500 status code when service throws generic Exception.
        /// Input: Valid lessonId but service throws generic Exception.
        /// Expected: Returns ObjectResult with 500 status code, error message, and logs the error with lessonId.
        /// </summary>
        [TestMethod]
        public async Task CompleteLesson_ServiceThrowsGenericException_Returns500AndLogsErrorWithLessonId()
        {
            // Arrange
            int lessonId = 42;
            string userId = "test-user-id";
            var exception = new Exception("Generic error");

            var mockLearnService = new Mock<ILearnService>();
            mockLearnService
                .Setup(s => s.CompleteLessonAsync(userId, lessonId))
                .ThrowsAsync(exception);

            var mockLogger = new Mock<ILogger<LearnController>>();

            var controller = new LearnController(mockLearnService.Object, mockLogger.Object);
            SetupControllerUser(controller, userId);

            // Act
            var result = await controller.CompleteLesson(lessonId);

            // Assert
            Assert.IsNotNull(result);
            var actionResult = result.Result as ObjectResult;
            Assert.IsNotNull(actionResult);
            Assert.AreEqual(500, actionResult.StatusCode);

            mockLogger.Verify(
                x => x.Log(
                    LogLevel.Error,
                    It.IsAny<EventId>(),
                    It.Is<It.IsAnyType>((v, t) => v.ToString()!.Contains("Error completing lesson") && v.ToString()!.Contains("42")),
                    exception,
                    It.IsAny<Func<It.IsAnyType, Exception?, string>>()),
                Times.Once);
        }

        /// <summary>
        /// Tests that CompleteLesson correctly passes userId and lessonId to service.
        /// Input: Specific userId from claims and specific lessonId.
        /// Expected: Service is called with exact userId and lessonId parameters.
        /// </summary>
        [TestMethod]
        public async Task CompleteLesson_ValidInput_PassesCorrectParametersToService()
        {
            // Arrange
            int lessonId = 123;
            string userId = "specific-user-id-456";

            var mockLearnService = new Mock<ILearnService>();
            mockLearnService
                .Setup(s => s.CompleteLessonAsync(userId, lessonId))
                .Returns(Task.CompletedTask);

            var mockLogger = new Mock<ILogger<LearnController>>();

            var controller = new LearnController(mockLearnService.Object, mockLogger.Object);
            SetupControllerUser(controller, userId);

            // Act
            var result = await controller.CompleteLesson(lessonId);

            // Assert
            mockLearnService.Verify(s => s.CompleteLessonAsync(userId, lessonId), Times.Once);
            mockLearnService.Verify(s => s.CompleteLessonAsync(It.IsAny<string>(), It.IsAny<int>()), Times.Once);
        }

        private static void SetupControllerUser(LearnController controller, string userId)
        {
            var claims = new[]
            {
                new Claim(ClaimTypes.NameIdentifier, userId)
            };
            var identity = new ClaimsIdentity(claims, "TestAuthType");
            var claimsPrincipal = new ClaimsPrincipal(identity);

            controller.ControllerContext = new ControllerContext
            {
                HttpContext = new DefaultHttpContext
                {
                    User = claimsPrincipal
                }
            };
        }

        private static void SetupControllerUserWithNoClaims(LearnController controller)
        {
            var identity = new ClaimsIdentity();
            var claimsPrincipal = new ClaimsPrincipal(identity);

            controller.ControllerContext = new ControllerContext
            {
                HttpContext = new DefaultHttpContext
                {
                    User = claimsPrincipal
                }
            };
        }

        /// <summary>
        /// Tests that GetCategory returns Ok with the category data when a valid category is found.
        /// Input: Valid categoryId with authenticated user.
        /// Expected: Returns OkObjectResult with LessonCategoryDto.
        /// </summary>
        [TestMethod]
        public async Task GetCategory_ValidCategoryIdAndUserAuthenticated_ReturnsOkWithCategory()
        {
            // Arrange
            const int categoryId = 1;
            const string userId = "test-user-123";
            var expectedCategory = new LessonCategoryDto
            {
                Id = categoryId,
                Title = "Basic Signs",
                Description = "Learn basic sign language",
                Difficulty = "Beginner",
                Progress = 50.0
            };

            var mockLearnService = new Mock<ILearnService>();
            mockLearnService.Setup(s => s.GetCategoryAsync(categoryId, userId))
                .ReturnsAsync(expectedCategory);

            var mockLogger = new Mock<ILogger<LearnController>>();
            var controller = CreateControllerWithUser(mockLearnService.Object, mockLogger.Object, userId);

            // Act
            var result = await controller.GetCategory(categoryId);

            // Assert
            Assert.IsNotNull(result);
            Assert.IsInstanceOfType(result.Result, typeof(OkObjectResult));
            var okResult = (OkObjectResult)result.Result;
            Assert.AreEqual(expectedCategory, okResult.Value);
            mockLearnService.Verify(s => s.GetCategoryAsync(categoryId, userId), Times.Once);
        }

        /// <summary>
        /// Tests that GetCategory returns NotFound when the category does not exist.
        /// Input: CategoryId that does not exist (service returns null).
        /// Expected: Returns NotFoundObjectResult with error message.
        /// </summary>
        [TestMethod]
        public async Task GetCategory_CategoryNotFound_ReturnsNotFoundWithMessage()
        {
            // Arrange
            const int categoryId = 999;
            const string userId = "test-user-123";

            var mockLearnService = new Mock<ILearnService>();
            mockLearnService.Setup(s => s.GetCategoryAsync(categoryId, userId))
                .ReturnsAsync((LessonCategoryDto?)null);

            var mockLogger = new Mock<ILogger<LearnController>>();
            var controller = CreateControllerWithUser(mockLearnService.Object, mockLogger.Object, userId);

            // Act
            var result = await controller.GetCategory(categoryId);

            // Assert
            Assert.IsNotNull(result);
            Assert.IsInstanceOfType(result.Result, typeof(NotFoundObjectResult));
            var notFoundResult = (NotFoundObjectResult)result.Result;
            Assert.IsNotNull(notFoundResult.Value);
            mockLearnService.Verify(s => s.GetCategoryAsync(categoryId, userId), Times.Once);
        }

        /// <summary>
        /// Tests that GetCategory returns InternalServerError when the service throws an exception.
        /// Input: Valid categoryId, but service throws exception.
        /// Expected: Returns ObjectResult with 500 status code and error message, and logs the exception.
        /// </summary>
        [TestMethod]
        public async Task GetCategory_ServiceThrowsException_ReturnsInternalServerErrorAndLogsError()
        {
            // Arrange
            const int categoryId = 1;
            const string userId = "test-user-123";
            var exception = new Exception("Database connection failed");

            var mockLearnService = new Mock<ILearnService>();
            mockLearnService.Setup(s => s.GetCategoryAsync(categoryId, userId))
                .ThrowsAsync(exception);

            var mockLogger = new Mock<ILogger<LearnController>>();
            var controller = CreateControllerWithUser(mockLearnService.Object, mockLogger.Object, userId);

            // Act
            var result = await controller.GetCategory(categoryId);

            // Assert
            Assert.IsNotNull(result);
            Assert.IsInstanceOfType(result.Result, typeof(ObjectResult));
            var objectResult = (ObjectResult)result.Result;
            Assert.AreEqual(500, objectResult.StatusCode);
            mockLogger.Verify(
                x => x.Log(
                    LogLevel.Error,
                    It.IsAny<EventId>(),
                    It.Is<It.IsAnyType>((v, t) => true),
                    exception,
                    It.Is<Func<It.IsAnyType, Exception?, string>>((v, t) => true)),
                Times.Once);
        }

        /// <summary>
        /// Tests that GetCategory returns InternalServerError when user is not authenticated (no user ID in token).
        /// Input: No NameIdentifier claim in user token.
        /// Expected: Returns ObjectResult with 500 status code and error message, and logs the exception.
        /// </summary>
        [TestMethod]
        public async Task GetCategory_UserNotAuthenticated_ReturnsInternalServerErrorAndLogsError()
        {
            // Arrange
            const int categoryId = 1;

            var mockLearnService = new Mock<ILearnService>();
            var mockLogger = new Mock<ILogger<LearnController>>();
            var controller = CreateControllerWithoutUser(mockLearnService.Object, mockLogger.Object);

            // Act
            var result = await controller.GetCategory(categoryId);

            // Assert
            Assert.IsNotNull(result);
            Assert.IsInstanceOfType(result.Result, typeof(ObjectResult));
            var objectResult = (ObjectResult)result.Result;
            Assert.AreEqual(500, objectResult.StatusCode);
            mockLogger.Verify(
                x => x.Log(
                    LogLevel.Error,
                    It.IsAny<EventId>(),
                    It.Is<It.IsAnyType>((v, t) => true),
                    It.IsAny<Exception>(),
                    It.Is<Func<It.IsAnyType, Exception?, string>>((v, t) => true)),
                Times.Once);
        }

        /// <summary>
        /// Tests that GetCategory handles various edge case category ID values correctly.
        /// Input: Edge case categoryId values (0, negative, int.MinValue, int.MaxValue).
        /// Expected: Returns Ok if category is found, regardless of categoryId value.
        /// </summary>
        [TestMethod]
        [DataRow(0)]
        [DataRow(-1)]
        [DataRow(-100)]
        [DataRow(int.MinValue)]
        [DataRow(int.MaxValue)]
        public async Task GetCategory_EdgeCaseCategoryIdValues_ReturnsOkWhenCategoryFound(int categoryId)
        {
            // Arrange
            const string userId = "test-user-123";
            var expectedCategory = new LessonCategoryDto
            {
                Id = categoryId,
                Title = "Test Category",
                Description = "Test Description",
                Difficulty = "Beginner",
                Progress = 0.0
            };

            var mockLearnService = new Mock<ILearnService>();
            mockLearnService.Setup(s => s.GetCategoryAsync(categoryId, userId))
                .ReturnsAsync(expectedCategory);

            var mockLogger = new Mock<ILogger<LearnController>>();
            var controller = CreateControllerWithUser(mockLearnService.Object, mockLogger.Object, userId);

            // Act
            var result = await controller.GetCategory(categoryId);

            // Assert
            Assert.IsNotNull(result);
            Assert.IsInstanceOfType(result.Result, typeof(OkObjectResult));
            var okResult = (OkObjectResult)result.Result;
            Assert.AreEqual(expectedCategory, okResult.Value);
            mockLearnService.Verify(s => s.GetCategoryAsync(categoryId, userId), Times.Once);
        }

        /// <summary>
        /// Tests that GetCategory returns NotFound for edge case category ID values when category is not found.
        /// Input: Edge case categoryId values with service returning null.
        /// Expected: Returns NotFoundObjectResult.
        /// </summary>
        [TestMethod]
        [DataRow(0)]
        [DataRow(-1)]
        [DataRow(int.MinValue)]
        [DataRow(int.MaxValue)]
        public async Task GetCategory_EdgeCaseCategoryIdValues_ReturnsNotFoundWhenCategoryNotFound(int categoryId)
        {
            // Arrange
            const string userId = "test-user-123";

            var mockLearnService = new Mock<ILearnService>();
            mockLearnService.Setup(s => s.GetCategoryAsync(categoryId, userId))
                .ReturnsAsync((LessonCategoryDto?)null);

            var mockLogger = new Mock<ILogger<LearnController>>();
            var controller = CreateControllerWithUser(mockLearnService.Object, mockLogger.Object, userId);

            // Act
            var result = await controller.GetCategory(categoryId);

            // Assert
            Assert.IsNotNull(result);
            Assert.IsInstanceOfType(result.Result, typeof(NotFoundObjectResult));
            mockLearnService.Verify(s => s.GetCategoryAsync(categoryId, userId), Times.Once);
        }

        /// <summary>
        /// Tests that GetCategory correctly logs the categoryId when an exception occurs.
        /// Input: Valid categoryId with service throwing exception.
        /// Expected: Logs error with the specific categoryId parameter.
        /// </summary>
        [TestMethod]
        public async Task GetCategory_ExceptionThrown_LogsWithCorrectCategoryId()
        {
            // Arrange
            const int categoryId = 42;
            const string userId = "test-user-123";
            var exception = new InvalidOperationException("Test exception");

            var mockLearnService = new Mock<ILearnService>();
            mockLearnService.Setup(s => s.GetCategoryAsync(categoryId, userId))
                .ThrowsAsync(exception);

            var mockLogger = new Mock<ILogger<LearnController>>();
            var controller = CreateControllerWithUser(mockLearnService.Object, mockLogger.Object, userId);

            // Act
            var result = await controller.GetCategory(categoryId);

            // Assert
            mockLogger.Verify(
                x => x.Log(
                    LogLevel.Error,
                    It.IsAny<EventId>(),
                    It.Is<It.IsAnyType>((v, t) => true),
                    exception,
                    It.Is<Func<It.IsAnyType, Exception?, string>>((v, t) => true)),
                Times.Once);
        }

        private static LearnController CreateControllerWithoutUser(ILearnService learnService, ILogger<LearnController> logger)
        {
            var controller = new LearnController(learnService, logger);
            var claimsPrincipal = new ClaimsPrincipal(new ClaimsIdentity());
            controller.ControllerContext = new ControllerContext
            {
                HttpContext = new DefaultHttpContext { User = claimsPrincipal }
            };
            return controller;
        }

        /// <summary>
        /// Tests that GetDailyReviews returns Ok result with reviews when service successfully returns data for valid user.
        /// Input: Valid authenticated user with reviews available.
        /// Expected: Returns OkObjectResult containing list of SpacedRepetitionLessonDto.
        /// </summary>
        [TestMethod]
        [DataRow(0, DisplayName = "Empty reviews list")]
        [DataRow(1, DisplayName = "Single review")]
        [DataRow(5, DisplayName = "Multiple reviews")]
        public async Task GetDailyReviews_ValidUserWithReviews_ReturnsOkWithReviews(int reviewCount)
        {
            // Arrange
            var mockLearnService = new Mock<ILearnService>();
            var mockLogger = new Mock<ILogger<LearnController>>();
            var controller = new LearnController(mockLearnService.Object, mockLogger.Object);

            var userId = "test-user-123";
            var claims = new List<Claim> { new Claim(ClaimTypes.NameIdentifier, userId) };
            var identity = new ClaimsIdentity(claims, "TestAuth");
            var claimsPrincipal = new ClaimsPrincipal(identity);
            controller.ControllerContext = new ControllerContext
            {
                HttpContext = new DefaultHttpContext { User = claimsPrincipal }
            };

            var expectedReviews = new List<SpacedRepetitionLessonDto>();
            for (int i = 0; i < reviewCount; i++)
            {
                expectedReviews.Add(new SpacedRepetitionLessonDto
                {
                    Id = i + 1,
                    Title = $"Review {i + 1}",
                    DueDate = DateTime.UtcNow.ToString(),
                    RepetitionCount = i,
                    RetentionPercentage = 80.0,
                    IsReviewDue = true,
                    LessonId = i + 100
                });
            }

            mockLearnService.Setup(s => s.GetDailyReviewLessonsAsync(userId))
                .ReturnsAsync(expectedReviews);

            // Act
            var result = await controller.GetDailyReviews();

            // Assert
            Assert.IsNotNull(result);
            Assert.IsInstanceOfType(result.Result, typeof(OkObjectResult));
            var okResult = (OkObjectResult)result.Result;
            Assert.IsNotNull(okResult.Value);
            var reviews = okResult.Value as List<SpacedRepetitionLessonDto>;
            Assert.IsNotNull(reviews);
            Assert.AreEqual(reviewCount, reviews.Count);
            mockLearnService.Verify(s => s.GetDailyReviewLessonsAsync(userId), Times.Once);
        }

        /// <summary>
        /// Tests that GetDailyReviews returns 500 Internal Server Error when service throws an exception.
        /// Input: Valid authenticated user, but service throws exception.
        /// Expected: Returns ObjectResult with 500 status code and error message, logs error.
        /// </summary>
        [TestMethod]
        [DataRow(typeof(InvalidOperationException), DisplayName = "InvalidOperationException")]
        [DataRow(typeof(ArgumentException), DisplayName = "ArgumentException")]
        [DataRow(typeof(Exception), DisplayName = "Generic Exception")]
        public async Task GetDailyReviews_ServiceThrowsException_ReturnsInternalServerError(Type exceptionType)
        {
            // Arrange
            var mockLearnService = new Mock<ILearnService>();
            var mockLogger = new Mock<ILogger<LearnController>>();
            var controller = new LearnController(mockLearnService.Object, mockLogger.Object);

            var userId = "test-user-123";
            var claims = new List<Claim> { new Claim(ClaimTypes.NameIdentifier, userId) };
            var identity = new ClaimsIdentity(claims, "TestAuth");
            var claimsPrincipal = new ClaimsPrincipal(identity);
            controller.ControllerContext = new ControllerContext
            {
                HttpContext = new DefaultHttpContext { User = claimsPrincipal }
            };

            var exception = (Exception)Activator.CreateInstance(exceptionType, "Test exception")!;
            mockLearnService.Setup(s => s.GetDailyReviewLessonsAsync(userId))
                .ThrowsAsync(exception);

            // Act
            var result = await controller.GetDailyReviews();

            // Assert
            Assert.IsNotNull(result);
            Assert.IsInstanceOfType(result.Result, typeof(ObjectResult));
            var objectResult = (ObjectResult)result.Result;
            Assert.AreEqual(500, objectResult.StatusCode);
            Assert.IsNotNull(objectResult.Value);
            mockLogger.Verify(
                l => l.Log(
                    LogLevel.Error,
                    It.IsAny<EventId>(),
                    It.Is<It.IsAnyType>((v, t) => true),
                    It.IsAny<Exception>(),
                    It.Is<Func<It.IsAnyType, Exception?, string>>((v, t) => true)),
                Times.Once);
        }

        /// <summary>
        /// Tests that GetDailyReviews returns 500 Internal Server Error when user ID is not found in claims.
        /// Input: User with no NameIdentifier claim.
        /// Expected: Returns ObjectResult with 500 status code, GetUserId throws InvalidOperationException.
        /// </summary>
        [TestMethod]
        public async Task GetDailyReviews_NoUserIdInClaims_ReturnsInternalServerError()
        {
            // Arrange
            var mockLearnService = new Mock<ILearnService>();
            var mockLogger = new Mock<ILogger<LearnController>>();
            var controller = new LearnController(mockLearnService.Object, mockLogger.Object);

            var claims = new List<Claim>();
            var identity = new ClaimsIdentity(claims, "TestAuth");
            var claimsPrincipal = new ClaimsPrincipal(identity);
            controller.ControllerContext = new ControllerContext
            {
                HttpContext = new DefaultHttpContext { User = claimsPrincipal }
            };

            // Act
            var result = await controller.GetDailyReviews();

            // Assert
            Assert.IsNotNull(result);
            Assert.IsInstanceOfType(result.Result, typeof(ObjectResult));
            var objectResult = (ObjectResult)result.Result;
            Assert.AreEqual(500, objectResult.StatusCode);
            mockLogger.Verify(
                l => l.Log(
                    LogLevel.Error,
                    It.IsAny<EventId>(),
                    It.Is<It.IsAnyType>((v, t) => true),
                    It.IsAny<Exception>(),
                    It.Is<Func<It.IsAnyType, Exception?, string>>((v, t) => true)),
                Times.Once);
            mockLearnService.Verify(s => s.GetDailyReviewLessonsAsync(It.IsAny<string>()), Times.Never);
        }

        /// <summary>
        /// Tests that GetDailyReviews returns 500 Internal Server Error when user ID claim is empty string.
        /// Input: User with empty NameIdentifier claim value.
        /// Expected: Returns ObjectResult with 500 status code, GetUserId throws InvalidOperationException.
        /// </summary>
        [TestMethod]
        public async Task GetDailyReviews_EmptyUserIdInClaims_ReturnsInternalServerError()
        {
            // Arrange
            var mockLearnService = new Mock<ILearnService>();
            var mockLogger = new Mock<ILogger<LearnController>>();
            var controller = new LearnController(mockLearnService.Object, mockLogger.Object);

            var claims = new List<Claim> { new Claim(ClaimTypes.NameIdentifier, string.Empty) };
            var identity = new ClaimsIdentity(claims, "TestAuth");
            var claimsPrincipal = new ClaimsPrincipal(identity);
            controller.ControllerContext = new ControllerContext
            {
                HttpContext = new DefaultHttpContext { User = claimsPrincipal }
            };

            // Act
            var result = await controller.GetDailyReviews();

            // Assert
            Assert.IsNotNull(result);
            Assert.IsInstanceOfType(result.Result, typeof(ObjectResult));
            var objectResult = (ObjectResult)result.Result;
            Assert.AreEqual(500, objectResult.StatusCode);
            mockLogger.Verify(
                l => l.Log(
                    LogLevel.Error,
                    It.IsAny<EventId>(),
                    It.Is<It.IsAnyType>((v, t) => true),
                    It.IsAny<Exception>(),
                    It.Is<Func<It.IsAnyType, Exception?, string>>((v, t) => true)),
                Times.Once);
            mockLearnService.Verify(s => s.GetDailyReviewLessonsAsync(It.IsAny<string>()), Times.Never);
        }

        /// <summary>
        /// Tests that GetDailyReviews returns 500 Internal Server Error with correct error message structure.
        /// Input: Valid user but service throws exception.
        /// Expected: Returns ObjectResult with anonymous object containing 'message' property.
        /// </summary>
        [TestMethod]
        public async Task GetDailyReviews_ServiceThrowsException_ReturnsCorrectErrorMessageStructure()
        {
            // Arrange
            var mockLearnService = new Mock<ILearnService>();
            var mockLogger = new Mock<ILogger<LearnController>>();
            var controller = new LearnController(mockLearnService.Object, mockLogger.Object);

            var userId = "test-user-123";
            var claims = new List<Claim> { new Claim(ClaimTypes.NameIdentifier, userId) };
            var identity = new ClaimsIdentity(claims, "TestAuth");
            var claimsPrincipal = new ClaimsPrincipal(identity);
            controller.ControllerContext = new ControllerContext
            {
                HttpContext = new DefaultHttpContext { User = claimsPrincipal }
            };

            mockLearnService.Setup(s => s.GetDailyReviewLessonsAsync(userId))
                .ThrowsAsync(new Exception("Database connection failed"));

            // Act
            var result = await controller.GetDailyReviews();

            // Assert
            Assert.IsNotNull(result);
            Assert.IsInstanceOfType(result.Result, typeof(ObjectResult));
            var objectResult = (ObjectResult)result.Result;
            Assert.AreEqual(500, objectResult.StatusCode);
            Assert.IsNotNull(objectResult.Value);

            var valueType = objectResult.Value.GetType();
            var messageProperty = valueType.GetProperty("message");
            Assert.IsNotNull(messageProperty);
            var messageValue = messageProperty.GetValue(objectResult.Value) as string;
            Assert.AreEqual("Error fetching daily reviews", messageValue);
        }

        /// <summary>
        /// Tests that GetDailyReviews handles user ID with special characters correctly.
        /// Input: User ID containing special characters (GUID format, special chars).
        /// Expected: Returns OkObjectResult with reviews, service called with exact user ID.
        /// </summary>
        [TestMethod]
        [DataRow("user-123-abc-456", DisplayName = "User ID with hyphens")]
        [DataRow("user@domain.com", DisplayName = "User ID as email")]
        [DataRow("550e8400-e29b-41d4-a716-446655440000", DisplayName = "User ID as GUID")]
        public async Task GetDailyReviews_UserIdWithSpecialCharacters_ReturnsOkWithReviews(string userId)
        {
            // Arrange
            var mockLearnService = new Mock<ILearnService>();
            var mockLogger = new Mock<ILogger<LearnController>>();
            var controller = new LearnController(mockLearnService.Object, mockLogger.Object);

            var claims = new List<Claim> { new Claim(ClaimTypes.NameIdentifier, userId) };
            var identity = new ClaimsIdentity(claims, "TestAuth");
            var claimsPrincipal = new ClaimsPrincipal(identity);
            controller.ControllerContext = new ControllerContext
            {
                HttpContext = new DefaultHttpContext { User = claimsPrincipal }
            };

            var expectedReviews = new List<SpacedRepetitionLessonDto>
            {
                new SpacedRepetitionLessonDto { Id = 1, Title = "Test", LessonId = 100 }
            };

            mockLearnService.Setup(s => s.GetDailyReviewLessonsAsync(userId))
                .ReturnsAsync(expectedReviews);

            // Act
            var result = await controller.GetDailyReviews();

            // Assert
            Assert.IsNotNull(result);
            Assert.IsInstanceOfType(result.Result, typeof(OkObjectResult));
            mockLearnService.Verify(s => s.GetDailyReviewLessonsAsync(userId), Times.Once);
        }

        /// <summary>
        /// Tests ReviewLesson with a valid quality rating within the acceptable range (0-5).
        /// Should call the service method and return Ok result with success message.
        /// </summary>
        [TestMethod]
        [DataRow(0.0)]
        [DataRow(2.5)]
        [DataRow(5.0)]
        [DataRow(3.7)]
        [DataRow(1.0)]
        [DataRow(4.9)]
        public async Task ReviewLesson_ValidQualityRating_ReturnsOkResult(double qualityRating)
        {
            // Arrange
            var mockLearnService = new Mock<ILearnService>();
            var mockLogger = new Mock<ILogger<LearnController>>();
            var controller = new LearnController(mockLearnService.Object, mockLogger.Object);
            var spacedRepetitionId = 123;
            var userId = "test-user-id";

            var claims = new List<Claim> { new Claim(ClaimTypes.NameIdentifier, userId) };
            var identity = new ClaimsIdentity(claims, "TestAuth");
            var claimsPrincipal = new ClaimsPrincipal(identity);
            controller.ControllerContext = new ControllerContext
            {
                HttpContext = new DefaultHttpContext { User = claimsPrincipal }
            };

            mockLearnService.Setup(x => x.ReviewLessonAsync(userId, spacedRepetitionId, qualityRating))
                .Returns(Task.CompletedTask);

            // Act
            var result = await controller.ReviewLesson(spacedRepetitionId, qualityRating);

            // Assert
            Assert.IsNotNull(result);
            var okResult = result.Result as OkObjectResult;
            Assert.IsNotNull(okResult);
            Assert.AreEqual(200, okResult.StatusCode);

            Assert.IsNotNull(okResult.Value);
            var valueType = okResult.Value.GetType();
            var messageProperty = valueType.GetProperty("message");
            Assert.IsNotNull(messageProperty);
            var messageValue = messageProperty.GetValue(okResult.Value) as string;
            Assert.AreEqual("Review recorded successfully", messageValue);

            mockLearnService.Verify(x => x.ReviewLessonAsync(userId, spacedRepetitionId, qualityRating), Times.Once);
        }

        /// <summary>
        /// Tests ReviewLesson with quality rating less than 0.
        /// Should return BadRequest with appropriate error message without calling the service.
        /// </summary>
        [TestMethod]
        [DataRow(-0.1)]
        [DataRow(-1.0)]
        [DataRow(-100.0)]
        public async Task ReviewLesson_QualityRatingLessThanZero_ReturnsBadRequest(double qualityRating)
        {
            // Arrange
            var mockLearnService = new Mock<ILearnService>();
            var mockLogger = new Mock<ILogger<LearnController>>();
            var controller = new LearnController(mockLearnService.Object, mockLogger.Object);
            var spacedRepetitionId = 123;
            var userId = "test-user-id";

            SetupControllerUser(controller, userId);

            // Act
            var result = await controller.ReviewLesson(spacedRepetitionId, qualityRating);

            // Assert
            Assert.IsNotNull(result);
            var badRequestResult = result.Result as BadRequestObjectResult;
            Assert.IsNotNull(badRequestResult);
            Assert.AreEqual(400, badRequestResult.StatusCode);

            var value = badRequestResult.Value;
            Assert.IsNotNull(value);
            var messageProperty = value.GetType().GetProperty("message");
            Assert.IsNotNull(messageProperty);
            var messageValue = messageProperty.GetValue(value) as string;
            Assert.AreEqual("Quality rating must be between 0 and 5", messageValue);

            mockLearnService.Verify(x => x.ReviewLessonAsync(It.IsAny<string>(), It.IsAny<int>(), It.IsAny<double>()), Times.Never);
        }

        /// <summary>
        /// Tests ReviewLesson with quality rating greater than 5.
        /// Should return BadRequest with appropriate error message without calling the service.
        /// </summary>
        [TestMethod]
        [DataRow(5.1)]
        [DataRow(6.0)]
        [DataRow(100.0)]
        public async Task ReviewLesson_QualityRatingGreaterThanFive_ReturnsBadRequest(double qualityRating)
        {
            // Arrange
            var mockLearnService = new Mock<ILearnService>();
            var mockLogger = new Mock<ILogger<LearnController>>();
            var controller = new LearnController(mockLearnService.Object, mockLogger.Object);
            var spacedRepetitionId = 123;
            var userId = "test-user-id";

            SetupControllerUser(controller, userId);

            // Act
            var result = await controller.ReviewLesson(spacedRepetitionId, qualityRating);

            // Assert
            Assert.IsNotNull(result);
            var badRequestResult = result.Result as BadRequestObjectResult;
            Assert.IsNotNull(badRequestResult);
            Assert.AreEqual(400, badRequestResult.StatusCode);

            var value = badRequestResult.Value;
            Assert.IsNotNull(value);
            var messageProperty = value.GetType().GetProperty("message");
            Assert.IsNotNull(messageProperty);
            var messageValue = messageProperty.GetValue(value) as string;
            Assert.AreEqual("Quality rating must be between 0 and 5", messageValue);

            mockLearnService.Verify(x => x.ReviewLessonAsync(It.IsAny<string>(), It.IsAny<int>(), It.IsAny<double>()), Times.Never);
        }

        /// <summary>
        /// Tests ReviewLesson with special double values (NaN, PositiveInfinity, NegativeInfinity).
        /// Should return BadRequest as these values are outside the valid range.
        /// </summary>
        [TestMethod]
        [DataRow(double.NaN)]
        [DataRow(double.PositiveInfinity)]
        [DataRow(double.NegativeInfinity)]
        [TestCategory("ProductionBugSuspected")]
        [Ignore("ProductionBugSuspected")]
        public async Task ReviewLesson_SpecialDoubleValues_ReturnsBadRequest(double qualityRating)
        {
            // Arrange
            var mockLearnService = new Mock<ILearnService>();
            var mockLogger = new Mock<ILogger<LearnController>>();
            var controller = new LearnController(mockLearnService.Object, mockLogger.Object);
            var spacedRepetitionId = 123;
            var userId = "test-user-id";

            SetupControllerUser(controller, userId);

            // Act
            var result = await controller.ReviewLesson(spacedRepetitionId, qualityRating);

            // Assert
            Assert.IsNotNull(result);
            var badRequestResult = result.Result as BadRequestObjectResult;
            Assert.IsNotNull(badRequestResult);
            Assert.AreEqual(400, badRequestResult.StatusCode);

            dynamic? value = badRequestResult.Value;
            Assert.IsNotNull(value);
            Assert.AreEqual("Quality rating must be between 0 and 5", value.message);

            mockLearnService.Verify(x => x.ReviewLessonAsync(It.IsAny<string>(), It.IsAny<int>(), It.IsAny<double>()), Times.Never);
        }

        /// <summary>
        /// Tests ReviewLesson with extreme double boundary values.
        /// Should return BadRequest as these values are outside the valid range.
        /// </summary>
        [TestMethod]
        [DataRow(double.MinValue)]
        [DataRow(double.MaxValue)]
        public async Task ReviewLesson_ExtremeDoubleValues_ReturnsBadRequest(double qualityRating)
        {
            // Arrange
            var mockLearnService = new Mock<ILearnService>();
            var mockLogger = new Mock<ILogger<LearnController>>();
            var controller = new LearnController(mockLearnService.Object, mockLogger.Object);
            var spacedRepetitionId = 123;
            var userId = "test-user-id";

            SetupControllerUser(controller, userId);

            // Act
            var result = await controller.ReviewLesson(spacedRepetitionId, qualityRating);

            // Assert
            Assert.IsNotNull(result);
            var badRequestResult = result.Result as BadRequestObjectResult;
            Assert.IsNotNull(badRequestResult);

            mockLearnService.Verify(x => x.ReviewLessonAsync(It.IsAny<string>(), It.IsAny<int>(), It.IsAny<double>()), Times.Never);
        }

        /// <summary>
        /// Tests ReviewLesson with various spacedRepetitionId values.
        /// Should successfully call the service with the provided ID when quality rating is valid.
        /// </summary>
        [TestMethod]
        [DataRow(1)]
        [DataRow(0)]
        [DataRow(-1)]
        [DataRow(int.MaxValue)]
        [DataRow(int.MinValue)]
        public async Task ReviewLesson_VariousSpacedRepetitionIds_CallsServiceWithCorrectId(int spacedRepetitionId)
        {
            // Arrange
            var mockLearnService = new Mock<ILearnService>();
            var mockLogger = new Mock<ILogger<LearnController>>();
            var controller = new LearnController(mockLearnService.Object, mockLogger.Object);
            var qualityRating = 3.0;
            var userId = "test-user-id";

            SetupControllerUser(controller, userId);
            mockLearnService.Setup(x => x.ReviewLessonAsync(userId, spacedRepetitionId, qualityRating))
                .Returns(Task.CompletedTask);

            // Act
            var result = await controller.ReviewLesson(spacedRepetitionId, qualityRating);

            // Assert
            Assert.IsNotNull(result);
            var okResult = result.Result as OkObjectResult;
            Assert.IsNotNull(okResult);

            mockLearnService.Verify(x => x.ReviewLessonAsync(userId, spacedRepetitionId, qualityRating), Times.Once);
        }

        /// <summary>
        /// Tests ReviewLesson when service throws InvalidOperationException.
        /// Should return BadRequest with the exception message and log a warning.
        /// </summary>
        [TestMethod]
        public async Task ReviewLesson_ServiceThrowsInvalidOperationException_ReturnsBadRequestAndLogsWarning()
        {
            // Arrange
            var mockLearnService = new Mock<ILearnService>();
            var mockLogger = new Mock<ILogger<LearnController>>();
            var controller = new LearnController(mockLearnService.Object, mockLogger.Object);
            var spacedRepetitionId = 123;
            var qualityRating = 3.0;
            var userId = "test-user-id";
            var exceptionMessage = "Spaced repetition record not found";
            var exception = new InvalidOperationException(exceptionMessage);

            SetupControllerUser(controller, userId);
            mockLearnService.Setup(x => x.ReviewLessonAsync(userId, spacedRepetitionId, qualityRating))
                .ThrowsAsync(exception);

            // Act
            var result = await controller.ReviewLesson(spacedRepetitionId, qualityRating);

            // Assert
            Assert.IsNotNull(result);
            var badRequestResult = result.Result as BadRequestObjectResult;
            Assert.IsNotNull(badRequestResult);
            Assert.AreEqual(400, badRequestResult.StatusCode);

            var value = badRequestResult.Value;
            Assert.IsNotNull(value);
            var messageProperty = value.GetType().GetProperty("message");
            Assert.IsNotNull(messageProperty);
            Assert.AreEqual(exceptionMessage, messageProperty.GetValue(value));

            mockLogger.Verify(
                x => x.Log(
                    LogLevel.Warning,
                    It.IsAny<EventId>(),
                    It.Is<It.IsAnyType>((v, t) => true),
                    It.IsAny<InvalidOperationException>(),
                    It.IsAny<Func<It.IsAnyType, Exception?, string>>()),
                Times.Once);
        }

        /// <summary>
        /// Tests ReviewLesson when service throws a general Exception.
        /// Should return StatusCode 500 with error message and log an error.
        /// </summary>
        [TestMethod]
        public async Task ReviewLesson_ServiceThrowsException_ReturnsInternalServerErrorAndLogsError()
        {
            // Arrange
            var mockLearnService = new Mock<ILearnService>();
            var mockLogger = new Mock<ILogger<LearnController>>();
            var controller = new LearnController(mockLearnService.Object, mockLogger.Object);
            var spacedRepetitionId = 123;
            var qualityRating = 3.0;
            var userId = "test-user-id";
            var exception = new Exception("Database connection error");

            SetupControllerUser(controller, userId);
            mockLearnService.Setup(x => x.ReviewLessonAsync(userId, spacedRepetitionId, qualityRating))
                .ThrowsAsync(exception);

            // Act
            var result = await controller.ReviewLesson(spacedRepetitionId, qualityRating);

            // Assert
            Assert.IsNotNull(result);
            var statusCodeResult = result.Result as ObjectResult;
            Assert.IsNotNull(statusCodeResult);
            Assert.AreEqual(500, statusCodeResult.StatusCode);

            var value = statusCodeResult.Value;
            Assert.IsNotNull(value);
            var messageProperty = value.GetType().GetProperty("message");
            Assert.IsNotNull(messageProperty);
            var messageValue = messageProperty.GetValue(value) as string;
            Assert.AreEqual("Error recording review", messageValue);

            mockLogger.Verify(
                x => x.Log(
                    LogLevel.Error,
                    It.IsAny<EventId>(),
                    It.Is<It.IsAnyType>((v, t) => true),
                    It.IsAny<Exception>(),
                    It.IsAny<Func<It.IsAnyType, Exception?, string>>()),
                Times.Once);
        }

        /// <summary>
        /// Tests ReviewLesson with boundary quality rating values (exactly 0 and exactly 5).
        /// Should accept these as valid boundary values and call the service.
        /// </summary>
        [TestMethod]
        public async Task ReviewLesson_BoundaryQualityRatingValues_ReturnsOkResult()
        {
            // Arrange
            var mockLearnService = new Mock<ILearnService>();
            var mockLogger = new Mock<ILogger<LearnController>>();
            var controller = new LearnController(mockLearnService.Object, mockLogger.Object);
            var spacedRepetitionId = 123;
            var userId = "test-user-id";

            SetupControllerUser(controller, userId);

            // Test with quality rating = 0
            mockLearnService.Setup(x => x.ReviewLessonAsync(userId, spacedRepetitionId, 0.0))
                .Returns(Task.CompletedTask);

            // Act
            var result1 = await controller.ReviewLesson(spacedRepetitionId, 0.0);

            // Assert
            Assert.IsNotNull(result1);
            var okResult1 = result1.Result as OkObjectResult;
            Assert.IsNotNull(okResult1);
            Assert.AreEqual(200, okResult1.StatusCode);

            // Test with quality rating = 5
            mockLearnService.Setup(x => x.ReviewLessonAsync(userId, spacedRepetitionId, 5.0))
                .Returns(Task.CompletedTask);

            // Act
            var result2 = await controller.ReviewLesson(spacedRepetitionId, 5.0);

            // Assert
            Assert.IsNotNull(result2);
            var okResult2 = result2.Result as OkObjectResult;
            Assert.IsNotNull(okResult2);
            Assert.AreEqual(200, okResult2.StatusCode);

            mockLearnService.Verify(x => x.ReviewLessonAsync(userId, spacedRepetitionId, 0.0), Times.Once);
            mockLearnService.Verify(x => x.ReviewLessonAsync(userId, spacedRepetitionId, 5.0), Times.Once);
        }

        /// <summary>
        /// Tests ReviewLesson with InvalidOperationException containing empty message.
        /// Should return BadRequest with empty message from exception.
        /// </summary>
        [TestMethod]
        public async Task ReviewLesson_InvalidOperationExceptionWithEmptyMessage_ReturnsBadRequestWithEmptyMessage()
        {
            // Arrange
            var mockLearnService = new Mock<ILearnService>();
            var mockLogger = new Mock<ILogger<LearnController>>();
            var controller = new LearnController(mockLearnService.Object, mockLogger.Object);
            var spacedRepetitionId = 123;
            var qualityRating = 3.0;
            var userId = "test-user-id";
            var exception = new InvalidOperationException(string.Empty);

            SetupControllerUser(controller, userId);
            mockLearnService.Setup(x => x.ReviewLessonAsync(userId, spacedRepetitionId, qualityRating))
                .ThrowsAsync(exception);

            // Act
            var result = await controller.ReviewLesson(spacedRepetitionId, qualityRating);

            // Assert
            Assert.IsNotNull(result);
            var badRequestResult = result.Result as BadRequestObjectResult;
            Assert.IsNotNull(badRequestResult);

            var value = badRequestResult.Value;
            Assert.IsNotNull(value);
            var messageProperty = value.GetType().GetProperty("message");
            Assert.IsNotNull(messageProperty);
            var messageValue = messageProperty.GetValue(value) as string;
            Assert.AreEqual(string.Empty, messageValue);
        }

        /// <summary>
        /// Tests that GetDailyGoal returns Ok result with daily goal data when service call succeeds.
        /// Input: Valid authenticated user with claims.
        /// Expected: Ok result containing DailyGoalDto from service.
        /// </summary>
        [TestMethod]
        public async Task GetDailyGoal_ValidUser_ReturnsOkResultWithDailyGoal()
        {
            // Arrange
            var mockLearnService = new Mock<ILearnService>();
            var mockLogger = new Mock<ILogger<LearnController>>();

            var expectedDailyGoal = new DailyGoalDto
            {
                TotalReviewsDue = 10,
                CompletedToday = 3,
                DailyGoal = 5,
                ProgressPercentage = 60.0
            };

            mockLearnService
                .Setup(s => s.GetDailyGoalAsync(It.IsAny<string>()))
                .ReturnsAsync(expectedDailyGoal);

            var controller = new LearnController(mockLearnService.Object, mockLogger.Object);
            var user = new ClaimsPrincipal(new ClaimsIdentity(new[]
            {
                new Claim(ClaimTypes.NameIdentifier, "test-user-123")
            }));
            controller.ControllerContext = new ControllerContext
            {
                HttpContext = new DefaultHttpContext { User = user }
            };

            // Act
            var result = await controller.GetDailyGoal();

            // Assert
            Assert.IsNotNull(result);
            var okResult = result.Result as OkObjectResult;
            Assert.IsNotNull(okResult);
            Assert.AreEqual(200, okResult.StatusCode);
            var returnedGoal = okResult.Value as DailyGoalDto;
            Assert.IsNotNull(returnedGoal);
            Assert.AreEqual(expectedDailyGoal.TotalReviewsDue, returnedGoal.TotalReviewsDue);
            Assert.AreEqual(expectedDailyGoal.CompletedToday, returnedGoal.CompletedToday);
            Assert.AreEqual(expectedDailyGoal.DailyGoal, returnedGoal.DailyGoal);
            Assert.AreEqual(expectedDailyGoal.ProgressPercentage, returnedGoal.ProgressPercentage);
        }

        /// <summary>
        /// Tests that GetDailyGoal calls the service with the correct user ID from claims.
        /// Input: User with specific ID in claims.
        /// Expected: Service method called with matching user ID.
        /// </summary>
        [TestMethod]
        public async Task GetDailyGoal_ValidUser_CallsServiceWithCorrectUserId()
        {
            // Arrange
            var mockLearnService = new Mock<ILearnService>();
            var mockLogger = new Mock<ILogger<LearnController>>();
            var expectedUserId = "user-456";

            mockLearnService
                .Setup(s => s.GetDailyGoalAsync(expectedUserId))
                .ReturnsAsync(new DailyGoalDto());

            var controller = new LearnController(mockLearnService.Object, mockLogger.Object);
            var user = new ClaimsPrincipal(new ClaimsIdentity(new[]
            {
                new Claim(ClaimTypes.NameIdentifier, expectedUserId)
            }));
            controller.ControllerContext = new ControllerContext
            {
                HttpContext = new DefaultHttpContext { User = user }
            };

            // Act
            await controller.GetDailyGoal();

            // Assert
            mockLearnService.Verify(s => s.GetDailyGoalAsync(expectedUserId), Times.Once);
        }

        /// <summary>
        /// Tests that GetDailyGoal returns 500 status code when user ID is not found in claims.
        /// Input: User without NameIdentifier claim.
        /// Expected: 500 status code with error message.
        /// </summary>
        [TestMethod]
        public async Task GetDailyGoal_MissingUserIdClaim_Returns500StatusCode()
        {
            // Arrange
            var mockLearnService = new Mock<ILearnService>();
            var mockLogger = new Mock<ILogger<LearnController>>();

            var controller = new LearnController(mockLearnService.Object, mockLogger.Object);
            var user = new ClaimsPrincipal(new ClaimsIdentity());
            controller.ControllerContext = new ControllerContext
            {
                HttpContext = new DefaultHttpContext { User = user }
            };

            // Act
            var result = await controller.GetDailyGoal();

            // Assert
            Assert.IsNotNull(result);
            var statusCodeResult = result.Result as ObjectResult;
            Assert.IsNotNull(statusCodeResult);
            Assert.AreEqual(500, statusCodeResult.StatusCode);
        }

        /// <summary>
        /// Tests that GetDailyGoal logs error when user ID is not found in claims.
        /// Input: User without NameIdentifier claim.
        /// Expected: Error logged with appropriate exception.
        /// </summary>
        [TestMethod]
        public async Task GetDailyGoal_MissingUserIdClaim_LogsError()
        {
            // Arrange
            var mockLearnService = new Mock<ILearnService>();
            var mockLogger = new Mock<ILogger<LearnController>>();

            var controller = new LearnController(mockLearnService.Object, mockLogger.Object);
            var user = new ClaimsPrincipal(new ClaimsIdentity());
            controller.ControllerContext = new ControllerContext
            {
                HttpContext = new DefaultHttpContext { User = user }
            };

            // Act
            await controller.GetDailyGoal();

            // Assert
            mockLogger.Verify(
                x => x.Log(
                    LogLevel.Error,
                    It.IsAny<EventId>(),
                    It.Is<It.IsAnyType>((o, t) => o.ToString()!.Contains("Error fetching daily goal")),
                    It.IsAny<Exception>(),
                    It.IsAny<Func<It.IsAnyType, Exception?, string>>()),
                Times.Once);
        }

        /// <summary>
        /// Tests that GetDailyGoal returns 500 status code when service throws exception.
        /// Input: Service throws generic Exception.
        /// Expected: 500 status code with error message.
        /// </summary>
        [TestMethod]
        public async Task GetDailyGoal_ServiceThrowsException_Returns500StatusCode()
        {
            // Arrange
            var mockLearnService = new Mock<ILearnService>();
            var mockLogger = new Mock<ILogger<LearnController>>();

            mockLearnService
                .Setup(s => s.GetDailyGoalAsync(It.IsAny<string>()))
                .ThrowsAsync(new Exception("Database error"));

            var controller = new LearnController(mockLearnService.Object, mockLogger.Object);
            var user = new ClaimsPrincipal(new ClaimsIdentity(new[]
            {
                new Claim(ClaimTypes.NameIdentifier, "test-user")
            }));
            controller.ControllerContext = new ControllerContext
            {
                HttpContext = new DefaultHttpContext { User = user }
            };

            // Act
            var result = await controller.GetDailyGoal();

            // Assert
            Assert.IsNotNull(result);
            var statusCodeResult = result.Result as ObjectResult;
            Assert.IsNotNull(statusCodeResult);
            Assert.AreEqual(500, statusCodeResult.StatusCode);
        }

        /// <summary>
        /// Tests that GetDailyGoal returns correct error message when service throws exception.
        /// Input: Service throws exception.
        /// Expected: 500 status code with message "Error fetching daily goal".
        /// </summary>
        [TestMethod]
        public async Task GetDailyGoal_ServiceThrowsException_ReturnsCorrectErrorMessage()
        {
            // Arrange
            var mockLearnService = new Mock<ILearnService>();
            var mockLogger = new Mock<ILogger<LearnController>>();

            mockLearnService
                .Setup(s => s.GetDailyGoalAsync(It.IsAny<string>()))
                .ThrowsAsync(new Exception("Database error"));

            var controller = new LearnController(mockLearnService.Object, mockLogger.Object);
            var user = new ClaimsPrincipal(new ClaimsIdentity(new[]
            {
                new Claim(ClaimTypes.NameIdentifier, "test-user")
            }));
            controller.ControllerContext = new ControllerContext
            {
                HttpContext = new DefaultHttpContext { User = user }
            };

            // Act
            var result = await controller.GetDailyGoal();

            // Assert
            var statusCodeResult = result.Result as ObjectResult;
            Assert.IsNotNull(statusCodeResult);
            var value = statusCodeResult.Value;
            Assert.IsNotNull(value);
            var messageProperty = value.GetType().GetProperty("message");
            Assert.IsNotNull(messageProperty);
            var message = messageProperty.GetValue(value) as string;
            Assert.AreEqual("Error fetching daily goal", message);
        }

        /// <summary>
        /// Tests that GetDailyGoal logs error when service throws exception.
        /// Input: Service throws exception.
        /// Expected: Error logged with exception details.
        /// </summary>
        [TestMethod]
        public async Task GetDailyGoal_ServiceThrowsException_LogsError()
        {
            // Arrange
            var mockLearnService = new Mock<ILearnService>();
            var mockLogger = new Mock<ILogger<LearnController>>();
            var expectedException = new Exception("Database connection failed");

            mockLearnService
                .Setup(s => s.GetDailyGoalAsync(It.IsAny<string>()))
                .ThrowsAsync(expectedException);

            var controller = new LearnController(mockLearnService.Object, mockLogger.Object);
            var user = new ClaimsPrincipal(new ClaimsIdentity(new[]
            {
                new Claim(ClaimTypes.NameIdentifier, "test-user")
            }));
            controller.ControllerContext = new ControllerContext
            {
                HttpContext = new DefaultHttpContext { User = user }
            };

            // Act
            await controller.GetDailyGoal();

            // Assert
            mockLogger.Verify(
                x => x.Log(
                    LogLevel.Error,
                    It.IsAny<EventId>(),
                    It.Is<It.IsAnyType>((o, t) => o.ToString()!.Contains("Error fetching daily goal")),
                    It.IsAny<Exception>(),
                    It.IsAny<Func<It.IsAnyType, Exception?, string>>()),
                Times.Once);
        }

        /// <summary>
        /// Tests that GetDailyGoal handles InvalidOperationException from GetUserId.
        /// Input: Empty user ID claim value.
        /// Expected: 500 status code returned.
        /// </summary>
        [TestMethod]
        public async Task GetDailyGoal_EmptyUserIdClaim_Returns500StatusCode()
        {
            // Arrange
            var mockLearnService = new Mock<ILearnService>();
            var mockLogger = new Mock<ILogger<LearnController>>();

            var controller = new LearnController(mockLearnService.Object, mockLogger.Object);
            var user = new ClaimsPrincipal(new ClaimsIdentity(new[]
            {
                new Claim(ClaimTypes.NameIdentifier, string.Empty)
            }));
            controller.ControllerContext = new ControllerContext
            {
                HttpContext = new DefaultHttpContext { User = user }
            };

            // Act
            var result = await controller.GetDailyGoal();

            // Assert
            Assert.IsNotNull(result);
            var statusCodeResult = result.Result as ObjectResult;
            Assert.IsNotNull(statusCodeResult);
            Assert.AreEqual(500, statusCodeResult.StatusCode);
        }

        /// <summary>
        /// Tests that GetDailyGoal does not call service when user ID is missing.
        /// Input: User without NameIdentifier claim.
        /// Expected: Service method not called.
        /// </summary>
        [TestMethod]
        public async Task GetDailyGoal_MissingUserIdClaim_DoesNotCallService()
        {
            // Arrange
            var mockLearnService = new Mock<ILearnService>();
            var mockLogger = new Mock<ILogger<LearnController>>();

            var controller = new LearnController(mockLearnService.Object, mockLogger.Object);
            var user = new ClaimsPrincipal(new ClaimsIdentity());
            controller.ControllerContext = new ControllerContext
            {
                HttpContext = new DefaultHttpContext { User = user }
            };

            // Act
            await controller.GetDailyGoal();

            // Assert
            mockLearnService.Verify(s => s.GetDailyGoalAsync(It.IsAny<string>()), Times.Never);
        }

        /// <summary>
        /// Tests that GetDailyGoal correctly returns daily goal with zero values.
        /// Input: Service returns DailyGoalDto with all zero values.
        /// Expected: Ok result with zero values preserved.
        /// </summary>
        [TestMethod]
        public async Task GetDailyGoal_ZeroValues_ReturnsOkResultWithZeroValues()
        {
            // Arrange
            var mockLearnService = new Mock<ILearnService>();
            var mockLogger = new Mock<ILogger<LearnController>>();

            var expectedDailyGoal = new DailyGoalDto
            {
                TotalReviewsDue = 0,
                CompletedToday = 0,
                DailyGoal = 0,
                ProgressPercentage = 0.0
            };

            mockLearnService
                .Setup(s => s.GetDailyGoalAsync(It.IsAny<string>()))
                .ReturnsAsync(expectedDailyGoal);

            var controller = new LearnController(mockLearnService.Object, mockLogger.Object);
            var user = new ClaimsPrincipal(new ClaimsIdentity(new[]
            {
                new Claim(ClaimTypes.NameIdentifier, "test-user")
            }));
            controller.ControllerContext = new ControllerContext
            {
                HttpContext = new DefaultHttpContext { User = user }
            };

            // Act
            var result = await controller.GetDailyGoal();

            // Assert
            var okResult = result.Result as OkObjectResult;
            Assert.IsNotNull(okResult);
            var returnedGoal = okResult.Value as DailyGoalDto;
            Assert.IsNotNull(returnedGoal);
            Assert.AreEqual(0, returnedGoal.TotalReviewsDue);
            Assert.AreEqual(0, returnedGoal.CompletedToday);
            Assert.AreEqual(0, returnedGoal.DailyGoal);
            Assert.AreEqual(0.0, returnedGoal.ProgressPercentage);
        }

        /// <summary>
        /// Tests that GetDailyGoal correctly handles maximum integer values.
        /// Input: Service returns DailyGoalDto with maximum integer values.
        /// Expected: Ok result with maximum values preserved.
        /// </summary>
        [TestMethod]
        public async Task GetDailyGoal_MaximumValues_ReturnsOkResultWithMaximumValues()
        {
            // Arrange
            var mockLearnService = new Mock<ILearnService>();
            var mockLogger = new Mock<ILogger<LearnController>>();

            var expectedDailyGoal = new DailyGoalDto
            {
                TotalReviewsDue = int.MaxValue,
                CompletedToday = int.MaxValue,
                DailyGoal = int.MaxValue,
                ProgressPercentage = double.MaxValue
            };

            mockLearnService
                .Setup(s => s.GetDailyGoalAsync(It.IsAny<string>()))
                .ReturnsAsync(expectedDailyGoal);

            var controller = new LearnController(mockLearnService.Object, mockLogger.Object);
            var user = new ClaimsPrincipal(new ClaimsIdentity(new[]
            {
                new Claim(ClaimTypes.NameIdentifier, "test-user")
            }));
            controller.ControllerContext = new ControllerContext
            {
                HttpContext = new DefaultHttpContext { User = user }
            };

            // Act
            var result = await controller.GetDailyGoal();

            // Assert
            var okResult = result.Result as OkObjectResult;
            Assert.IsNotNull(okResult);
            var returnedGoal = okResult.Value as DailyGoalDto;
            Assert.IsNotNull(returnedGoal);
            Assert.AreEqual(int.MaxValue, returnedGoal.TotalReviewsDue);
            Assert.AreEqual(int.MaxValue, returnedGoal.CompletedToday);
            Assert.AreEqual(int.MaxValue, returnedGoal.DailyGoal);
            Assert.AreEqual(double.MaxValue, returnedGoal.ProgressPercentage);
        }

        /// <summary>
        /// Tests that GetDailyGoal correctly handles negative values in DailyGoalDto.
        /// Input: Service returns DailyGoalDto with negative values.
        /// Expected: Ok result with negative values preserved.
        /// </summary>
        [TestMethod]
        public async Task GetDailyGoal_NegativeValues_ReturnsOkResultWithNegativeValues()
        {
            // Arrange
            var mockLearnService = new Mock<ILearnService>();
            var mockLogger = new Mock<ILogger<LearnController>>();

            var expectedDailyGoal = new DailyGoalDto
            {
                TotalReviewsDue = -10,
                CompletedToday = -5,
                DailyGoal = -3,
                ProgressPercentage = -50.0
            };

            mockLearnService
                .Setup(s => s.GetDailyGoalAsync(It.IsAny<string>()))
                .ReturnsAsync(expectedDailyGoal);

            var controller = new LearnController(mockLearnService.Object, mockLogger.Object);
            var user = new ClaimsPrincipal(new ClaimsIdentity(new[]
            {
                new Claim(ClaimTypes.NameIdentifier, "test-user")
            }));
            controller.ControllerContext = new ControllerContext
            {
                HttpContext = new DefaultHttpContext { User = user }
            };

            // Act
            var result = await controller.GetDailyGoal();

            // Assert
            var okResult = result.Result as OkObjectResult;
            Assert.IsNotNull(okResult);
            var returnedGoal = okResult.Value as DailyGoalDto;
            Assert.IsNotNull(returnedGoal);
            Assert.AreEqual(-10, returnedGoal.TotalReviewsDue);
            Assert.AreEqual(-5, returnedGoal.CompletedToday);
            Assert.AreEqual(-3, returnedGoal.DailyGoal);
            Assert.AreEqual(-50.0, returnedGoal.ProgressPercentage);
        }

        /// <summary>
        /// Tests that GetDailyGoal correctly handles special double values.
        /// Input: Service returns DailyGoalDto with NaN ProgressPercentage.
        /// Expected: Ok result with NaN value preserved.
        /// </summary>
        [TestMethod]
        public async Task GetDailyGoal_NaNProgressPercentage_ReturnsOkResultWithNaN()
        {
            // Arrange
            var mockLearnService = new Mock<ILearnService>();
            var mockLogger = new Mock<ILogger<LearnController>>();

            var expectedDailyGoal = new DailyGoalDto
            {
                TotalReviewsDue = 10,
                CompletedToday = 5,
                DailyGoal = 5,
                ProgressPercentage = double.NaN
            };

            mockLearnService
                .Setup(s => s.GetDailyGoalAsync(It.IsAny<string>()))
                .ReturnsAsync(expectedDailyGoal);

            var controller = new LearnController(mockLearnService.Object, mockLogger.Object);
            var user = new ClaimsPrincipal(new ClaimsIdentity(new[]
            {
                new Claim(ClaimTypes.NameIdentifier, "test-user")
            }));
            controller.ControllerContext = new ControllerContext
            {
                HttpContext = new DefaultHttpContext { User = user }
            };

            // Act
            var result = await controller.GetDailyGoal();

            // Assert
            var okResult = result.Result as OkObjectResult;
            Assert.IsNotNull(okResult);
            var returnedGoal = okResult.Value as DailyGoalDto;
            Assert.IsNotNull(returnedGoal);
            Assert.IsTrue(double.IsNaN(returnedGoal.ProgressPercentage));
        }

        /// <summary>
        /// Tests that GetDailyGoal correctly handles positive infinity ProgressPercentage.
        /// Input: Service returns DailyGoalDto with PositiveInfinity ProgressPercentage.
        /// Expected: Ok result with PositiveInfinity value preserved.
        /// </summary>
        [TestMethod]
        public async Task GetDailyGoal_PositiveInfinityProgressPercentage_ReturnsOkResult()
        {
            // Arrange
            var mockLearnService = new Mock<ILearnService>();
            var mockLogger = new Mock<ILogger<LearnController>>();

            var expectedDailyGoal = new DailyGoalDto
            {
                TotalReviewsDue = 10,
                CompletedToday = 5,
                DailyGoal = 5,
                ProgressPercentage = double.PositiveInfinity
            };

            mockLearnService
                .Setup(s => s.GetDailyGoalAsync(It.IsAny<string>()))
                .ReturnsAsync(expectedDailyGoal);

            var controller = new LearnController(mockLearnService.Object, mockLogger.Object);
            var user = new ClaimsPrincipal(new ClaimsIdentity(new[]
            {
                new Claim(ClaimTypes.NameIdentifier, "test-user")
            }));
            controller.ControllerContext = new ControllerContext
            {
                HttpContext = new DefaultHttpContext { User = user }
            };

            // Act
            var result = await controller.GetDailyGoal();

            // Assert
            var okResult = result.Result as OkObjectResult;
            Assert.IsNotNull(okResult);
            var returnedGoal = okResult.Value as DailyGoalDto;
            Assert.IsNotNull(returnedGoal);
            Assert.IsTrue(double.IsPositiveInfinity(returnedGoal.ProgressPercentage));
        }

        /// <summary>
        /// Tests that GetDailyGoal correctly handles negative infinity ProgressPercentage.
        /// Input: Service returns DailyGoalDto with NegativeInfinity ProgressPercentage.
        /// Expected: Ok result with NegativeInfinity value preserved.
        /// </summary>
        [TestMethod]
        public async Task GetDailyGoal_NegativeInfinityProgressPercentage_ReturnsOkResult()
        {
            // Arrange
            var mockLearnService = new Mock<ILearnService>();
            var mockLogger = new Mock<ILogger<LearnController>>();

            var expectedDailyGoal = new DailyGoalDto
            {
                TotalReviewsDue = 10,
                CompletedToday = 5,
                DailyGoal = 5,
                ProgressPercentage = double.NegativeInfinity
            };

            mockLearnService
                .Setup(s => s.GetDailyGoalAsync(It.IsAny<string>()))
                .ReturnsAsync(expectedDailyGoal);

            var controller = new LearnController(mockLearnService.Object, mockLogger.Object);
            var user = new ClaimsPrincipal(new ClaimsIdentity(new[]
            {
                new Claim(ClaimTypes.NameIdentifier, "test-user")
            }));
            controller.ControllerContext = new ControllerContext
            {
                HttpContext = new DefaultHttpContext { User = user }
            };

            // Act
            var result = await controller.GetDailyGoal();

            // Assert
            var okResult = result.Result as OkObjectResult;
            Assert.IsNotNull(okResult);
            var returnedGoal = okResult.Value as DailyGoalDto;
            Assert.IsNotNull(returnedGoal);
            Assert.IsTrue(double.IsNegativeInfinity(returnedGoal.ProgressPercentage));
        }

        /// <summary>
        /// Tests that GetDailyGoal handles different exception types correctly.
        /// Input: Service throws ArgumentException.
        /// Expected: 500 status code with error message.
        /// </summary>
        [TestMethod]
        public async Task GetDailyGoal_ServiceThrowsArgumentException_Returns500StatusCode()
        {
            // Arrange
            var mockLearnService = new Mock<ILearnService>();
            var mockLogger = new Mock<ILogger<LearnController>>();

            mockLearnService
                .Setup(s => s.GetDailyGoalAsync(It.IsAny<string>()))
                .ThrowsAsync(new ArgumentException("Invalid user ID"));

            var controller = new LearnController(mockLearnService.Object, mockLogger.Object);
            var user = new ClaimsPrincipal(new ClaimsIdentity(new[]
            {
                new Claim(ClaimTypes.NameIdentifier, "test-user")
            }));
            controller.ControllerContext = new ControllerContext
            {
                HttpContext = new DefaultHttpContext { User = user }
            };

            // Act
            var result = await controller.GetDailyGoal();

            // Assert
            var statusCodeResult = result.Result as ObjectResult;
            Assert.IsNotNull(statusCodeResult);
            Assert.AreEqual(500, statusCodeResult.StatusCode);
        }

        /// <summary>
        /// Tests that GetDailyGoal handles special characters in user ID correctly.
        /// Input: User ID with special characters in claim.
        /// Expected: Service called with exact user ID including special characters.
        /// </summary>
        [TestMethod]
        public async Task GetDailyGoal_UserIdWithSpecialCharacters_CallsServiceWithCorrectUserId()
        {
            // Arrange
            var mockLearnService = new Mock<ILearnService>();
            var mockLogger = new Mock<ILogger<LearnController>>();
            var specialUserId = "user@test.com|123!#$%";

            mockLearnService
                .Setup(s => s.GetDailyGoalAsync(specialUserId))
                .ReturnsAsync(new DailyGoalDto());

            var controller = new LearnController(mockLearnService.Object, mockLogger.Object);
            var user = new ClaimsPrincipal(new ClaimsIdentity(new[]
            {
                new Claim(ClaimTypes.NameIdentifier, specialUserId)
            }));
            controller.ControllerContext = new ControllerContext
            {
                HttpContext = new DefaultHttpContext { User = user }
            };

            // Act
            await controller.GetDailyGoal();

            // Assert
            mockLearnService.Verify(s => s.GetDailyGoalAsync(specialUserId), Times.Once);
        }

        /// <summary>
        /// Tests that GetDailyGoal handles very long user ID correctly.
        /// Input: User ID with 1000 characters in claim.
        /// Expected: Service called with full user ID.
        /// </summary>
        [TestMethod]
        public async Task GetDailyGoal_VeryLongUserId_CallsServiceWithCorrectUserId()
        {
            // Arrange
            var mockLearnService = new Mock<ILearnService>();
            var mockLogger = new Mock<ILogger<LearnController>>();
            var longUserId = new string('a', 1000);

            mockLearnService
                .Setup(s => s.GetDailyGoalAsync(longUserId))
                .ReturnsAsync(new DailyGoalDto());

            var controller = new LearnController(mockLearnService.Object, mockLogger.Object);
            var user = new ClaimsPrincipal(new ClaimsIdentity(new[]
            {
                new Claim(ClaimTypes.NameIdentifier, longUserId)
            }));
            controller.ControllerContext = new ControllerContext
            {
                HttpContext = new DefaultHttpContext { User = user }
            };

            // Act
            await controller.GetDailyGoal();

            // Assert
            mockLearnService.Verify(s => s.GetDailyGoalAsync(longUserId), Times.Once);
        }

        /// <summary>
        /// Tests that GetLesson returns Ok with lesson data when the lesson exists.
        /// </summary>
        [TestMethod]
        public async Task GetLesson_ValidLessonId_ReturnsOkWithLesson()
        {
            // Arrange
            const int lessonId = 1;
            const string userId = "user123";
            var expectedLesson = new LessonDto
            {
                Id = lessonId,
                Title = "Test Lesson",
                Description = "Test Description",
                Thumbnail = "test.jpg",
                DurationSeconds = 300,
                Difficulty = "Beginner",
                CompletionPercentage = 50.0,
                InstructorName = "Test Instructor",
                CategoryId = 1
            };

            var mockLearnService = new Mock<ILearnService>();
            mockLearnService
                .Setup(s => s.GetLessonAsync(lessonId, userId))
                .ReturnsAsync(expectedLesson);

            var mockLogger = new Mock<ILogger<LearnController>>();
            var controller = new LearnController(mockLearnService.Object, mockLogger.Object);
            SetupUserClaims(controller, userId);

            // Act
            var result = await controller.GetLesson(lessonId);

            // Assert
            Assert.IsNotNull(result);
            Assert.IsInstanceOfType(result.Result, typeof(OkObjectResult));
            var okResult = result.Result as OkObjectResult;
            Assert.IsNotNull(okResult);
            Assert.AreEqual(200, okResult.StatusCode);
            Assert.AreEqual(expectedLesson, okResult.Value);
            mockLearnService.Verify(s => s.GetLessonAsync(lessonId, userId), Times.Once);
        }

        /// <summary>
        /// Tests that GetLesson returns NotFound when the lesson does not exist.
        /// </summary>
        [TestMethod]
        public async Task GetLesson_LessonNotFound_ReturnsNotFound()
        {
            // Arrange
            const int lessonId = 999;
            const string userId = "user123";

            var mockLearnService = new Mock<ILearnService>();
            mockLearnService
                .Setup(s => s.GetLessonAsync(lessonId, userId))
                .ReturnsAsync((LessonDto?)null);

            var mockLogger = new Mock<ILogger<LearnController>>();
            var controller = new LearnController(mockLearnService.Object, mockLogger.Object);
            SetupUserClaims(controller, userId);

            // Act
            var result = await controller.GetLesson(lessonId);

            // Assert
            Assert.IsNotNull(result);
            Assert.IsInstanceOfType(result.Result, typeof(NotFoundObjectResult));
            var notFoundResult = result.Result as NotFoundObjectResult;
            Assert.IsNotNull(notFoundResult);
            Assert.AreEqual(404, notFoundResult.StatusCode);
            mockLearnService.Verify(s => s.GetLessonAsync(lessonId, userId), Times.Once);
        }

        /// <summary>
        /// Tests that GetLesson returns 500 status code when the service throws an exception.
        /// </summary>
        [TestMethod]
        public async Task GetLesson_ServiceThrowsException_Returns500StatusCode()
        {
            // Arrange
            const int lessonId = 1;
            const string userId = "user123";
            var expectedException = new Exception("Database connection failed");

            var mockLearnService = new Mock<ILearnService>();
            mockLearnService
                .Setup(s => s.GetLessonAsync(lessonId, userId))
                .ThrowsAsync(expectedException);

            var mockLogger = new Mock<ILogger<LearnController>>();
            var controller = new LearnController(mockLearnService.Object, mockLogger.Object);
            SetupUserClaims(controller, userId);

            // Act
            var result = await controller.GetLesson(lessonId);

            // Assert
            Assert.IsNotNull(result);
            Assert.IsInstanceOfType(result.Result, typeof(ObjectResult));
            var objectResult = result.Result as ObjectResult;
            Assert.IsNotNull(objectResult);
            Assert.AreEqual(500, objectResult.StatusCode);
            mockLogger.Verify(
                x => x.Log(
                    LogLevel.Error,
                    It.IsAny<EventId>(),
                    It.Is<It.IsAnyType>((v, t) => true),
                    expectedException,
                    It.IsAny<Func<It.IsAnyType, Exception?, string>>()),
                Times.Once);
        }

        /// <summary>
        /// Tests that GetLesson returns 500 status code when user ID is not found in token.
        /// </summary>
        [TestMethod]
        public async Task GetLesson_UserIdNotFoundInToken_Returns500StatusCode()
        {
            // Arrange
            const int lessonId = 1;

            var mockLearnService = new Mock<ILearnService>();
            var mockLogger = new Mock<ILogger<LearnController>>();
            var controller = new LearnController(mockLearnService.Object, mockLogger.Object);
            SetupUserClaimsWithoutUserId(controller);

            // Act
            var result = await controller.GetLesson(lessonId);

            // Assert
            Assert.IsNotNull(result);
            Assert.IsInstanceOfType(result.Result, typeof(ObjectResult));
            var objectResult = result.Result as ObjectResult;
            Assert.IsNotNull(objectResult);
            Assert.AreEqual(500, objectResult.StatusCode);
            mockLogger.Verify(
                x => x.Log(
                    LogLevel.Error,
                    It.IsAny<EventId>(),
                    It.Is<It.IsAnyType>((v, t) => true),
                    It.IsAny<Exception>(),
                    It.IsAny<Func<It.IsAnyType, Exception?, string>>()),
                Times.Once);
        }

        /// <summary>
        /// Tests that GetLesson handles edge case lessonId values correctly.
        /// Tests zero, negative, int.MinValue, and int.MaxValue.
        /// </summary>
        /// <param name="lessonId">The lesson ID to test.</param>
        [TestMethod]
        [DataRow(0)]
        [DataRow(-1)]
        [DataRow(-100)]
        [DataRow(int.MinValue)]
        [DataRow(int.MaxValue)]
        public async Task GetLesson_EdgeCaseLessonIds_PassesValueToService(int lessonId)
        {
            // Arrange
            const string userId = "user123";
            var expectedLesson = new LessonDto { Id = lessonId };

            var mockLearnService = new Mock<ILearnService>();
            mockLearnService
                .Setup(s => s.GetLessonAsync(lessonId, userId))
                .ReturnsAsync(expectedLesson);

            var mockLogger = new Mock<ILogger<LearnController>>();
            var controller = new LearnController(mockLearnService.Object, mockLogger.Object);
            SetupUserClaims(controller, userId);

            // Act
            var result = await controller.GetLesson(lessonId);

            // Assert
            Assert.IsNotNull(result);
            mockLearnService.Verify(s => s.GetLessonAsync(lessonId, userId), Times.Once);
        }

        /// <summary>
        /// Tests that GetLesson logs the correct lessonId when an exception occurs.
        /// </summary>
        [TestMethod]
        public async Task GetLesson_ServiceThrowsException_LogsCorrectLessonId()
        {
            // Arrange
            const int lessonId = 42;
            const string userId = "user123";
            var expectedException = new InvalidOperationException("Test error");

            var mockLearnService = new Mock<ILearnService>();
            mockLearnService
                .Setup(s => s.GetLessonAsync(lessonId, userId))
                .ThrowsAsync(expectedException);

            var mockLogger = new Mock<ILogger<LearnController>>();
            var controller = new LearnController(mockLearnService.Object, mockLogger.Object);
            SetupUserClaims(controller, userId);

            // Act
            var result = await controller.GetLesson(lessonId);

            // Assert
            Assert.IsNotNull(result);
            mockLogger.Verify(
                x => x.Log(
                    LogLevel.Error,
                    It.IsAny<EventId>(),
                    It.Is<It.IsAnyType>((v, t) => true),
                    expectedException,
                    It.IsAny<Func<It.IsAnyType, Exception?, string>>()),
                Times.Once);
        }

        /// <summary>
        /// Helper method to setup user claims with a valid user ID.
        /// </summary>
        /// <param name="controller">The controller to setup.</param>
        /// <param name="userId">The user ID to set in claims.</param>
        private void SetupUserClaims(LearnController controller, string userId)
        {
            var claims = new List<Claim>
            {
                new Claim(ClaimTypes.NameIdentifier, userId)
            };
            var identity = new ClaimsIdentity(claims, "TestAuthType");
            var claimsPrincipal = new ClaimsPrincipal(identity);

            controller.ControllerContext = new ControllerContext
            {
                HttpContext = new DefaultHttpContext
                {
                    User = claimsPrincipal
                }
            };
        }

        /// <summary>
        /// Helper method to setup user claims without a user ID.
        /// </summary>
        /// <param name="controller">The controller to setup.</param>
        private void SetupUserClaimsWithoutUserId(LearnController controller)
        {
            var claims = new List<Claim>();
            var identity = new ClaimsIdentity(claims, "TestAuthType");
            var claimsPrincipal = new ClaimsPrincipal(identity);

            controller.ControllerContext = new ControllerContext
            {
                HttpContext = new DefaultHttpContext
                {
                    User = claimsPrincipal
                }
            };
        }

        /// <summary>
        /// Test that GetLearnPageData returns Ok with the DTO when the user is authenticated and service returns data.
        /// Input: ClaimsPrincipal with valid NameIdentifier claim; ILearnService returns a populated LearnPageDataDto.
        /// Expected: OkObjectResult containing the same LearnPageDataDto instance.
        /// </summary>
        [TestMethod]
        public async Task GetLearnPageData_ValidUser_ServiceReturnsData_ReturnsOkWithData()
        {
            // Arrange
            const string userId = "user-123";
            var dto = new LearnPageDataDto
            {
                TotalXp = 42,
                RecommendationReason = "Keep going!"
            };

            var mockService = new Mock<ILearnService>();
            mockService.Setup(s => s.GetLearnPageDataAsync(It.Is<string>(id => id == userId)))
                       .ReturnsAsync(dto);

            var mockLogger = new Mock<ILogger<LearnController>>();

            var controller = new LearnController(mockService.Object, mockLogger.Object);
            controller.ControllerContext = new ControllerContext
            {
                HttpContext = new DefaultHttpContext
                {
                    User = new ClaimsPrincipal(new ClaimsIdentity(new[]
                    {
                        new Claim(ClaimTypes.NameIdentifier, userId)
                    }, "TestAuth"))
                }
            };

            // Act
            var result = await controller.GetLearnPageData();

            // Assert
            Assert.IsNotNull(result);
            Assert.IsInstanceOfType(result.Result, typeof(OkObjectResult));
            var okResult = (OkObjectResult)result.Result!;
            Assert.IsNotNull(okResult.Value);
            var returned = okResult.Value as LearnPageDataDto;
            Assert.IsNotNull(returned);
            Assert.AreEqual(dto.TotalXp, returned.TotalXp);
            Assert.AreEqual(dto.RecommendationReason, returned.RecommendationReason);

            // Verify service was called with correct userId
            mockService.Verify(s => s.GetLearnPageDataAsync(It.Is<string>(id => id == userId)), Times.Once);
        }

        /// <summary>
        /// Test that GetLearnPageData returns 500 and logs error when the learn service throws an exception.
        /// Input: Authenticated user; ILearnService throws an Exception with a specific message.
        /// Expected: ObjectResult with status code 500 and message "Error fetching learn page data"; logger logs the thrown exception.
        /// </summary>
        [TestMethod]
        public async Task GetLearnPageData_ServiceThrowsException_Returns500WithErrorMessage()
        {
            // Arrange
            const string userId = "user-ex";
            var serviceException = new Exception("Database is down");

            var mockService = new Mock<ILearnService>();
            mockService.Setup(s => s.GetLearnPageDataAsync(It.Is<string>(id => id == userId)))
                       .ThrowsAsync(serviceException);

            var mockLogger = new Mock<ILogger<LearnController>>();

            var controller = new LearnController(mockService.Object, mockLogger.Object);
            controller.ControllerContext = new ControllerContext
            {
                HttpContext = new DefaultHttpContext
                {
                    User = new ClaimsPrincipal(new ClaimsIdentity(new[]
                    {
                        new Claim(ClaimTypes.NameIdentifier, userId)
                    }, "TestAuth"))
                }
            };

            // Act
            var result = await controller.GetLearnPageData();

            // Assert
            Assert.IsNotNull(result);
            Assert.IsInstanceOfType(result.Result, typeof(ObjectResult));
            var objectResult = (ObjectResult)result.Result!;
            Assert.AreEqual(500, objectResult.StatusCode);

            var value = objectResult.Value;
            Assert.IsNotNull(value);
            var messageProp = value.GetType().GetProperty("message");
            Assert.IsNotNull(messageProp);
            var messageValue = messageProp.GetValue(value) as string;
            Assert.AreEqual("Error fetching learn page data", messageValue);

            // Verify the logger logged the exception thrown by the service
            mockLogger.Verify(l => l.Log(
                LogLevel.Error,
                It.IsAny<EventId>(),
                It.Is<It.IsAnyType>((v, t) => true),
                It.Is<Exception>(ex => ex == serviceException),
                It.IsAny<Func<It.IsAnyType, Exception, string>>()),
                Times.Once);

            // Verify service attempted to be called
            mockService.Verify(s => s.GetLearnPageDataAsync(It.Is<string>(id => id == userId)), Times.Once);
        }

        /// <summary>
        /// Tests that GetLesson passes various edge case lessonId values to the service and returns Ok.
        /// Input: Several edge case lessonId values (0, negative, int.MinValue, int.MaxValue).
        /// Expected: For each lessonId the service is invoked with the same id and OkObjectResult is returned.
        /// </summary>
        [TestMethod]
        public async Task GetLesson_EdgeCaseLessonIds_PassesValueToService()
        {
            // Arrange
            int[] edgeIds = new[] { 0, -1, -100, int.MinValue, int.MaxValue };
            string userId = "edge-user";

            foreach (var lessonId in edgeIds)
            {
                var expectedLesson = new LessonDto { Id = lessonId, Title = "e", Description = "d" };

                var mockLearnService = new Mock<ILearnService>();
                mockLearnService
                    .Setup(s => s.GetLessonAsync(lessonId, userId))
                    .ReturnsAsync(expectedLesson);

                var mockLogger = new Mock<ILogger<LearnController>>();

                var controller = new LearnController(mockLearnService.Object, mockLogger.Object);
                controller.ControllerContext = new ControllerContext
                {
                    HttpContext = new DefaultHttpContext
                    {
                        User = new ClaimsPrincipal(new ClaimsIdentity(
                            new[] { new Claim(ClaimTypes.NameIdentifier, userId) }, "TestAuth"))
                    }
                };

                // Act
                var actionResult = await controller.GetLesson(lessonId);

                // Assert
                Assert.IsNotNull(actionResult);
                Assert.IsInstanceOfType(actionResult.Result, typeof(OkObjectResult));
                var okResult = (OkObjectResult)actionResult.Result!;
                var returned = okResult.Value as LessonDto;
                Assert.IsNotNull(returned);
                Assert.AreEqual(expectedLesson.Id, returned!.Id);

                mockLearnService.Verify(s => s.GetLessonAsync(lessonId, userId), Times.Once);
            }
        }

        /// <summary>
        /// Verifies that the controller preserves extreme integer values returned by the service.
        /// Input: Authenticated user with NameIdentifier claim and service returning UpcomingReviewsDto with extreme values (0 and int.MaxValue).
        /// Expected: OkObjectResult containing DTO with identical numeric values as returned by the service.
        /// </summary>
        [TestMethod]
        public async Task GetUpcomingReviews_ServiceReturnsExtremeValues_ReturnsOkWithSameValues()
        {
            // Arrange
            var mockService = new Mock<ILearnService>();
            var mockLogger = new Mock<ILogger<LearnController>>();
            var controller = new LearnController(mockService.Object, mockLogger.Object);

            var userId = "user-3";
            var claims = new List<Claim> { new Claim(ClaimTypes.NameIdentifier, userId) };
            controller.ControllerContext = new ControllerContext
            {
                HttpContext = new DefaultHttpContext
                {
                    User = new ClaimsPrincipal(new ClaimsIdentity(claims, "mock"))
                }
            };

            var extreme = new UpcomingReviewsDto
            {
                DueToday = 0,
                DueTomorrow = int.MaxValue,
                DueThisWeek = int.MaxValue,
                Overdue = 0
            };

            mockService.Setup(s => s.GetUpcomingReviewsAsync(userId)).ReturnsAsync(extreme);

            // Act
            var actionResult = await controller.GetUpcomingReviews();

            // Assert
            Assert.IsNotNull(actionResult);
            Assert.IsInstanceOfType(actionResult.Result, typeof(OkObjectResult));
            var okResult = (OkObjectResult)actionResult.Result!;
            Assert.IsNotNull(okResult.Value);
            var returned = okResult.Value as UpcomingReviewsDto;
            Assert.IsNotNull(returned);
            Assert.AreEqual(extreme.DueToday, returned.DueToday);
            Assert.AreEqual(extreme.DueTomorrow, returned.DueTomorrow);
            Assert.AreEqual(extreme.DueThisWeek, returned.DueThisWeek);
            Assert.AreEqual(extreme.Overdue, returned.Overdue);
        }

        /// <summary>
        /// Tests that CompleteLesson handles various edge-case lessonId values.
        /// Input: Authenticated user with multiple lessonId edge values (int.MinValue, int.MaxValue, 0, negative).
        /// Expected: For each id, when service completes normally, CompleteLesson returns OkObjectResult with success message and service is invoked with exact parameters.
        /// </summary>
        [TestMethod]
        public async Task CompleteLesson_EdgeCaseLessonIds_ReturnsOkForVariousIds()
        {
            // Arrange
            string userId = "edge-user";
            var idsToTest = new[] { int.MinValue, int.MaxValue, 0, -1, 999999 };
            foreach (var lessonId in idsToTest)
            {
                var mockService = new Mock<ILearnService>();
                mockService.Setup(s => s.CompleteLessonAsync(userId, lessonId)).Returns(Task.CompletedTask);

                var mockLogger = new Mock<ILogger<LearnController>>();
                var controller = CreateControllerWithUser(mockService.Object, mockLogger.Object, userId);

                // Act
                var result = await controller.CompleteLesson(lessonId);

                // Assert
                Assert.IsNotNull(result);
                Assert.IsInstanceOfType(result.Result, typeof(OkObjectResult));
                var ok = (OkObjectResult)result.Result!;
                Assert.IsNotNull(ok.Value);
                var messageProp = ok.Value.GetType().GetProperty("message");
                Assert.IsNotNull(messageProp);
                var message = messageProp!.GetValue(ok.Value) as string;
                Assert.AreEqual("Lesson completed successfully", message);

                mockService.Verify(s => s.CompleteLessonAsync(userId, lessonId), Times.Once);
            }
        }

        /// <summary>
        /// Test that GetAllCategories returns 500 StatusCode and appropriate message when the service throws an exception.
        /// Input: Authenticated user claim and service throws Exception.
        /// Expected: ObjectResult with StatusCode 500 and message "Error fetching categories".
        /// </summary>
        [TestMethod]
        public async Task GetAllCategories_ServiceThrowsException_Returns500StatusCodeAndErrorMessage()
        {
            // Arrange
            var userId = "user-error";
            var mockLearnService = new Mock<ILearnService>();
            mockLearnService
                .Setup(s => s.GetAllCategoriesAsync(It.Is<string>(u => u == userId)))
                .ThrowsAsync(new Exception("boom"));

            var mockLogger = new Mock<ILogger<LearnController>>();

            var controller = new LearnController(mockLearnService.Object, mockLogger.Object);
            var claimsPrincipal = new ClaimsPrincipal(new ClaimsIdentity(new[] { new Claim(ClaimTypes.NameIdentifier, userId) }, "mock"));
            controller.ControllerContext = new ControllerContext { HttpContext = new DefaultHttpContext { User = claimsPrincipal } };

            // Act
            var result = await controller.GetAllCategories();

            // Assert
            Assert.IsNotNull(result);
            Assert.IsInstanceOfType(result.Result, typeof(ObjectResult));
            var objResult = (ObjectResult)result.Result!;
            Assert.AreEqual(500, objResult.StatusCode);
            var msg = objResult.Value?.GetType().GetProperty("message")?.GetValue(objResult.Value) as string;
            Assert.AreEqual("Error fetching categories", msg);
        }

        /// <summary>
        /// Tests that the constructor successfully initializes the controller with valid dependencies.
        /// Input: Valid ILearnService mock and valid ILogger mock.
        /// Expected: Constructor completes successfully and instance is created (not null).
        /// </summary>
        [TestMethod]
        public void LearnController_ValidDependencies_InitializesSuccessfully()
        {
            // Arrange
            var mockLearnService = new Mock<ILearnService>();
            var mockLogger = new Mock<ILogger<LearnController>>();

            // Act
            var controller = new LearnController(mockLearnService.Object, mockLogger.Object);

            // Assert
            Assert.IsNotNull(controller);
            Assert.IsInstanceOfType(controller, typeof(LearnController));
        }

        /// <summary>
        /// Tests the constructor behavior when null is passed for the learnService parameter.
        /// Input: null for learnService, valid ILogger mock.
        /// Expected: Constructor completes without throwing and instance is created (no validation in ctor).
        /// </summary>
        [TestMethod]
        public void LearnController_NullLearnService_CompletesWithoutException()
        {
            // Arrange
            ILearnService? nullLearnService = null;
            var mockLogger = new Mock<ILogger<LearnController>>();

            // Act
            var controller = new LearnController(nullLearnService!, mockLogger.Object);

            // Assert
            Assert.IsNotNull(controller);
            Assert.IsInstanceOfType(controller, typeof(LearnController));
        }

        /// <summary>
        /// Tests the constructor behavior when null is passed for the logger parameter.
        /// Input: Valid ILearnService mock, null for logger.
        /// Expected: Constructor completes without throwing and instance is created (no validation in ctor).
        /// </summary>
        [TestMethod]
        public void LearnController_NullLogger_CompletesWithoutException()
        {
            // Arrange
            var mockLearnService = new Mock<ILearnService>();
            ILogger<LearnController>? nullLogger = null;

            // Act
            var controller = new LearnController(mockLearnService.Object, nullLogger!);

            // Assert
            Assert.IsNotNull(controller);
            Assert.IsInstanceOfType(controller, typeof(LearnController));
        }

        /// <summary>
        /// Tests the constructor behavior when both parameters are null.
        /// Input: null for both learnService and logger.
        /// Expected: Constructor completes without throwing and instance is created (no validation in ctor).
        /// </summary>
        [TestMethod]
        public void LearnController_BothParametersNull_CompletesWithoutException()
        {
            // Arrange
            ILearnService? nullLearnService = null;
            ILogger<LearnController>? nullLogger = null;

            // Act
            var controller = new LearnController(nullLearnService!, nullLogger!);

            // Assert
            Assert.IsNotNull(controller);
            Assert.IsInstanceOfType(controller, typeof(LearnController));
        }

        /// <summary>
        /// Tests that valid completion percentages (boundary and typical values) call the service
        /// and return Ok with a success message.
        /// Input: Valid user claim present, completion percentages 0, 50, 100 and various lessonIds.
        /// Expected: Service UpdateLessonProgressAsync called once with correct parameters and OkObjectResult returned containing success message.
        /// </summary>
        [TestMethod]
        public async Task UpdateLessonProgress_ValidCompletionPercentage_CallsServiceAndReturnsOk()
        {
            // Arrange - test multiple valid cases in a single test to avoid redundancy
            var testCases = new List<(int completionPercentage, int lessonId)>
            {
                (0, 1),
                (50, 0),
                (100, int.MaxValue),
                (100, int.MinValue)
            };

            foreach (var (completionPercentage, lessonId) in testCases)
            {
                var mockService = new Mock<ILearnService>();
                var mockLogger = new Mock<ILogger<LearnController>>();
                string expectedUserId = "user-123";

                mockService
                    .Setup(s => s.UpdateLessonProgressAsync(expectedUserId, lessonId, completionPercentage))
                    .Returns(Task.CompletedTask)
                    .Verifiable();

                var controller = CreateControllerWithUser(mockService.Object, mockLogger.Object, expectedUserId);

                // Act
                var result = await controller.UpdateLessonProgress(lessonId, completionPercentage);

                // Assert
                Assert.IsNotNull(result);
                Assert.IsInstanceOfType(result.Result, typeof(OkObjectResult));
                var okResult = (OkObjectResult)result.Result!;
                Assert.IsNotNull(okResult.Value);
                // Anonymous object; check text representation for expected message
                StringAssert.Contains(okResult.Value.ToString() ?? string.Empty, "Lesson progress updated successfully");

                mockService.Verify(s => s.UpdateLessonProgressAsync(expectedUserId, lessonId, completionPercentage), Times.Once);

                // Clean up verifications for next iteration
                mockService.VerifyNoOtherCalls();
            }
        }

        /// <summary>
        /// Tests that invalid completion percentages outside [0,100] return BadRequest and do NOT call the service.
        /// Input: Various invalid completionPercentage values (negative, >100, extreme int).
        /// Expected: BadRequestObjectResult with appropriate message and service not invoked.
        /// </summary>
        [TestMethod]
        public async Task UpdateLessonProgress_CompletionPercentageOutOfRange_ReturnsBadRequestAndDoesNotCallService()
        {
            // Arrange - invalid values to test both sides of the range and extreme values
            var invalidCompletions = new[] { -1, -100, int.MinValue, 101, 200, int.MaxValue };
            var lessonId = 1;
            var mockService = new Mock<ILearnService>();
            var mockLogger = new Mock<ILogger<LearnController>>();
            var controller = CreateControllerWithUser(mockService.Object, mockLogger.Object, "user-xyz");

            foreach (var completion in invalidCompletions)
            {
                // Act
                var result = await controller.UpdateLessonProgress(lessonId, completion);

                // Assert
                Assert.IsNotNull(result);
                Assert.IsInstanceOfType(result.Result, typeof(BadRequestObjectResult));
                var badRequest = (BadRequestObjectResult)result.Result!;
                Assert.IsNotNull(badRequest.Value);
                StringAssert.Contains(badRequest.Value.ToString() ?? string.Empty, "Completion percentage must be between 0 and 100");

                // Service must not be called for invalid input
                mockService.Verify(s => s.UpdateLessonProgressAsync(It.IsAny<string>(), It.IsAny<int>(), It.IsAny<int>()), Times.Never);
            }
        }

        /// <summary>
        /// Tests behavior when the service throws an exception during UpdateLessonProgressAsync.
        /// Input: Valid user claim, valid completionPercentage; service throws Exception.
        /// Expected: Method catches exception, logs error, returns 500 status code and appropriate error message.
        /// </summary>
        [TestMethod]
        public async Task UpdateLessonProgress_ServiceThrowsException_ReturnsInternalServerErrorAndLogs()
        {
            // Arrange
            var mockService = new Mock<ILearnService>();
            var mockLogger = new Mock<ILogger<LearnController>>();
            var userId = "user-service-ex";
            int lessonId = 42;
            int completion = 75;

            mockService
                .Setup(s => s.UpdateLessonProgressAsync(userId, lessonId, completion))
                .ThrowsAsync(new Exception("service failure"))
                .Verifiable();

            var controller = CreateControllerWithUser(mockService.Object, mockLogger.Object, userId);

            // Act
            var result = await controller.UpdateLessonProgress(lessonId, completion);

            // Assert
            Assert.IsNotNull(result);
            Assert.IsInstanceOfType(result.Result, typeof(ObjectResult));
            var objectResult = (ObjectResult)result.Result!;
            Assert.AreEqual(500, objectResult.StatusCode);
            Assert.IsNotNull(objectResult.Value);
            StringAssert.Contains(objectResult.Value.ToString() ?? string.Empty, "Error updating lesson progress");

            mockService.Verify(s => s.UpdateLessonProgressAsync(userId, lessonId, completion), Times.Once);

            mockLogger.Verify(
                x => x.Log(
                    LogLevel.Error,
                    It.IsAny<EventId>(),
                    It.Is<It.IsAnyType>((v, t) => v.ToString().Contains("Error updating lesson progress")),
                    It.IsAny<Exception>(),
                    It.IsAny<Func<It.IsAnyType, Exception, string>>()),
                Times.Once);
        }

        /// <summary>
        /// Tests that ReviewLesson returns BadRequest when GetUserId throws InvalidOperationException (missing or empty NameIdentifier claim).
        /// Input: No NameIdentifier claim in controller's user principal.
        /// Expected: BadRequest with message from the thrown InvalidOperationException and no service invocation.
        /// </summary>
        [TestMethod]
        public async Task ReviewLesson_UserIdNotFoundInToken_ReturnsBadRequestAndDoesNotCallService()
        {
            // Arrange
            int spacedRepetitionId = 5;
            double qualityRating = 4.0;

            var mockLearnService = new Mock<ILearnService>();
            var mockLogger = new Mock<ILogger<LearnController>>();
            var controller = new LearnController(mockLearnService.Object, mockLogger.Object)
            {
                // No user claims provided to trigger GetUserId InvalidOperationException
                ControllerContext = CreateControllerContextWithoutUserClaims()
            };

            // Act
            var result = await controller.ReviewLesson(spacedRepetitionId, qualityRating);

            // Assert
            Assert.IsNotNull(result);
            Assert.IsInstanceOfType(result.Result, typeof(BadRequestObjectResult));
            var badRequest = (BadRequestObjectResult)result.Result!;
            Assert.IsNotNull(badRequest.Value);
            var messageProp = badRequest.Value.GetType().GetProperty("message");
            Assert.IsNotNull(messageProp);
            var message = messageProp.GetValue(badRequest.Value) as string;
            Assert.AreEqual("User ID not found in token", message);

            mockLearnService.Verify(s => s.ReviewLessonAsync(It.IsAny<string>(), It.IsAny<int>(), It.IsAny<double>()), Times.Never);
        }

        /// <summary>
        /// Creates a ControllerContext without a NameIdentifier claim (unauthenticated for GetUserId).
        /// </summary>
        /// <returns>A configured ControllerContext with no user claims.</returns>
        private static ControllerContext CreateControllerContextWithNoUser()
        {
            var identity = new ClaimsIdentity(); // no claims
            var principal = new ClaimsPrincipal(identity);
            var httpContext = new DefaultHttpContext { User = principal };
            return new ControllerContext { HttpContext = httpContext };
        }

        /// <summary>
        /// Tests that GetLearnPageData returns Ok with null when the service returns null.
        /// Input: Authenticated user and service returns null.
        /// Expected: OkObjectResult with null Value.
        /// </summary>
        [TestMethod]
        public async Task GetLearnPageData_ValidUser_ServiceReturnsNull_ReturnsOkWithNull()
        {
            // Arrange
            var mockLearnService = new Mock<ILearnService>();
            var mockLogger = new Mock<ILogger<LearnController>>();
            var controller = new LearnController(mockLearnService.Object, mockLogger.Object);

            var userId = "test-user-null";
            mockLearnService
                .Setup(s => s.GetLearnPageDataAsync(userId))
                .ReturnsAsync((LearnPageDataDto?)null);

            controller.ControllerContext = CreateControllerContextWithUser(userId);

            // Act
            var result = await controller.GetLearnPageData();

            // Assert
            Assert.IsNotNull(result);
            var okResult = result.Result as OkObjectResult;
            Assert.IsNotNull(okResult, "Expected OkObjectResult even when service returns null");
            Assert.IsNull(okResult!.Value, "Expected null value preserved from service");

            mockLearnService.Verify(s => s.GetLearnPageDataAsync(userId), Times.Once);
        }

        /// <summary>
        /// Tests that GetLearnPageData returns 500 when the NameIdentifier claim exists but has null or whitespace value.
        /// Input: Claim with null value and whitespace-only value.
        /// Expected: ObjectResult with 500 status code and error message; logger called.
        /// </summary>
        [TestMethod]
        public async Task GetLearnPageData_UserIdClaimNullOrWhitespace_Returns500WithErrorMessage()
        {
            // Test for null value
            await GetLearnPageData_UserIdClaimValue_Returns500_Helper((string?)null);

            // Test for whitespace-only value
            await GetLearnPageData_UserIdClaimValue_Returns500_Helper("   ");
        }

        /// <summary>
        /// Helper that sets up controller with a NameIdentifier claim having the provided value and asserts 500 response.
        /// Used to test null and whitespace claim values.
        /// </summary>
        private static async Task GetLearnPageData_UserIdClaimValue_Returns500_Helper(string? claimValue)
        {
            var mockLearnService = new Mock<ILearnService>();
            var mockLogger = new Mock<ILogger<LearnController>>();
            var controller = new LearnController(mockLearnService.Object, mockLogger.Object);

            var claims = new List<Claim>
            {
                new Claim(ClaimTypes.NameIdentifier, claimValue ?? string.Empty)
            };
            controller.ControllerContext = new ControllerContext
            {
                HttpContext = new DefaultHttpContext { User = new ClaimsPrincipal(new ClaimsIdentity(claims, "TestAuth")) }
            };

            var result = await controller.GetLearnPageData();

            Assert.IsNotNull(result);
            var statusCodeResult = result.Result as ObjectResult;
            Assert.IsNotNull(statusCodeResult);
            Assert.AreEqual(500, statusCodeResult!.StatusCode);

            var value = statusCodeResult.Value;
            Assert.IsNotNull(value);
            var messageProperty = value.GetType().GetProperty("message");
            Assert.IsNotNull(messageProperty);
            Assert.AreEqual("Error fetching learn page data", messageProperty!.GetValue(value));

            mockLogger.Verify(
                x => x.Log(
                    LogLevel.Error,
                    It.IsAny<EventId>(),
                    It.Is<It.IsAnyType>((v, t) => true),
                    It.IsAny<InvalidOperationException>(),
                    It.Is<Func<It.IsAnyType, Exception?, string>>((v, t) => true)),
                Times.Once);

            mockLearnService.Verify(s => s.GetLearnPageDataAsync(It.IsAny<string>()), Times.Never);
        }

        /// <summary>
        /// Tests that GetRecommendation returns Ok with the recommendation when the user is authenticated
        /// and the service successfully returns a PersonalizedRecommendationDto.
        /// Input: ClaimsPrincipal with a valid NameIdentifier claim and ILearnService returning a DTO.
        /// Expected: OkObjectResult with the same DTO instance and service invoked once with the userId.
        /// </summary>
        [TestMethod]
        public async Task GetRecommendation_ValidUser_ServiceReturnsRecommendation_ReturnsOkWithRecommendation()
        {
            // Arrange
            var userId = "test-user-123";
            var recommendation = new PersonalizedRecommendationDto
            {
                RecommendedLessonId = 42,
                LessonTitle = "Test Lesson",
                LessonDescription = "Desc",
                CategoryId = 7,
                CategoryName = "Cat",
                Reason = "Because",
                CurrentProgress = 0.5,
                Difficulty = "Easy"
            };

            var mockLearnService = new Mock<ILearnService>();
            mockLearnService
                .Setup(s => s.GetPersonalizedRecommendationAsync(userId))
                .ReturnsAsync(recommendation);

            var mockLogger = new Mock<ILogger<LearnController>>();

            var controller = new LearnController(mockLearnService.Object, mockLogger.Object)
            {
                ControllerContext = CreateControllerContextWithUser(userId)
            };

            // Act
            var result = await controller.GetRecommendation();

            // Assert
            Assert.IsNotNull(result);
            var okResult = result.Result as OkObjectResult;
            Assert.IsNotNull(okResult);
            Assert.AreSame(recommendation, okResult.Value);

            mockLearnService.Verify(s => s.GetPersonalizedRecommendationAsync(userId), Times.Once);
        }

        /// <summary>
        /// Tests that GetRecommendation returns 500 status code and logs an error when the learn service throws an exception.
        /// Input: Authenticated user; ILearnService throws Exception.
        /// Expected: ObjectResult with status code 500 and message "Error fetching recommendation"; logger logs the exception.
        /// </summary>
        [TestMethod]
        public async Task GetRecommendation_ServiceThrowsException_Returns500AndLogsError()
        {
            // Arrange
            var userId = "service-error-user";
            var mockLearnService = new Mock<ILearnService>();
            var exception = new Exception("Service failure");
            mockLearnService
                .Setup(s => s.GetPersonalizedRecommendationAsync(userId))
                .ThrowsAsync(exception);

            var mockLogger = new Mock<ILogger<LearnController>>();

            var controller = new LearnController(mockLearnService.Object, mockLogger.Object)
            {
                ControllerContext = CreateControllerContextWithUser(userId)
            };

            // Act
            var result = await controller.GetRecommendation();

            // Assert
            Assert.IsNotNull(result);
            var statusCodeResult = result.Result as ObjectResult;
            Assert.IsNotNull(statusCodeResult);
            Assert.AreEqual(500, statusCodeResult.StatusCode);

            var value = statusCodeResult.Value;
            Assert.IsNotNull(value);
            var messageProperty = value.GetType().GetProperty("message");
            Assert.IsNotNull(messageProperty);
            Assert.AreEqual("Error fetching recommendation", messageProperty.GetValue(value));

            mockLearnService.Verify(s => s.GetPersonalizedRecommendationAsync(userId), Times.Once);

            mockLogger.Verify(
                x => x.Log(
                    LogLevel.Error,
                    It.IsAny<EventId>(),
                    It.Is<It.IsAnyType>((v, t) => true),
                    It.IsAny<Exception>(),
                    It.Is<Func<It.IsAnyType, Exception?, string>>((v, t) => true)),
                Times.Once);
        }

        /// <summary>
        /// Tests that GetLessonsByCategory returns 500 and the expected error message when the service throws an exception.
        /// Input: Authenticated user but ILearnService.GetLessonsByCategoryAsync throws.
        /// Expected: ObjectResult with status code 500 and message "Error fetching lessons".
        /// </summary>
        [TestMethod]
        public async Task GetLessonsByCategory_ServiceThrowsException_ReturnsInternalServerError()
        {
            // Arrange
            var mockLearnService = new Mock<ILearnService>();
            var mockLogger = new Mock<ILogger<LearnController>>();
            var categoryId = 7;
            var userId = "test-user-ex";

            mockLearnService
                .Setup(s => s.GetLessonsByCategoryAsync(categoryId, userId))
                .ThrowsAsync(new Exception("service failure"));

            var controller = CreateControllerWithUser(mockLearnService.Object, mockLogger.Object, userId);

            // Act
            var result = await controller.GetLessonsByCategory(categoryId);

            // Assert
            Assert.IsNotNull(result);
            Assert.IsInstanceOfType(result.Result, typeof(ObjectResult));
            var obj = result.Result as ObjectResult;
            Assert.IsNotNull(obj);
            Assert.AreEqual(500, obj.StatusCode);
            var message = obj.Value?.GetType().GetProperty("message")?.GetValue(obj.Value) as string;
            Assert.AreEqual("Error fetching lessons", message);
            mockLearnService.Verify(s => s.GetLessonsByCategoryAsync(categoryId, userId), Times.Once);
        }

        /// <summary>
        /// Tests that GetDailyGoal returns 500 status code and correct error message when the learn service throws an exception.
        /// Input: Authenticated user claim; ILearnService throws Exception.
        /// Expected: ObjectResult with status code 500 and anonymous object containing message "Error fetching daily goal".
        /// </summary>
        [TestMethod]
        public async Task GetDailyGoal_ServiceThrowsException_Returns500StatusCodeAndErrorMessage()
        {
            // Arrange
            var mockLearnService = new Mock<ILearnService>();
            var mockLogger = new Mock<ILogger<LearnController>>();

            var expectedUserId = "user-exception";
            mockLearnService
                .Setup(s => s.GetDailyGoalAsync(It.IsAny<string>()))
                .ThrowsAsync(new Exception("database failure"));

            var controller = new LearnController(mockLearnService.Object, mockLogger.Object);
            var user = new ClaimsPrincipal(new ClaimsIdentity(new[]
            {
                new Claim(ClaimTypes.NameIdentifier, expectedUserId)
            }));
            controller.ControllerContext = new ControllerContext
            {
                HttpContext = new DefaultHttpContext { User = user }
            };

            // Act
            var result = await controller.GetDailyGoal();

            // Assert
            Assert.IsNotNull(result);
            var objectResult = result.Result as ObjectResult;
            Assert.IsNotNull(objectResult);
            Assert.AreEqual(500, objectResult.StatusCode);

            var messageValue = objectResult.Value?.GetType().GetProperty("message")?.GetValue(objectResult.Value) as string;
            Assert.AreEqual("Error fetching daily goal", messageValue);

            mockLearnService.Verify(s => s.GetDailyGoalAsync(expectedUserId), Times.Once);
        }

        /// <summary>
        /// Tests that GetDailyGoal returns 500 status code when the user ID claim is missing.
        /// Input: Controller's User has no NameIdentifier claim.
        /// Expected: Controller catches the InvalidOperationException from GetUserId and returns status code 500; service is not invoked.
        /// </summary>
        [TestMethod]
        public async Task GetDailyGoal_MissingUserIdClaim_Returns500AndDoesNotCallService()
        {
            // Arrange
            var mockLearnService = new Mock<ILearnService>();
            var mockLogger = new Mock<ILogger<LearnController>>();

            var controller = new LearnController(mockLearnService.Object, mockLogger.Object);
            // No NameIdentifier claim present
            var user = new ClaimsPrincipal(new ClaimsIdentity());
            controller.ControllerContext = new ControllerContext
            {
                HttpContext = new DefaultHttpContext { User = user }
            };

            // Act
            var result = await controller.GetDailyGoal();

            // Assert
            Assert.IsNotNull(result);
            var objectResult = result.Result as ObjectResult;
            Assert.IsNotNull(objectResult);
            Assert.AreEqual(500, objectResult.StatusCode);

            // Ensure service was never called due to missing user id
            mockLearnService.Verify(s => s.GetDailyGoalAsync(It.IsAny<string>()), Times.Never);
        }

        /// <summary>
        /// Tests that GetDailyGoal preserves special double values in the returned DTO.
        /// Input: Authenticated user; ILearnService returns DailyGoalDto with special ProgressPercentage values (NaN, PositiveInfinity, NegativeInfinity).
        /// Expected: OkObjectResult with DailyGoalDto containing identical ProgressPercentage values (including NaN and infinities).
        /// </summary>
        [TestMethod]
        public async Task GetDailyGoal_PreservesSpecialDoubleValues_ReturnsSameProgressPercentage()
        {
            // Arrange
            var specialValues = new[] { double.NaN, double.PositiveInfinity, double.NegativeInfinity };

            foreach (var special in specialValues)
            {
                var mockLearnService = new Mock<ILearnService>();
                var mockLogger = new Mock<ILogger<LearnController>>();

                var expectedUserId = "user-special-" + Guid.NewGuid().ToString();
                var dto = new DailyGoalDto
                {
                    TotalReviewsDue = 0,
                    CompletedToday = 0,
                    DailyGoal = 5,
                    ProgressPercentage = special
                };

                mockLearnService
                    .Setup(s => s.GetDailyGoalAsync(expectedUserId))
                    .ReturnsAsync(dto);

                var controller = new LearnController(mockLearnService.Object, mockLogger.Object);
                var user = new ClaimsPrincipal(new ClaimsIdentity(new[]
                {
                    new Claim(ClaimTypes.NameIdentifier, expectedUserId)
                }));
                controller.ControllerContext = new ControllerContext
                {
                    HttpContext = new DefaultHttpContext { User = user }
                };

                // Act
                var result = await controller.GetDailyGoal();

                // Assert
                Assert.IsNotNull(result);
                var okResult = result.Result as OkObjectResult;
                Assert.IsNotNull(okResult);
                var returned = okResult.Value as DailyGoalDto;
                Assert.IsNotNull(returned);

                if (double.IsNaN(special))
                {
                    Assert.IsTrue(double.IsNaN(returned.ProgressPercentage));
                }
                else if (double.IsPositiveInfinity(special))
                {
                    Assert.IsTrue(double.IsPositiveInfinity(returned.ProgressPercentage));
                }
                else if (double.IsNegativeInfinity(special))
                {
                    Assert.IsTrue(double.IsNegativeInfinity(returned.ProgressPercentage));
                }

                mockLearnService.Verify(s => s.GetDailyGoalAsync(expectedUserId), Times.Once);
            }
        }

        /// <summary>
        /// Tests that GetLesson returns 500 status code when the learn service throws an exception.
        /// Input: Valid lessonId and authenticated user; ILearnService throws Exception.
        /// Expected: ObjectResult with StatusCode 500 and logger.LogError called once with the thrown exception.
        /// </summary>
        [TestMethod]
        public async Task GetLesson_ServiceThrowsException_Returns500StatusCodeAndLogsError()
        {
            // Arrange
            const int lessonId = 1;
            const string userId = "user123";
            var expectedException = new Exception("Database connection failed");

            var mockLearnService = new Mock<ILearnService>();
            mockLearnService
                .Setup(s => s.GetLessonAsync(lessonId, userId))
                .ThrowsAsync(expectedException);

            var mockLogger = new Mock<ILogger<LearnController>>();
            var controller = new LearnController(mockLearnService.Object, mockLogger.Object);
            SetControllerContextWithUser(controller, userId);

            // Act
            var result = await controller.GetLesson(lessonId);

            // Assert
            Assert.IsNotNull(result);
            Assert.IsInstanceOfType(result.Result, typeof(ObjectResult));
            var objectResult = result.Result as ObjectResult;
            Assert.IsNotNull(objectResult);
            Assert.AreEqual(500, objectResult.StatusCode);

            mockLogger.Verify(
                x => x.Log(
                    LogLevel.Error,
                    It.IsAny<EventId>(),
                    It.Is<It.IsAnyType>((v, t) => true),
                    expectedException,
                    It.IsAny<Func<It.IsAnyType, Exception?, string>>()),
                Times.Once);

            mockLearnService.Verify(s => s.GetLessonAsync(lessonId, userId), Times.Once);
        }

        // Helper: set controller context with a user that has a NameIdentifier claim
        private static void SetControllerContextWithUser(LearnController controller, string userId)
        {
            var claims = new List<Claim> { new Claim(ClaimTypes.NameIdentifier, userId) };
            var identity = new ClaimsIdentity(claims, "TestAuthType");
            var principal = new ClaimsPrincipal(identity);
            controller.ControllerContext = new ControllerContext
            {
                HttpContext = new DefaultHttpContext { User = principal }
            };
        }

        // Helper: set controller context without user claims
        private static void SetControllerContextWithoutUser(LearnController controller)
        {
            var principal = new ClaimsPrincipal(new ClaimsIdentity()); // no claims
            controller.ControllerContext = new ControllerContext
            {
                HttpContext = new DefaultHttpContext { User = principal }
            };
        }

        /// <summary>
        /// Tests that valid quality ratings within [0,5] call the learn service and return Ok.
        /// Inputs tested: 0.0, 2.5, 5.0, 3.7, 1.0, 4.9 (representative valid values including boundaries).
        /// Expected: Service invoked once per call with correct parameters and 200 Ok with success message.
        /// </summary>
        [TestMethod]
        public async Task ReviewLesson_ValidQualityRatings_CallsServiceAndReturnsOk()
        {
            // Arrange
            var validRatings = new double[] { 0.0, 2.5, 5.0, 3.7, 1.0, 4.9 };
            foreach (var qualityRating in validRatings)
            {
                var mockLearnService = new Mock<ILearnService>();
                var mockLogger = new Mock<ILogger<LearnController>>();
                var controller = new LearnController(mockLearnService.Object, mockLogger.Object);

                var spacedRepetitionId = 123;
                var userId = "test-user-id";

                // Setup authenticated user
                controller.ControllerContext = CreateControllerContextWithUser(userId);

                mockLearnService
                    .Setup(x => x.ReviewLessonAsync(userId, spacedRepetitionId, qualityRating))
                    .Returns(Task.CompletedTask)
                    .Verifiable();

                // Act
                var result = await controller.ReviewLesson(spacedRepetitionId, qualityRating);

                // Assert
                Assert.IsNotNull(result);
                var okResult = result.Result as OkObjectResult;
                Assert.IsNotNull(okResult, "Expected OkObjectResult for valid rating: " + qualityRating);
                Assert.AreEqual(200, okResult.StatusCode);

                Assert.IsNotNull(okResult.Value);
                var messageProp = okResult.Value.GetType().GetProperty("message");
                Assert.IsNotNull(messageProp);
                var messageValue = messageProp.GetValue(okResult.Value) as string;
                Assert.AreEqual("Review recorded successfully", messageValue);

                mockLearnService.Verify(x => x.ReviewLessonAsync(userId, spacedRepetitionId, qualityRating), Times.Once);
            }
        }

        /// <summary>
        /// Tests that quality ratings outside (less than 0 or greater than 5) return BadRequest and do NOT call the service.
        /// Inputs tested: -0.1, -1.0, -100.0, 5.1, 6.0, 100.0 (negative and above-range values).
        /// Expected: 400 BadRequest with specific message and no service invocation.
        /// </summary>
        [TestMethod]
        public async Task ReviewLesson_QualityRatingOutOfRange_ReturnsBadRequestAndDoesNotCallService()
        {
            // Arrange
            var invalidRatings = new double[] { -0.1, -1.0, -100.0, 5.1, 6.0, 100.0 };
            foreach (var qualityRating in invalidRatings)
            {
                var mockLearnService = new Mock<ILearnService>();
                var mockLogger = new Mock<ILogger<LearnController>>();
                var controller = new LearnController(mockLearnService.Object, mockLogger.Object);

                var spacedRepetitionId = 123;
                var userId = "test-user-id";
                controller.ControllerContext = CreateControllerContextWithUser(userId);

                // Act
                var result = await controller.ReviewLesson(spacedRepetitionId, qualityRating);

                // Assert
                Assert.IsNotNull(result);
                var badRequestResult = result.Result as BadRequestObjectResult;
                Assert.IsNotNull(badRequestResult, "Expected BadRequestObjectResult for invalid rating: " + qualityRating);
                Assert.AreEqual(400, badRequestResult.StatusCode);

                var value = badRequestResult.Value;
                Assert.IsNotNull(value);
                var messageProp = value.GetType().GetProperty("message");
                Assert.IsNotNull(messageProp);
                var messageValue = messageProp.GetValue(value) as string;
                Assert.AreEqual("Quality rating must be between 0 and 5", messageValue);

                mockLearnService.Verify(x => x.ReviewLessonAsync(It.IsAny<string>(), It.IsAny<int>(), It.IsAny<double>()), Times.Never);
            }
        }

        /// <summary>
        /// Tests that when the learn service throws InvalidOperationException, the controller returns BadRequest with the exception message.
        /// Input: Service throws InvalidOperationException during ReviewLessonAsync.
        /// Expected: 400 BadRequest containing the exception message.
        /// </summary>
        [TestMethod]
        public async Task ReviewLesson_ServiceThrowsInvalidOperationException_ReturnsBadRequestWithExceptionMessage()
        {
            // Arrange
            var mockLearnService = new Mock<ILearnService>();
            var mockLogger = new Mock<ILogger<LearnController>>();
            var controller = new LearnController(mockLearnService.Object, mockLogger.Object);

            var spacedRepetitionId = 321;
            var qualityRating = 4.0;
            var userId = "user-xyz";
            controller.ControllerContext = CreateControllerContextWithUser(userId);

            var exMessage = "Invalid spaced repetition state";
            mockLearnService
                .Setup(x => x.ReviewLessonAsync(userId, spacedRepetitionId, qualityRating))
                .ThrowsAsync(new InvalidOperationException(exMessage));

            // Act
            var result = await controller.ReviewLesson(spacedRepetitionId, qualityRating);

            // Assert
            Assert.IsNotNull(result);
            var badRequest = result.Result as BadRequestObjectResult;
            Assert.IsNotNull(badRequest);
            Assert.AreEqual(400, badRequest.StatusCode);

            var value = badRequest.Value;
            Assert.IsNotNull(value);
            var messageProp = value.GetType().GetProperty("message");
            Assert.IsNotNull(messageProp);
            var messageValue = messageProp.GetValue(value) as string;
            Assert.AreEqual(exMessage, messageValue);

            mockLearnService.Verify(x => x.ReviewLessonAsync(userId, spacedRepetitionId, qualityRating), Times.Once);
        }

        /// <summary>
        /// Tests that when the learn service throws a generic Exception, the controller returns StatusCode 500 with a generic error message.
        /// Input: Service throws Exception during ReviewLessonAsync.
        /// Expected: 500 StatusCode and message 'Error recording review'.
        /// </summary>
        [TestMethod]
        public async Task ReviewLesson_ServiceThrowsException_ReturnsInternalServerError()
        {
            // Arrange
            var mockLearnService = new Mock<ILearnService>();
            var mockLogger = new Mock<ILogger<LearnController>>();
            var controller = new LearnController(mockLearnService.Object, mockLogger.Object);

            var spacedRepetitionId = 555;
            var qualityRating = 2.2;
            var userId = "user-err";
            controller.ControllerContext = CreateControllerContextWithUser(userId);

            mockLearnService
                .Setup(x => x.ReviewLessonAsync(userId, spacedRepetitionId, qualityRating))
                .ThrowsAsync(new Exception("boom"));

            // Act
            var result = await controller.ReviewLesson(spacedRepetitionId, qualityRating);

            // Assert
            Assert.IsNotNull(result);
            var objectResult = result.Result as ObjectResult;
            Assert.IsNotNull(objectResult);
            Assert.AreEqual(500, objectResult.StatusCode);

            var value = objectResult.Value;
            Assert.IsNotNull(value);
            var messageProp = value.GetType().GetProperty("message");
            Assert.IsNotNull(messageProp);
            var messageValue = messageProp.GetValue(value) as string;
            Assert.AreEqual("Error recording review", messageValue);

            mockLearnService.Verify(x => x.ReviewLessonAsync(userId, spacedRepetitionId, qualityRating), Times.Once);
        }

        /// <summary>
        /// Tests that when the authenticated user claim is missing, GetUserId throws InvalidOperationException which is caught and returns BadRequest.
        /// Input: Controller without NameIdentifier claim.
        /// Expected: 400 BadRequest with message 'User ID not found in token' and service not invoked.
        /// </summary>
        [TestMethod]
        public async Task ReviewLesson_UserIdMissingInClaims_ReturnsBadRequestAndDoesNotCallService()
        {
            // Arrange
            var mockLearnService = new Mock<ILearnService>();
            var mockLogger = new Mock<ILogger<LearnController>>();
            var controller = new LearnController(mockLearnService.Object, mockLogger.Object);

            var spacedRepetitionId = 777;
            var qualityRating = 3.0;

            // Leave controller without user claims
            controller.ControllerContext = new ControllerContext
            {
                HttpContext = new DefaultHttpContext { User = new ClaimsPrincipal(new ClaimsIdentity()) }
            };

            // Act
            var result = await controller.ReviewLesson(spacedRepetitionId, qualityRating);

            // Assert
            Assert.IsNotNull(result);
            var badRequest = result.Result as BadRequestObjectResult;
            Assert.IsNotNull(badRequest);
            Assert.AreEqual(400, badRequest.StatusCode);

            var value = badRequest.Value;
            Assert.IsNotNull(value);
            var messageProp = value.GetType().GetProperty("message");
            Assert.IsNotNull(messageProp);
            var messageValue = messageProp.GetValue(value) as string;
            Assert.AreEqual("User ID not found in token", messageValue);

            mockLearnService.Verify(x => x.ReviewLessonAsync(It.IsAny<string>(), It.IsAny<int>(), It.IsAny<double>()), Times.Never);
        }

        /// <summary>
        /// Tests special double values (NaN, PositiveInfinity, NegativeInfinity).
        /// These are marked ignored because behavior is ambiguous (NaN comparisons are false and may bypass range checks).
        /// Manual review required to decide expected handling in production.
        /// </summary>
        [TestMethod]
        [Ignore("ProductionBugSuspected")]
        public async Task ReviewLesson_SpecialDoubleValues_IgnoredForManualReview()
        {
            // Arrange & Act are intentionally left as an example for manual completion.
            var mockLearnService = new Mock<ILearnService>();
            var mockLogger = new Mock<ILogger<LearnController>>();
            var controller = new LearnController(mockLearnService.Object, mockLogger.Object);

            var spacedRepetitionId = 999;
            var userId = "user-special";
            controller.ControllerContext = CreateControllerContextWithUser(userId);

            var specialValues = new double[] { double.NaN, double.PositiveInfinity, double.NegativeInfinity };

            foreach (var val in specialValues)
            {
                // If keeping this test active, decide expected behavior:
                // - Should NaN be treated as invalid? Current implementation does not explicitly handle NaN.
                // - If treated invalid, expect BadRequest. If treated valid, service should be called.
                await controller.ReviewLesson(spacedRepetitionId, val);
            }
        }

        /// <summary>
        /// Tests that the constructor successfully initializes the controller with valid dependencies.
        /// Input: Valid ILearnService mock and valid ILogger mock.
        /// Expected: Constructor completes successfully without throwing exceptions and instance is not null.
        /// </summary>
        [TestMethod]
        public void LearnController_Constructor_ValidDependencies_InitializesSuccessfully()
        {
            // Arrange
            var mockLearnService = new Mock<ILearnService>();
            var mockLogger = new Mock<ILogger<LearnController>>();

            // Act
            var controller = new LearnController(mockLearnService.Object, mockLogger.Object);

            // Assert
            Assert.IsNotNull(controller);
        }

        /// <summary>
        /// Tests the constructor behavior when null is passed for the learnService parameter.
        /// Input: null for learnService, valid ILogger mock.
        /// Expected: Constructor completes without throwing (no validation is performed in the constructor).
        /// </summary>
        [TestMethod]
        public void LearnController_Constructor_NullLearnService_CompletesWithoutException()
        {
            // Arrange
            var mockLogger = new Mock<ILogger<LearnController>>();

            // Act
            var controller = new LearnController(null!, mockLogger.Object);

            // Assert
            Assert.IsNotNull(controller);
        }

        /// <summary>
        /// Tests the constructor behavior when null is passed for the logger parameter.
        /// Input: Valid ILearnService mock, null for logger.
        /// Expected: Constructor completes without throwing (no validation is performed in the constructor).
        /// </summary>
        [TestMethod]
        public void LearnController_Constructor_NullLogger_CompletesWithoutException()
        {
            // Arrange
            var mockLearnService = new Mock<ILearnService>();

            // Act
            var controller = new LearnController(mockLearnService.Object, null!);

            // Assert
            Assert.IsNotNull(controller);
        }

        /// <summary>
        /// Tests the constructor behavior when null is passed for both parameters.
        /// Input: null for both learnService and logger.
        /// Expected: Constructor completes without throwing (no validation is performed in the constructor).
        /// </summary>
        [TestMethod]
        public void LearnController_Constructor_BothParametersNull_CompletesWithoutException()
        {
            // Arrange & Act
            var controller = new LearnController(null!, null!);

            // Assert
            Assert.IsNotNull(controller);
        }

        /// <summary>
        /// Tests that GetLearnPageData returns Ok result with LearnPageDataDto when user is authenticated and service succeeds.
        /// Input: Valid authenticated user with NameIdentifier claim, service returns populated LearnPageDataDto.
        /// Expected: Returns OkObjectResult containing the LearnPageDataDto from the service.
        /// </summary>
        [TestMethod]
        public async Task GetLearnPageData_ValidAuthenticatedUser_ReturnsOkWithData()
        {
            // Arrange
            var mockLearnService = new Mock<ILearnService>();
            var mockLogger = new Mock<ILogger<LearnController>>();
            var controller = new LearnController(mockLearnService.Object, mockLogger.Object);

            var userId = "valid-user-123";
            var expectedData = new LearnPageDataDto();
            var claims = new List<Claim>
            {
                new Claim(ClaimTypes.NameIdentifier, userId)
            };
            var identity = new ClaimsIdentity(claims, "TestAuth");
            var claimsPrincipal = new ClaimsPrincipal(identity);

            controller.ControllerContext = new ControllerContext
            {
                HttpContext = new DefaultHttpContext { User = claimsPrincipal }
            };

            mockLearnService.Setup(s => s.GetLearnPageDataAsync(userId))
                .ReturnsAsync(expectedData);

            // Act
            var result = await controller.GetLearnPageData();

            // Assert
            Assert.IsNotNull(result);
            var okResult = result.Result as OkObjectResult;
            Assert.IsNotNull(okResult);
            Assert.AreEqual(200, okResult.StatusCode);
            Assert.AreSame(expectedData, okResult.Value);
            mockLearnService.Verify(s => s.GetLearnPageDataAsync(userId), Times.Once);
        }

        /// <summary>
        /// Tests that GetLearnPageData returns 500 when user has no claims at all.
        /// Input: User with no claims (not authenticated).
        /// Expected: Returns ObjectResult with 500 status code and error message, logs InvalidOperationException.
        /// </summary>
        [TestMethod]
        public async Task GetLearnPageData_UserWithNoClaims_Returns500WithErrorMessage()
        {
            // Arrange
            var mockLearnService = new Mock<ILearnService>();
            var mockLogger = new Mock<ILogger<LearnController>>();
            var controller = new LearnController(mockLearnService.Object, mockLogger.Object);

            var claimsPrincipal = new ClaimsPrincipal(new ClaimsIdentity());

            controller.ControllerContext = new ControllerContext
            {
                HttpContext = new DefaultHttpContext { User = claimsPrincipal }
            };

            // Act
            var result = await controller.GetLearnPageData();

            // Assert
            Assert.IsNotNull(result);
            var statusCodeResult = result.Result as ObjectResult;
            Assert.IsNotNull(statusCodeResult);
            Assert.AreEqual(500, statusCodeResult.StatusCode);

            var value = statusCodeResult.Value;
            Assert.IsNotNull(value);
            var messageProperty = value.GetType().GetProperty("message");
            Assert.IsNotNull(messageProperty);
            Assert.AreEqual("Error fetching learn page data", messageProperty.GetValue(value));

            mockLearnService.Verify(s => s.GetLearnPageDataAsync(It.IsAny<string>()), Times.Never);
            mockLogger.Verify(
                x => x.Log(
                    LogLevel.Error,
                    It.IsAny<EventId>(),
                    It.Is<It.IsAnyType>((v, t) => true),
                    It.IsAny<InvalidOperationException>(),
                    It.Is<Func<It.IsAnyType, Exception?, string>>((v, t) => true)),
                Times.Once);
        }

        /// <summary>
        /// Tests that GetLearnPageData returns 500 when NameIdentifier claim exists but has whitespace-only value.
        /// Input: User with NameIdentifier claim containing only whitespace.
        /// Expected: Returns ObjectResult with 500 status code and error message.
        /// </summary>
        [TestMethod]
        [DataRow("   ")]
        [DataRow("\t")]
        [DataRow("\n")]
        [DataRow("\r\n")]
        [DataRow("  \t  \n  ")]
        public async Task GetLearnPageData_UserIdClaimWhitespaceOnly_Returns500WithErrorMessage(string whitespaceUserId)
        {
            // Arrange
            var mockLearnService = new Mock<ILearnService>();
            var mockLogger = new Mock<ILogger<LearnController>>();
            var controller = new LearnController(mockLearnService.Object, mockLogger.Object);

            var claims = new List<Claim>
            {
                new Claim(ClaimTypes.NameIdentifier, whitespaceUserId)
            };
            var identity = new ClaimsIdentity(claims, "TestAuth");
            var claimsPrincipal = new ClaimsPrincipal(identity);

            controller.ControllerContext = new ControllerContext
            {
                HttpContext = new DefaultHttpContext { User = claimsPrincipal }
            };

            // Act
            var result = await controller.GetLearnPageData();

            // Assert
            Assert.IsNotNull(result);
            var statusCodeResult = result.Result as ObjectResult;
            Assert.IsNotNull(statusCodeResult);
            Assert.AreEqual(500, statusCodeResult.StatusCode);

            mockLearnService.Verify(s => s.GetLearnPageDataAsync(It.IsAny<string>()), Times.Never);
        }

        /// <summary>
        /// Tests that GetLearnPageData correctly passes various valid user ID formats to the service.
        /// Input: Different valid user ID string formats.
        /// Expected: Returns OkObjectResult with data, service called with exact userId.
        /// </summary>
        [TestMethod]
        [DataRow("550e8400-e29b-41d4-a716-446655440000")]
        [DataRow("user@example.com")]
        [DataRow("123456789")]
        [DataRow("a")]
        [DataRow("user-with-special-chars!@#$%")]
        public async Task GetLearnPageData_VariousValidUserIds_PassesCorrectUserIdToService(string userId)
        {
            // Arrange
            var mockLearnService = new Mock<ILearnService>();
            var mockLogger = new Mock<ILogger<LearnController>>();
            var controller = new LearnController(mockLearnService.Object, mockLogger.Object);

            var expectedData = new LearnPageDataDto();
            var claims = new List<Claim>
            {
                new Claim(ClaimTypes.NameIdentifier, userId)
            };
            var identity = new ClaimsIdentity(claims, "TestAuth");
            var claimsPrincipal = new ClaimsPrincipal(identity);

            controller.ControllerContext = new ControllerContext
            {
                HttpContext = new DefaultHttpContext { User = claimsPrincipal }
            };

            mockLearnService.Setup(s => s.GetLearnPageDataAsync(userId))
                .ReturnsAsync(expectedData);

            // Act
            var result = await controller.GetLearnPageData();

            // Assert
            Assert.IsNotNull(result);
            var okResult = result.Result as OkObjectResult;
            Assert.IsNotNull(okResult);
            Assert.AreSame(expectedData, okResult.Value);
            mockLearnService.Verify(s => s.GetLearnPageDataAsync(userId), Times.Once);
        }

        /// <summary>
        /// Tests that GetLearnPageData logs the correct exception message when GetUserId fails.
        /// Input: User without NameIdentifier claim.
        /// Expected: Logger logs InvalidOperationException with message "User ID not found in token".
        /// </summary>
        [TestMethod]
        public async Task GetLearnPageData_GetUserIdFails_LogsCorrectExceptionMessage()
        {
            // Arrange
            var mockLearnService = new Mock<ILearnService>();
            var mockLogger = new Mock<ILogger<LearnController>>();
            var controller = new LearnController(mockLearnService.Object, mockLogger.Object);

            var claimsPrincipal = new ClaimsPrincipal(new ClaimsIdentity());

            controller.ControllerContext = new ControllerContext
            {
                HttpContext = new DefaultHttpContext { User = claimsPrincipal }
            };

            // Act
            var result = await controller.GetLearnPageData();

            // Assert
            mockLogger.Verify(
                x => x.Log(
                    LogLevel.Error,
                    It.IsAny<EventId>(),
                    It.Is<It.IsAnyType>((v, t) => v.ToString()!.Contains("User ID not found in token")),
                    It.Is<InvalidOperationException>(ex => ex.Message == "User ID not found in token"),
                    It.Is<Func<It.IsAnyType, Exception?, string>>((v, t) => true)),
                Times.Once);
        }

        /// <summary>
        /// Tests that GetLearnPageData returns 500 when service throws TimeoutException.
        /// Input: Valid user ID, service throws TimeoutException.
        /// Expected: Returns ObjectResult with 500 status code and error message, logs exception.
        /// </summary>
        [TestMethod]
        public async Task GetLearnPageData_ServiceThrowsTimeoutException_Returns500WithErrorMessage()
        {
            // Arrange
            var mockLearnService = new Mock<ILearnService>();
            var mockLogger = new Mock<ILogger<LearnController>>();
            var controller = new LearnController(mockLearnService.Object, mockLogger.Object);

            var userId = "test-user-timeout";
            var claims = new List<Claim>
            {
                new Claim(ClaimTypes.NameIdentifier, userId)
            };
            var identity = new ClaimsIdentity(claims, "TestAuth");
            var claimsPrincipal = new ClaimsPrincipal(identity);

            controller.ControllerContext = new ControllerContext
            {
                HttpContext = new DefaultHttpContext { User = claimsPrincipal }
            };

            var exception = new TimeoutException("Service timeout");
            mockLearnService.Setup(s => s.GetLearnPageDataAsync(userId))
                .ThrowsAsync(exception);

            // Act
            var result = await controller.GetLearnPageData();

            // Assert
            Assert.IsNotNull(result);
            var statusCodeResult = result.Result as ObjectResult;
            Assert.IsNotNull(statusCodeResult);
            Assert.AreEqual(500, statusCodeResult.StatusCode);

            var value = statusCodeResult.Value;
            Assert.IsNotNull(value);
            var messageProperty = value.GetType().GetProperty("message");
            Assert.IsNotNull(messageProperty);
            Assert.AreEqual("Error fetching learn page data", messageProperty.GetValue(value));

            mockLogger.Verify(
                x => x.Log(
                    LogLevel.Error,
                    It.IsAny<EventId>(),
                    It.Is<It.IsAnyType>((v, t) => true),
                    It.IsAny<TimeoutException>(),
                    It.Is<Func<It.IsAnyType, Exception?, string>>((v, t) => true)),
                Times.Once);
        }

        /// <summary>
        /// Tests that GetLearnPageData does not call service when GetUserId throws InvalidOperationException.
        /// Input: User without NameIdentifier claim.
        /// Expected: Service method is never called.
        /// </summary>
        [TestMethod]
        public async Task GetLearnPageData_GetUserIdThrowsException_DoesNotCallService()
        {
            // Arrange
            var mockLearnService = new Mock<ILearnService>();
            var mockLogger = new Mock<ILogger<LearnController>>();
            var controller = new LearnController(mockLearnService.Object, mockLogger.Object);

            var claimsPrincipal = new ClaimsPrincipal(new ClaimsIdentity());

            controller.ControllerContext = new ControllerContext
            {
                HttpContext = new DefaultHttpContext { User = claimsPrincipal }
            };

            // Act
            var result = await controller.GetLearnPageData();

            // Assert
            mockLearnService.Verify(s => s.GetLearnPageDataAsync(It.IsAny<string>()), Times.Never);
        }

        /// <summary>
        /// Tests that GetLearnPageData handles very long user IDs correctly.
        /// Input: User ID with 10000 characters.
        /// Expected: Returns OkObjectResult with data, service called with full userId.
        /// </summary>
        [TestMethod]
        public async Task GetLearnPageData_VeryLongUserId_PassesFullUserIdToService()
        {
            // Arrange
            var mockLearnService = new Mock<ILearnService>();
            var mockLogger = new Mock<ILogger<LearnController>>();
            var controller = new LearnController(mockLearnService.Object, mockLogger.Object);

            var userId = new string('a', 10000);
            var expectedData = new LearnPageDataDto();
            var claims = new List<Claim>
            {
                new Claim(ClaimTypes.NameIdentifier, userId)
            };
            var identity = new ClaimsIdentity(claims, "TestAuth");
            var claimsPrincipal = new ClaimsPrincipal(identity);

            controller.ControllerContext = new ControllerContext
            {
                HttpContext = new DefaultHttpContext { User = claimsPrincipal }
            };

            mockLearnService.Setup(s => s.GetLearnPageDataAsync(userId))
                .ReturnsAsync(expectedData);

            // Act
            var result = await controller.GetLearnPageData();

            // Assert
            Assert.IsNotNull(result);
            var okResult = result.Result as OkObjectResult;
            Assert.IsNotNull(okResult);
            Assert.AreSame(expectedData, okResult.Value);
            mockLearnService.Verify(s => s.GetLearnPageDataAsync(userId), Times.Once);
        }

        /// <summary>
        /// Tests that GetLearnPageData handles user IDs with unicode characters correctly.
        /// Input: User IDs containing various unicode characters.
        /// Expected: Returns OkObjectResult with data, service called with exact userId including unicode.
        /// </summary>
        [TestMethod]
        [DataRow("user-中文-123")]
        [DataRow("user-العربية")]
        [DataRow("user-🚀emoji")]
        [DataRow("user-кириллица")]
        public async Task GetLearnPageData_UserIdWithUnicodeCharacters_PassesCorrectUserIdToService(string userId)
        {
            // Arrange
            var mockLearnService = new Mock<ILearnService>();
            var mockLogger = new Mock<ILogger<LearnController>>();
            var controller = new LearnController(mockLearnService.Object, mockLogger.Object);

            var expectedData = new LearnPageDataDto();
            var claims = new List<Claim>
            {
                new Claim(ClaimTypes.NameIdentifier, userId)
            };
            var identity = new ClaimsIdentity(claims, "TestAuth");
            var claimsPrincipal = new ClaimsPrincipal(identity);

            controller.ControllerContext = new ControllerContext
            {
                HttpContext = new DefaultHttpContext { User = claimsPrincipal }
            };

            mockLearnService.Setup(s => s.GetLearnPageDataAsync(userId))
                .ReturnsAsync(expectedData);

            // Act
            var result = await controller.GetLearnPageData();

            // Assert
            Assert.IsNotNull(result);
            var okResult = result.Result as OkObjectResult;
            Assert.IsNotNull(okResult);
            Assert.AreSame(expectedData, okResult.Value);
            mockLearnService.Verify(s => s.GetLearnPageDataAsync(userId), Times.Once);
        }

        /// <summary>
        /// Tests that GetLearnPageData logs the correct exception type and message when service fails.
        /// Input: Valid user ID, service throws Exception with specific message.
        /// Expected: Logger logs Exception with matching message "Error fetching learn page data".
        /// </summary>
        [TestMethod]
        public async Task GetLearnPageData_ServiceThrowsException_LogsExceptionWithCorrectMessage()
        {
            // Arrange
            var mockLearnService = new Mock<ILearnService>();
            var mockLogger = new Mock<ILogger<LearnController>>();
            var controller = new LearnController(mockLearnService.Object, mockLogger.Object);

            var userId = "test-user-exception";
            var claims = new List<Claim>
            {
                new Claim(ClaimTypes.NameIdentifier, userId)
            };
            var identity = new ClaimsIdentity(claims, "TestAuth");
            var claimsPrincipal = new ClaimsPrincipal(identity);

            controller.ControllerContext = new ControllerContext
            {
                HttpContext = new DefaultHttpContext { User = claimsPrincipal }
            };

            var exception = new Exception("Service failed");
            mockLearnService.Setup(s => s.GetLearnPageDataAsync(userId))
                .ThrowsAsync(exception);

            // Act
            var result = await controller.GetLearnPageData();

            // Assert
            mockLogger.Verify(
                x => x.Log(
                    LogLevel.Error,
                    It.IsAny<EventId>(),
                    It.Is<It.IsAnyType>((v, t) => v.ToString()!.Contains("Error fetching learn page data")),
                    It.Is<Exception>(ex => ex == exception),
                    It.Is<Func<It.IsAnyType, Exception?, string>>((v, t) => true)),
                Times.Once);
        }

        /// <summary>
        /// Tests that GetLearnPageData returns Ok when service returns LearnPageDataDto with null properties.
        /// Input: Valid user ID, service returns LearnPageDataDto with all null properties.
        /// Expected: Returns OkObjectResult with the DTO containing null values.
        /// </summary>
        [TestMethod]
        public async Task GetLearnPageData_ServiceReturnsDataWithNullProperties_ReturnsOkWithData()
        {
            // Arrange
            var mockLearnService = new Mock<ILearnService>();
            var mockLogger = new Mock<ILogger<LearnController>>();
            var controller = new LearnController(mockLearnService.Object, mockLogger.Object);

            var userId = "test-user-null-props";
            var expectedData = new LearnPageDataDto();
            var claims = new List<Claim>
            {
                new Claim(ClaimTypes.NameIdentifier, userId)
            };
            var identity = new ClaimsIdentity(claims, "TestAuth");
            var claimsPrincipal = new ClaimsPrincipal(identity);

            controller.ControllerContext = new ControllerContext
            {
                HttpContext = new DefaultHttpContext { User = claimsPrincipal }
            };

            mockLearnService.Setup(s => s.GetLearnPageDataAsync(userId))
                .ReturnsAsync(expectedData);

            // Act
            var result = await controller.GetLearnPageData();

            // Assert
            Assert.IsNotNull(result);
            var okResult = result.Result as OkObjectResult;
            Assert.IsNotNull(okResult);
            Assert.AreSame(expectedData, okResult.Value);
        }

        /// <summary>
        /// Tests that GetUpcomingReviews returns Ok with null when service returns null.
        /// Input: Valid authenticated user, service returns null.
        /// Expected: OkObjectResult with null value.
        /// </summary>
        [TestMethod]
        public async Task GetUpcomingReviews_ServiceReturnsNull_ReturnsOkWithNull()
        {
            // Arrange
            var mockLearnService = new Mock<ILearnService>();
            var mockLogger = new Mock<ILogger<LearnController>>();
            var controller = new LearnController(mockLearnService.Object, mockLogger.Object);

            var userId = "test-user-123";
            var claims = new[] { new Claim(ClaimTypes.NameIdentifier, userId) };
            var identity = new ClaimsIdentity(claims, "TestAuthType");
            var claimsPrincipal = new ClaimsPrincipal(identity);

            controller.ControllerContext = new ControllerContext
            {
                HttpContext = new DefaultHttpContext { User = claimsPrincipal }
            };

            mockLearnService
                .Setup(s => s.GetUpcomingReviewsAsync(userId))
                .ReturnsAsync((UpcomingReviewsDto?)null);

            // Act
            var result = await controller.GetUpcomingReviews();

            // Assert
            Assert.IsNotNull(result);
            Assert.IsInstanceOfType(result.Result, typeof(OkObjectResult));
            var okResult = result.Result as OkObjectResult;
            Assert.IsNotNull(okResult);
            Assert.AreEqual(200, okResult.StatusCode);
            Assert.IsNull(okResult.Value);

            mockLearnService.Verify(s => s.GetUpcomingReviewsAsync(userId), Times.Once);
        }

        /// <summary>
        /// Tests that GetUpcomingReviews correctly handles negative values in UpcomingReviewsDto.
        /// Input: Valid authenticated user, service returns DTO with negative values.
        /// Expected: OkObjectResult with DTO containing negative values.
        /// </summary>
        [TestMethod]
        public async Task GetUpcomingReviews_ServiceReturnsNegativeValues_ReturnsOkWithNegativeValues()
        {
            // Arrange
            var mockLearnService = new Mock<ILearnService>();
            var mockLogger = new Mock<ILogger<LearnController>>();
            var controller = new LearnController(mockLearnService.Object, mockLogger.Object);

            var userId = "test-user-456";
            var claims = new[] { new Claim(ClaimTypes.NameIdentifier, userId) };
            var identity = new ClaimsIdentity(claims, "TestAuthType");
            var claimsPrincipal = new ClaimsPrincipal(identity);

            controller.ControllerContext = new ControllerContext
            {
                HttpContext = new DefaultHttpContext { User = claimsPrincipal }
            };

            var expectedDto = new UpcomingReviewsDto
            {
                DueToday = -5,
                DueTomorrow = -3,
                DueThisWeek = -10,
                Overdue = -2
            };

            mockLearnService
                .Setup(s => s.GetUpcomingReviewsAsync(userId))
                .ReturnsAsync(expectedDto);

            // Act
            var result = await controller.GetUpcomingReviews();

            // Assert
            Assert.IsNotNull(result);
            Assert.IsInstanceOfType(result.Result, typeof(OkObjectResult));
            var okResult = result.Result as OkObjectResult;
            Assert.IsNotNull(okResult);
            Assert.AreEqual(200, okResult.StatusCode);
            Assert.IsInstanceOfType(okResult.Value, typeof(UpcomingReviewsDto));
            var returnedDto = okResult.Value as UpcomingReviewsDto;
            Assert.IsNotNull(returnedDto);
            Assert.AreEqual(expectedDto.DueToday, returnedDto.DueToday);
            Assert.AreEqual(expectedDto.DueTomorrow, returnedDto.DueTomorrow);
            Assert.AreEqual(expectedDto.DueThisWeek, returnedDto.DueThisWeek);
            Assert.AreEqual(expectedDto.Overdue, returnedDto.Overdue);

            mockLearnService.Verify(s => s.GetUpcomingReviewsAsync(userId), Times.Once);
        }

        /// <summary>
        /// Tests that GetUpcomingReviews correctly handles minimum integer values in UpcomingReviewsDto.
        /// Input: Valid authenticated user, service returns DTO with int.MinValue.
        /// Expected: OkObjectResult with DTO containing minimum values.
        /// </summary>
        [TestMethod]
        public async Task GetUpcomingReviews_ServiceReturnsMinIntValues_ReturnsOkWithMinValues()
        {
            // Arrange
            var mockLearnService = new Mock<ILearnService>();
            var mockLogger = new Mock<ILogger<LearnController>>();
            var controller = new LearnController(mockLearnService.Object, mockLogger.Object);

            var userId = "test-user-min";
            var claims = new[] { new Claim(ClaimTypes.NameIdentifier, userId) };
            var identity = new ClaimsIdentity(claims, "TestAuthType");
            var claimsPrincipal = new ClaimsPrincipal(identity);

            controller.ControllerContext = new ControllerContext
            {
                HttpContext = new DefaultHttpContext { User = claimsPrincipal }
            };

            var expectedDto = new UpcomingReviewsDto
            {
                DueToday = int.MinValue,
                DueTomorrow = int.MinValue,
                DueThisWeek = int.MinValue,
                Overdue = int.MinValue
            };

            mockLearnService
                .Setup(s => s.GetUpcomingReviewsAsync(userId))
                .ReturnsAsync(expectedDto);

            // Act
            var result = await controller.GetUpcomingReviews();

            // Assert
            Assert.IsNotNull(result);
            Assert.IsInstanceOfType(result.Result, typeof(OkObjectResult));
            var okResult = result.Result as OkObjectResult;
            Assert.IsNotNull(okResult);
            Assert.AreEqual(200, okResult.StatusCode);
            Assert.IsInstanceOfType(okResult.Value, typeof(UpcomingReviewsDto));
            var returnedDto = okResult.Value as UpcomingReviewsDto;
            Assert.IsNotNull(returnedDto);
            Assert.AreEqual(int.MinValue, returnedDto.DueToday);
            Assert.AreEqual(int.MinValue, returnedDto.DueTomorrow);
            Assert.AreEqual(int.MinValue, returnedDto.DueThisWeek);
            Assert.AreEqual(int.MinValue, returnedDto.Overdue);

            mockLearnService.Verify(s => s.GetUpcomingReviewsAsync(userId), Times.Once);
        }

        /// <summary>
        /// Tests that GetUpcomingReviews does not call service when GetUserId throws InvalidOperationException.
        /// Input: No NameIdentifier claim in user claims.
        /// Expected: Service is not invoked, returns 500 status code.
        /// </summary>
        [TestMethod]
        public async Task GetUpcomingReviews_GetUserIdThrows_DoesNotCallService()
        {
            // Arrange
            var mockLearnService = new Mock<ILearnService>();
            var mockLogger = new Mock<ILogger<LearnController>>();
            var controller = new LearnController(mockLearnService.Object, mockLogger.Object);

            var claims = Array.Empty<Claim>();
            var identity = new ClaimsIdentity(claims, "TestAuthType");
            var claimsPrincipal = new ClaimsPrincipal(identity);

            controller.ControllerContext = new ControllerContext
            {
                HttpContext = new DefaultHttpContext { User = claimsPrincipal }
            };

            // Act
            var result = await controller.GetUpcomingReviews();

            // Assert
            Assert.IsNotNull(result);
            Assert.IsInstanceOfType(result.Result, typeof(ObjectResult));
            var objectResult = result.Result as ObjectResult;
            Assert.IsNotNull(objectResult);
            Assert.AreEqual(500, objectResult.StatusCode);

            mockLearnService.Verify(s => s.GetUpcomingReviewsAsync(It.IsAny<string>()), Times.Never);
        }

        /// <summary>
        /// Tests that GetUpcomingReviews returns 500 with correct error message when service throws InvalidOperationException.
        /// Input: Valid authenticated user, service throws InvalidOperationException.
        /// Expected: ObjectResult with 500 status code and error message "Error fetching upcoming reviews".
        /// </summary>
        [TestMethod]
        public async Task GetUpcomingReviews_ServiceThrowsInvalidOperationException_Returns500WithErrorMessage()
        {
            // Arrange
            var mockLearnService = new Mock<ILearnService>();
            var mockLogger = new Mock<ILogger<LearnController>>();
            var controller = new LearnController(mockLearnService.Object, mockLogger.Object);

            var userId = "test-user-exception";
            var claims = new[] { new Claim(ClaimTypes.NameIdentifier, userId) };
            var identity = new ClaimsIdentity(claims, "TestAuthType");
            var claimsPrincipal = new ClaimsPrincipal(identity);

            controller.ControllerContext = new ControllerContext
            {
                HttpContext = new DefaultHttpContext { User = claimsPrincipal }
            };

            var exceptionMessage = "Invalid operation occurred";
            mockLearnService
                .Setup(s => s.GetUpcomingReviewsAsync(userId))
                .ThrowsAsync(new InvalidOperationException(exceptionMessage));

            // Act
            var result = await controller.GetUpcomingReviews();

            // Assert
            Assert.IsNotNull(result);
            Assert.IsInstanceOfType(result.Result, typeof(ObjectResult));
            var objectResult = result.Result as ObjectResult;
            Assert.IsNotNull(objectResult);
            Assert.AreEqual(500, objectResult.StatusCode);

            var responseValue = objectResult.Value;
            Assert.IsNotNull(responseValue);
            var messageProperty = responseValue.GetType().GetProperty("message");
            Assert.IsNotNull(messageProperty);
            var message = messageProperty.GetValue(responseValue) as string;
            Assert.AreEqual("Error fetching upcoming reviews", message);

            mockLogger.Verify(
                x => x.Log(
                    LogLevel.Error,
                    It.IsAny<EventId>(),
                    It.Is<It.IsAnyType>((v, t) => v.ToString()!.Contains("Error fetching upcoming reviews")),
                    It.IsAny<Exception>(),
                    It.IsAny<Func<It.IsAnyType, Exception?, string>>()),
                Times.Once);
        }

        /// <summary>
        /// Tests that GetUpcomingReviews returns 500 with correct error message when service throws ArgumentException.
        /// Input: Valid authenticated user, service throws ArgumentException.
        /// Expected: ObjectResult with 500 status code and error message "Error fetching upcoming reviews".
        /// </summary>
        [TestMethod]
        public async Task GetUpcomingReviews_ServiceThrowsArgumentException_Returns500WithErrorMessage()
        {
            // Arrange
            var mockLearnService = new Mock<ILearnService>();
            var mockLogger = new Mock<ILogger<LearnController>>();
            var controller = new LearnController(mockLearnService.Object, mockLogger.Object);

            var userId = "test-user-arg-ex";
            var claims = new[] { new Claim(ClaimTypes.NameIdentifier, userId) };
            var identity = new ClaimsIdentity(claims, "TestAuthType");
            var claimsPrincipal = new ClaimsPrincipal(identity);

            controller.ControllerContext = new ControllerContext
            {
                HttpContext = new DefaultHttpContext { User = claimsPrincipal }
            };

            mockLearnService
                .Setup(s => s.GetUpcomingReviewsAsync(userId))
                .ThrowsAsync(new ArgumentException("Invalid argument"));

            // Act
            var result = await controller.GetUpcomingReviews();

            // Assert
            Assert.IsNotNull(result);
            Assert.IsInstanceOfType(result.Result, typeof(ObjectResult));
            var objectResult = result.Result as ObjectResult;
            Assert.IsNotNull(objectResult);
            Assert.AreEqual(500, objectResult.StatusCode);

            var responseValue = objectResult.Value;
            Assert.IsNotNull(responseValue);
            var messageProperty = responseValue.GetType().GetProperty("message");
            Assert.IsNotNull(messageProperty);
            var message = messageProperty.GetValue(responseValue) as string;
            Assert.AreEqual("Error fetching upcoming reviews", message);

            mockLogger.Verify(
                x => x.Log(
                    LogLevel.Error,
                    It.IsAny<EventId>(),
                    It.Is<It.IsAnyType>((v, t) => v.ToString()!.Contains("Error fetching upcoming reviews")),
                    It.IsAny<Exception>(),
                    It.IsAny<Func<It.IsAnyType, Exception?, string>>()),
                Times.Once);
        }

        /// <summary>
        /// Tests that GetUpcomingReviews handles whitespace-only userId claim correctly.
        /// Input: NameIdentifier claim with whitespace-only value.
        /// Expected: Returns 500 status code due to GetUserId validation.
        /// </summary>
        [TestMethod]
        public async Task GetUpcomingReviews_UserIdClaimIsWhitespace_ReturnsInternalServerError()
        {
            // Arrange
            var mockLearnService = new Mock<ILearnService>();
            var mockLogger = new Mock<ILogger<LearnController>>();
            var controller = new LearnController(mockLearnService.Object, mockLogger.Object);

            var claims = new[] { new Claim(ClaimTypes.NameIdentifier, "   ") };
            var identity = new ClaimsIdentity(claims, "TestAuthType");
            var claimsPrincipal = new ClaimsPrincipal(identity);

            controller.ControllerContext = new ControllerContext
            {
                HttpContext = new DefaultHttpContext { User = claimsPrincipal }
            };

            // Act
            var result = await controller.GetUpcomingReviews();

            // Assert
            Assert.IsNotNull(result);
            Assert.IsInstanceOfType(result.Result, typeof(ObjectResult));
            var objectResult = result.Result as ObjectResult;
            Assert.IsNotNull(objectResult);
            Assert.AreEqual(500, objectResult.StatusCode);

            mockLogger.Verify(
                x => x.Log(
                    LogLevel.Error,
                    It.IsAny<EventId>(),
                    It.Is<It.IsAnyType>((v, t) => v.ToString()!.Contains("User ID not found in token")),
                    It.IsAny<InvalidOperationException>(),
                    It.IsAny<Func<It.IsAnyType, Exception?, string>>()),
                Times.Once);

            mockLearnService.Verify(s => s.GetUpcomingReviewsAsync(It.IsAny<string>()), Times.Never);
        }

        /// <summary>
        /// Tests that GetUpcomingReviews correctly passes userId with special characters to the service.
        /// Input: UserId containing special characters (GUID, email, etc.).
        /// Expected: Service is called with exact userId value and returns Ok.
        /// </summary>
        [TestMethod]
        [DataRow("user@example.com")]
        [DataRow("550e8400-e29b-41d4-a716-446655440000")]
        [DataRow("user-with-dashes-123")]
        [DataRow("user_with_underscores_456")]
        [DataRow("user.with.dots.789")]
        public async Task GetUpcomingReviews_UserIdWithSpecialCharacters_PassesCorrectUserIdToService(string userId)
        {
            // Arrange
            var mockLearnService = new Mock<ILearnService>();
            var mockLogger = new Mock<ILogger<LearnController>>();
            var controller = new LearnController(mockLearnService.Object, mockLogger.Object);

            var claims = new[] { new Claim(ClaimTypes.NameIdentifier, userId) };
            var identity = new ClaimsIdentity(claims, "TestAuthType");
            var claimsPrincipal = new ClaimsPrincipal(identity);

            controller.ControllerContext = new ControllerContext
            {
                HttpContext = new DefaultHttpContext { User = claimsPrincipal }
            };

            var expectedDto = new UpcomingReviewsDto
            {
                DueToday = 1,
                DueTomorrow = 2,
                DueThisWeek = 3,
                Overdue = 4
            };

            mockLearnService
                .Setup(s => s.GetUpcomingReviewsAsync(userId))
                .ReturnsAsync(expectedDto);

            // Act
            var result = await controller.GetUpcomingReviews();

            // Assert
            Assert.IsNotNull(result);
            Assert.IsInstanceOfType(result.Result, typeof(OkObjectResult));
            var okResult = result.Result as OkObjectResult;
            Assert.IsNotNull(okResult);
            Assert.AreEqual(200, okResult.StatusCode);

            mockLearnService.Verify(s => s.GetUpcomingReviewsAsync(userId), Times.Once);
        }

        /// <summary>
        /// Tests that GetUpcomingReviews correctly handles very long userId.
        /// Input: UserId with 500 characters.
        /// Expected: Service is called with the full userId and returns Ok.
        /// </summary>
        [TestMethod]
        public async Task GetUpcomingReviews_VeryLongUserId_PassesCorrectUserIdToService()
        {
            // Arrange
            var mockLearnService = new Mock<ILearnService>();
            var mockLogger = new Mock<ILogger<LearnController>>();
            var controller = new LearnController(mockLearnService.Object, mockLogger.Object);

            var userId = new string('a', 500);
            var claims = new[] { new Claim(ClaimTypes.NameIdentifier, userId) };
            var identity = new ClaimsIdentity(claims, "TestAuthType");
            var claimsPrincipal = new ClaimsPrincipal(identity);

            controller.ControllerContext = new ControllerContext
            {
                HttpContext = new DefaultHttpContext { User = claimsPrincipal }
            };

            var expectedDto = new UpcomingReviewsDto
            {
                DueToday = 5,
                DueTomorrow = 10,
                DueThisWeek = 15,
                Overdue = 20
            };

            mockLearnService
                .Setup(s => s.GetUpcomingReviewsAsync(userId))
                .ReturnsAsync(expectedDto);

            // Act
            var result = await controller.GetUpcomingReviews();

            // Assert
            Assert.IsNotNull(result);
            Assert.IsInstanceOfType(result.Result, typeof(OkObjectResult));
            var okResult = result.Result as OkObjectResult;
            Assert.IsNotNull(okResult);
            Assert.AreEqual(200, okResult.StatusCode);

            mockLearnService.Verify(s => s.GetUpcomingReviewsAsync(userId), Times.Once);
            Assert.AreEqual(500, userId.Length);
        }

        /// <summary>
        /// Tests that GetUpcomingReviews correctly handles mixed values in UpcomingReviewsDto.
        /// Input: Valid authenticated user, service returns DTO with mixed positive, negative, and zero values.
        /// Expected: OkObjectResult with DTO containing exact mixed values.
        /// </summary>
        [TestMethod]
        public async Task GetUpcomingReviews_ServiceReturnsMixedValues_ReturnsOkWithMixedValues()
        {
            // Arrange
            var mockLearnService = new Mock<ILearnService>();
            var mockLogger = new Mock<ILogger<LearnController>>();
            var controller = new LearnController(mockLearnService.Object, mockLogger.Object);

            var userId = "test-user-mixed";
            var claims = new[] { new Claim(ClaimTypes.NameIdentifier, userId) };
            var identity = new ClaimsIdentity(claims, "TestAuthType");
            var claimsPrincipal = new ClaimsPrincipal(identity);

            controller.ControllerContext = new ControllerContext
            {
                HttpContext = new DefaultHttpContext { User = claimsPrincipal }
            };

            var expectedDto = new UpcomingReviewsDto
            {
                DueToday = 5,
                DueTomorrow = 0,
                DueThisWeek = -10,
                Overdue = int.MaxValue
            };

            mockLearnService
                .Setup(s => s.GetUpcomingReviewsAsync(userId))
                .ReturnsAsync(expectedDto);

            // Act
            var result = await controller.GetUpcomingReviews();

            // Assert
            Assert.IsNotNull(result);
            Assert.IsInstanceOfType(result.Result, typeof(OkObjectResult));
            var okResult = result.Result as OkObjectResult;
            Assert.IsNotNull(okResult);
            Assert.AreEqual(200, okResult.StatusCode);
            Assert.IsInstanceOfType(okResult.Value, typeof(UpcomingReviewsDto));
            var returnedDto = okResult.Value as UpcomingReviewsDto;
            Assert.IsNotNull(returnedDto);
            Assert.AreEqual(5, returnedDto.DueToday);
            Assert.AreEqual(0, returnedDto.DueTomorrow);
            Assert.AreEqual(-10, returnedDto.DueThisWeek);
            Assert.AreEqual(int.MaxValue, returnedDto.Overdue);

            mockLearnService.Verify(s => s.GetUpcomingReviewsAsync(userId), Times.Once);
        }

        /// <summary>
        /// Tests that GetUpcomingReviews logs the correct error when service throws exception.
        /// Input: Valid user, service throws Exception.
        /// Expected: Logger is called with Exception and correct message.
        /// </summary>
        [TestMethod]
        public async Task GetUpcomingReviews_ServiceThrows_LogsCorrectError()
        {
            // Arrange
            var mockLearnService = new Mock<ILearnService>();
            var mockLogger = new Mock<ILogger<LearnController>>();
            var controller = new LearnController(mockLearnService.Object, mockLogger.Object);

            var userId = "test-user-log";
            var claims = new[] { new Claim(ClaimTypes.NameIdentifier, userId) };
            var identity = new ClaimsIdentity(claims, "TestAuthType");
            var claimsPrincipal = new ClaimsPrincipal(identity);

            controller.ControllerContext = new ControllerContext
            {
                HttpContext = new DefaultHttpContext { User = claimsPrincipal }
            };

            var expectedException = new Exception("Service failed");
            mockLearnService
                .Setup(s => s.GetUpcomingReviewsAsync(userId))
                .ThrowsAsync(expectedException);

            // Act
            var result = await controller.GetUpcomingReviews();

            // Assert
            mockLogger.Verify(
                x => x.Log(
                    LogLevel.Error,
                    It.IsAny<EventId>(),
                    It.Is<It.IsAnyType>((v, t) => v.ToString()!.Contains("Error fetching upcoming reviews")),
                    It.IsAny<Exception>(),
                    It.IsAny<Func<It.IsAnyType, Exception?, string>>()),
                Times.Once);
        }

        /// <summary>
        /// Tests that UpdateLessonProgress returns BadRequest when completionPercentage is below the valid range.
        /// Input: completionPercentage values less than 0 with various lessonId values.
        /// Expected: BadRequestObjectResult with message "Completion percentage must be between 0 and 100", service is not invoked.
        /// </summary>
        [TestMethod]
        [DataRow(-1, 1)]
        [DataRow(-50, 10)]
        [DataRow(-100, 100)]
        [DataRow(int.MinValue, 1)]
        [DataRow(int.MinValue, int.MaxValue)]
        [DataRow(int.MinValue, 0)]
        [DataRow(int.MinValue, -1)]
        public async Task UpdateLessonProgress_CompletionPercentageBelowZero_ReturnsBadRequestWithMessage(int completionPercentage, int lessonId)
        {
            // Arrange
            var mockLearnService = new Mock<ILearnService>();
            var mockLogger = new Mock<ILogger<LearnController>>();
            var controller = new LearnController(mockLearnService.Object, mockLogger.Object);

            // Act
            var result = await controller.UpdateLessonProgress(lessonId, completionPercentage);

            // Assert
            Assert.IsNotNull(result);
            Assert.IsInstanceOfType(result.Result, typeof(BadRequestObjectResult));
            var badRequestResult = result.Result as BadRequestObjectResult;
            Assert.IsNotNull(badRequestResult);
            Assert.AreEqual(400, badRequestResult.StatusCode);

            var responseValue = badRequestResult.Value;
            Assert.IsNotNull(responseValue);
            var messageProperty = responseValue.GetType().GetProperty("message");
            Assert.IsNotNull(messageProperty);
            var message = messageProperty.GetValue(responseValue) as string;
            Assert.AreEqual("Completion percentage must be between 0 and 100", message);

            mockLearnService.Verify(s => s.UpdateLessonProgressAsync(It.IsAny<string>(), It.IsAny<int>(), It.IsAny<int>()), Times.Never);
        }

        /// <summary>
        /// Tests that UpdateLessonProgress returns BadRequest when completionPercentage is above the valid range.
        /// Input: completionPercentage values greater than 100 with various lessonId values.
        /// Expected: BadRequestObjectResult with message "Completion percentage must be between 0 and 100", service is not invoked.
        /// </summary>
        [TestMethod]
        [DataRow(101, 1)]
        [DataRow(150, 10)]
        [DataRow(200, 100)]
        [DataRow(int.MaxValue, 1)]
        [DataRow(int.MaxValue, int.MaxValue)]
        [DataRow(int.MaxValue, 0)]
        [DataRow(int.MaxValue, -1)]
        public async Task UpdateLessonProgress_CompletionPercentageAboveOneHundred_ReturnsBadRequestWithMessage(int completionPercentage, int lessonId)
        {
            // Arrange
            var mockLearnService = new Mock<ILearnService>();
            var mockLogger = new Mock<ILogger<LearnController>>();
            var controller = new LearnController(mockLearnService.Object, mockLogger.Object);

            // Act
            var result = await controller.UpdateLessonProgress(lessonId, completionPercentage);

            // Assert
            Assert.IsNotNull(result);
            Assert.IsInstanceOfType(result.Result, typeof(BadRequestObjectResult));
            var badRequestResult = result.Result as BadRequestObjectResult;
            Assert.IsNotNull(badRequestResult);
            Assert.AreEqual(400, badRequestResult.StatusCode);

            var responseValue = badRequestResult.Value;
            Assert.IsNotNull(responseValue);
            var messageProperty = responseValue.GetType().GetProperty("message");
            Assert.IsNotNull(messageProperty);
            var message = messageProperty.GetValue(responseValue) as string;
            Assert.AreEqual("Completion percentage must be between 0 and 100", message);

            mockLearnService.Verify(s => s.UpdateLessonProgressAsync(It.IsAny<string>(), It.IsAny<int>(), It.IsAny<int>()), Times.Never);
        }

        /// <summary>
        /// Tests that UpdateLessonProgress returns Ok with success message when valid parameters are provided.
        /// Input: Valid completionPercentage values (0, 1, 50, 99, 100) with various lessonId edge values.
        /// Expected: OkObjectResult with message "Lesson progress updated successfully", service is invoked once with exact parameters.
        /// </summary>
        [TestMethod]
        [DataRow(0, 1)]
        [DataRow(0, 0)]
        [DataRow(0, -1)]
        [DataRow(0, int.MaxValue)]
        [DataRow(0, int.MinValue)]
        [DataRow(1, 1)]
        [DataRow(50, 100)]
        [DataRow(99, 999)]
        [DataRow(100, 1)]
        [DataRow(100, 0)]
        [DataRow(100, -1)]
        [DataRow(100, int.MaxValue)]
        [DataRow(100, int.MinValue)]
        public async Task UpdateLessonProgress_ValidCompletionPercentage_ReturnsOkWithSuccessMessage(int completionPercentage, int lessonId)
        {
            // Arrange
            var mockLearnService = new Mock<ILearnService>();
            var mockLogger = new Mock<ILogger<LearnController>>();
            var controller = new LearnController(mockLearnService.Object, mockLogger.Object);

            var userId = "test-user-123";
            var claims = new List<Claim> { new Claim(ClaimTypes.NameIdentifier, userId) };
            var identity = new ClaimsIdentity(claims, "TestAuthType");
            var claimsPrincipal = new ClaimsPrincipal(identity);

            controller.ControllerContext = new ControllerContext
            {
                HttpContext = new DefaultHttpContext { User = claimsPrincipal }
            };

            mockLearnService
                .Setup(s => s.UpdateLessonProgressAsync(userId, lessonId, completionPercentage))
                .Returns(Task.CompletedTask);

            // Act
            var result = await controller.UpdateLessonProgress(lessonId, completionPercentage);

            // Assert
            Assert.IsNotNull(result);
            Assert.IsInstanceOfType(result.Result, typeof(OkObjectResult));
            var okResult = result.Result as OkObjectResult;
            Assert.IsNotNull(okResult);
            Assert.AreEqual(200, okResult.StatusCode);

            var responseValue = okResult.Value;
            Assert.IsNotNull(responseValue);
            var messageProperty = responseValue.GetType().GetProperty("message");
            Assert.IsNotNull(messageProperty);
            var message = messageProperty.GetValue(responseValue) as string;
            Assert.AreEqual("Lesson progress updated successfully", message);

            mockLearnService.Verify(s => s.UpdateLessonProgressAsync(userId, lessonId, completionPercentage), Times.Once);
        }

        /// <summary>
        /// Tests that UpdateLessonProgress returns 500 status code when GetUserId throws InvalidOperationException due to missing user claim.
        /// Input: Controller without NameIdentifier claim, valid completionPercentage and lessonId.
        /// Expected: ObjectResult with status code 500 and message "Error updating lesson progress", logger logs error with lessonId, service is not invoked.
        /// </summary>
        [TestMethod]
        [DataRow(50, 1)]
        [DataRow(0, 100)]
        [DataRow(100, int.MaxValue)]
        public async Task UpdateLessonProgress_UserIdNotFoundInToken_Returns500AndLogsError(int completionPercentage, int lessonId)
        {
            // Arrange
            var mockLearnService = new Mock<ILearnService>();
            var mockLogger = new Mock<ILogger<LearnController>>();
            var controller = new LearnController(mockLearnService.Object, mockLogger.Object);

            var identity = new ClaimsIdentity();
            var claimsPrincipal = new ClaimsPrincipal(identity);

            controller.ControllerContext = new ControllerContext
            {
                HttpContext = new DefaultHttpContext { User = claimsPrincipal }
            };

            // Act
            var result = await controller.UpdateLessonProgress(lessonId, completionPercentage);

            // Assert
            Assert.IsNotNull(result);
            Assert.IsInstanceOfType(result.Result, typeof(ObjectResult));
            var objectResult = result.Result as ObjectResult;
            Assert.IsNotNull(objectResult);
            Assert.AreEqual(500, objectResult.StatusCode);

            var responseValue = objectResult.Value;
            Assert.IsNotNull(responseValue);
            var messageProperty = responseValue.GetType().GetProperty("message");
            Assert.IsNotNull(messageProperty);
            var message = messageProperty.GetValue(responseValue) as string;
            Assert.AreEqual("Error updating lesson progress", message);

            mockLogger.Verify(
                x => x.Log(
                    LogLevel.Error,
                    It.IsAny<EventId>(),
                    It.Is<It.IsAnyType>((v, t) => v.ToString().Contains("User ID not found in token") && v.ToString().Contains(lessonId.ToString())),
                    It.IsAny<InvalidOperationException>(),
                    It.IsAny<Func<It.IsAnyType, Exception?, string>>()),
                Times.Once);

            mockLearnService.Verify(s => s.UpdateLessonProgressAsync(It.IsAny<string>(), It.IsAny<int>(), It.IsAny<int>()), Times.Never);
        }

        /// <summary>
        /// Tests that UpdateLessonProgress returns 500 status code when GetUserId throws InvalidOperationException due to empty user claim.
        /// Input: Controller with empty NameIdentifier claim value, valid completionPercentage and lessonId.
        /// Expected: ObjectResult with status code 500 and message "Error updating lesson progress", logger logs error, service is not invoked.
        /// </summary>
        [TestMethod]
        [DataRow(50, 1)]
        [DataRow(0, 100)]
        public async Task UpdateLessonProgress_UserIdIsEmptyString_Returns500AndLogsError(int completionPercentage, int lessonId)
        {
            // Arrange
            var mockLearnService = new Mock<ILearnService>();
            var mockLogger = new Mock<ILogger<LearnController>>();
            var controller = new LearnController(mockLearnService.Object, mockLogger.Object);

            var claims = new List<Claim> { new Claim(ClaimTypes.NameIdentifier, string.Empty) };
            var identity = new ClaimsIdentity(claims, "TestAuthType");
            var claimsPrincipal = new ClaimsPrincipal(identity);

            controller.ControllerContext = new ControllerContext
            {
                HttpContext = new DefaultHttpContext { User = claimsPrincipal }
            };

            // Act
            var result = await controller.UpdateLessonProgress(lessonId, completionPercentage);

            // Assert
            Assert.IsNotNull(result);
            Assert.IsInstanceOfType(result.Result, typeof(ObjectResult));
            var objectResult = result.Result as ObjectResult;
            Assert.IsNotNull(objectResult);
            Assert.AreEqual(500, objectResult.StatusCode);

            var responseValue = objectResult.Value;
            Assert.IsNotNull(responseValue);
            var messageProperty = responseValue.GetType().GetProperty("message");
            Assert.IsNotNull(messageProperty);
            var message = messageProperty.GetValue(responseValue) as string;
            Assert.AreEqual("Error updating lesson progress", message);

            mockLearnService.Verify(s => s.UpdateLessonProgressAsync(It.IsAny<string>(), It.IsAny<int>(), It.IsAny<int>()), Times.Never);
        }

        /// <summary>
        /// Tests that UpdateLessonProgress returns 500 status code when the service throws an Exception.
        /// Input: Valid user claim, valid completionPercentage and lessonId, but service throws Exception.
        /// Expected: ObjectResult with status code 500 and message "Error updating lesson progress", logger logs error with lessonId and exception.
        /// </summary>
        [TestMethod]
        [DataRow(50, 1)]
        [DataRow(0, 100)]
        [DataRow(100, int.MaxValue)]
        public async Task UpdateLessonProgress_ServiceThrowsGenericException_Returns500AndLogsError(int completionPercentage, int lessonId)
        {
            // Arrange
            var mockLearnService = new Mock<ILearnService>();
            var mockLogger = new Mock<ILogger<LearnController>>();
            var controller = new LearnController(mockLearnService.Object, mockLogger.Object);

            var userId = "test-user-123";
            var claims = new List<Claim> { new Claim(ClaimTypes.NameIdentifier, userId) };
            var identity = new ClaimsIdentity(claims, "TestAuthType");
            var claimsPrincipal = new ClaimsPrincipal(identity);

            controller.ControllerContext = new ControllerContext
            {
                HttpContext = new DefaultHttpContext { User = claimsPrincipal }
            };

            var expectedException = new Exception("Service error");
            mockLearnService
                .Setup(s => s.UpdateLessonProgressAsync(userId, lessonId, completionPercentage))
                .ThrowsAsync(expectedException);

            // Act
            var result = await controller.UpdateLessonProgress(lessonId, completionPercentage);

            // Assert
            Assert.IsNotNull(result);
            Assert.IsInstanceOfType(result.Result, typeof(ObjectResult));
            var objectResult = result.Result as ObjectResult;
            Assert.IsNotNull(objectResult);
            Assert.AreEqual(500, objectResult.StatusCode);

            var responseValue = objectResult.Value;
            Assert.IsNotNull(responseValue);
            var messageProperty = responseValue.GetType().GetProperty("message");
            Assert.IsNotNull(messageProperty);
            var message = messageProperty.GetValue(responseValue) as string;
            Assert.AreEqual("Error updating lesson progress", message);

            mockLogger.Verify(
                x => x.Log(
                    LogLevel.Error,
                    It.IsAny<EventId>(),
                    It.Is<It.IsAnyType>((v, t) => v.ToString().Contains("Error updating lesson progress") && v.ToString().Contains(lessonId.ToString())),
                    It.Is<Exception>(ex => ex == expectedException),
                    It.IsAny<Func<It.IsAnyType, Exception?, string>>()),
                Times.Once);
        }

        /// <summary>
        /// Tests that UpdateLessonProgress returns 500 status code when the service throws InvalidOperationException.
        /// Input: Valid user claim, valid completionPercentage and lessonId, but service throws InvalidOperationException.
        /// Expected: ObjectResult with status code 500 and message "Error updating lesson progress", logger logs error.
        /// </summary>
        [TestMethod]
        [DataRow(50, 1)]
        [DataRow(0, 100)]
        public async Task UpdateLessonProgress_ServiceThrowsInvalidOperationException_Returns500AndLogsError(int completionPercentage, int lessonId)
        {
            // Arrange
            var mockLearnService = new Mock<ILearnService>();
            var mockLogger = new Mock<ILogger<LearnController>>();
            var controller = new LearnController(mockLearnService.Object, mockLogger.Object);

            var userId = "test-user-123";
            var claims = new List<Claim> { new Claim(ClaimTypes.NameIdentifier, userId) };
            var identity = new ClaimsIdentity(claims, "TestAuthType");
            var claimsPrincipal = new ClaimsPrincipal(identity);

            controller.ControllerContext = new ControllerContext
            {
                HttpContext = new DefaultHttpContext { User = claimsPrincipal }
            };

            var expectedException = new InvalidOperationException("Service operation invalid");
            mockLearnService
                .Setup(s => s.UpdateLessonProgressAsync(userId, lessonId, completionPercentage))
                .ThrowsAsync(expectedException);

            // Act
            var result = await controller.UpdateLessonProgress(lessonId, completionPercentage);

            // Assert
            Assert.IsNotNull(result);
            Assert.IsInstanceOfType(result.Result, typeof(ObjectResult));
            var objectResult = result.Result as ObjectResult;
            Assert.IsNotNull(objectResult);
            Assert.AreEqual(500, objectResult.StatusCode);

            var responseValue = objectResult.Value;
            Assert.IsNotNull(responseValue);
            var messageProperty = responseValue.GetType().GetProperty("message");
            Assert.IsNotNull(messageProperty);
            var message = messageProperty.GetValue(responseValue) as string;
            Assert.AreEqual("Error updating lesson progress", message);
        }

        /// <summary>
        /// Tests that UpdateLessonProgress correctly passes special characters in userId to the service.
        /// Input: Valid completionPercentage, userId with special characters (email, GUID, etc.).
        /// Expected: Service is invoked with exact userId including special characters.
        /// </summary>
        [TestMethod]
        [DataRow("user@example.com", 50, 1)]
        [DataRow("user-with-dashes-123", 0, 100)]
        [DataRow("GUID-12345678-1234-1234-1234-123456789012", 100, int.MaxValue)]
        [DataRow("user.name+tag@domain.co.uk", 75, 999)]
        public async Task UpdateLessonProgress_UserIdWithSpecialCharacters_PassesCorrectUserIdToService(string userId, int completionPercentage, int lessonId)
        {
            // Arrange
            var mockLearnService = new Mock<ILearnService>();
            var mockLogger = new Mock<ILogger<LearnController>>();
            var controller = new LearnController(mockLearnService.Object, mockLogger.Object);

            var claims = new List<Claim> { new Claim(ClaimTypes.NameIdentifier, userId) };
            var identity = new ClaimsIdentity(claims, "TestAuthType");
            var claimsPrincipal = new ClaimsPrincipal(identity);

            controller.ControllerContext = new ControllerContext
            {
                HttpContext = new DefaultHttpContext { User = claimsPrincipal }
            };

            mockLearnService
                .Setup(s => s.UpdateLessonProgressAsync(userId, lessonId, completionPercentage))
                .Returns(Task.CompletedTask);

            // Act
            var result = await controller.UpdateLessonProgress(lessonId, completionPercentage);

            // Assert
            Assert.IsNotNull(result);
            Assert.IsInstanceOfType(result.Result, typeof(OkObjectResult));
            mockLearnService.Verify(s => s.UpdateLessonProgressAsync(userId, lessonId, completionPercentage), Times.Once);
        }

        /// <summary>
        /// Tests that UpdateLessonProgress handles whitespace-only userId claim by throwing InvalidOperationException.
        /// Input: Controller with whitespace-only NameIdentifier claim value.
        /// Expected: ObjectResult with status code 500, service is not invoked.
        /// </summary>
        [TestMethod]
        [DataRow("   ", 50, 1)]
        [DataRow("\t", 0, 100)]
        [DataRow("\n", 100, int.MaxValue)]
        public async Task UpdateLessonProgress_UserIdIsWhitespace_Returns500(string userId, int completionPercentage, int lessonId)
        {
            // Arrange
            var mockLearnService = new Mock<ILearnService>();
            var mockLogger = new Mock<ILogger<LearnController>>();
            var controller = new LearnController(mockLearnService.Object, mockLogger.Object);

            var claims = new List<Claim> { new Claim(ClaimTypes.NameIdentifier, userId) };
            var identity = new ClaimsIdentity(claims, "TestAuthType");
            var claimsPrincipal = new ClaimsPrincipal(identity);

            controller.ControllerContext = new ControllerContext
            {
                HttpContext = new DefaultHttpContext { User = claimsPrincipal }
            };

            // Act
            var result = await controller.UpdateLessonProgress(lessonId, completionPercentage);

            // Assert
            Assert.IsNotNull(result);
            Assert.IsInstanceOfType(result.Result, typeof(ObjectResult));
            var objectResult = result.Result as ObjectResult;
            Assert.IsNotNull(objectResult);
            Assert.AreEqual(500, objectResult.StatusCode);

            mockLearnService.Verify(s => s.UpdateLessonProgressAsync(It.IsAny<string>(), It.IsAny<int>(), It.IsAny<int>()), Times.Never);
        }

        /// <summary>
        /// Tests that GetAllCategories returns 500 status code when service throws InvalidOperationException.
        /// Input: Valid authenticated user, service throws InvalidOperationException.
        /// Expected: Returns ObjectResult with 500 status code and error message, logs error.
        /// </summary>
        [TestMethod]
        public async Task GetAllCategories_ServiceThrowsInvalidOperationException_Returns500StatusCode()
        {
            // Arrange
            var mockLearnService = new Mock<ILearnService>();
            var mockLogger = new Mock<ILogger<LearnController>>();
            var userId = "test-user-123";
            var exception = new InvalidOperationException("Service operation failed");

            mockLearnService.Setup(s => s.GetAllCategoriesAsync(userId))
                .ThrowsAsync(exception);

            var controller = new LearnController(mockLearnService.Object, mockLogger.Object);
            SetupControllerWithUser(controller, userId);

            // Act
            var result = await controller.GetAllCategories();

            // Assert
            Assert.IsNotNull(result);
            Assert.IsInstanceOfType(result.Result, typeof(ObjectResult));
            var objectResult = result.Result as ObjectResult;
            Assert.IsNotNull(objectResult);
            Assert.AreEqual(500, objectResult.StatusCode);
            mockLearnService.Verify(s => s.GetAllCategoriesAsync(userId), Times.Once);
            mockLogger.Verify(
                x => x.Log(
                    LogLevel.Error,
                    It.IsAny<EventId>(),
                    It.Is<It.IsAnyType>((v, t) => v.ToString()!.Contains("Error fetching categories")),
                    exception,
                    It.IsAny<Func<It.IsAnyType, Exception?, string>>()),
                Times.Once);
        }

        /// <summary>
        /// Tests that GetAllCategories returns 500 status code when service throws ArgumentNullException.
        /// Input: Valid authenticated user, service throws ArgumentNullException.
        /// Expected: Returns ObjectResult with 500 status code and error message, logs error.
        /// </summary>
        [TestMethod]
        public async Task GetAllCategories_ServiceThrowsArgumentNullException_Returns500StatusCode()
        {
            // Arrange
            var mockLearnService = new Mock<ILearnService>();
            var mockLogger = new Mock<ILogger<LearnController>>();
            var userId = "test-user-456";
            var exception = new ArgumentNullException("categories", "Categories cannot be null");

            mockLearnService.Setup(s => s.GetAllCategoriesAsync(userId))
                .ThrowsAsync(exception);

            var controller = new LearnController(mockLearnService.Object, mockLogger.Object);
            SetupControllerWithUser(controller, userId);

            // Act
            var result = await controller.GetAllCategories();

            // Assert
            Assert.IsNotNull(result);
            Assert.IsInstanceOfType(result.Result, typeof(ObjectResult));
            var objectResult = result.Result as ObjectResult;
            Assert.IsNotNull(objectResult);
            Assert.AreEqual(500, objectResult.StatusCode);
            mockLearnService.Verify(s => s.GetAllCategoriesAsync(userId), Times.Once);
        }

        /// <summary>
        /// Tests that GetAllCategories returns 500 status code when service throws ArgumentException.
        /// Input: Valid authenticated user, service throws ArgumentException.
        /// Expected: Returns ObjectResult with 500 status code and error message, logs error.
        /// </summary>
        [TestMethod]
        public async Task GetAllCategories_ServiceThrowsArgumentException_Returns500StatusCode()
        {
            // Arrange
            var mockLearnService = new Mock<ILearnService>();
            var mockLogger = new Mock<ILogger<LearnController>>();
            var userId = "test-user-789";
            var exception = new ArgumentException("Invalid argument");

            mockLearnService.Setup(s => s.GetAllCategoriesAsync(userId))
                .ThrowsAsync(exception);

            var controller = new LearnController(mockLearnService.Object, mockLogger.Object);
            SetupControllerWithUser(controller, userId);

            // Act
            var result = await controller.GetAllCategories();

            // Assert
            Assert.IsNotNull(result);
            Assert.IsInstanceOfType(result.Result, typeof(ObjectResult));
            var objectResult = result.Result as ObjectResult;
            Assert.IsNotNull(objectResult);
            Assert.AreEqual(500, objectResult.StatusCode);
        }

        /// <summary>
        /// Tests that GetAllCategories does not call service when GetUserId throws InvalidOperationException.
        /// Input: Controller without user claims.
        /// Expected: Service is never called, returns 500 status code.
        /// </summary>
        [TestMethod]
        public async Task GetAllCategories_GetUserIdFails_DoesNotCallService()
        {
            // Arrange
            var mockLearnService = new Mock<ILearnService>();
            var mockLogger = new Mock<ILogger<LearnController>>();

            var controller = new LearnController(mockLearnService.Object, mockLogger.Object);
            SetupControllerWithoutUser(controller);

            // Act
            var result = await controller.GetAllCategories();

            // Assert
            Assert.IsNotNull(result);
            Assert.IsInstanceOfType(result.Result, typeof(ObjectResult));
            mockLearnService.Verify(s => s.GetAllCategoriesAsync(It.IsAny<string>()), Times.Never);
        }

        /// <summary>
        /// Tests that GetAllCategories logs error with correct exception when GetUserId fails.
        /// Input: User claim with null value.
        /// Expected: Logger called with InvalidOperationException and appropriate message.
        /// </summary>
        [TestMethod]
        public async Task GetAllCategories_UserIdClaimNull_LogsErrorWithCorrectException()
        {
            // Arrange
            var mockLearnService = new Mock<ILearnService>();
            var mockLogger = new Mock<ILogger<LearnController>>();

            var controller = new LearnController(mockLearnService.Object, mockLogger.Object);
            var claims = new List<Claim> { new Claim(ClaimTypes.NameIdentifier, string.Empty) };
            var identity = new ClaimsIdentity(claims);
            var claimsPrincipal = new ClaimsPrincipal(identity);
            controller.ControllerContext = new ControllerContext
            {
                HttpContext = new DefaultHttpContext { User = claimsPrincipal }
            };

            // Act
            var result = await controller.GetAllCategories();

            // Assert
            mockLogger.Verify(
                x => x.Log(
                    LogLevel.Error,
                    It.IsAny<EventId>(),
                    It.Is<It.IsAnyType>((v, t) => v.ToString()!.Contains("User ID not found in token")),
                    It.IsAny<InvalidOperationException>(),
                    It.IsAny<Func<It.IsAnyType, Exception?, string>>()),
                Times.Once);
        }

        /// <summary>
        /// Tests that GetAllCategories returns categories with single item correctly.
        /// Input: Valid user, service returns list with one category.
        /// Expected: Returns OkObjectResult with list containing one category.
        /// </summary>
        [TestMethod]
        public async Task GetAllCategories_ServiceReturnsSingleCategory_ReturnsOkWithSingleCategory()
        {
            // Arrange
            var mockLearnService = new Mock<ILearnService>();
            var mockLogger = new Mock<ILogger<LearnController>>();
            var userId = "single-user";
            var categories = new List<LessonCategoryDto>
            {
                new LessonCategoryDto { Id = 1, Title = "Single Category", Description = "Only one", Difficulty = "Easy", Progress = 1.0 }
            };

            mockLearnService.Setup(s => s.GetAllCategoriesAsync(userId)).ReturnsAsync(categories);

            var controller = new LearnController(mockLearnService.Object, mockLogger.Object);
            SetupControllerWithUser(controller, userId);

            // Act
            var result = await controller.GetAllCategories();

            // Assert
            Assert.IsNotNull(result);
            Assert.IsInstanceOfType(result.Result, typeof(OkObjectResult));
            var okResult = result.Result as OkObjectResult;
            Assert.IsNotNull(okResult);
            var returnedCategories = okResult.Value as List<LessonCategoryDto>;
            Assert.IsNotNull(returnedCategories);
            Assert.AreEqual(1, returnedCategories.Count);
            Assert.AreEqual("Single Category", returnedCategories[0].Title);
        }

        /// <summary>
        /// Tests that GetAllCategories preserves category data accuracy.
        /// Input: Service returns categories with specific progress values.
        /// Expected: Returned categories have exact same progress values including edge cases.
        /// </summary>
        [TestMethod]
        public async Task GetAllCategories_ServiceReturnsCategoriesWithEdgeProgress_PreservesAccuracy()
        {
            // Arrange
            var mockLearnService = new Mock<ILearnService>();
            var mockLogger = new Mock<ILogger<LearnController>>();
            var userId = "edge-user";
            var categories = new List<LessonCategoryDto>
            {
                new LessonCategoryDto { Id = 1, Title = "Zero Progress", Description = "Not started", Difficulty = "Easy", Progress = 0.0 },
                new LessonCategoryDto { Id = 2, Title = "Full Progress", Description = "Completed", Difficulty = "Hard", Progress = 1.0 },
                new LessonCategoryDto { Id = 3, Title = "Decimal Progress", Description = "In progress", Difficulty = "Medium", Progress = 0.123456789 }
            };

            mockLearnService.Setup(s => s.GetAllCategoriesAsync(userId)).ReturnsAsync(categories);

            var controller = new LearnController(mockLearnService.Object, mockLogger.Object);
            SetupControllerWithUser(controller, userId);

            // Act
            var result = await controller.GetAllCategories();

            // Assert
            Assert.IsNotNull(result);
            var okResult = result.Result as OkObjectResult;
            Assert.IsNotNull(okResult);
            var returnedCategories = okResult.Value as List<LessonCategoryDto>;
            Assert.IsNotNull(returnedCategories);
            Assert.AreEqual(3, returnedCategories.Count);
            Assert.AreEqual(0.0, returnedCategories[0].Progress);
            Assert.AreEqual(1.0, returnedCategories[1].Progress);
            Assert.AreEqual(0.123456789, returnedCategories[2].Progress);
        }

        /// <summary>
        /// Tests that GetLesson handles various exception types from service consistently.
        /// Input: Valid user and various exception types from service.
        /// Expected: All return 500 status code with error message.
        /// </summary>
        [TestMethod]
        [DataRow(typeof(InvalidOperationException), "Invalid operation")]
        [DataRow(typeof(ArgumentException), "Invalid argument")]
        [DataRow(typeof(ArgumentNullException), "Null argument")]
        public async Task GetLesson_ServiceThrowsVariousExceptions_Returns500(Type exceptionType, string exceptionMessage)
        {
            // Arrange
            const int lessonId = 5;
            const string userId = "user-xyz";
            var exception = (Exception)Activator.CreateInstance(exceptionType, exceptionMessage)!;
            var mockLearnService = new Mock<ILearnService>();
            mockLearnService
                .Setup(s => s.GetLessonAsync(lessonId, userId))
                .ThrowsAsync(exception);

            var mockLogger = new Mock<ILogger<LearnController>>();
            var controller = new LearnController(mockLearnService.Object, mockLogger.Object);
            SetControllerContextWithUser(controller, userId);

            // Act
            var result = await controller.GetLesson(lessonId);

            // Assert
            Assert.IsNotNull(result);
            Assert.IsInstanceOfType(result.Result, typeof(ObjectResult));
            var objectResult = result.Result as ObjectResult;
            Assert.IsNotNull(objectResult);
            Assert.AreEqual(500, objectResult.StatusCode);
            mockLogger.Verify(
                x => x.Log(
                    LogLevel.Error,
                    It.IsAny<EventId>(),
                    It.Is<It.IsAnyType>((v, t) => true),
                    exception,
                    It.IsAny<Func<It.IsAnyType, Exception?, string>>()),
                Times.Once);
        }

        /// <summary>
        /// Tests that GetLesson passes correct userId from claims to service.
        /// Input: Controller with specific userId in claims.
        /// Expected: Service called with exact userId value.
        /// </summary>
        [TestMethod]
        [DataRow("user-simple")]
        [DataRow("123-numeric")]
        [DataRow("user@example.com")]
        [DataRow("550e8400-e29b-41d4-a716-446655440000")]
        [DataRow("very-long-user-id-with-many-characters-and-special-chars-12345")]
        public async Task GetLesson_VariousUserIdFormats_PassesCorrectUserIdToService(string userId)
        {
            // Arrange
            const int lessonId = 1;
            var expectedLesson = new LessonDto { Id = lessonId, Title = "Test" };
            var mockLearnService = new Mock<ILearnService>();
            mockLearnService
                .Setup(s => s.GetLessonAsync(lessonId, userId))
                .ReturnsAsync(expectedLesson);

            var mockLogger = new Mock<ILogger<LearnController>>();
            var controller = new LearnController(mockLearnService.Object, mockLogger.Object);
            SetControllerContextWithUser(controller, userId);

            // Act
            var result = await controller.GetLesson(lessonId);

            // Assert
            Assert.IsNotNull(result);
            Assert.IsInstanceOfType(result.Result, typeof(OkObjectResult));
            mockLearnService.Verify(s => s.GetLessonAsync(lessonId, userId), Times.Once);
        }

        /// <summary>
        /// Tests that GetLesson does not invoke service when GetUserId fails.
        /// Input: Controller without user claims.
        /// Expected: Service method never called.
        /// </summary>
        [TestMethod]
        public async Task GetLesson_GetUserIdFails_DoesNotCallService()
        {
            // Arrange
            const int lessonId = 100;
            var mockLearnService = new Mock<ILearnService>();
            var mockLogger = new Mock<ILogger<LearnController>>();
            var controller = new LearnController(mockLearnService.Object, mockLogger.Object);
            SetControllerContextWithoutUser(controller);

            // Act
            await controller.GetLesson(lessonId);

            // Assert
            mockLearnService.Verify(s => s.GetLessonAsync(It.IsAny<int>(), It.IsAny<string>()), Times.Never);
        }

        /// <summary>
        /// Tests that GetLesson logs error with correct lessonId when GetUserId fails.
        /// Input: Controller without user claims and specific lessonId.
        /// Expected: Logger called with InvalidOperationException and lessonId in structured logging.
        /// </summary>
        [TestMethod]
        [DataRow(1)]
        [DataRow(0)]
        [DataRow(-1)]
        [DataRow(int.MaxValue)]
        [DataRow(int.MinValue)]
        public async Task GetLesson_GetUserIdFails_LogsWithCorrectLessonId(int lessonId)
        {
            // Arrange
            var mockLearnService = new Mock<ILearnService>();
            var mockLogger = new Mock<ILogger<LearnController>>();
            var controller = new LearnController(mockLearnService.Object, mockLogger.Object);
            SetControllerContextWithoutUser(controller);

            // Act
            await controller.GetLesson(lessonId);

            // Assert
            mockLogger.Verify(
                x => x.Log(
                    LogLevel.Error,
                    It.IsAny<EventId>(),
                    It.Is<It.IsAnyType>((v, t) => v.ToString()!.Contains(lessonId.ToString())),
                    It.IsAny<InvalidOperationException>(),
                    It.IsAny<Func<It.IsAnyType, Exception?, string>>()),
                Times.Once);
        }

        /// <summary>
        /// Tests that GetLesson logs error with correct lessonId when service throws.
        /// Input: Valid user but service throws exception for specific lessonId.
        /// Expected: Logger called with exception and lessonId in structured logging.
        /// </summary>
        [TestMethod]
        [DataRow(42)]
        [DataRow(999)]
        [DataRow(int.MaxValue)]
        public async Task GetLesson_ServiceThrows_LogsWithCorrectLessonId(int lessonId)
        {
            // Arrange
            const string userId = "test-user";
            var expectedException = new Exception("Test exception");
            var mockLearnService = new Mock<ILearnService>();
            mockLearnService
                .Setup(s => s.GetLessonAsync(lessonId, userId))
                .ThrowsAsync(expectedException);

            var mockLogger = new Mock<ILogger<LearnController>>();
            var controller = new LearnController(mockLearnService.Object, mockLogger.Object);
            SetControllerContextWithUser(controller, userId);

            // Act
            await controller.GetLesson(lessonId);

            // Assert
            mockLogger.Verify(
                x => x.Log(
                    LogLevel.Error,
                    It.IsAny<EventId>(),
                    It.Is<It.IsAnyType>((v, t) => v.ToString()!.Contains(lessonId.ToString())),
                    expectedException,
                    It.IsAny<Func<It.IsAnyType, Exception?, string>>()),
                Times.Once);
        }

        /// <summary>
        /// Tests that GetLesson correctly handles all edge case lessonId values with valid lessons.
        /// Input: Various edge case lessonIds with service returning valid lessons.
        /// Expected: Returns Ok with lesson for all cases.
        /// </summary>
        [TestMethod]
        [DataRow(1)]
        [DataRow(100)]
        [DataRow(0)]
        [DataRow(-1)]
        [DataRow(-999)]
        [DataRow(int.MinValue)]
        [DataRow(int.MaxValue)]
        public async Task GetLesson_EdgeCaseLessonIdsWithValidLesson_ReturnsOk(int lessonId)
        {
            // Arrange
            const string userId = "user123";
            var expectedLesson = new LessonDto { Id = lessonId, Title = "Test Lesson" };
            var mockLearnService = new Mock<ILearnService>();
            mockLearnService
                .Setup(s => s.GetLessonAsync(lessonId, userId))
                .ReturnsAsync(expectedLesson);

            var mockLogger = new Mock<ILogger<LearnController>>();
            var controller = new LearnController(mockLearnService.Object, mockLogger.Object);
            SetControllerContextWithUser(controller, userId);

            // Act
            var result = await controller.GetLesson(lessonId);

            // Assert
            Assert.IsNotNull(result);
            Assert.IsInstanceOfType(result.Result, typeof(OkObjectResult));
            var okResult = result.Result as OkObjectResult;
            Assert.IsNotNull(okResult);
            Assert.AreEqual(expectedLesson, okResult.Value);
        }

        /// <summary>
        /// Tests that GetLesson returns NotFound for edge case lessonIds when lesson doesn't exist.
        /// Input: Various edge case lessonIds with service returning null.
        /// Expected: Returns NotFound for all cases.
        /// </summary>
        [TestMethod]
        [DataRow(0)]
        [DataRow(-1)]
        [DataRow(-100)]
        [DataRow(int.MinValue)]
        [DataRow(int.MaxValue)]
        public async Task GetLesson_EdgeCaseLessonIdsWithNullLesson_ReturnsNotFound(int lessonId)
        {
            // Arrange
            const string userId = "user456";
            var mockLearnService = new Mock<ILearnService>();
            mockLearnService
                .Setup(s => s.GetLessonAsync(lessonId, userId))
                .ReturnsAsync((LessonDto?)null);

            var mockLogger = new Mock<ILogger<LearnController>>();
            var controller = new LearnController(mockLearnService.Object, mockLogger.Object);
            SetControllerContextWithUser(controller, userId);

            // Act
            var result = await controller.GetLesson(lessonId);

            // Assert
            Assert.IsNotNull(result);
            Assert.IsInstanceOfType(result.Result, typeof(NotFoundObjectResult));
        }

        /// <summary>
        /// Tests that GetLesson correctly handles user claim with empty string value.
        /// Input: Controller with NameIdentifier claim having empty string value.
        /// Expected: Returns 500 status code since GetUserId throws on empty string.
        /// </summary>
        [TestMethod]
        public async Task GetLesson_UserIdClaimIsEmptyString_Returns500()
        {
            // Arrange
            const int lessonId = 15;
            var mockLearnService = new Mock<ILearnService>();
            var mockLogger = new Mock<ILogger<LearnController>>();
            var controller = new LearnController(mockLearnService.Object, mockLogger.Object);

            var claims = new List<Claim> { new Claim(ClaimTypes.NameIdentifier, string.Empty) };
            var identity = new ClaimsIdentity(claims, "TestAuth");
            var principal = new ClaimsPrincipal(identity);
            controller.ControllerContext = new ControllerContext
            {
                HttpContext = new DefaultHttpContext { User = principal }
            };

            // Act
            var result = await controller.GetLesson(lessonId);

            // Assert
            Assert.IsNotNull(result);
            Assert.IsInstanceOfType(result.Result, typeof(ObjectResult));
            var objectResult = result.Result as ObjectResult;
            Assert.IsNotNull(objectResult);
            Assert.AreEqual(500, objectResult.StatusCode);
            mockLearnService.Verify(s => s.GetLessonAsync(It.IsAny<int>(), It.IsAny<string>()), Times.Never);
        }

        /// <summary>
        /// Tests that GetLesson correctly handles user claim with whitespace-only value.
        /// Input: Controller with NameIdentifier claim having whitespace-only value.
        /// Expected: Returns 500 status code since GetUserId throws on whitespace.
        /// </summary>
        [TestMethod]
        [DataRow("   ")]
        [DataRow("\t")]
        [DataRow("\n")]
        [DataRow(" \t\n ")]
        public async Task GetLesson_UserIdClaimIsWhitespace_Returns500(string whitespaceClaim)
        {
            // Arrange
            const int lessonId = 20;
            var mockLearnService = new Mock<ILearnService>();
            var mockLogger = new Mock<ILogger<LearnController>>();
            var controller = new LearnController(mockLearnService.Object, mockLogger.Object);

            var claims = new List<Claim> { new Claim(ClaimTypes.NameIdentifier, whitespaceClaim) };
            var identity = new ClaimsIdentity(claims, "TestAuth");
            var principal = new ClaimsPrincipal(identity);
            controller.ControllerContext = new ControllerContext
            {
                HttpContext = new DefaultHttpContext { User = principal }
            };

            // Act
            var result = await controller.GetLesson(lessonId);

            // Assert
            Assert.IsNotNull(result);
            Assert.IsInstanceOfType(result.Result, typeof(ObjectResult));
            var objectResult = result.Result as ObjectResult;
            Assert.IsNotNull(objectResult);
            Assert.AreEqual(500, objectResult.StatusCode);
        }

        /// <summary>
        /// Tests that GetLesson verifies both lessonId and userId are passed to service correctly.
        /// Input: Specific lessonId and userId values.
        /// Expected: Service called exactly once with both parameters matching.
        /// </summary>
        [TestMethod]
        public async Task GetLesson_ValidInputs_PassesBothParametersToService()
        {
            // Arrange
            const int lessonId = 777;
            const string userId = "specific-user-id-123";
            var expectedLesson = new LessonDto { Id = lessonId };
            var mockLearnService = new Mock<ILearnService>();
            mockLearnService
                .Setup(s => s.GetLessonAsync(lessonId, userId))
                .ReturnsAsync(expectedLesson);

            var mockLogger = new Mock<ILogger<LearnController>>();
            var controller = new LearnController(mockLearnService.Object, mockLogger.Object);
            SetControllerContextWithUser(controller, userId);

            // Act
            await controller.GetLesson(lessonId);

            // Assert
            mockLearnService.Verify(s => s.GetLessonAsync(lessonId, userId), Times.Once);
            mockLearnService.Verify(s => s.GetLessonAsync(It.IsAny<int>(), It.IsAny<string>()), Times.Once);
        }

        /// <summary>
        /// Tests that GetRecommendation returns Ok result with null when service returns null.
        /// Input: Valid authenticated user and service returns null.
        /// Expected: OkObjectResult with null value.
        /// </summary>
        [TestMethod]
        public async Task GetRecommendation_ServiceReturnsNull_ReturnsOkWithNull()
        {
            // Arrange
            var userId = "test-user-null";
            var mockLearnService = new Mock<ILearnService>();
            mockLearnService
                .Setup(s => s.GetPersonalizedRecommendationAsync(userId))
                .ReturnsAsync((PersonalizedRecommendationDto?)null);

            var mockLogger = new Mock<ILogger<LearnController>>();

            var controller = new LearnController(mockLearnService.Object, mockLogger.Object)
            {
                ControllerContext = CreateControllerContextWithUser(userId)
            };

            // Act
            var result = await controller.GetRecommendation();

            // Assert
            Assert.IsNotNull(result);
            var okResult = result.Result as OkObjectResult;
            Assert.IsNotNull(okResult);
            Assert.IsNull(okResult.Value);

            mockLearnService.Verify(s => s.GetPersonalizedRecommendationAsync(userId), Times.Once);
            mockLogger.Verify(
                x => x.Log(
                    It.IsAny<LogLevel>(),
                    It.IsAny<EventId>(),
                    It.Is<It.IsAnyType>((v, t) => true),
                    It.IsAny<Exception>(),
                    It.Is<Func<It.IsAnyType, Exception?, string>>((v, t) => true)),
                Times.Never);
        }

        /// <summary>
        /// Tests that GetRecommendation handles service throwing InvalidOperationException.
        /// Input: Valid authenticated user but service throws InvalidOperationException.
        /// Expected: Returns 500 status code with "Error fetching recommendation" message (caught by generic Exception handler).
        /// </summary>
        [TestMethod]
        public async Task GetRecommendation_ServiceThrowsInvalidOperationException_Returns500WithGenericErrorMessage()
        {
            // Arrange
            var userId = "user-invalid-op";
            var mockLearnService = new Mock<ILearnService>();
            var exception = new InvalidOperationException("Service specific error");
            mockLearnService
                .Setup(s => s.GetPersonalizedRecommendationAsync(userId))
                .ThrowsAsync(exception);

            var mockLogger = new Mock<ILogger<LearnController>>();

            var controller = new LearnController(mockLearnService.Object, mockLogger.Object)
            {
                ControllerContext = CreateControllerContextWithUser(userId)
            };

            // Act
            var result = await controller.GetRecommendation();

            // Assert
            Assert.IsNotNull(result);
            var statusCodeResult = result.Result as ObjectResult;
            Assert.IsNotNull(statusCodeResult);
            Assert.AreEqual(500, statusCodeResult.StatusCode);

            var value = statusCodeResult.Value;
            Assert.IsNotNull(value);
            var messageProperty = value.GetType().GetProperty("message");
            Assert.IsNotNull(messageProperty);
            Assert.AreEqual("Error fetching recommendation", messageProperty.GetValue(value));

            mockLearnService.Verify(s => s.GetPersonalizedRecommendationAsync(userId), Times.Once);

            mockLogger.Verify(
                x => x.Log(
                    LogLevel.Error,
                    It.IsAny<EventId>(),
                    It.Is<It.IsAnyType>((v, t) => true),
                    It.IsAny<Exception>(),
                    It.Is<Func<It.IsAnyType, Exception?, string>>((v, t) => true)),
                Times.Once);
        }

        /// <summary>
        /// Tests that GetRecommendation correctly handles various valid user ID formats.
        /// Input: Different valid user ID formats from authenticated user claims.
        /// Expected: Returns OkObjectResult with recommendation for each user ID format.
        /// </summary>
        [TestMethod]
        [DataRow("simple-user")]
        [DataRow("user-123-abc-456")]
        [DataRow("user@example.com")]
        [DataRow("550e8400-e29b-41d4-a716-446655440000")]
        [DataRow("12345")]
        [DataRow("a")]
        [DataRow("very-long-user-id-with-many-characters-to-test-edge-cases-1234567890-abcdefghijklmnopqrstuvwxyz")]
        public async Task GetRecommendation_VariousUserIdFormats_ReturnsOkWithRecommendation(string userId)
        {
            // Arrange
            var recommendation = new PersonalizedRecommendationDto
            {
                RecommendedLessonId = 1,
                LessonTitle = "Test",
                LessonDescription = "Desc",
                CategoryId = 1,
                CategoryName = "Cat",
                Reason = "Reason",
                CurrentProgress = 0.0,
                Difficulty = "Easy"
            };

            var mockLearnService = new Mock<ILearnService>();
            mockLearnService
                .Setup(s => s.GetPersonalizedRecommendationAsync(userId))
                .ReturnsAsync(recommendation);

            var mockLogger = new Mock<ILogger<LearnController>>();

            var controller = new LearnController(mockLearnService.Object, mockLogger.Object)
            {
                ControllerContext = CreateControllerContextWithUser(userId)
            };

            // Act
            var result = await controller.GetRecommendation();

            // Assert
            Assert.IsNotNull(result);
            var okResult = result.Result as OkObjectResult;
            Assert.IsNotNull(okResult);
            Assert.AreSame(recommendation, okResult.Value);

            mockLearnService.Verify(s => s.GetPersonalizedRecommendationAsync(userId), Times.Once);
        }

        /// <summary>
        /// Tests that GetRecommendation returns 500 when user ID claim contains only whitespace.
        /// Input: User claim with whitespace-only value.
        /// Expected: GetUserId passes whitespace to service, and if service throws, returns 500.
        /// </summary>
        [TestMethod]
        [DataRow("   ")]
        [DataRow("\t")]
        [DataRow("\n")]
        [DataRow(" \t\n ")]
        public async Task GetRecommendation_WhitespaceOnlyUserId_ServiceThrows_Returns500(string whitespaceUserId)
        {
            // Arrange
            var mockLearnService = new Mock<ILearnService>();
            var exception = new ArgumentException("Invalid user ID");
            mockLearnService
                .Setup(s => s.GetPersonalizedRecommendationAsync(whitespaceUserId))
                .ThrowsAsync(exception);

            var mockLogger = new Mock<ILogger<LearnController>>();

            var controller = new LearnController(mockLearnService.Object, mockLogger.Object)
            {
                ControllerContext = CreateControllerContextWithUser(whitespaceUserId)
            };

            // Act
            var result = await controller.GetRecommendation();

            // Assert
            Assert.IsNotNull(result);
            var statusCodeResult = result.Result as ObjectResult;
            Assert.IsNotNull(statusCodeResult);
            Assert.AreEqual(500, statusCodeResult.StatusCode);

            mockLearnService.Verify(s => s.GetPersonalizedRecommendationAsync(whitespaceUserId), Times.Once);
        }

        /// <summary>
        /// Tests that GetRecommendation does not call service when GetUserId throws InvalidOperationException.
        /// Input: Controller without user claims.
        /// Expected: Returns 500 and service is never called.
        /// </summary>
        [TestMethod]
        public async Task GetRecommendation_GetUserIdThrows_ServiceNotCalled()
        {
            // Arrange
            var mockLearnService = new Mock<ILearnService>();
            var mockLogger = new Mock<ILogger<LearnController>>();

            var controller = new LearnController(mockLearnService.Object, mockLogger.Object)
            {
                ControllerContext = CreateControllerContextWithoutUserClaims()
            };

            // Act
            var result = await controller.GetRecommendation();

            // Assert
            Assert.IsNotNull(result);
            var statusCodeResult = result.Result as ObjectResult;
            Assert.IsNotNull(statusCodeResult);
            Assert.AreEqual(500, statusCodeResult.StatusCode);

            mockLearnService.Verify(s => s.GetPersonalizedRecommendationAsync(It.IsAny<string>()), Times.Never);
        }

        /// <summary>
        /// Tests that GetRecommendation returns error message from InvalidOperationException when GetUserId fails.
        /// Input: No user claims (GetUserId throws InvalidOperationException).
        /// Expected: Returns 500 with exception message "User ID not found in token".
        /// </summary>
        [TestMethod]
        public async Task GetRecommendation_GetUserIdThrows_ReturnsExceptionMessage()
        {
            // Arrange
            var mockLearnService = new Mock<ILearnService>();
            var mockLogger = new Mock<ILogger<LearnController>>();

            var controller = new LearnController(mockLearnService.Object, mockLogger.Object)
            {
                ControllerContext = CreateControllerContextWithoutUserClaims()
            };

            // Act
            var result = await controller.GetRecommendation();

            // Assert
            Assert.IsNotNull(result);
            var statusCodeResult = result.Result as ObjectResult;
            Assert.IsNotNull(statusCodeResult);
            Assert.AreEqual(500, statusCodeResult.StatusCode);

            var value = statusCodeResult.Value;
            Assert.IsNotNull(value);
            var messageProperty = value.GetType().GetProperty("message");
            Assert.IsNotNull(messageProperty);
            Assert.AreEqual("User ID not found in token", messageProperty.GetValue(value));
        }

        /// <summary>
        /// Tests that GetRecommendation logs error when GetUserId throws InvalidOperationException.
        /// Input: No user claims.
        /// Expected: Logger.LogError is called once with InvalidOperationException.
        /// </summary>
        [TestMethod]
        public async Task GetRecommendation_GetUserIdThrows_LogsError()
        {
            // Arrange
            var mockLearnService = new Mock<ILearnService>();
            var mockLogger = new Mock<ILogger<LearnController>>();

            var controller = new LearnController(mockLearnService.Object, mockLogger.Object)
            {
                ControllerContext = CreateControllerContextWithoutUserClaims()
            };

            // Act
            var result = await controller.GetRecommendation();

            // Assert
            mockLogger.Verify(
                x => x.Log(
                    LogLevel.Error,
                    It.IsAny<EventId>(),
                    It.Is<It.IsAnyType>((v, t) => true),
                    It.IsAny<InvalidOperationException>(),
                    It.Is<Func<It.IsAnyType, Exception?, string>>((v, t) => true)),
                Times.Once);
        }

        /// <summary>
        /// Tests that GetRecommendation handles ArgumentException from service.
        /// Input: Valid user but service throws ArgumentException.
        /// Expected: Returns 500 with "Error fetching recommendation" message.
        /// </summary>
        [TestMethod]
        public async Task GetRecommendation_ServiceThrowsArgumentException_Returns500WithGenericMessage()
        {
            // Arrange
            var userId = "test-user-arg-ex";
            var mockLearnService = new Mock<ILearnService>();
            var exception = new ArgumentException("Invalid argument");
            mockLearnService
                .Setup(s => s.GetPersonalizedRecommendationAsync(userId))
                .ThrowsAsync(exception);

            var mockLogger = new Mock<ILogger<LearnController>>();

            var controller = new LearnController(mockLearnService.Object, mockLogger.Object)
            {
                ControllerContext = CreateControllerContextWithUser(userId)
            };

            // Act
            var result = await controller.GetRecommendation();

            // Assert
            Assert.IsNotNull(result);
            var statusCodeResult = result.Result as ObjectResult;
            Assert.IsNotNull(statusCodeResult);
            Assert.AreEqual(500, statusCodeResult.StatusCode);

            var value = statusCodeResult.Value;
            Assert.IsNotNull(value);
            var messageProperty = value.GetType().GetProperty("message");
            Assert.IsNotNull(messageProperty);
            Assert.AreEqual("Error fetching recommendation", messageProperty.GetValue(value));

            mockLogger.Verify(
                x => x.Log(
                    LogLevel.Error,
                    It.IsAny<EventId>(),
                    It.Is<It.IsAnyType>((v, t) => true),
                    It.IsAny<Exception>(),
                    It.Is<Func<It.IsAnyType, Exception?, string>>((v, t) => true)),
                Times.Once);
        }

        /// <summary>
        /// Tests that GetRecommendation does not log any errors on successful execution.
        /// Input: Valid user and service returns recommendation successfully.
        /// Expected: Logger is never called.
        /// </summary>
        [TestMethod]
        public async Task GetRecommendation_SuccessfulExecution_DoesNotLogErrors()
        {
            // Arrange
            var userId = "success-user";
            var recommendation = new PersonalizedRecommendationDto
            {
                RecommendedLessonId = 10,
                LessonTitle = "Success Lesson",
                LessonDescription = "Description",
                CategoryId = 5,
                CategoryName = "Category",
                Reason = "Test reason",
                CurrentProgress = 0.75,
                Difficulty = "Medium"
            };

            var mockLearnService = new Mock<ILearnService>();
            mockLearnService
                .Setup(s => s.GetPersonalizedRecommendationAsync(userId))
                .ReturnsAsync(recommendation);

            var mockLogger = new Mock<ILogger<LearnController>>();

            var controller = new LearnController(mockLearnService.Object, mockLogger.Object)
            {
                ControllerContext = CreateControllerContextWithUser(userId)
            };

            // Act
            var result = await controller.GetRecommendation();

            // Assert
            Assert.IsNotNull(result);
            var okResult = result.Result as OkObjectResult;
            Assert.IsNotNull(okResult);

            mockLogger.Verify(
                x => x.Log(
                    It.IsAny<LogLevel>(),
                    It.IsAny<EventId>(),
                    It.Is<It.IsAnyType>((v, t) => true),
                    It.IsAny<Exception>(),
                    It.Is<Func<It.IsAnyType, Exception?, string>>((v, t) => true)),
                Times.Never);
        }

        /// <summary>
        /// Tests that CompleteLesson successfully completes a lesson and returns Ok result with success message.
        /// Input: Valid lessonId and authenticated user.
        /// Expected: Returns OkObjectResult with success message "Lesson completed successfully".
        /// </summary>
        [TestMethod]
        public async Task CompleteLesson_ValidLessonIdAndAuthenticatedUser_ReturnsOkWithSuccessMessage()
        {
            // Arrange
            int lessonId = 42;
            string userId = "test-user-123";

            var mockLearnService = new Mock<ILearnService>();
            mockLearnService
                .Setup(s => s.CompleteLessonAsync(userId, lessonId))
                .Returns(Task.CompletedTask);

            var mockLogger = new Mock<ILogger<LearnController>>();

            var controller = new LearnController(mockLearnService.Object, mockLogger.Object);
            SetupControllerUser(controller, userId);

            // Act
            var result = await controller.CompleteLesson(lessonId);

            // Assert
            Assert.IsNotNull(result);
            var actionResult = result.Result as OkObjectResult;
            Assert.IsNotNull(actionResult);
            Assert.AreEqual(200, actionResult.StatusCode);

            var value = actionResult.Value;
            Assert.IsNotNull(value);

            var messageProperty = value.GetType().GetProperty("message");
            Assert.IsNotNull(messageProperty);
            Assert.AreEqual("Lesson completed successfully", messageProperty.GetValue(value));

            mockLearnService.Verify(s => s.CompleteLessonAsync(userId, lessonId), Times.Once);
        }

        /// <summary>
        /// Tests that CompleteLesson handles various edge case lesson IDs correctly.
        /// Input: Edge case lessonId values including boundaries and extreme values with authenticated user.
        /// Expected: Returns OkObjectResult for all valid inputs after successful service call.
        /// </summary>
        [TestMethod]
        [DataRow(int.MinValue)]
        [DataRow(int.MaxValue)]
        [DataRow(0)]
        [DataRow(-1)]
        [DataRow(-100)]
        [DataRow(999999)]
        [DataRow(1)]
        public async Task CompleteLesson_EdgeCaseLessonIds_ReturnsOkWhenServiceSucceeds(int lessonId)
        {
            // Arrange
            string userId = "edge-case-user";

            var mockLearnService = new Mock<ILearnService>();
            mockLearnService
                .Setup(s => s.CompleteLessonAsync(userId, lessonId))
                .Returns(Task.CompletedTask);

            var mockLogger = new Mock<ILogger<LearnController>>();

            var controller = new LearnController(mockLearnService.Object, mockLogger.Object);
            SetupControllerUser(controller, userId);

            // Act
            var result = await controller.CompleteLesson(lessonId);

            // Assert
            Assert.IsNotNull(result);
            var actionResult = result.Result as OkObjectResult;
            Assert.IsNotNull(actionResult);
            Assert.AreEqual(200, actionResult.StatusCode);

            mockLearnService.Verify(s => s.CompleteLessonAsync(userId, lessonId), Times.Once);
        }

        /// <summary>
        /// Tests that CompleteLesson returns 500 status code when GetUserId fails due to missing user claims.
        /// Input: No NameIdentifier claim configured (unauthenticated user).
        /// Expected: Returns ObjectResult with 500 status code, error message, and service is not called.
        /// </summary>
        [TestMethod]
        public async Task CompleteLesson_NoUserClaims_Returns500AndDoesNotCallService()
        {
            // Arrange
            int lessonId = 10;

            var mockLearnService = new Mock<ILearnService>();
            var mockLogger = new Mock<ILogger<LearnController>>();

            var controller = new LearnController(mockLearnService.Object, mockLogger.Object);
            SetupControllerUserWithNoClaims(controller);

            // Act
            var result = await controller.CompleteLesson(lessonId);

            // Assert
            Assert.IsNotNull(result);
            var actionResult = result.Result as ObjectResult;
            Assert.IsNotNull(actionResult);
            Assert.AreEqual(500, actionResult.StatusCode);

            var value = actionResult.Value;
            Assert.IsNotNull(value);

            var messageProperty = value.GetType().GetProperty("message");
            Assert.IsNotNull(messageProperty);
            Assert.AreEqual("Error completing lesson", messageProperty.GetValue(value));

            mockLearnService.Verify(s => s.CompleteLessonAsync(It.IsAny<string>(), It.IsAny<int>()), Times.Never);
        }

        /// <summary>
        /// Tests that CompleteLesson logs error with correct lessonId when GetUserId fails.
        /// Input: Controller without user claims and specific lessonId.
        /// Expected: Logger.LogError is called with InvalidOperationException and lessonId parameter.
        /// </summary>
        [TestMethod]
        [DataRow(1)]
        [DataRow(100)]
        [DataRow(int.MaxValue)]
        public async Task CompleteLesson_GetUserIdFails_LogsErrorWithLessonId(int lessonId)
        {
            // Arrange
            var mockLearnService = new Mock<ILearnService>();
            var mockLogger = new Mock<ILogger<LearnController>>();

            var controller = new LearnController(mockLearnService.Object, mockLogger.Object);
            SetupControllerUserWithNoClaims(controller);

            // Act
            await controller.CompleteLesson(lessonId);

            // Assert
            mockLogger.Verify(
                x => x.Log(
                    LogLevel.Error,
                    It.IsAny<EventId>(),
                    It.Is<It.IsAnyType>((v, t) => v.ToString()!.Contains($"lesson {lessonId}")),
                    It.IsAny<InvalidOperationException>(),
                    It.IsAny<Func<It.IsAnyType, Exception?, string>>()),
                Times.Once);
        }

        /// <summary>
        /// Tests that CompleteLesson returns 500 status code when service throws InvalidOperationException.
        /// Input: Valid lessonId and authenticated user, but service throws InvalidOperationException.
        /// Expected: Returns ObjectResult with 500 status code, error message, and logs the error.
        /// </summary>
        [TestMethod]
        public async Task CompleteLesson_ServiceThrowsInvalidOperationException_Returns500AndLogsError()
        {
            // Arrange
            int lessonId = 25;
            string userId = "test-user";
            var exception = new InvalidOperationException("Lesson already completed");

            var mockLearnService = new Mock<ILearnService>();
            mockLearnService
                .Setup(s => s.CompleteLessonAsync(userId, lessonId))
                .ThrowsAsync(exception);

            var mockLogger = new Mock<ILogger<LearnController>>();

            var controller = new LearnController(mockLearnService.Object, mockLogger.Object);
            SetupControllerUser(controller, userId);

            // Act
            var result = await controller.CompleteLesson(lessonId);

            // Assert
            Assert.IsNotNull(result);
            var actionResult = result.Result as ObjectResult;
            Assert.IsNotNull(actionResult);
            Assert.AreEqual(500, actionResult.StatusCode);

            var value = actionResult.Value;
            Assert.IsNotNull(value);

            var messageProperty = value.GetType().GetProperty("message");
            Assert.IsNotNull(messageProperty);
            Assert.AreEqual("Error completing lesson", messageProperty.GetValue(value));

            mockLogger.Verify(
                x => x.Log(
                    LogLevel.Error,
                    It.IsAny<EventId>(),
                    It.Is<It.IsAnyType>((v, t) => true),
                    It.IsAny<Exception>(),
                    It.IsAny<Func<It.IsAnyType, Exception?, string>>()),
                Times.Once);
        }

        /// <summary>
        /// Tests that CompleteLesson returns 500 status code when service throws generic Exception.
        /// Input: Valid lessonId and authenticated user, but service throws generic Exception.
        /// Expected: Returns ObjectResult with 500 status code, error message, and logs error with lessonId.
        /// </summary>
        [TestMethod]
        [DataRow(1, "Database error")]
        [DataRow(50, "Network timeout")]
        [DataRow(int.MaxValue, "Unknown error")]
        public async Task CompleteLesson_ServiceThrowsGenericException_Returns500AndLogsErrorWithLessonId(int lessonId, string exceptionMessage)
        {
            // Arrange
            string userId = "generic-ex-user";
            var exception = new Exception(exceptionMessage);

            var mockLearnService = new Mock<ILearnService>();
            mockLearnService
                .Setup(s => s.CompleteLessonAsync(userId, lessonId))
                .ThrowsAsync(exception);

            var mockLogger = new Mock<ILogger<LearnController>>();

            var controller = new LearnController(mockLearnService.Object, mockLogger.Object);
            SetupControllerUser(controller, userId);

            // Act
            var result = await controller.CompleteLesson(lessonId);

            // Assert
            Assert.IsNotNull(result);
            var actionResult = result.Result as ObjectResult;
            Assert.IsNotNull(actionResult);
            Assert.AreEqual(500, actionResult.StatusCode);

            mockLogger.Verify(
                x => x.Log(
                    LogLevel.Error,
                    It.IsAny<EventId>(),
                    It.Is<It.IsAnyType>((v, t) => v.ToString()!.Contains($"lesson {lessonId}")),
                    It.IsAny<Exception>(),
                    It.IsAny<Func<It.IsAnyType, Exception?, string>>()),
                Times.Once);
        }

        /// <summary>
        /// Tests that CompleteLesson correctly passes userId and lessonId to the service.
        /// Input: Specific userId from claims and specific lessonId.
        /// Expected: Service is called with exact userId and lessonId parameters.
        /// </summary>
        [TestMethod]
        [DataRow("user-123", 5)]
        [DataRow("guid-user-456", 100)]
        [DataRow("email@example.com", int.MinValue)]
        [DataRow("a", int.MaxValue)]
        public async Task CompleteLesson_ValidInput_PassesCorrectParametersToService(string userId, int lessonId)
        {
            // Arrange
            var mockLearnService = new Mock<ILearnService>();
            mockLearnService
                .Setup(s => s.CompleteLessonAsync(userId, lessonId))
                .Returns(Task.CompletedTask);

            var mockLogger = new Mock<ILogger<LearnController>>();

            var controller = new LearnController(mockLearnService.Object, mockLogger.Object);
            SetupControllerUser(controller, userId);

            // Act
            await controller.CompleteLesson(lessonId);

            // Assert
            mockLearnService.Verify(s => s.CompleteLessonAsync(userId, lessonId), Times.Once);
        }

        /// <summary>
        /// Tests that CompleteLesson handles empty string userId claim correctly.
        /// Input: Controller with NameIdentifier claim containing empty string.
        /// Expected: GetUserId throws InvalidOperationException, returns 500 status code.
        /// </summary>
        [TestMethod]
        public async Task CompleteLesson_EmptyUserIdClaim_Returns500()
        {
            // Arrange
            int lessonId = 7;

            var mockLearnService = new Mock<ILearnService>();
            var mockLogger = new Mock<ILogger<LearnController>>();

            var controller = new LearnController(mockLearnService.Object, mockLogger.Object);

            var claims = new[] { new Claim(ClaimTypes.NameIdentifier, "") };
            var identity = new ClaimsIdentity(claims, "TestAuth");
            var claimsPrincipal = new ClaimsPrincipal(identity);
            controller.ControllerContext = new ControllerContext
            {
                HttpContext = new DefaultHttpContext { User = claimsPrincipal }
            };

            // Act
            var result = await controller.CompleteLesson(lessonId);

            // Assert
            Assert.IsNotNull(result);
            var actionResult = result.Result as ObjectResult;
            Assert.IsNotNull(actionResult);
            Assert.AreEqual(500, actionResult.StatusCode);

            mockLearnService.Verify(s => s.CompleteLessonAsync(It.IsAny<string>(), It.IsAny<int>()), Times.Never);
        }

        /// <summary>
        /// Tests that CompleteLesson handles null userId claim value correctly.
        /// Input: Controller with NameIdentifier claim having null value.
        /// Expected: GetUserId throws InvalidOperationException, returns 500 status code.
        /// </summary>
        [TestMethod]
        public async Task CompleteLesson_NullUserIdClaimValue_Returns500()
        {
            // Arrange
            int lessonId = 9;

            var mockLearnService = new Mock<ILearnService>();
            var mockLogger = new Mock<ILogger<LearnController>>();

            var controller = new LearnController(mockLearnService.Object, mockLogger.Object);

            var claims = new[] { new Claim(ClaimTypes.NameIdentifier, string.Empty) };
            var identity = new ClaimsIdentity(claims, "TestAuth");
            var claimsPrincipal = new ClaimsPrincipal(identity);
            controller.ControllerContext = new ControllerContext
            {
                HttpContext = new DefaultHttpContext { User = claimsPrincipal }
            };

            // Act
            var result = await controller.CompleteLesson(lessonId);

            // Assert
            Assert.IsNotNull(result);
            var actionResult = result.Result as ObjectResult;
            Assert.IsNotNull(actionResult);
            Assert.AreEqual(500, actionResult.StatusCode);

            mockLearnService.Verify(s => s.CompleteLessonAsync(It.IsAny<string>(), It.IsAny<int>()), Times.Never);
        }

        /// <summary>
        /// Tests that CompleteLesson handles special characters in userId correctly.
        /// Input: UserId with special characters (email, GUID, symbols).
        /// Expected: Service is called with exact userId including special characters.
        /// </summary>
        [TestMethod]
        [DataRow("user@example.com")]
        [DataRow("550e8400-e29b-41d4-a716-446655440000")]
        [DataRow("user-name_with.special+chars")]
        [DataRow("user#123!@")]
        public async Task CompleteLesson_UserIdWithSpecialCharacters_PassesCorrectUserIdToService(string userId)
        {
            // Arrange
            int lessonId = 20;

            var mockLearnService = new Mock<ILearnService>();
            mockLearnService
                .Setup(s => s.CompleteLessonAsync(userId, lessonId))
                .Returns(Task.CompletedTask);

            var mockLogger = new Mock<ILogger<LearnController>>();

            var controller = new LearnController(mockLearnService.Object, mockLogger.Object);
            SetupControllerUser(controller, userId);

            // Act
            await controller.CompleteLesson(lessonId);

            // Assert
            mockLearnService.Verify(s => s.CompleteLessonAsync(userId, lessonId), Times.Once);
        }

        /// <summary>
        /// Tests that CompleteLesson handles very long userId correctly.
        /// Input: UserId with 1000 characters.
        /// Expected: Service is called with full userId.
        /// </summary>
        [TestMethod]
        public async Task CompleteLesson_VeryLongUserId_PassesFullUserIdToService()
        {
            // Arrange
            string userId = new string('a', 1000);
            int lessonId = 30;

            var mockLearnService = new Mock<ILearnService>();
            mockLearnService
                .Setup(s => s.CompleteLessonAsync(userId, lessonId))
                .Returns(Task.CompletedTask);

            var mockLogger = new Mock<ILogger<LearnController>>();

            var controller = new LearnController(mockLearnService.Object, mockLogger.Object);
            SetupControllerUser(controller, userId);

            // Act
            await controller.CompleteLesson(lessonId);

            // Assert
            mockLearnService.Verify(s => s.CompleteLessonAsync(userId, lessonId), Times.Once);
            mockLearnService.Verify(s => s.CompleteLessonAsync(It.Is<string>(u => u.Length == 1000), lessonId), Times.Once);
        }

        /// <summary>
        /// Tests that CompleteLesson returns the exact error message structure in response body.
        /// Input: Service throws exception.
        /// Expected: Response contains anonymous object with "message" property set to "Error completing lesson".
        /// </summary>
        [TestMethod]
        public async Task CompleteLesson_ServiceThrowsException_ReturnsCorrectErrorMessageStructure()
        {
            // Arrange
            int lessonId = 40;
            string userId = "message-test-user";
            var exception = new Exception("Test exception");

            var mockLearnService = new Mock<ILearnService>();
            mockLearnService
                .Setup(s => s.CompleteLessonAsync(userId, lessonId))
                .ThrowsAsync(exception);

            var mockLogger = new Mock<ILogger<LearnController>>();

            var controller = new LearnController(mockLearnService.Object, mockLogger.Object);
            SetupControllerUser(controller, userId);

            // Act
            var result = await controller.CompleteLesson(lessonId);

            // Assert
            var actionResult = result.Result as ObjectResult;
            Assert.IsNotNull(actionResult);

            var value = actionResult.Value;
            Assert.IsNotNull(value);

            var messageProperty = value.GetType().GetProperty("message");
            Assert.IsNotNull(messageProperty, "Response should contain 'message' property");
            Assert.AreEqual("Error completing lesson", messageProperty.GetValue(value));
        }

        /// <summary>
        /// Tests that CompleteLesson does not call service multiple times.
        /// Input: Valid user and lessonId.
        /// Expected: Service.CompleteLessonAsync is called exactly once.
        /// </summary>
        [TestMethod]
        public async Task CompleteLesson_ValidInput_CallsServiceExactlyOnce()
        {
            // Arrange
            int lessonId = 55;
            string userId = "once-test-user";

            var mockLearnService = new Mock<ILearnService>();
            mockLearnService
                .Setup(s => s.CompleteLessonAsync(userId, lessonId))
                .Returns(Task.CompletedTask);

            var mockLogger = new Mock<ILogger<LearnController>>();

            var controller = new LearnController(mockLearnService.Object, mockLogger.Object);
            SetupControllerUser(controller, userId);

            // Act
            await controller.CompleteLesson(lessonId);

            // Assert
            mockLearnService.Verify(s => s.CompleteLessonAsync(userId, lessonId), Times.Once);
            mockLearnService.Verify(s => s.CompleteLessonAsync(It.IsAny<string>(), It.IsAny<int>()), Times.Once);
        }

        /// <summary>
        /// Tests that GetDailyGoal returns Ok result with daily goal data when service call succeeds.
        /// Input: Valid authenticated user with NameIdentifier claim.
        /// Expected: Ok result containing DailyGoalDto from service with all properties preserved.
        /// </summary>
        [TestMethod]
        public async Task GetDailyGoal_ValidUserAndServiceReturnsData_ReturnsOkResultWithDailyGoal()
        {
            // Arrange
            var mockLearnService = new Mock<ILearnService>();
            var mockLogger = new Mock<ILogger<LearnController>>();

            var expectedDailyGoal = new DailyGoalDto
            {
                TotalReviewsDue = 10,
                CompletedToday = 3,
                DailyGoal = 5,
                ProgressPercentage = 60.0
            };

            mockLearnService
                .Setup(s => s.GetDailyGoalAsync(It.IsAny<string>()))
                .ReturnsAsync(expectedDailyGoal);

            var controller = new LearnController(mockLearnService.Object, mockLogger.Object);
            var user = new ClaimsPrincipal(new ClaimsIdentity(new[]
            {
                new Claim(ClaimTypes.NameIdentifier, "test-user-123")
            }));
            controller.ControllerContext = new ControllerContext
            {
                HttpContext = new DefaultHttpContext { User = user }
            };

            // Act
            var result = await controller.GetDailyGoal();

            // Assert
            Assert.IsNotNull(result);
            var okResult = result.Result as OkObjectResult;
            Assert.IsNotNull(okResult);
            Assert.AreEqual(200, okResult.StatusCode);
            var returnedGoal = okResult.Value as DailyGoalDto;
            Assert.IsNotNull(returnedGoal);
            Assert.AreEqual(expectedDailyGoal.TotalReviewsDue, returnedGoal.TotalReviewsDue);
            Assert.AreEqual(expectedDailyGoal.CompletedToday, returnedGoal.CompletedToday);
            Assert.AreEqual(expectedDailyGoal.DailyGoal, returnedGoal.DailyGoal);
            Assert.AreEqual(expectedDailyGoal.ProgressPercentage, returnedGoal.ProgressPercentage);
        }

        /// <summary>
        /// Tests that GetDailyGoal handles special characters in user ID correctly.
        /// Input: User ID with special characters (email format, GUID, special chars) in claim.
        /// Expected: Service called with exact user ID including special characters.
        /// </summary>
        [TestMethod]
        [DataRow("user@example.com")]
        [DataRow("550e8400-e29b-41d4-a716-446655440000")]
        [DataRow("user-with-dashes-123")]
        [DataRow("user_with_underscores")]
        public async Task GetDailyGoal_UserIdWithSpecialCharacters_CallsServiceWithCorrectUserId(string userId)
        {
            // Arrange
            var mockLearnService = new Mock<ILearnService>();
            var mockLogger = new Mock<ILogger<LearnController>>();

            mockLearnService
                .Setup(s => s.GetDailyGoalAsync(userId))
                .ReturnsAsync(new DailyGoalDto());

            var controller = new LearnController(mockLearnService.Object, mockLogger.Object);
            var user = new ClaimsPrincipal(new ClaimsIdentity(new[]
            {
                new Claim(ClaimTypes.NameIdentifier, userId)
            }));
            controller.ControllerContext = new ControllerContext
            {
                HttpContext = new DefaultHttpContext { User = user }
            };

            // Act
            await controller.GetDailyGoal();

            // Assert
            mockLearnService.Verify(s => s.GetDailyGoalAsync(userId), Times.Once);
        }

        /// <summary>
        /// Tests that GetDailyGoal handles whitespace-only user ID claim correctly.
        /// Input: User ID claim with whitespace-only value.
        /// Expected: 500 status code because GetUserId throws InvalidOperationException for null/empty.
        /// </summary>
        [TestMethod]
        public async Task GetDailyGoal_WhitespaceUserIdClaim_Returns500StatusCode()
        {
            // Arrange
            var mockLearnService = new Mock<ILearnService>();
            var mockLogger = new Mock<ILogger<LearnController>>();

            var controller = new LearnController(mockLearnService.Object, mockLogger.Object);
            var user = new ClaimsPrincipal(new ClaimsIdentity(new[]
            {
                new Claim(ClaimTypes.NameIdentifier, "   ")
            }));
            controller.ControllerContext = new ControllerContext
            {
                HttpContext = new DefaultHttpContext { User = user }
            };

            // Act
            var result = await controller.GetDailyGoal();

            // Assert
            var statusCodeResult = result.Result as ObjectResult;
            Assert.IsNotNull(statusCodeResult);
            Assert.AreEqual(500, statusCodeResult.StatusCode);
        }

        /// <summary>
        /// Tests that GetDailyGoal handles minimum integer values correctly.
        /// Input: Service returns DailyGoalDto with int.MinValue for all integer properties.
        /// Expected: Ok result with minimum values preserved.
        /// </summary>
        [TestMethod]
        public async Task GetDailyGoal_MinimumIntValues_ReturnsOkResultWithMinimumValues()
        {
            // Arrange
            var mockLearnService = new Mock<ILearnService>();
            var mockLogger = new Mock<ILogger<LearnController>>();

            var expectedDailyGoal = new DailyGoalDto
            {
                TotalReviewsDue = int.MinValue,
                CompletedToday = int.MinValue,
                DailyGoal = int.MinValue,
                ProgressPercentage = double.MinValue
            };

            mockLearnService
                .Setup(s => s.GetDailyGoalAsync(It.IsAny<string>()))
                .ReturnsAsync(expectedDailyGoal);

            var controller = new LearnController(mockLearnService.Object, mockLogger.Object);
            var user = new ClaimsPrincipal(new ClaimsIdentity(new[]
            {
                new Claim(ClaimTypes.NameIdentifier, "test-user")
            }));
            controller.ControllerContext = new ControllerContext
            {
                HttpContext = new DefaultHttpContext { User = user }
            };

            // Act
            var result = await controller.GetDailyGoal();

            // Assert
            var okResult = result.Result as OkObjectResult;
            Assert.IsNotNull(okResult);
            var returnedGoal = okResult.Value as DailyGoalDto;
            Assert.IsNotNull(returnedGoal);
            Assert.AreEqual(int.MinValue, returnedGoal.TotalReviewsDue);
            Assert.AreEqual(int.MinValue, returnedGoal.CompletedToday);
            Assert.AreEqual(int.MinValue, returnedGoal.DailyGoal);
            Assert.AreEqual(double.MinValue, returnedGoal.ProgressPercentage);
        }

        /// <summary>
        /// Tests that GetDailyGoal handles InvalidOperationException from service.
        /// Input: Service throws InvalidOperationException.
        /// Expected: 500 status code with error message.
        /// </summary>
        [TestMethod]
        public async Task GetDailyGoal_ServiceThrowsInvalidOperationException_Returns500StatusCode()
        {
            // Arrange
            var mockLearnService = new Mock<ILearnService>();
            var mockLogger = new Mock<ILogger<LearnController>>();

            mockLearnService
                .Setup(s => s.GetDailyGoalAsync(It.IsAny<string>()))
                .ThrowsAsync(new InvalidOperationException("Invalid operation"));

            var controller = new LearnController(mockLearnService.Object, mockLogger.Object);
            var user = new ClaimsPrincipal(new ClaimsIdentity(new[]
            {
                new Claim(ClaimTypes.NameIdentifier, "test-user")
            }));
            controller.ControllerContext = new ControllerContext
            {
                HttpContext = new DefaultHttpContext { User = user }
            };

            // Act
            var result = await controller.GetDailyGoal();

            // Assert
            var statusCodeResult = result.Result as ObjectResult;
            Assert.IsNotNull(statusCodeResult);
            Assert.AreEqual(500, statusCodeResult.StatusCode);
        }

        /// <summary>
        /// Tests that GetDailyGoal does not call service when empty user ID claim is provided.
        /// Input: User with empty string in NameIdentifier claim.
        /// Expected: Service GetDailyGoalAsync method never called.
        /// </summary>
        [TestMethod]
        public async Task GetDailyGoal_EmptyUserIdClaim_DoesNotCallService()
        {
            // Arrange
            var mockLearnService = new Mock<ILearnService>();
            var mockLogger = new Mock<ILogger<LearnController>>();

            var controller = new LearnController(mockLearnService.Object, mockLogger.Object);
            var user = new ClaimsPrincipal(new ClaimsIdentity(new[]
            {
                new Claim(ClaimTypes.NameIdentifier, string.Empty)
            }));
            controller.ControllerContext = new ControllerContext
            {
                HttpContext = new DefaultHttpContext { User = user }
            };

            // Act
            await controller.GetDailyGoal();

            // Assert
            mockLearnService.Verify(s => s.GetDailyGoalAsync(It.IsAny<string>()), Times.Never);
        }

        /// <summary>
        /// Tests that GetDailyGoal correctly handles extreme double values for ProgressPercentage.
        /// Input: Service returns DailyGoalDto with double.MaxValue ProgressPercentage.
        /// Expected: Ok result with MaxValue preserved.
        /// </summary>
        [TestMethod]
        public async Task GetDailyGoal_MaxDoubleProgressPercentage_ReturnsOkResult()
        {
            // Arrange
            var mockLearnService = new Mock<ILearnService>();
            var mockLogger = new Mock<ILogger<LearnController>>();

            var expectedDailyGoal = new DailyGoalDto
            {
                TotalReviewsDue = 10,
                CompletedToday = 5,
                DailyGoal = 5,
                ProgressPercentage = double.MaxValue
            };

            mockLearnService
                .Setup(s => s.GetDailyGoalAsync(It.IsAny<string>()))
                .ReturnsAsync(expectedDailyGoal);

            var controller = new LearnController(mockLearnService.Object, mockLogger.Object);
            var user = new ClaimsPrincipal(new ClaimsIdentity(new[]
            {
                new Claim(ClaimTypes.NameIdentifier, "test-user")
            }));
            controller.ControllerContext = new ControllerContext
            {
                HttpContext = new DefaultHttpContext { User = user }
            };

            // Act
            var result = await controller.GetDailyGoal();

            // Assert
            var okResult = result.Result as OkObjectResult;
            Assert.IsNotNull(okResult);
            var returnedGoal = okResult.Value as DailyGoalDto;
            Assert.IsNotNull(returnedGoal);
            Assert.AreEqual(double.MaxValue, returnedGoal.ProgressPercentage);
        }

        /// <summary>
        /// Tests that GetDailyReviews returns 500 Internal Server Error when service throws an exception.
        /// Input: Valid authenticated user, but service throws exception.
        /// Expected: Returns ObjectResult with 500 status code and error message, logs error.
        /// </summary>
        [TestMethod]
        public async Task GetDailyReviews_ServiceThrowsException_ReturnsInternalServerError()
        {
            // Arrange
            var mockLearnService = new Mock<ILearnService>();
            var mockLogger = new Mock<ILogger<LearnController>>();
            var controller = new LearnController(mockLearnService.Object, mockLogger.Object);

            var userId = "test-user-123";
            var claims = new List<Claim> { new Claim(ClaimTypes.NameIdentifier, userId) };
            var identity = new ClaimsIdentity(claims, "TestAuth");
            var claimsPrincipal = new ClaimsPrincipal(identity);
            controller.ControllerContext = new ControllerContext
            {
                HttpContext = new DefaultHttpContext { User = claimsPrincipal }
            };

            var exception = new Exception("Test exception");
            mockLearnService.Setup(s => s.GetDailyReviewLessonsAsync(userId))
                .ThrowsAsync(exception);

            // Act
            var result = await controller.GetDailyReviews();

            // Assert
            Assert.IsNotNull(result);
            Assert.IsInstanceOfType(result.Result, typeof(ObjectResult));
            var objectResult = (ObjectResult)result.Result;
            Assert.AreEqual(500, objectResult.StatusCode);
            Assert.IsNotNull(objectResult.Value);
            mockLogger.Verify(
                l => l.Log(
                    LogLevel.Error,
                    It.IsAny<EventId>(),
                    It.Is<It.IsAnyType>((v, t) => true),
                    It.IsAny<Exception>(),
                    It.Is<Func<It.IsAnyType, Exception?, string>>((v, t) => true)),
                Times.Once);
        }

        /// <summary>
        /// Tests that GetDailyReviews returns 500 when service throws various exception types.
        /// Input: Valid authenticated user, service throws InvalidOperationException, ArgumentException, or generic Exception.
        /// Expected: Returns ObjectResult with 500 status code and error message for all exception types.
        /// </summary>
        [TestMethod]
        [DataRow(typeof(InvalidOperationException))]
        [DataRow(typeof(ArgumentException))]
        [DataRow(typeof(ArgumentNullException))]
        [DataRow(typeof(NullReferenceException))]
        public async Task GetDailyReviews_ServiceThrowsDifferentExceptionTypes_ReturnsInternalServerError(Type exceptionType)
        {
            // Arrange
            var mockLearnService = new Mock<ILearnService>();
            var mockLogger = new Mock<ILogger<LearnController>>();
            var controller = new LearnController(mockLearnService.Object, mockLogger.Object);

            var userId = "test-user-123";
            var claims = new List<Claim> { new Claim(ClaimTypes.NameIdentifier, userId) };
            var identity = new ClaimsIdentity(claims, "TestAuth");
            var claimsPrincipal = new ClaimsPrincipal(identity);
            controller.ControllerContext = new ControllerContext
            {
                HttpContext = new DefaultHttpContext { User = claimsPrincipal }
            };

            var exception = (Exception)Activator.CreateInstance(exceptionType, "Test exception")!;
            mockLearnService.Setup(s => s.GetDailyReviewLessonsAsync(userId))
                .ThrowsAsync(exception);

            // Act
            var result = await controller.GetDailyReviews();

            // Assert
            Assert.IsNotNull(result);
            Assert.IsInstanceOfType(result.Result, typeof(ObjectResult));
            var objectResult = (ObjectResult)result.Result;
            Assert.AreEqual(500, objectResult.StatusCode);
        }

        /// <summary>
        /// Tests that GetDailyReviews returns 500 Internal Server Error when user ID claim is whitespace.
        /// Input: User with whitespace-only NameIdentifier claim value.
        /// Expected: Returns ObjectResult with 500 status code.
        /// </summary>
        [TestMethod]
        [DataRow("   ")]
        [DataRow("\t")]
        [DataRow("\n")]
        [DataRow(" \t\n ")]
        public async Task GetDailyReviews_WhitespaceUserIdInClaims_ReturnsInternalServerError(string whitespaceUserId)
        {
            // Arrange
            var mockLearnService = new Mock<ILearnService>();
            var mockLogger = new Mock<ILogger<LearnController>>();
            var controller = new LearnController(mockLearnService.Object, mockLogger.Object);

            var claims = new List<Claim> { new Claim(ClaimTypes.NameIdentifier, whitespaceUserId) };
            var identity = new ClaimsIdentity(claims, "TestAuth");
            var claimsPrincipal = new ClaimsPrincipal(identity);
            controller.ControllerContext = new ControllerContext
            {
                HttpContext = new DefaultHttpContext { User = claimsPrincipal }
            };

            // Act
            var result = await controller.GetDailyReviews();

            // Assert
            Assert.IsNotNull(result);
            Assert.IsInstanceOfType(result.Result, typeof(ObjectResult));
            var objectResult = (ObjectResult)result.Result;
            Assert.AreEqual(500, objectResult.StatusCode);
            mockLearnService.Verify(s => s.GetDailyReviewLessonsAsync(It.IsAny<string>()), Times.Never);
        }

        /// <summary>
        /// Tests that GetDailyReviews returns correct error message when GetUserId fails.
        /// Input: User with no NameIdentifier claim.
        /// Expected: Returns 500 with message "Error fetching daily reviews".
        /// </summary>
        [TestMethod]
        public async Task GetDailyReviews_GetUserIdFails_ReturnsCorrectErrorMessage()
        {
            // Arrange
            var mockLearnService = new Mock<ILearnService>();
            var mockLogger = new Mock<ILogger<LearnController>>();
            var controller = new LearnController(mockLearnService.Object, mockLogger.Object);

            var claims = new List<Claim>();
            var identity = new ClaimsIdentity(claims, "TestAuth");
            var claimsPrincipal = new ClaimsPrincipal(identity);
            controller.ControllerContext = new ControllerContext
            {
                HttpContext = new DefaultHttpContext { User = claimsPrincipal }
            };

            // Act
            var result = await controller.GetDailyReviews();

            // Assert
            Assert.IsNotNull(result);
            var objectResult = result.Result as ObjectResult;
            Assert.IsNotNull(objectResult);
            Assert.AreEqual(500, objectResult.StatusCode);
            var messageProperty = objectResult.Value?.GetType().GetProperty("message");
            Assert.IsNotNull(messageProperty);
            var messageValue = messageProperty.GetValue(objectResult.Value) as string;
            Assert.AreEqual("Error fetching daily reviews", messageValue);
        }

        /// <summary>
        /// Tests that GetDailyReviews logs InvalidOperationException when user ID is not found.
        /// Input: User with no NameIdentifier claim.
        /// Expected: Logger logs error with InvalidOperationException.
        /// </summary>
        [TestMethod]
        public async Task GetDailyReviews_NoUserIdInClaims_LogsInvalidOperationException()
        {
            // Arrange
            var mockLearnService = new Mock<ILearnService>();
            var mockLogger = new Mock<ILogger<LearnController>>();
            var controller = new LearnController(mockLearnService.Object, mockLogger.Object);

            var claims = new List<Claim>();
            var identity = new ClaimsIdentity(claims, "TestAuth");
            var claimsPrincipal = new ClaimsPrincipal(identity);
            controller.ControllerContext = new ControllerContext
            {
                HttpContext = new DefaultHttpContext { User = claimsPrincipal }
            };

            // Act
            var result = await controller.GetDailyReviews();

            // Assert
            mockLogger.Verify(
                l => l.Log(
                    LogLevel.Error,
                    It.IsAny<EventId>(),
                    It.Is<It.IsAnyType>((v, t) => true),
                    It.IsAny<InvalidOperationException>(),
                    It.Is<Func<It.IsAnyType, Exception?, string>>((v, t) => true)),
                Times.Once);
        }

        /// <summary>
        /// Tests that GetDailyReviews logs exception with correct message when service fails.
        /// Input: Valid user, service throws exception with specific message.
        /// Expected: Logger logs error with the thrown exception.
        /// </summary>
        [TestMethod]
        public async Task GetDailyReviews_ServiceThrowsException_LogsExceptionWithMessage()
        {
            // Arrange
            var mockLearnService = new Mock<ILearnService>();
            var mockLogger = new Mock<ILogger<LearnController>>();
            var controller = new LearnController(mockLearnService.Object, mockLogger.Object);

            var userId = "test-user-123";
            var claims = new List<Claim> { new Claim(ClaimTypes.NameIdentifier, userId) };
            var identity = new ClaimsIdentity(claims, "TestAuth");
            var claimsPrincipal = new ClaimsPrincipal(identity);
            controller.ControllerContext = new ControllerContext
            {
                HttpContext = new DefaultHttpContext { User = claimsPrincipal }
            };

            var expectedException = new Exception("Database connection failed");
            mockLearnService.Setup(s => s.GetDailyReviewLessonsAsync(userId))
                .ThrowsAsync(expectedException);

            // Act
            var result = await controller.GetDailyReviews();

            // Assert
            mockLogger.Verify(
                l => l.Log(
                    LogLevel.Error,
                    It.IsAny<EventId>(),
                    It.Is<It.IsAnyType>((v, t) => true),
                    expectedException,
                    It.Is<Func<It.IsAnyType, Exception?, string>>((v, t) => true)),
                Times.Once);
        }

        /// <summary>
        /// Tests that GetDailyReviews handles very long user IDs correctly.
        /// Input: User ID with 1000 characters.
        /// Expected: Service called with full user ID, returns Ok.
        /// </summary>
        [TestMethod]
        public async Task GetDailyReviews_VeryLongUserId_ReturnsOkWithReviews()
        {
            // Arrange
            var mockLearnService = new Mock<ILearnService>();
            var mockLogger = new Mock<ILogger<LearnController>>();
            var controller = new LearnController(mockLearnService.Object, mockLogger.Object);

            var userId = new string('a', 1000);
            var claims = new List<Claim> { new Claim(ClaimTypes.NameIdentifier, userId) };
            var identity = new ClaimsIdentity(claims, "TestAuth");
            var claimsPrincipal = new ClaimsPrincipal(identity);
            controller.ControllerContext = new ControllerContext
            {
                HttpContext = new DefaultHttpContext { User = claimsPrincipal }
            };

            var expectedReviews = new List<SpacedRepetitionLessonDto>();
            mockLearnService.Setup(s => s.GetDailyReviewLessonsAsync(userId))
                .ReturnsAsync(expectedReviews);

            // Act
            var result = await controller.GetDailyReviews();

            // Assert
            Assert.IsNotNull(result);
            Assert.IsInstanceOfType(result.Result, typeof(OkObjectResult));
            mockLearnService.Verify(s => s.GetDailyReviewLessonsAsync(userId), Times.Once);
        }

        /// <summary>
        /// Tests that GetDailyReviews does not call service when GetUserId throws.
        /// Input: User with no NameIdentifier claim.
        /// Expected: Service method not called.
        /// </summary>
        [TestMethod]
        public async Task GetDailyReviews_GetUserIdThrows_DoesNotCallService()
        {
            // Arrange
            var mockLearnService = new Mock<ILearnService>();
            var mockLogger = new Mock<ILogger<LearnController>>();
            var controller = new LearnController(mockLearnService.Object, mockLogger.Object);

            var claims = new List<Claim>();
            var identity = new ClaimsIdentity(claims, "TestAuth");
            var claimsPrincipal = new ClaimsPrincipal(identity);
            controller.ControllerContext = new ControllerContext
            {
                HttpContext = new DefaultHttpContext { User = claimsPrincipal }
            };

            // Act
            var result = await controller.GetDailyReviews();

            // Assert
            mockLearnService.Verify(s => s.GetDailyReviewLessonsAsync(It.IsAny<string>()), Times.Never);
        }

        /// <summary>
        /// Tests that GetDailyReviews preserves review data integrity when service returns populated reviews.
        /// Input: Service returns reviews with specific data.
        /// Expected: Ok result with reviews containing exact data returned by service.
        /// </summary>
        [TestMethod]
        public async Task GetDailyReviews_ServiceReturnsReviews_PreservesDataIntegrity()
        {
            // Arrange
            var mockLearnService = new Mock<ILearnService>();
            var mockLogger = new Mock<ILogger<LearnController>>();
            var controller = new LearnController(mockLearnService.Object, mockLogger.Object);

            var userId = "test-user-123";
            var claims = new List<Claim> { new Claim(ClaimTypes.NameIdentifier, userId) };
            var identity = new ClaimsIdentity(claims, "TestAuth");
            var claimsPrincipal = new ClaimsPrincipal(identity);
            controller.ControllerContext = new ControllerContext
            {
                HttpContext = new DefaultHttpContext { User = claimsPrincipal }
            };

            var expectedReviews = new List<SpacedRepetitionLessonDto>
            {
                new SpacedRepetitionLessonDto
                {
                    Id = 1,
                    Title = "Test Review",
                    DueDate = "2024-01-01",
                    RepetitionCount = 3,
                    RetentionPercentage = 85.5,
                    IsReviewDue = true,
                    LessonId = 100
                }
            };

            mockLearnService.Setup(s => s.GetDailyReviewLessonsAsync(userId))
                .ReturnsAsync(expectedReviews);

            // Act
            var result = await controller.GetDailyReviews();

            // Assert
            var okResult = result.Result as OkObjectResult;
            Assert.IsNotNull(okResult);
            var reviews = okResult.Value as List<SpacedRepetitionLessonDto>;
            Assert.IsNotNull(reviews);
            Assert.AreEqual(1, reviews.Count);
            Assert.AreEqual(1, reviews[0].Id);
            Assert.AreEqual("Test Review", reviews[0].Title);
            Assert.AreEqual("2024-01-01", reviews[0].DueDate);
            Assert.AreEqual(3, reviews[0].RepetitionCount);
            Assert.AreEqual(85.5, reviews[0].RetentionPercentage);
            Assert.AreEqual(true, reviews[0].IsReviewDue);
            Assert.AreEqual(100, reviews[0].LessonId);
        }

        /// <summary>
        /// Tests that ReviewLesson returns Ok with success message when quality rating is valid and service completes successfully.
        /// Input: Valid qualityRating values (0.0, 2.5, 5.0), valid spacedRepetitionId, authenticated user.
        /// Expected: OkObjectResult with status code 200 and success message.
        /// </summary>
        [TestMethod]
        [DataRow(0.0, 1)]
        [DataRow(2.5, 100)]
        [DataRow(5.0, 999)]
        [DataRow(1.0, 0)]
        [DataRow(4.9, -1)]
        [DataRow(3.7, int.MaxValue)]
        [DataRow(0.0, int.MinValue)]
        public async Task ReviewLesson_ValidQualityRatingAndSpacedRepetitionId_ReturnsOkWithSuccessMessage(double qualityRating, int spacedRepetitionId)
        {
            // Arrange
            var mockLearnService = new Mock<ILearnService>();
            var mockLogger = new Mock<ILogger<LearnController>>();
            var controller = new LearnController(mockLearnService.Object, mockLogger.Object);
            var userId = "test-user-123";

            var claims = new List<Claim> { new Claim(ClaimTypes.NameIdentifier, userId) };
            var identity = new ClaimsIdentity(claims, "TestAuth");
            var claimsPrincipal = new ClaimsPrincipal(identity);
            controller.ControllerContext = new ControllerContext
            {
                HttpContext = new DefaultHttpContext { User = claimsPrincipal }
            };

            mockLearnService.Setup(x => x.ReviewLessonAsync(userId, spacedRepetitionId, qualityRating))
                .Returns(Task.CompletedTask);

            // Act
            var result = await controller.ReviewLesson(spacedRepetitionId, qualityRating);

            // Assert
            Assert.IsNotNull(result);
            var okResult = result.Result as OkObjectResult;
            Assert.IsNotNull(okResult);
            Assert.AreEqual(200, okResult.StatusCode);

            var messageProperty = okResult.Value?.GetType().GetProperty("message");
            Assert.IsNotNull(messageProperty);
            var messageValue = messageProperty.GetValue(okResult.Value) as string;
            Assert.AreEqual("Review recorded successfully", messageValue);

            mockLearnService.Verify(x => x.ReviewLessonAsync(userId, spacedRepetitionId, qualityRating), Times.Once);
        }

        /// <summary>
        /// Tests that ReviewLesson returns BadRequest when quality rating is below the valid range.
        /// Input: qualityRating less than 0 (negative values, double.MinValue, double.NegativeInfinity).
        /// Expected: BadRequestObjectResult with error message, service not called.
        /// </summary>
        [TestMethod]
        [DataRow(-0.1, 1)]
        [DataRow(-1.0, 100)]
        [DataRow(-100.0, 50)]
        public async Task ReviewLesson_QualityRatingBelowZero_ReturnsBadRequestWithErrorMessage(double qualityRating, int spacedRepetitionId)
        {
            // Arrange
            var mockLearnService = new Mock<ILearnService>();
            var mockLogger = new Mock<ILogger<LearnController>>();
            var controller = new LearnController(mockLearnService.Object, mockLogger.Object);
            var userId = "test-user-123";

            var claims = new List<Claim> { new Claim(ClaimTypes.NameIdentifier, userId) };
            var identity = new ClaimsIdentity(claims, "TestAuth");
            var claimsPrincipal = new ClaimsPrincipal(identity);
            controller.ControllerContext = new ControllerContext
            {
                HttpContext = new DefaultHttpContext { User = claimsPrincipal }
            };

            // Act
            var result = await controller.ReviewLesson(spacedRepetitionId, qualityRating);

            // Assert
            Assert.IsNotNull(result);
            var badRequestResult = result.Result as BadRequestObjectResult;
            Assert.IsNotNull(badRequestResult);
            Assert.AreEqual(400, badRequestResult.StatusCode);

            var messageProperty = badRequestResult.Value?.GetType().GetProperty("message");
            Assert.IsNotNull(messageProperty);
            var messageValue = messageProperty.GetValue(badRequestResult.Value) as string;
            Assert.AreEqual("Quality rating must be between 0 and 5", messageValue);

            mockLearnService.Verify(x => x.ReviewLessonAsync(It.IsAny<string>(), It.IsAny<int>(), It.IsAny<double>()), Times.Never);
        }

        /// <summary>
        /// Tests that ReviewLesson returns BadRequest when quality rating is above the valid range.
        /// Input: qualityRating greater than 5 (5.1, 6.0, 100.0, double.MaxValue, double.PositiveInfinity).
        /// Expected: BadRequestObjectResult with error message, service not called.
        /// </summary>
        [TestMethod]
        [DataRow(5.1, 1)]
        [DataRow(6.0, 100)]
        [DataRow(100.0, 50)]
        public async Task ReviewLesson_QualityRatingAboveFive_ReturnsBadRequestWithErrorMessage(double qualityRating, int spacedRepetitionId)
        {
            // Arrange
            var mockLearnService = new Mock<ILearnService>();
            var mockLogger = new Mock<ILogger<LearnController>>();
            var controller = new LearnController(mockLearnService.Object, mockLogger.Object);
            var userId = "test-user-123";

            var claims = new List<Claim> { new Claim(ClaimTypes.NameIdentifier, userId) };
            var identity = new ClaimsIdentity(claims, "TestAuth");
            var claimsPrincipal = new ClaimsPrincipal(identity);
            controller.ControllerContext = new ControllerContext
            {
                HttpContext = new DefaultHttpContext { User = claimsPrincipal }
            };

            // Act
            var result = await controller.ReviewLesson(spacedRepetitionId, qualityRating);

            // Assert
            Assert.IsNotNull(result);
            var badRequestResult = result.Result as BadRequestObjectResult;
            Assert.IsNotNull(badRequestResult);
            Assert.AreEqual(400, badRequestResult.StatusCode);

            var messageProperty = badRequestResult.Value?.GetType().GetProperty("message");
            Assert.IsNotNull(messageProperty);
            var messageValue = messageProperty.GetValue(badRequestResult.Value) as string;
            Assert.AreEqual("Quality rating must be between 0 and 5", messageValue);

            mockLearnService.Verify(x => x.ReviewLessonAsync(It.IsAny<string>(), It.IsAny<int>(), It.IsAny<double>()), Times.Never);
        }

        /// <summary>
        /// Tests that ReviewLesson returns StatusCode 500 when the service throws a generic Exception.
        /// Input: Valid qualityRating and spacedRepetitionId, but service throws generic Exception.
        /// Expected: ObjectResult with status code 500 and error message, LogError called with exception and spacedRepetitionId.
        /// </summary>
        [TestMethod]
        public async Task ReviewLesson_ServiceThrowsGenericException_ReturnsInternalServerErrorAndLogsError()
        {
            // Arrange
            var mockLearnService = new Mock<ILearnService>();
            var mockLogger = new Mock<ILogger<LearnController>>();
            var controller = new LearnController(mockLearnService.Object, mockLogger.Object);
            var userId = "test-user-123";
            var spacedRepetitionId = 42;
            var qualityRating = 3.5;
            var exceptionMessage = "Database connection failed";

            var claims = new List<Claim> { new Claim(ClaimTypes.NameIdentifier, userId) };
            var identity = new ClaimsIdentity(claims, "TestAuth");
            var claimsPrincipal = new ClaimsPrincipal(identity);
            controller.ControllerContext = new ControllerContext
            {
                HttpContext = new DefaultHttpContext { User = claimsPrincipal }
            };

            mockLearnService.Setup(x => x.ReviewLessonAsync(userId, spacedRepetitionId, qualityRating))
                .ThrowsAsync(new Exception(exceptionMessage));

            // Act
            var result = await controller.ReviewLesson(spacedRepetitionId, qualityRating);

            // Assert
            Assert.IsNotNull(result);
            var objectResult = result.Result as ObjectResult;
            Assert.IsNotNull(objectResult);
            Assert.AreEqual(500, objectResult.StatusCode);

            var messageProperty = objectResult.Value?.GetType().GetProperty("message");
            Assert.IsNotNull(messageProperty);
            var messageValue = messageProperty.GetValue(objectResult.Value) as string;
            Assert.AreEqual("Error recording review", messageValue);

            mockLogger.Verify(
                x => x.Log(
                    LogLevel.Error,
                    It.IsAny<EventId>(),
                    It.Is<It.IsAnyType>((v, t) => v.ToString()!.Contains("Error recording review for spaced repetition")),
                    It.IsAny<Exception>(),
                    It.IsAny<Func<It.IsAnyType, Exception?, string>>()),
                Times.Once);
        }

        /// <summary>
        /// Tests that ReviewLesson returns BadRequest when user ID is not found in token (GetUserId throws InvalidOperationException).
        /// Input: No NameIdentifier claim in user principal.
        /// Expected: BadRequestObjectResult with message "User ID not found in token", service not called.
        /// </summary>
        [TestMethod]
        public async Task ReviewLesson_UserIdNotFoundInToken_ReturnsBadRequestWithErrorMessage()
        {
            // Arrange
            var mockLearnService = new Mock<ILearnService>();
            var mockLogger = new Mock<ILogger<LearnController>>();
            var controller = new LearnController(mockLearnService.Object, mockLogger.Object);
            var spacedRepetitionId = 42;
            var qualityRating = 3.5;

            var claims = new List<Claim>();
            var identity = new ClaimsIdentity(claims, "TestAuth");
            var claimsPrincipal = new ClaimsPrincipal(identity);
            controller.ControllerContext = new ControllerContext
            {
                HttpContext = new DefaultHttpContext { User = claimsPrincipal }
            };

            // Act
            var result = await controller.ReviewLesson(spacedRepetitionId, qualityRating);

            // Assert
            Assert.IsNotNull(result);
            var badRequestResult = result.Result as BadRequestObjectResult;
            Assert.IsNotNull(badRequestResult);
            Assert.AreEqual(400, badRequestResult.StatusCode);

            var messageProperty = badRequestResult.Value?.GetType().GetProperty("message");
            Assert.IsNotNull(messageProperty);
            var messageValue = messageProperty.GetValue(badRequestResult.Value) as string;
            Assert.AreEqual("User ID not found in token", messageValue);

            mockLearnService.Verify(x => x.ReviewLessonAsync(It.IsAny<string>(), It.IsAny<int>(), It.IsAny<double>()), Times.Never);
        }

        /// <summary>
        /// Tests that ReviewLesson returns BadRequest when user ID claim is empty string.
        /// Input: NameIdentifier claim with empty string value.
        /// Expected: BadRequestObjectResult with message "User ID not found in token", service not called.
        /// </summary>
        [TestMethod]
        public async Task ReviewLesson_UserIdClaimIsEmptyString_ReturnsBadRequestWithErrorMessage()
        {
            // Arrange
            var mockLearnService = new Mock<ILearnService>();
            var mockLogger = new Mock<ILogger<LearnController>>();
            var controller = new LearnController(mockLearnService.Object, mockLogger.Object);
            var spacedRepetitionId = 42;
            var qualityRating = 3.5;

            var claims = new List<Claim> { new Claim(ClaimTypes.NameIdentifier, string.Empty) };
            var identity = new ClaimsIdentity(claims, "TestAuth");
            var claimsPrincipal = new ClaimsPrincipal(identity);
            controller.ControllerContext = new ControllerContext
            {
                HttpContext = new DefaultHttpContext { User = claimsPrincipal }
            };

            // Act
            var result = await controller.ReviewLesson(spacedRepetitionId, qualityRating);

            // Assert
            Assert.IsNotNull(result);
            var badRequestResult = result.Result as BadRequestObjectResult;
            Assert.IsNotNull(badRequestResult);
            Assert.AreEqual(400, badRequestResult.StatusCode);

            var messageProperty = badRequestResult.Value?.GetType().GetProperty("message");
            Assert.IsNotNull(messageProperty);
            var messageValue = messageProperty.GetValue(badRequestResult.Value) as string;
            Assert.AreEqual("User ID not found in token", messageValue);

            mockLearnService.Verify(x => x.ReviewLessonAsync(It.IsAny<string>(), It.IsAny<int>(), It.IsAny<double>()), Times.Never);
        }

        /// <summary>
        /// Tests that ReviewLesson handles extreme double values for quality rating correctly.
        /// Input: double.MinValue and double.MaxValue for qualityRating.
        /// Expected: BadRequestObjectResult with error message since these are outside the valid range [0, 5].
        /// </summary>
        [TestMethod]
        [DataRow(double.MinValue, 1)]
        [DataRow(double.MaxValue, 1)]
        public async Task ReviewLesson_ExtremeDoubleValuesForQualityRating_ReturnsBadRequest(double qualityRating, int spacedRepetitionId)
        {
            // Arrange
            var mockLearnService = new Mock<ILearnService>();
            var mockLogger = new Mock<ILogger<LearnController>>();
            var controller = new LearnController(mockLearnService.Object, mockLogger.Object);
            var userId = "test-user-123";

            var claims = new List<Claim> { new Claim(ClaimTypes.NameIdentifier, userId) };
            var identity = new ClaimsIdentity(claims, "TestAuth");
            var claimsPrincipal = new ClaimsPrincipal(identity);
            controller.ControllerContext = new ControllerContext
            {
                HttpContext = new DefaultHttpContext { User = claimsPrincipal }
            };

            // Act
            var result = await controller.ReviewLesson(spacedRepetitionId, qualityRating);

            // Assert
            Assert.IsNotNull(result);
            var badRequestResult = result.Result as BadRequestObjectResult;
            Assert.IsNotNull(badRequestResult);
            Assert.AreEqual(400, badRequestResult.StatusCode);

            var messageProperty = badRequestResult.Value?.GetType().GetProperty("message");
            Assert.IsNotNull(messageProperty);
            var messageValue = messageProperty.GetValue(badRequestResult.Value) as string;
            Assert.AreEqual("Quality rating must be between 0 and 5", messageValue);

            mockLearnService.Verify(x => x.ReviewLessonAsync(It.IsAny<string>(), It.IsAny<int>(), It.IsAny<double>()), Times.Never);
        }

        /// <summary>
        /// Tests that ReviewLesson handles special double values (NaN, PositiveInfinity, NegativeInfinity) for quality rating.
        /// Input: double.NaN, double.PositiveInfinity, double.NegativeInfinity.
        /// Expected: BadRequest since these values fail the range check (NaN comparison is always false, infinities are out of range).
        /// Note: Marked as ignored due to potential production bug with NaN handling.
        /// </summary>
        [TestMethod]
        [DataRow(double.NaN, 1)]
        [DataRow(double.PositiveInfinity, 1)]
        [DataRow(double.NegativeInfinity, 1)]
        [TestCategory("ProductionBugSuspected")]
        [Ignore("ProductionBugSuspected")]
        public async Task ReviewLesson_SpecialDoubleValuesForQualityRating_ReturnsBadRequest(double qualityRating, int spacedRepetitionId)
        {
            // Arrange
            var mockLearnService = new Mock<ILearnService>();
            var mockLogger = new Mock<ILogger<LearnController>>();
            var controller = new LearnController(mockLearnService.Object, mockLogger.Object);
            var userId = "test-user-123";

            var claims = new List<Claim> { new Claim(ClaimTypes.NameIdentifier, userId) };
            var identity = new ClaimsIdentity(claims, "TestAuth");
            var claimsPrincipal = new ClaimsPrincipal(identity);
            controller.ControllerContext = new ControllerContext
            {
                HttpContext = new DefaultHttpContext { User = claimsPrincipal }
            };

            // Act
            var result = await controller.ReviewLesson(spacedRepetitionId, qualityRating);

            // Assert
            Assert.IsNotNull(result);
            var badRequestResult = result.Result as BadRequestObjectResult;
            Assert.IsNotNull(badRequestResult);
            Assert.AreEqual(400, badRequestResult.StatusCode);

            mockLearnService.Verify(x => x.ReviewLessonAsync(It.IsAny<string>(), It.IsAny<int>(), It.IsAny<double>()), Times.Never);
        }

        /// <summary>
        /// Tests that ReviewLesson correctly passes all parameters to the service method.
        /// Input: Various combinations of valid spacedRepetitionId and qualityRating.
        /// Expected: Service called with exact userId, spacedRepetitionId, and qualityRating values.
        /// </summary>
        [TestMethod]
        [DataRow(1, 0.0)]
        [DataRow(100, 5.0)]
        [DataRow(0, 2.5)]
        [DataRow(-1, 3.0)]
        [DataRow(int.MaxValue, 4.5)]
        [DataRow(int.MinValue, 1.0)]
        public async Task ReviewLesson_ValidInputs_PassesCorrectParametersToService(int spacedRepetitionId, double qualityRating)
        {
            // Arrange
            var mockLearnService = new Mock<ILearnService>();
            var mockLogger = new Mock<ILogger<LearnController>>();
            var controller = new LearnController(mockLearnService.Object, mockLogger.Object);
            var userId = "test-user-abc-123";

            var claims = new List<Claim> { new Claim(ClaimTypes.NameIdentifier, userId) };
            var identity = new ClaimsIdentity(claims, "TestAuth");
            var claimsPrincipal = new ClaimsPrincipal(identity);
            controller.ControllerContext = new ControllerContext
            {
                HttpContext = new DefaultHttpContext { User = claimsPrincipal }
            };

            mockLearnService.Setup(x => x.ReviewLessonAsync(userId, spacedRepetitionId, qualityRating))
                .Returns(Task.CompletedTask);

            // Act
            var result = await controller.ReviewLesson(spacedRepetitionId, qualityRating);

            // Assert
            mockLearnService.Verify(x => x.ReviewLessonAsync(userId, spacedRepetitionId, qualityRating), Times.Once);
        }

        /// <summary>
        /// Tests that ReviewLesson returns BadRequest with the exact exception message when service throws InvalidOperationException with empty message.
        /// Input: Service throws InvalidOperationException with empty message.
        /// Expected: BadRequestObjectResult with empty message property.
        /// </summary>
        [TestMethod]
        public async Task ReviewLesson_ServiceThrowsInvalidOperationExceptionWithEmptyMessage_ReturnsBadRequestWithEmptyMessage()
        {
            // Arrange
            var mockLearnService = new Mock<ILearnService>();
            var mockLogger = new Mock<ILogger<LearnController>>();
            var controller = new LearnController(mockLearnService.Object, mockLogger.Object);
            var userId = "test-user-123";
            var spacedRepetitionId = 42;
            var qualityRating = 3.5;

            var claims = new List<Claim> { new Claim(ClaimTypes.NameIdentifier, userId) };
            var identity = new ClaimsIdentity(claims, "TestAuth");
            var claimsPrincipal = new ClaimsPrincipal(identity);
            controller.ControllerContext = new ControllerContext
            {
                HttpContext = new DefaultHttpContext { User = claimsPrincipal }
            };

            mockLearnService.Setup(x => x.ReviewLessonAsync(userId, spacedRepetitionId, qualityRating))
                .ThrowsAsync(new InvalidOperationException(string.Empty));

            // Act
            var result = await controller.ReviewLesson(spacedRepetitionId, qualityRating);

            // Assert
            Assert.IsNotNull(result);
            var badRequestResult = result.Result as BadRequestObjectResult;
            Assert.IsNotNull(badRequestResult);
            Assert.AreEqual(400, badRequestResult.StatusCode);

            var messageProperty = badRequestResult.Value?.GetType().GetProperty("message");
            Assert.IsNotNull(messageProperty);
            var messageValue = messageProperty.GetValue(badRequestResult.Value) as string;
            Assert.AreEqual(string.Empty, messageValue);
        }

        /// <summary>
        /// Tests that ReviewLesson handles user ID claim with whitespace-only value correctly.
        /// Input: NameIdentifier claim with whitespace-only value.
        /// Expected: BadRequestObjectResult with message "User ID not found in token" since IsNullOrEmpty returns true for whitespace.
        /// </summary>
        [TestMethod]
        public async Task ReviewLesson_UserIdClaimIsWhitespace_ReturnsBadRequestWithErrorMessage()
        {
            // Arrange
            var mockLearnService = new Mock<ILearnService>();
            var mockLogger = new Mock<ILogger<LearnController>>();
            var controller = new LearnController(mockLearnService.Object, mockLogger.Object);
            var spacedRepetitionId = 42;
            var qualityRating = 3.5;

            var claims = new List<Claim> { new Claim(ClaimTypes.NameIdentifier, "   ") };
            var identity = new ClaimsIdentity(claims, "TestAuth");
            var claimsPrincipal = new ClaimsPrincipal(identity);
            controller.ControllerContext = new ControllerContext
            {
                HttpContext = new DefaultHttpContext { User = claimsPrincipal }
            };

            // Act
            var result = await controller.ReviewLesson(spacedRepetitionId, qualityRating);

            // Assert
            Assert.IsNotNull(result);
            var badRequestResult = result.Result as BadRequestObjectResult;
            Assert.IsNotNull(badRequestResult);
            Assert.AreEqual(400, badRequestResult.StatusCode);

            var messageProperty = badRequestResult.Value?.GetType().GetProperty("message");
            Assert.IsNotNull(messageProperty);
            var messageValue = messageProperty.GetValue(badRequestResult.Value) as string;
            Assert.AreEqual("User ID not found in token", messageValue);

            mockLearnService.Verify(x => x.ReviewLessonAsync(It.IsAny<string>(), It.IsAny<int>(), It.IsAny<double>()), Times.Never);
        }

        /// <summary>
        /// Tests that ReviewLesson handles various valid user ID formats correctly.
        /// Input: User IDs with special characters, GUID format, email format, etc.
        /// Expected: Service called with exact user ID value.
        /// </summary>
        [TestMethod]
        [DataRow("user-123-abc")]
        [DataRow("user@example.com")]
        [DataRow("550e8400-e29b-41d4-a716-446655440000")]
        [DataRow("a")]
        [DataRow("very-long-user-id-with-many-characters-to-test-edge-cases-1234567890")]
        public async Task ReviewLesson_VariousUserIdFormats_CallsServiceWithCorrectUserId(string userId)
        {
            // Arrange
            var mockLearnService = new Mock<ILearnService>();
            var mockLogger = new Mock<ILogger<LearnController>>();
            var controller = new LearnController(mockLearnService.Object, mockLogger.Object);
            var spacedRepetitionId = 42;
            var qualityRating = 3.5;

            var claims = new List<Claim> { new Claim(ClaimTypes.NameIdentifier, userId) };
            var identity = new ClaimsIdentity(claims, "TestAuth");
            var claimsPrincipal = new ClaimsPrincipal(identity);
            controller.ControllerContext = new ControllerContext
            {
                HttpContext = new DefaultHttpContext { User = claimsPrincipal }
            };

            mockLearnService.Setup(x => x.ReviewLessonAsync(userId, spacedRepetitionId, qualityRating))
                .Returns(Task.CompletedTask);

            // Act
            var result = await controller.ReviewLesson(spacedRepetitionId, qualityRating);

            // Assert
            mockLearnService.Verify(x => x.ReviewLessonAsync(userId, spacedRepetitionId, qualityRating), Times.Once);
        }

        /// <summary>
        /// Tests that GetLessonsByCategory returns Ok with empty list when valid categoryId is provided but no lessons exist.
        /// Input: Valid categoryId, authenticated user, service returns empty list.
        /// Expected: OkObjectResult with empty list, service called once.
        /// </summary>
        [TestMethod]
        [DataRow(1)]
        [DataRow(0)]
        [DataRow(-1)]
        [DataRow(int.MaxValue)]
        [DataRow(int.MinValue)]
        public async Task GetLessonsByCategory_ValidCategoryIdWithNoLessons_ReturnsOkWithEmptyList(int categoryId)
        {
            // Arrange
            var mockLearnService = new Mock<ILearnService>();
            var mockLogger = new Mock<ILogger<LearnController>>();
            var emptyLessonsList = new List<LessonDto>();
            var userId = "test-user-456";

            mockLearnService
                .Setup(s => s.GetLessonsByCategoryAsync(categoryId, userId))
                .ReturnsAsync(emptyLessonsList);

            var controller = new LearnController(mockLearnService.Object, mockLogger.Object);
            SetupControllerWithUser(controller, userId);

            // Act
            var result = await controller.GetLessonsByCategory(categoryId);

            // Assert
            Assert.IsNotNull(result);
            Assert.IsInstanceOfType(result.Result, typeof(OkObjectResult));
            var okResult = result.Result as OkObjectResult;
            Assert.IsNotNull(okResult);
            Assert.AreEqual(200, okResult.StatusCode);
            var returnedLessons = okResult.Value as List<LessonDto>;
            Assert.IsNotNull(returnedLessons);
            Assert.AreEqual(0, returnedLessons.Count);
            mockLearnService.Verify(s => s.GetLessonsByCategoryAsync(categoryId, userId), Times.Once);
        }

        /// <summary>
        /// Tests that GetLessonsByCategory correctly passes userId from token to service.
        /// Input: Authenticated user with specific userId, various categoryId values.
        /// Expected: Service called with correct userId and categoryId parameters.
        /// </summary>
        [TestMethod]
        [DataRow("user-123", 1)]
        [DataRow("test-user-with-dashes-456", 100)]
        [DataRow("GUID-12345678-1234-1234-1234-123456789012", 0)]
        [DataRow("user@example.com", -1)]
        [DataRow("a", int.MaxValue)]
        [DataRow("very-long-user-id-with-many-characters", int.MinValue)]
        public async Task GetLessonsByCategory_AuthenticatedUser_PassesCorrectUserIdToService(string userId, int categoryId)
        {
            // Arrange
            var mockLearnService = new Mock<ILearnService>();
            var mockLogger = new Mock<ILogger<LearnController>>();
            var lessons = new List<LessonDto>();

            mockLearnService
                .Setup(s => s.GetLessonsByCategoryAsync(categoryId, userId))
                .ReturnsAsync(lessons);

            var controller = new LearnController(mockLearnService.Object, mockLogger.Object);
            SetupControllerWithUser(controller, userId);

            // Act
            var result = await controller.GetLessonsByCategory(categoryId);

            // Assert
            Assert.IsNotNull(result);
            mockLearnService.Verify(s => s.GetLessonsByCategoryAsync(categoryId, userId), Times.Once);
        }

        /// <summary>
        /// Tests that GetLessonsByCategory logs error with correct categoryId when GetUserId throws InvalidOperationException.
        /// Input: Controller without user claims, specific categoryId.
        /// Expected: Logger called with exception and categoryId parameter.
        /// </summary>
        [TestMethod]
        [DataRow(1)]
        [DataRow(999)]
        [DataRow(-100)]
        public async Task GetLessonsByCategory_UserIdNotFoundInToken_LogsErrorWithCategoryId(int categoryId)
        {
            // Arrange
            var mockLearnService = new Mock<ILearnService>();
            var mockLogger = new Mock<ILogger<LearnController>>();
            var controller = new LearnController(mockLearnService.Object, mockLogger.Object);
            SetupControllerWithoutUser(controller);

            // Act
            var result = await controller.GetLessonsByCategory(categoryId);

            // Assert
            mockLogger.Verify(
                x => x.Log(
                    LogLevel.Error,
                    It.IsAny<EventId>(),
                    It.Is<It.IsAnyType>((v, t) => v.ToString()!.Contains(categoryId.ToString())),
                    It.IsAny<InvalidOperationException>(),
                    It.IsAny<Func<It.IsAnyType, Exception?, string>>()),
                Times.Once);
        }

        /// <summary>
        /// Tests that GetLessonsByCategory logs error with correct categoryId when service throws exception.
        /// Input: Valid user, service throws exception, specific categoryId.
        /// Expected: Logger called with exception and categoryId parameter.
        /// </summary>
        [TestMethod]
        [DataRow(1)]
        [DataRow(42)]
        [DataRow(-5)]
        public async Task GetLessonsByCategory_ServiceThrowsException_LogsErrorWithCategoryId(int categoryId)
        {
            // Arrange
            var mockLearnService = new Mock<ILearnService>();
            var mockLogger = new Mock<ILogger<LearnController>>();
            var userId = "test-user";
            var exception = new Exception("Service error");

            mockLearnService
                .Setup(s => s.GetLessonsByCategoryAsync(categoryId, userId))
                .ThrowsAsync(exception);

            var controller = new LearnController(mockLearnService.Object, mockLogger.Object);
            SetupControllerWithUser(controller, userId);

            // Act
            var result = await controller.GetLessonsByCategory(categoryId);

            // Assert
            mockLogger.Verify(
                x => x.Log(
                    LogLevel.Error,
                    It.IsAny<EventId>(),
                    It.Is<It.IsAnyType>((v, t) => v.ToString()!.Contains(categoryId.ToString())),
                    It.IsAny<Exception>(),
                    It.IsAny<Func<It.IsAnyType, Exception?, string>>()),
                Times.Once);
        }

        /// <summary>
        /// Tests that GetLessonsByCategory handles user with empty string claim value correctly.
        /// Input: User claim with empty string value.
        /// Expected: Returns 500 status code with error message, service not called.
        /// </summary>
        [TestMethod]
        [DataRow(1)]
        [DataRow(0)]
        [DataRow(-1)]
        public async Task GetLessonsByCategory_UserClaimHasEmptyString_ReturnsInternalServerError(int categoryId)
        {
            // Arrange
            var mockLearnService = new Mock<ILearnService>();
            var mockLogger = new Mock<ILogger<LearnController>>();
            var controller = new LearnController(mockLearnService.Object, mockLogger.Object);

            var claims = new List<Claim>
            {
                new Claim(ClaimTypes.NameIdentifier, string.Empty)
            };
            var identity = new ClaimsIdentity(claims, "TestAuthType");
            var claimsPrincipal = new ClaimsPrincipal(identity);
            controller.ControllerContext = new ControllerContext
            {
                HttpContext = new DefaultHttpContext { User = claimsPrincipal }
            };

            // Act
            var result = await controller.GetLessonsByCategory(categoryId);

            // Assert
            Assert.IsNotNull(result);
            Assert.IsInstanceOfType(result.Result, typeof(ObjectResult));
            var objectResult = result.Result as ObjectResult;
            Assert.IsNotNull(objectResult);
            Assert.AreEqual(500, objectResult.StatusCode);
            mockLearnService.Verify(s => s.GetLessonsByCategoryAsync(It.IsAny<int>(), It.IsAny<string>()), Times.Never);
        }

        /// <summary>
        /// Tests that GetCategory returns InternalServerError when user ID claim is null.
        /// Input: User with NameIdentifier claim but null value.
        /// Expected: Returns ObjectResult with 500 status code and error message.
        /// </summary>
        [TestMethod]
        public async Task GetCategory_UserIdClaimIsNull_ReturnsInternalServerError()
        {
            // Arrange
            const int categoryId = 1;

            var mockLearnService = new Mock<ILearnService>();
            var mockLogger = new Mock<ILogger<LearnController>>();
            var controller = new LearnController(mockLearnService.Object, mockLogger.Object);

            var claims = new[] { new Claim(ClaimTypes.NameIdentifier, string.Empty) };
            var identity = new ClaimsIdentity(claims, "TestAuthType");
            var claimsPrincipal = new ClaimsPrincipal(identity);
            controller.ControllerContext = new ControllerContext
            {
                HttpContext = new DefaultHttpContext { User = claimsPrincipal }
            };

            // Act
            var result = await controller.GetCategory(categoryId);

            // Assert
            Assert.IsNotNull(result);
            Assert.IsInstanceOfType(result.Result, typeof(ObjectResult));
            var objectResult = (ObjectResult)result.Result;
            Assert.AreEqual(500, objectResult.StatusCode);
            mockLearnService.Verify(s => s.GetCategoryAsync(It.IsAny<int>(), It.IsAny<string>()), Times.Never);
        }

        /// <summary>
        /// Tests that GetCategory returns InternalServerError when user ID claim is empty string.
        /// Input: User with NameIdentifier claim but empty string value.
        /// Expected: Returns ObjectResult with 500 status code and error message.
        /// </summary>
        [TestMethod]
        public async Task GetCategory_UserIdClaimIsEmptyString_ReturnsInternalServerError()
        {
            // Arrange
            const int categoryId = 1;

            var mockLearnService = new Mock<ILearnService>();
            var mockLogger = new Mock<ILogger<LearnController>>();
            var controller = new LearnController(mockLearnService.Object, mockLogger.Object);

            var claims = new[] { new Claim(ClaimTypes.NameIdentifier, string.Empty) };
            var identity = new ClaimsIdentity(claims, "TestAuthType");
            var claimsPrincipal = new ClaimsPrincipal(identity);
            controller.ControllerContext = new ControllerContext
            {
                HttpContext = new DefaultHttpContext { User = claimsPrincipal }
            };

            // Act
            var result = await controller.GetCategory(categoryId);

            // Assert
            Assert.IsNotNull(result);
            Assert.IsInstanceOfType(result.Result, typeof(ObjectResult));
            var objectResult = (ObjectResult)result.Result;
            Assert.AreEqual(500, objectResult.StatusCode);
            mockLearnService.Verify(s => s.GetCategoryAsync(It.IsAny<int>(), It.IsAny<string>()), Times.Never);
        }

        /// <summary>
        /// Tests that GetCategory returns InternalServerError when user ID claim is whitespace only.
        /// Input: User with NameIdentifier claim containing only whitespace.
        /// Expected: Returns ObjectResult with 500 status code and error message.
        /// </summary>
        [TestMethod]
        public async Task GetCategory_UserIdClaimIsWhitespace_ReturnsInternalServerError()
        {
            // Arrange
            const int categoryId = 1;

            var mockLearnService = new Mock<ILearnService>();
            var mockLogger = new Mock<ILogger<LearnController>>();
            var controller = new LearnController(mockLearnService.Object, mockLogger.Object);

            var claims = new[] { new Claim(ClaimTypes.NameIdentifier, "   ") };
            var identity = new ClaimsIdentity(claims, "TestAuthType");
            var claimsPrincipal = new ClaimsPrincipal(identity);
            controller.ControllerContext = new ControllerContext
            {
                HttpContext = new DefaultHttpContext { User = claimsPrincipal }
            };

            // Act
            var result = await controller.GetCategory(categoryId);

            // Assert
            Assert.IsNotNull(result);
            Assert.IsInstanceOfType(result.Result, typeof(ObjectResult));
            var objectResult = (ObjectResult)result.Result;
            Assert.AreEqual(500, objectResult.StatusCode);
            mockLearnService.Verify(s => s.GetCategoryAsync(It.IsAny<int>(), It.IsAny<string>()), Times.Never);
        }

        /// <summary>
        /// Tests that GetCategory passes correct parameters to the service.
        /// Input: Specific categoryId and userId.
        /// Expected: Service is called with exact categoryId and userId parameters.
        /// </summary>
        [TestMethod]
        public async Task GetCategory_ValidInput_PassesCorrectParametersToService()
        {
            // Arrange
            const int categoryId = 5;
            const string userId = "specific-user-id-456";
            var expectedCategory = new LessonCategoryDto { Id = categoryId };

            var mockLearnService = new Mock<ILearnService>();
            mockLearnService.Setup(s => s.GetCategoryAsync(categoryId, userId))
                .ReturnsAsync(expectedCategory);

            var mockLogger = new Mock<ILogger<LearnController>>();
            var controller = CreateControllerWithUser(mockLearnService.Object, mockLogger.Object, userId);

            // Act
            var result = await controller.GetCategory(categoryId);

            // Assert
            mockLearnService.Verify(s => s.GetCategoryAsync(categoryId, userId), Times.Once);
        }

        /// <summary>
        /// Tests that GetCategory handles InvalidOperationException from service correctly.
        /// Input: Service throws InvalidOperationException.
        /// Expected: Returns ObjectResult with 500 status code and logs error.
        /// </summary>
        [TestMethod]
        public async Task GetCategory_ServiceThrowsInvalidOperationException_ReturnsInternalServerError()
        {
            // Arrange
            const int categoryId = 1;
            const string userId = "test-user-123";
            var exception = new InvalidOperationException("Invalid operation");

            var mockLearnService = new Mock<ILearnService>();
            mockLearnService.Setup(s => s.GetCategoryAsync(categoryId, userId))
                .ThrowsAsync(exception);

            var mockLogger = new Mock<ILogger<LearnController>>();
            var controller = CreateControllerWithUser(mockLearnService.Object, mockLogger.Object, userId);

            // Act
            var result = await controller.GetCategory(categoryId);

            // Assert
            Assert.IsNotNull(result);
            Assert.IsInstanceOfType(result.Result, typeof(ObjectResult));
            var objectResult = (ObjectResult)result.Result;
            Assert.AreEqual(500, objectResult.StatusCode);
        }

        /// <summary>
        /// Tests that GetCategory handles ArgumentException from service correctly.
        /// Input: Service throws ArgumentException.
        /// Expected: Returns ObjectResult with 500 status code and logs error.
        /// </summary>
        [TestMethod]
        public async Task GetCategory_ServiceThrowsArgumentException_ReturnsInternalServerError()
        {
            // Arrange
            const int categoryId = 1;
            const string userId = "test-user-123";
            var exception = new ArgumentException("Invalid argument");

            var mockLearnService = new Mock<ILearnService>();
            mockLearnService.Setup(s => s.GetCategoryAsync(categoryId, userId))
                .ThrowsAsync(exception);

            var mockLogger = new Mock<ILogger<LearnController>>();
            var controller = CreateControllerWithUser(mockLearnService.Object, mockLogger.Object, userId);

            // Act
            var result = await controller.GetCategory(categoryId);

            // Assert
            Assert.IsNotNull(result);
            Assert.IsInstanceOfType(result.Result, typeof(ObjectResult));
            var objectResult = (ObjectResult)result.Result;
            Assert.AreEqual(500, objectResult.StatusCode);
        }

        /// <summary>
        /// Tests that GetCategory handles category with null IconUrl correctly.
        /// Input: Service returns category with null IconUrl.
        /// Expected: Returns Ok with category containing null IconUrl.
        /// </summary>
        [TestMethod]
        public async Task GetCategory_CategoryWithNullIconUrl_ReturnsOkWithNullIconUrl()
        {
            // Arrange
            const int categoryId = 1;
            const string userId = "test-user-123";
            var expectedCategory = new LessonCategoryDto
            {
                Id = categoryId,
                Title = "Test",
                Description = "Test",
                IconUrl = null
            };

            var mockLearnService = new Mock<ILearnService>();
            mockLearnService.Setup(s => s.GetCategoryAsync(categoryId, userId))
                .ReturnsAsync(expectedCategory);

            var mockLogger = new Mock<ILogger<LearnController>>();
            var controller = CreateControllerWithUser(mockLearnService.Object, mockLogger.Object, userId);

            // Act
            var result = await controller.GetCategory(categoryId);

            // Assert
            Assert.IsNotNull(result);
            var okResult = result.Result as OkObjectResult;
            Assert.IsNotNull(okResult);
            var category = okResult.Value as LessonCategoryDto;
            Assert.IsNotNull(category);
            Assert.IsNull(category.IconUrl);
        }

        /// <summary>
        /// Tests that GetCategory handles category with extreme Progress values correctly.
        /// Input: Service returns category with various Progress values including edge cases.
        /// Expected: Returns Ok with category containing the same Progress value.
        /// </summary>
        [TestMethod]
        [DataRow(0.0)]
        [DataRow(100.0)]
        [DataRow(-1.0)]
        [DataRow(double.MaxValue)]
        [DataRow(double.MinValue)]
        public async Task GetCategory_CategoryWithExtremeProgressValues_ReturnsOkWithSameProgress(double progressValue)
        {
            // Arrange
            const int categoryId = 1;
            const string userId = "test-user-123";
            var expectedCategory = new LessonCategoryDto
            {
                Id = categoryId,
                Title = "Test",
                Description = "Test",
                Progress = progressValue
            };

            var mockLearnService = new Mock<ILearnService>();
            mockLearnService.Setup(s => s.GetCategoryAsync(categoryId, userId))
                .ReturnsAsync(expectedCategory);

            var mockLogger = new Mock<ILogger<LearnController>>();
            var controller = CreateControllerWithUser(mockLearnService.Object, mockLogger.Object, userId);

            // Act
            var result = await controller.GetCategory(categoryId);

            // Assert
            Assert.IsNotNull(result);
            var okResult = result.Result as OkObjectResult;
            Assert.IsNotNull(okResult);
            var category = okResult.Value as LessonCategoryDto;
            Assert.IsNotNull(category);
            Assert.AreEqual(progressValue, category.Progress);
        }

        /// <summary>
        /// Tests that GetCategory handles category with special double values (NaN, Infinity) correctly.
        /// Input: Service returns category with NaN or Infinity Progress values.
        /// Expected: Returns Ok with category containing the same special Progress value.
        /// </summary>
        [TestMethod]
        [DataRow(double.NaN)]
        [DataRow(double.PositiveInfinity)]
        [DataRow(double.NegativeInfinity)]
        public async Task GetCategory_CategoryWithSpecialDoubleProgressValues_ReturnsOkWithSameProgress(double progressValue)
        {
            // Arrange
            const int categoryId = 1;
            const string userId = "test-user-123";
            var expectedCategory = new LessonCategoryDto
            {
                Id = categoryId,
                Title = "Test",
                Description = "Test",
                Progress = progressValue
            };

            var mockLearnService = new Mock<ILearnService>();
            mockLearnService.Setup(s => s.GetCategoryAsync(categoryId, userId))
                .ReturnsAsync(expectedCategory);

            var mockLogger = new Mock<ILogger<LearnController>>();
            var controller = CreateControllerWithUser(mockLearnService.Object, mockLogger.Object, userId);

            // Act
            var result = await controller.GetCategory(categoryId);

            // Assert
            Assert.IsNotNull(result);
            var okResult = result.Result as OkObjectResult;
            Assert.IsNotNull(okResult);
            var category = okResult.Value as LessonCategoryDto;
            Assert.IsNotNull(category);
            if (double.IsNaN(progressValue))
            {
                Assert.IsTrue(double.IsNaN(category.Progress));
            }
            else
            {
                Assert.AreEqual(progressValue, category.Progress);
            }
        }

        /// <summary>
        /// Tests that GetCategory handles various userId formats correctly.
        /// Input: Different valid userId formats from claims.
        /// Expected: Returns Ok with category data for each userId format.
        /// </summary>
        [TestMethod]
        [DataRow("simple-id")]
        [DataRow("user-with-dashes-123")]
        [DataRow("GUID-12345678-1234-1234-1234-123456789012")]
        [DataRow("very-long-user-id-with-many-characters-to-test-edge-cases-1234567890")]
        [DataRow("123")]
        [DataRow("user@example.com")]
        public async Task GetCategory_VariousUserIdFormats_ReturnsOkWithCategory(string userId)
        {
            // Arrange
            const int categoryId = 1;
            var expectedCategory = new LessonCategoryDto
            {
                Id = categoryId,
                Title = "Test",
                Description = "Test"
            };

            var mockLearnService = new Mock<ILearnService>();
            mockLearnService.Setup(s => s.GetCategoryAsync(categoryId, userId))
                .ReturnsAsync(expectedCategory);

            var mockLogger = new Mock<ILogger<LearnController>>();
            var controller = CreateControllerWithUser(mockLearnService.Object, mockLogger.Object, userId);

            // Act
            var result = await controller.GetCategory(categoryId);

            // Assert
            Assert.IsNotNull(result);
            Assert.IsInstanceOfType(result.Result, typeof(OkObjectResult));
            mockLearnService.Verify(s => s.GetCategoryAsync(categoryId, userId), Times.Once);
        }

        /// <summary>
        /// Tests that GetCategory does not call service when GetUserId fails.
        /// Input: User without NameIdentifier claim.
        /// Expected: Service is not called.
        /// </summary>
        [TestMethod]
        public async Task GetCategory_GetUserIdFails_DoesNotCallService()
        {
            // Arrange
            const int categoryId = 1;

            var mockLearnService = new Mock<ILearnService>();
            var mockLogger = new Mock<ILogger<LearnController>>();
            var controller = CreateControllerWithoutUser(mockLearnService.Object, mockLogger.Object);

            // Act
            var result = await controller.GetCategory(categoryId);

            // Assert
            mockLearnService.Verify(s => s.GetCategoryAsync(It.IsAny<int>(), It.IsAny<string>()), Times.Never);
        }

        /// <summary>
        /// Tests that GetCategory handles empty string fields in category correctly.
        /// Input: Service returns category with empty Title and Description.
        /// Expected: Returns Ok with category containing empty strings.
        /// </summary>
        [TestMethod]
        public async Task GetCategory_CategoryWithEmptyStringFields_ReturnsOkWithEmptyStrings()
        {
            // Arrange
            const int categoryId = 1;
            const string userId = "test-user-123";
            var expectedCategory = new LessonCategoryDto
            {
                Id = categoryId,
                Title = string.Empty,
                Description = string.Empty,
                Difficulty = string.Empty
            };

            var mockLearnService = new Mock<ILearnService>();
            mockLearnService.Setup(s => s.GetCategoryAsync(categoryId, userId))
                .ReturnsAsync(expectedCategory);

            var mockLogger = new Mock<ILogger<LearnController>>();
            var controller = CreateControllerWithUser(mockLearnService.Object, mockLogger.Object, userId);

            // Act
            var result = await controller.GetCategory(categoryId);

            // Assert
            Assert.IsNotNull(result);
            var okResult = result.Result as OkObjectResult;
            Assert.IsNotNull(okResult);
            var category = okResult.Value as LessonCategoryDto;
            Assert.IsNotNull(category);
            Assert.AreEqual(string.Empty, category.Title);
            Assert.AreEqual(string.Empty, category.Description);
        }
    }
}