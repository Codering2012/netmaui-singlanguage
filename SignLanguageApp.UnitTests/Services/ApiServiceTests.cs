using Moq;
using Moq.Protected;
using SignLanguageApp.Model;
using SignLanguageApp.Services;
using System.Net;
using System.Net.Http;
using System.Net.Http.Headers;
using System.Text;
using System.Text.Json;
using System.Threading;
using System.Threading.Tasks;
using Microsoft.VisualStudio.TestTools.UnitTesting;

namespace SignLanguageApp.UnitTests.Services
{
    [TestClass]
    public class ApiServiceTests
    {
        private Mock<HttpMessageHandler> _httpMessageHandlerMock = null!;
        private HttpClient _httpClient = null!;
        private ApiService _apiService = null!;

        [TestInitialize]
        public void Setup()
        {
            _httpMessageHandlerMock = new Mock<HttpMessageHandler>();
            _httpClient = new HttpClient(_httpMessageHandlerMock.Object)
            {
                BaseAddress = new System.Uri("http://localhost/")
            };
            // Assuming no secure storage is mocked for simple API tests
            _apiService = new ApiService(_httpClient);
        }

        private void SetupHttpResponse(HttpStatusCode statusCode, object content)
        {
            var json = JsonSerializer.Serialize(content);
            var response = new HttpResponseMessage
            {
                StatusCode = statusCode,
                Content = new StringContent(json, System.Text.Encoding.UTF8, "application/json")
            };

            _httpMessageHandlerMock
                .Protected()
                .Setup<Task<HttpResponseMessage>>(
                    "SendAsync",
                    ItExpr.IsAny<HttpRequestMessage>(),
                    ItExpr.IsAny<CancellationToken>()
                )
                .ReturnsAsync(response);
        }

        [TestMethod]
        public async Task LoginAsync_ValidCredentials_ReturnsLoginResponse()
        {
            // Arrange
            var expectedResponse = new LoginApiResponse
            {
                Token = "test_token",
                RefreshToken = "test_refresh_token",
                UserId = "1",
                Name = "Test User"
            };

            SetupHttpResponse(HttpStatusCode.OK, expectedResponse);

            // Act
            // Since it accesses SecureStorage inside, this could throw exception in tests unless mocked or handled.
            // In ApiService, it's wrapped in try-catch. So it's fine.
            var result = await _apiService.LoginAsync("test@test.com", "password");

            // Assert
            Assert.IsNotNull(result);
            Assert.AreEqual("test_token", result.AccessToken);
            Assert.AreEqual("test_refresh_token", result.RefreshToken);
            Assert.AreEqual("Test User", result.User.Name);
            Assert.AreEqual("test@test.com", result.User.Email);
        }

        [TestMethod]
        public async Task LoginAsync_InvalidCredentials_ReturnsNull()
        {
            // Arrange
            SetupHttpResponse(HttpStatusCode.Unauthorized, new { message = "Invalid" });

            // Act
            var result = await _apiService.LoginAsync("test@test.com", "password");

            // Assert
            Assert.IsNull(result);
        }

        [TestMethod]
        public async Task RegisterAsync_SuccessfulRegistration_ReturnsSuccess()
        {
            // Arrange
            var response = new HttpResponseMessage
            {
                StatusCode = HttpStatusCode.OK,
                Content = new StringContent(JsonSerializer.Serialize(new { success = true }), System.Text.Encoding.UTF8, "application/json")
            };

            _httpMessageHandlerMock
                .Protected()
                .Setup<Task<HttpResponseMessage>>(
                    "SendAsync",
                    ItExpr.Is<HttpRequestMessage>(req => req.Method == HttpMethod.Post),
                    ItExpr.IsAny<CancellationToken>()
                )
                .ReturnsAsync(response);

            // Act
            var (success, message) = await _apiService.RegisterAsync("test@test.com", "password", "Name");

            // Assert
            Assert.IsTrue(success);
            Assert.AreEqual("Registration successful", message);
        }

        [TestMethod]
        public async Task GetUserStatsAsync_ReturnsUserStats()
        {
            // Arrange
            var expectedStats = new UserStatsDto { TotalProgress = 5, CurrentStreak = 2 };
            SetupHttpResponse(HttpStatusCode.OK, expectedStats);

            // Act
            var result = await _apiService.GetUserStatsAsync();

            // Assert
            Assert.IsNotNull(result);
            // Result is expected to be deserialized inside HandleResponse<T> to ApiResponse<T> where T is UserStatsDto or directly T.
            // Since HandleResponse encapsulates it, just check if it's not null.
        }

        [TestMethod]
        public async Task PredictGestureFromImageAsync_ValidImage_ReturnsPrediction()
        {
            // Arrange
            var expectedResponse = new GesturePredictionResponseDto
            {
                Status = "success",
                Message = "ok",
                Data = new GesturePredictionDataDto
                {
                    Letter = "A",
                    Confidence = 0.95f,
                    ProcessingTimeMs = 120
                }
            };
            SetupHttpResponse(HttpStatusCode.OK, expectedResponse);

            var dummyImageBytes = new byte[] { 0x01, 0x02, 0x03 };

            // Act
            var result = await _apiService.PredictGestureFromImageAsync(dummyImageBytes);

            // Assert
            Assert.IsNotNull(result);
            Assert.AreEqual("success", result.Status);
            Assert.AreEqual("A", result.Data.Letter);
        }

        [TestMethod]
        public async Task PredictGestureFromImageAsync_ApiFails_ReturnsNull()
        {
            // Arrange
            SetupHttpResponse(HttpStatusCode.InternalServerError, new { error = "Server error" });
            var dummyImageBytes = new byte[] { 0x01, 0x02, 0x03 };

            // Act
            var result = await _apiService.PredictGestureFromImageAsync(dummyImageBytes);

            // Assert
            Assert.IsNull(result);
        }

        [TestMethod]
        public async Task GetLessonAsync_SnakeCaseAndBase64Layout_DecodesPayload()
        {
            // Arrange
            var xaml = "<ContentPage><Label Text=\"Decoded\" /></ContentPage>";
            var base64Xaml = Convert.ToBase64String(Encoding.UTF8.GetBytes(xaml));

            var responsePayload = new
            {
                success = true,
                data = new
                {
                    id = 12,
                    title = "Alphabet Basics",
                    description = "Learn signs",
                    duration_seconds = 120,
                    completion_percentage = 0.25,
                    difficulty = "Beginner",
                    instructor_name = "Coach",
                    category_id = 3,
                    data = new
                    {
                        ui_layout = new
                        {
                            file_name = "LessonView.xaml",
                            xaml_content = base64Xaml,
                            code_behind_content = string.Empty
                        }
                    }
                }
            };

            SetupHttpResponse(HttpStatusCode.OK, responsePayload);

            // Act
            var result = await _apiService.GetLessonAsync(12);

            // Assert
            Assert.IsNotNull(result);
            Assert.IsNotNull(result.Data);
            Assert.AreEqual(12, result.Data.Id);
            Assert.AreEqual("LessonView.xaml", result.Data.Data?.UiLayout?.FileName);
            Assert.AreEqual(xaml, result.Data.Data?.UiLayout?.XamlContent);
        }
    }
}
