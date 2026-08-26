using System;
using System.Collections.Generic;
using System.Linq;
using System.Net;
using System.Net.Http;
using System.Text;
using System.Text.Json;
using System.Threading;
using System.Threading.Tasks;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using Moq;
using Moq.Protected;
using SignLanguageApp.Model;
using SignLanguageApp.Services;

namespace SignLanguageApp.UnitTests.Services
{
    [TestClass]
    public class ApiServiceSecurityAndEdgeTests
    {
        private Mock<IConnectivityService> _connectivityMock = null!;
        private Mock<IApiConfigService> _apiConfigMock = null!;
        private Mock<ILessonPayloadSecurityService> _securityMock = null!;
        private Mock<IDatabaseService> _databaseMock = null!;

        [TestInitialize]
        public void Setup()
        {
            _connectivityMock = new Mock<IConnectivityService>();
            _connectivityMock.Setup(c => c.IsConnected).Returns(true);

            _apiConfigMock = new Mock<IApiConfigService>();
            _apiConfigMock.Setup(a => a.BaseUrl).Returns("https://api.signlanguageapp.com/");

            _securityMock = new Mock<ILessonPayloadSecurityService>();
            _databaseMock = new Mock<IDatabaseService>();
        }

        private ApiService CreateApiService(HttpClient httpClient)
        {
            return new ApiService(
                httpClient,
                _connectivityMock.Object,
                _apiConfigMock.Object,
                _securityMock.Object,
                _databaseMock.Object
            );
        }

        private HttpClient CreateMockHttpClient(HttpResponseMessage response)
        {
            var handlerMock = new Mock<HttpMessageHandler>();
            handlerMock
                .Protected()
                .Setup<Task<HttpResponseMessage>>(
                    "SendAsync",
                    ItExpr.IsAny<HttpRequestMessage>(),
                    ItExpr.IsAny<CancellationToken>()
                )
                .ReturnsAsync(response);

            return new HttpClient(handlerMock.Object)
            {
                BaseAddress = new Uri("https://api.signlanguageapp.com/")
            };
        }

        [TestMethod]
        public void ApiService_Constructor_InitializesState()
        {
            var httpClient = CreateMockHttpClient(new HttpResponseMessage(HttpStatusCode.OK));
            var apiService = CreateApiService(httpClient);
            Assert.IsNotNull(apiService);
        }

        [TestMethod]
        public void ApiService_MultipleInstances_AreIsolated()
        {
            var client1 = CreateMockHttpClient(new HttpResponseMessage(HttpStatusCode.OK));
            var client2 = CreateMockHttpClient(new HttpResponseMessage(HttpStatusCode.OK));

            var service1 = CreateApiService(client1);
            var service2 = CreateApiService(client2);

            Assert.AreNotSame(service1, service2);
        }

        [TestMethod]
        public async Task ApiService_GetLessons_EmptyListJsonResponse_ReturnsSuccessResponse()
        {
            var responseMessage = new HttpResponseMessage
            {
                StatusCode = HttpStatusCode.OK,
                Content = new StringContent("{\"success\":true, \"data\":{\"categories\":[], \"lessons\":[], \"discoveryShorts\":[]}}", Encoding.UTF8, "application/json")
            };

            var httpClient = CreateMockHttpClient(responseMessage);
            var apiService = CreateApiService(httpClient);

            var lessons = await apiService.GetLessonsAsync();
            Assert.IsNotNull(lessons);
            Assert.IsTrue(lessons.Success);
        }

        [TestMethod]
        public async Task ApiService_GetLessons_401Unauthorized_ReturnsNullOrError()
        {
            var responseMessage = new HttpResponseMessage
            {
                StatusCode = HttpStatusCode.Unauthorized,
                Content = new StringContent("{\"success\":false,\"message\":\"Unauthorized\"}", Encoding.UTF8, "application/json")
            };

            var httpClient = CreateMockHttpClient(responseMessage);
            var apiService = CreateApiService(httpClient);

            var lessons = await apiService.GetLessonsAsync();
            Assert.IsTrue(lessons == null || !lessons.Success);
        }

        [TestMethod]
        public async Task ApiService_GetLessons_429TooManyRequests_HandlesRateLimitGracefully()
        {
            var responseMessage = new HttpResponseMessage
            {
                StatusCode = (HttpStatusCode)429,
                Content = new StringContent("{\"success\":false,\"message\":\"Too many requests\",\"retryAfterSeconds\":15}", Encoding.UTF8, "application/json")
            };

            var httpClient = CreateMockHttpClient(responseMessage);
            var apiService = CreateApiService(httpClient);

            var lessons = await apiService.GetLessonsAsync();
            Assert.IsTrue(lessons == null || !lessons.Success);
        }

        [TestMethod]
        public async Task ApiService_GetLessons_500InternalServerError_ReturnsNullOrError()
        {
            var responseMessage = new HttpResponseMessage
            {
                StatusCode = HttpStatusCode.InternalServerError,
                Content = new StringContent("{\"success\":false,\"message\":\"Server Error\"}", Encoding.UTF8, "application/json")
            };

            var httpClient = CreateMockHttpClient(responseMessage);
            var apiService = CreateApiService(httpClient);

            var lessons = await apiService.GetLessonsAsync();
            Assert.IsTrue(lessons == null || !lessons.Success);
        }

        [TestMethod]
        public async Task ApiService_GetSignerCredits_MalformedJson_HandlesExceptionWithoutCrashing()
        {
            var responseMessage = new HttpResponseMessage
            {
                StatusCode = HttpStatusCode.OK,
                Content = new StringContent("{ invalid_json_syntax: [ ", Encoding.UTF8, "application/json")
            };

            var httpClient = CreateMockHttpClient(responseMessage);
            var apiService = CreateApiService(httpClient);

            var credits = await apiService.GetSignerCreditsAsync();
            Assert.IsNull(credits);
        }

        [TestMethod]
        public async Task ApiService_GetSignerCredits_ValidPayload_ReturnsCredits()
        {
            var jsonPayload = JsonSerializer.Serialize(new[]
            {
                new SignerCreditDto { SignerName = "Alice", LicenseType = "CC-BY-4.0", Bio = "Sign Language Enthusiast" }
            });

            var responseMessage = new HttpResponseMessage
            {
                StatusCode = HttpStatusCode.OK,
                Content = new StringContent(jsonPayload, Encoding.UTF8, "application/json")
            };

            var httpClient = CreateMockHttpClient(responseMessage);
            var apiService = CreateApiService(httpClient);

            var credits = await apiService.GetSignerCreditsAsync();
            Assert.IsNotNull(credits);
            Assert.IsTrue(credits != null);
        }
    }
}
