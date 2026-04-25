using System;
using System.IO;
using System.Threading;
using System.Threading.Tasks;

using Microsoft.AspNetCore.Http;
using Microsoft.AspNetCore.Mvc;
using Microsoft.Extensions.Logging;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using Moq;
using SignLanguageApi.Controllers;
using SignLanguageApi.Dtos;
using SignLanguageApi.Services;

namespace SignLanguageApi.Controllers.UnitTests
{
    /// <summary>
    /// Unit tests for the GestureController class.
    /// </summary>
    [TestClass]
    public class GestureControllerTests
    {
        /// <summary>
        /// Tests that the constructor successfully creates an instance when provided with valid dependencies.
        /// Input: Valid mocked IGestureRecognitionService and ILogger instances.
        /// Expected: Constructor completes successfully and instance is created.
        /// </summary>
        [TestMethod]
        public void GestureController_ValidDependencies_CreatesInstance()
        {
            // Arrange
            Mock<IGestureRecognitionService> mockGestureService = new Mock<IGestureRecognitionService>();
            Mock<ILogger<GestureController>> mockLogger = new Mock<ILogger<GestureController>>();

            // Act
            GestureController controller = new GestureController(mockGestureService.Object, mockLogger.Object);

            // Assert
            Assert.IsNotNull(controller);
        }

        /// <summary>
        /// Tests that Predict returns BadRequest when image is null
        /// </summary>
        [TestMethod]
        public async Task Predict_NullImage_ReturnsBadRequest()
        {
            // Arrange
            var mockGestureService = new Mock<IGestureRecognitionService>();
            var mockLogger = new Mock<ILogger<GestureController>>();
            var controller = new GestureController(mockGestureService.Object, mockLogger.Object);

            // Act
            var result = await controller.Predict(null!, CancellationToken.None);

            // Assert
            Assert.IsNotNull(result);
            Assert.IsInstanceOfType(result.Result, typeof(BadRequestObjectResult));
            var badRequestResult = (BadRequestObjectResult)result.Result!;
            var response = badRequestResult.Value as GesturePredictionResponseDto;
            Assert.IsNotNull(response);
            Assert.AreEqual("error", response.Status);
            Assert.AreEqual("No image file provided. Please upload a JPEG image.", response.Message);
            Assert.IsNull(response.Data);
        }

        /// <summary>
        /// Tests that Predict returns BadRequest when image length is zero
        /// </summary>
        [TestMethod]
        public async Task Predict_EmptyImage_ReturnsBadRequest()
        {
            // Arrange
            var mockGestureService = new Mock<IGestureRecognitionService>();
            var mockLogger = new Mock<ILogger<GestureController>>();
            var controller = new GestureController(mockGestureService.Object, mockLogger.Object);
            var mockImage = new Mock<IFormFile>();
            mockImage.Setup(x => x.Length).Returns(0);

            // Act
            var result = await controller.Predict(mockImage.Object, CancellationToken.None);

            // Assert
            Assert.IsNotNull(result);
            Assert.IsInstanceOfType(result.Result, typeof(BadRequestObjectResult));
            var badRequestResult = (BadRequestObjectResult)result.Result!;
            var response = badRequestResult.Value as GesturePredictionResponseDto;
            Assert.IsNotNull(response);
            Assert.AreEqual("error", response.Status);
            Assert.AreEqual("No image file provided. Please upload a JPEG image.", response.Message);
            Assert.IsNull(response.Data);
        }

        /// <summary>
        /// Tests that Predict returns BadRequest when image size exceeds 5MB limit
        /// </summary>
        [TestMethod]
        [DataRow(5 * 1024 * 1024 + 1)]
        [DataRow(10 * 1024 * 1024)]
        [DataRow(long.MaxValue)]
        public async Task Predict_OversizedImage_ReturnsBadRequest(long imageSize)
        {
            // Arrange
            var mockGestureService = new Mock<IGestureRecognitionService>();
            var mockLogger = new Mock<ILogger<GestureController>>();
            var controller = new GestureController(mockGestureService.Object, mockLogger.Object);
            var mockImage = new Mock<IFormFile>();
            mockImage.Setup(x => x.Length).Returns(imageSize);
            mockImage.Setup(x => x.ContentType).Returns("image/jpeg");

            // Act
            var result = await controller.Predict(mockImage.Object, CancellationToken.None);

            // Assert
            Assert.IsNotNull(result);
            Assert.IsInstanceOfType(result.Result, typeof(BadRequestObjectResult));
            var badRequestResult = (BadRequestObjectResult)result.Result!;
            var response = badRequestResult.Value as GesturePredictionResponseDto;
            Assert.IsNotNull(response);
            Assert.AreEqual("error", response.Status);
            Assert.IsTrue(response.Message.Contains("Image size must be less than 5MB"));
            Assert.IsNull(response.Data);
        }

        /// <summary>
        /// Tests that Predict returns BadRequest for invalid content types
        /// </summary>
        [TestMethod]
        [DataRow("image/png")]
        [DataRow("image/gif")]
        [DataRow("text/plain")]
        [DataRow("application/json")]
        [DataRow("")]
        [DataRow("invalid")]
        public async Task Predict_InvalidContentType_ReturnsBadRequest(string contentType)
        {
            // Arrange
            var mockGestureService = new Mock<IGestureRecognitionService>();
            var mockLogger = new Mock<ILogger<GestureController>>();
            var controller = new GestureController(mockGestureService.Object, mockLogger.Object);
            var mockImage = new Mock<IFormFile>();
            mockImage.Setup(x => x.Length).Returns(1024);
            mockImage.Setup(x => x.ContentType).Returns(contentType);

            // Act
            var result = await controller.Predict(mockImage.Object, CancellationToken.None);

            // Assert
            Assert.IsNotNull(result);
            Assert.IsInstanceOfType(result.Result, typeof(BadRequestObjectResult));
            var badRequestResult = (BadRequestObjectResult)result.Result!;
            var response = badRequestResult.Value as GesturePredictionResponseDto;
            Assert.IsNotNull(response);
            Assert.AreEqual("error", response.Status);
            Assert.AreEqual("Invalid file type. Please upload a JPEG image.", response.Message);
            Assert.IsNull(response.Data);
        }

        /// <summary>
        /// Tests that Predict accepts valid JPEG content type with different casing
        /// </summary>
        [TestMethod]
        [DataRow("image/jpeg")]
        [DataRow("IMAGE/JPEG")]
        [DataRow("Image/Jpeg")]
        [DataRow("image/JPEG")]
        public async Task Predict_ValidJpegContentType_ProcessesSuccessfully(string contentType)
        {
            // Arrange
            var mockGestureService = new Mock<IGestureRecognitionService>();
            var mockLogger = new Mock<ILogger<GestureController>>();
            var controller = new GestureController(mockGestureService.Object, mockLogger.Object);
            var imageData = new byte[] { 0xFF, 0xD8, 0xFF, 0xE0 };
            var mockImage = CreateMockFormFile(imageData, contentType);
            var expectedResponse = new GesturePredictionResponseDto
            {
                Status = "success",
                Message = "Gesture predicted",
                Data = new GesturePredictionDataDto()
            };
            mockGestureService.Setup(x => x.PredictGestureAsync(It.IsAny<byte[]>(), It.IsAny<CancellationToken>()))
                .ReturnsAsync(expectedResponse);

            // Act
            var result = await controller.Predict(mockImage.Object, CancellationToken.None);

            // Assert
            Assert.IsNotNull(result);
            Assert.IsInstanceOfType(result.Result, typeof(OkObjectResult));
            var okResult = (OkObjectResult)result.Result!;
            Assert.AreEqual(expectedResponse, okResult.Value);
        }

        /// <summary>
        /// Tests that Predict accepts application/octet-stream content type
        /// </summary>
        [TestMethod]
        [DataRow("application/octet-stream")]
        [DataRow("APPLICATION/OCTET-STREAM")]
        [DataRow("Application/Octet-Stream")]
        public async Task Predict_OctetStreamContentType_ProcessesSuccessfully(string contentType)
        {
            // Arrange
            var mockGestureService = new Mock<IGestureRecognitionService>();
            var mockLogger = new Mock<ILogger<GestureController>>();
            var controller = new GestureController(mockGestureService.Object, mockLogger.Object);
            var imageData = new byte[] { 0xFF, 0xD8, 0xFF, 0xE0 };
            var mockImage = CreateMockFormFile(imageData, contentType);
            var expectedResponse = new GesturePredictionResponseDto
            {
                Status = "success",
                Message = "Gesture predicted",
                Data = new GesturePredictionDataDto()
            };
            mockGestureService.Setup(x => x.PredictGestureAsync(It.IsAny<byte[]>(), It.IsAny<CancellationToken>()))
                .ReturnsAsync(expectedResponse);

            // Act
            var result = await controller.Predict(mockImage.Object, CancellationToken.None);

            // Assert
            Assert.IsNotNull(result);
            Assert.IsInstanceOfType(result.Result, typeof(OkObjectResult));
            var okResult = (OkObjectResult)result.Result!;
            Assert.AreEqual(expectedResponse, okResult.Value);
        }

        /// <summary>
        /// Tests that Predict processes image at exact 5MB boundary successfully
        /// </summary>
        [TestMethod]
        public async Task Predict_ImageAtExactMaxSize_ProcessesSuccessfully()
        {
            // Arrange
            var mockGestureService = new Mock<IGestureRecognitionService>();
            var mockLogger = new Mock<ILogger<GestureController>>();
            var controller = new GestureController(mockGestureService.Object, mockLogger.Object);
            var exactMaxSize = 5 * 1024 * 1024;
            var imageData = new byte[exactMaxSize];
            var mockImage = CreateMockFormFile(imageData, "image/jpeg");
            var expectedResponse = new GesturePredictionResponseDto
            {
                Status = "success",
                Message = "Gesture predicted",
                Data = new GesturePredictionDataDto()
            };
            mockGestureService.Setup(x => x.PredictGestureAsync(It.IsAny<byte[]>(), It.IsAny<CancellationToken>()))
                .ReturnsAsync(expectedResponse);

            // Act
            var result = await controller.Predict(mockImage.Object, CancellationToken.None);

            // Assert
            Assert.IsNotNull(result);
            Assert.IsInstanceOfType(result.Result, typeof(OkObjectResult));
        }

        /// <summary>
        /// Tests that Predict calls gesture service with correct image data
        /// </summary>
        [TestMethod]
        public async Task Predict_ValidImage_CallsServiceWithCorrectData()
        {
            // Arrange
            var mockGestureService = new Mock<IGestureRecognitionService>();
            var mockLogger = new Mock<ILogger<GestureController>>();
            var controller = new GestureController(mockGestureService.Object, mockLogger.Object);
            var imageData = new byte[] { 0x01, 0x02, 0x03, 0x04, 0x05 };
            var mockImage = CreateMockFormFile(imageData, "image/jpeg");
            var expectedResponse = new GesturePredictionResponseDto
            {
                Status = "success",
                Message = "Gesture predicted",
                Data = new GesturePredictionDataDto()
            };
            byte[]? capturedData = null;
            mockGestureService.Setup(x => x.PredictGestureAsync(It.IsAny<byte[]>(), It.IsAny<CancellationToken>()))
                .Callback<byte[], CancellationToken>((data, ct) => capturedData = data)
                .ReturnsAsync(expectedResponse);

            // Act
            await controller.Predict(mockImage.Object, CancellationToken.None);

            // Assert
            mockGestureService.Verify(x => x.PredictGestureAsync(It.IsAny<byte[]>(), It.IsAny<CancellationToken>()), Times.Once);
            Assert.IsNotNull(capturedData);
            CollectionAssert.AreEqual(imageData, capturedData);
        }

        /// <summary>
        /// Tests that Predict passes cancellation token to the service
        /// </summary>
        [TestMethod]
        public async Task Predict_WithCancellationToken_PassesTokenToService()
        {
            // Arrange
            var mockGestureService = new Mock<IGestureRecognitionService>();
            var mockLogger = new Mock<ILogger<GestureController>>();
            var controller = new GestureController(mockGestureService.Object, mockLogger.Object);
            var imageData = new byte[] { 0x01, 0x02, 0x03 };
            var mockImage = CreateMockFormFile(imageData, "image/jpeg");
            var expectedResponse = new GesturePredictionResponseDto
            {
                Status = "success",
                Message = "Gesture predicted",
                Data = new GesturePredictionDataDto()
            };
            var cts = new CancellationTokenSource();
            CancellationToken capturedToken = default;
            mockGestureService.Setup(x => x.PredictGestureAsync(It.IsAny<byte[]>(), It.IsAny<CancellationToken>()))
                .Callback<byte[], CancellationToken>((data, ct) => capturedToken = ct)
                .ReturnsAsync(expectedResponse);

            // Act
            await controller.Predict(mockImage.Object, cts.Token);

            // Assert
            Assert.AreEqual(cts.Token, capturedToken);
        }

        /// <summary>
        /// Tests that Predict returns 408 status when operation is cancelled
        /// </summary>
        [TestMethod]
        public async Task Predict_OperationCancelled_Returns408Status()
        {
            // Arrange
            var mockGestureService = new Mock<IGestureRecognitionService>();
            var mockLogger = new Mock<ILogger<GestureController>>();
            var controller = new GestureController(mockGestureService.Object, mockLogger.Object);
            var imageData = new byte[] { 0x01, 0x02, 0x03 };
            var mockImage = CreateMockFormFile(imageData, "image/jpeg");
            mockGestureService.Setup(x => x.PredictGestureAsync(It.IsAny<byte[]>(), It.IsAny<CancellationToken>()))
                .ThrowsAsync(new OperationCanceledException());

            // Act
            var result = await controller.Predict(mockImage.Object, CancellationToken.None);

            // Assert
            Assert.IsNotNull(result);
            Assert.IsInstanceOfType(result.Result, typeof(ObjectResult));
            var objectResult = (ObjectResult)result.Result!;
            Assert.AreEqual(408, objectResult.StatusCode);
            var response = objectResult.Value as GesturePredictionResponseDto;
            Assert.IsNotNull(response);
            Assert.AreEqual("error", response.Status);
            Assert.AreEqual("Request timeout: Processing took too long", response.Message);
            Assert.IsNull(response.Data);
        }

        /// <summary>
        /// Tests that Predict returns 500 status when service throws generic exception
        /// </summary>
        [TestMethod]
        public async Task Predict_ServiceThrowsException_Returns500Status()
        {
            // Arrange
            var mockGestureService = new Mock<IGestureRecognitionService>();
            var mockLogger = new Mock<ILogger<GestureController>>();
            var controller = new GestureController(mockGestureService.Object, mockLogger.Object);
            var imageData = new byte[] { 0x01, 0x02, 0x03 };
            var mockImage = CreateMockFormFile(imageData, "image/jpeg");
            mockGestureService.Setup(x => x.PredictGestureAsync(It.IsAny<byte[]>(), It.IsAny<CancellationToken>()))
                .ThrowsAsync(new InvalidOperationException("Service error"));

            // Act
            var result = await controller.Predict(mockImage.Object, CancellationToken.None);

            // Assert
            Assert.IsNotNull(result);
            Assert.IsInstanceOfType(result.Result, typeof(ObjectResult));
            var objectResult = (ObjectResult)result.Result!;
            Assert.AreEqual(500, objectResult.StatusCode);
            var response = objectResult.Value as GesturePredictionResponseDto;
            Assert.IsNotNull(response);
            Assert.AreEqual("error", response.Status);
            Assert.AreEqual("An error occurred during gesture prediction", response.Message);
            Assert.IsNull(response.Data);
        }

        /// <summary>
        /// Tests that Predict handles IOException from stream operations
        /// </summary>
        [TestMethod]
        public async Task Predict_StreamThrowsIOException_Returns500Status()
        {
            // Arrange
            var mockGestureService = new Mock<IGestureRecognitionService>();
            var mockLogger = new Mock<ILogger<GestureController>>();
            var controller = new GestureController(mockGestureService.Object, mockLogger.Object);
            var mockImage = new Mock<IFormFile>();
            mockImage.Setup(x => x.Length).Returns(1024);
            mockImage.Setup(x => x.ContentType).Returns("image/jpeg");
            mockImage.Setup(x => x.CopyToAsync(It.IsAny<Stream>(), It.IsAny<CancellationToken>()))
                .ThrowsAsync(new IOException("Stream error"));

            // Act
            var result = await controller.Predict(mockImage.Object, CancellationToken.None);

            // Assert
            Assert.IsNotNull(result);
            Assert.IsInstanceOfType(result.Result, typeof(ObjectResult));
            var objectResult = (ObjectResult)result.Result!;
            Assert.AreEqual(500, objectResult.StatusCode);
        }

        /// <summary>
        /// Tests that Predict processes minimal valid image successfully
        /// </summary>
        [TestMethod]
        public async Task Predict_MinimalValidImage_ProcessesSuccessfully()
        {
            // Arrange
            var mockGestureService = new Mock<IGestureRecognitionService>();
            var mockLogger = new Mock<ILogger<GestureController>>();
            var controller = new GestureController(mockGestureService.Object, mockLogger.Object);
            var imageData = new byte[] { 0x01 };
            var mockImage = CreateMockFormFile(imageData, "image/jpeg");
            var expectedResponse = new GesturePredictionResponseDto
            {
                Status = "success",
                Message = "Gesture predicted",
                Data = new GesturePredictionDataDto()
            };
            mockGestureService.Setup(x => x.PredictGestureAsync(It.IsAny<byte[]>(), It.IsAny<CancellationToken>()))
                .ReturnsAsync(expectedResponse);

            // Act
            var result = await controller.Predict(mockImage.Object, CancellationToken.None);

            // Assert
            Assert.IsNotNull(result);
            Assert.IsInstanceOfType(result.Result, typeof(OkObjectResult));
            var okResult = (OkObjectResult)result.Result!;
            Assert.AreEqual(expectedResponse, okResult.Value);
        }

        /// <summary>
        /// Tests that Predict returns success response with complete gesture data
        /// </summary>
        [TestMethod]
        public async Task Predict_ValidImage_ReturnsCompleteResponse()
        {
            // Arrange
            var mockGestureService = new Mock<IGestureRecognitionService>();
            var mockLogger = new Mock<ILogger<GestureController>>();
            var controller = new GestureController(mockGestureService.Object, mockLogger.Object);
            var imageData = new byte[] { 0xFF, 0xD8, 0xFF, 0xE0 };
            var mockImage = CreateMockFormFile(imageData, "image/jpeg");
            var expectedResponse = new GesturePredictionResponseDto
            {
                Status = "success",
                Message = "Gesture A detected",
                Data = new GesturePredictionDataDto
                {
                    Count = 21,
                    Coordinates = new System.Collections.Generic.List<CoordinateDto>
                    {
                        new CoordinateDto { X = 0.5, Y = 0.5 }
                    },
                    Letter = "A",
                    Confidence = 0.95f,
                    ProcessingTimeMs = 123.45
                }
            };
            mockGestureService.Setup(x => x.PredictGestureAsync(It.IsAny<byte[]>(), It.IsAny<CancellationToken>()))
                .ReturnsAsync(expectedResponse);

            // Act
            var result = await controller.Predict(mockImage.Object, CancellationToken.None);

            // Assert
            Assert.IsNotNull(result);
            Assert.IsInstanceOfType(result.Result, typeof(OkObjectResult));
            var okResult = (OkObjectResult)result.Result!;
            var response = okResult.Value as GesturePredictionResponseDto;
            Assert.IsNotNull(response);
            Assert.AreEqual("success", response.Status);
            Assert.AreEqual("Gesture A detected", response.Message);
            Assert.IsNotNull(response.Data);
            Assert.AreEqual(21, response.Data.Count);
            Assert.AreEqual("A", response.Data.Letter);
            Assert.AreEqual(0.95f, response.Data.Confidence);
        }

        /// <summary>
        /// Helper method to create a mock IFormFile with specified data and content type
        /// </summary>
        private static Mock<IFormFile> CreateMockFormFile(byte[] data, string contentType)
        {
            var mockFile = new Mock<IFormFile>();
            var memoryStream = new MemoryStream(data);
            mockFile.Setup(x => x.Length).Returns(data.Length);
            mockFile.Setup(x => x.ContentType).Returns(contentType);
            mockFile.Setup(x => x.CopyToAsync(It.IsAny<Stream>(), It.IsAny<CancellationToken>()))
                .Callback<Stream, CancellationToken>((stream, ct) =>
                {
                    memoryStream.Position = 0;
                    memoryStream.CopyTo(stream);
                })
                .Returns(Task.CompletedTask);
            return mockFile;
        }

        /// <summary>
        /// Tests that the Health method returns an OkObjectResult with status "healthy".
        /// </summary>
        [TestMethod]
        public void Health_WhenCalled_ReturnsOkResultWithHealthyStatus()
        {
            // Arrange
            var mockGestureService = new Mock<IGestureRecognitionService>();
            var mockLogger = new Mock<ILogger<GestureController>>();
            var controller = new GestureController(mockGestureService.Object, mockLogger.Object);

            // Act
            var result = controller.Health();

            // Assert
            Assert.IsNotNull(result);
            Assert.IsInstanceOfType(result.Result, typeof(OkObjectResult));
            var okResult = (OkObjectResult)result.Result;
            Assert.IsNotNull(okResult.Value);

            // Use reflection to access anonymous type properties
            var valueType = okResult.Value.GetType();
            var statusProperty = valueType.GetProperty("status");
            Assert.IsNotNull(statusProperty, "status property should exist");
            var statusValue = statusProperty.GetValue(okResult.Value);
            Assert.AreEqual("healthy", statusValue);
        }

        /// <summary>
        /// Tests that the Health method returns an object with service property set to "GestureRecognitionService".
        /// </summary>
        [TestMethod]
        public void Health_WhenCalled_ReturnsObjectWithCorrectServiceName()
        {
            // Arrange
            var mockGestureService = new Mock<IGestureRecognitionService>();
            var mockLogger = new Mock<ILogger<GestureController>>();
            var controller = new GestureController(mockGestureService.Object, mockLogger.Object);

            // Act
            var result = controller.Health();

            // Assert
            var okResult = (OkObjectResult)result.Result;
            var value = okResult.Value;
            var serviceProperty = value.GetType().GetProperty("service");
            var serviceValue = serviceProperty.GetValue(value);
            Assert.AreEqual("GestureRecognitionService", serviceValue);
        }

        /// <summary>
        /// Tests that the Health method returns a timestamp that is approximately the current UTC time.
        /// Verifies that the timestamp is within 5 seconds of DateTime.UtcNow.
        /// </summary>
        [TestMethod]
        public void Health_WhenCalled_ReturnsTimestampCloseToCurrentUtcTime()
        {
            // Arrange
            var mockGestureService = new Mock<IGestureRecognitionService>();
            var mockLogger = new Mock<ILogger<GestureController>>();
            var controller = new GestureController(mockGestureService.Object, mockLogger.Object);
            var beforeCall = DateTime.UtcNow;

            // Act
            var result = controller.Health();

            // Assert
            var afterCall = DateTime.UtcNow;
            var okResult = (OkObjectResult)result.Result;
            var value = okResult.Value;
            var timestampProperty = value.GetType().GetProperty("timestamp");
            DateTime timestamp = (DateTime)timestampProperty.GetValue(value);

            Assert.IsTrue(timestamp >= beforeCall, "Timestamp should be after or equal to the time before the call");
            Assert.IsTrue(timestamp <= afterCall, "Timestamp should be before or equal to the time after the call");
        }

        /// <summary>
        /// Tests that the Health method returns an ActionResult with all three expected properties.
        /// Validates the complete structure of the health check response.
        /// </summary>
        [TestMethod]
        public void Health_WhenCalled_ReturnsObjectWithAllExpectedProperties()
        {
            // Arrange
            var mockGestureService = new Mock<IGestureRecognitionService>();
            var mockLogger = new Mock<ILogger<GestureController>>();
            var controller = new GestureController(mockGestureService.Object, mockLogger.Object);

            // Act
            var result = controller.Health();

            // Assert
            Assert.IsNotNull(result);
            Assert.IsInstanceOfType(result.Result, typeof(OkObjectResult));
            var okResult = (OkObjectResult)result.Result;
            Assert.IsNotNull(okResult.Value);

            var value = okResult.Value;
            var statusProp = value.GetType().GetProperty("status");
            var serviceProp = value.GetType().GetProperty("service");
            var timestampProp = value.GetType().GetProperty("timestamp");

            Assert.IsNotNull(statusProp, "status property should exist");
            Assert.IsNotNull(serviceProp, "service property should exist");
            Assert.IsNotNull(timestampProp, "timestamp property should exist");

            Assert.AreEqual("healthy", statusProp.GetValue(value));
            Assert.AreEqual("GestureRecognitionService", serviceProp.GetValue(value));
            Assert.IsInstanceOfType(timestampProp.GetValue(value), typeof(DateTime));
        }

        /// <summary>
        /// Tests that multiple calls to the Health method return consistent status and service values.
        /// Input: None (method has no parameters).
        /// Expected: Multiple calls return identical status and service values.
        /// </summary>
        [TestMethod]
        public void Health_MultipleCalls_ReturnsConsistentValues()
        {
            // Arrange
            var mockGestureService = new Mock<IGestureRecognitionService>();
            var mockLogger = new Mock<ILogger<GestureController>>();
            var controller = new GestureController(mockGestureService.Object, mockLogger.Object);

            // Act
            var result1 = controller.Health();
            var result2 = controller.Health();

            // Assert
            var okResult1 = (OkObjectResult)result1.Result;
            var okResult2 = (OkObjectResult)result2.Result;

            var value1 = okResult1.Value;
            var value2 = okResult2.Value;

            var status1 = value1.GetType().GetProperty("status").GetValue(value1);
            var status2 = value2.GetType().GetProperty("status").GetValue(value2);
            Assert.AreEqual(status1, status2, "Status should be consistent across calls");

            var service1 = value1.GetType().GetProperty("service").GetValue(value1);
            var service2 = value2.GetType().GetProperty("service").GetValue(value2);
            Assert.AreEqual(service1, service2, "Service should be consistent across calls");
        }

        /// <summary>
        /// Tests that the Health method returns a timestamp with DateTime kind set to Utc.
        /// Input: None (method has no parameters).
        /// Expected: Timestamp has Kind property set to DateTimeKind.Utc.
        /// </summary>
        [TestMethod]
        public void Health_WhenCalled_ReturnsTimestampWithUtcKind()
        {
            // Arrange
            var mockGestureService = new Mock<IGestureRecognitionService>();
            var mockLogger = new Mock<ILogger<GestureController>>();
            var controller = new GestureController(mockGestureService.Object, mockLogger.Object);

            // Act
            var result = controller.Health();

            // Assert
            var okResult = (OkObjectResult)result.Result;
            var value = okResult.Value;
            var timestampProperty = value.GetType().GetProperty("timestamp");
            DateTime timestamp = (DateTime)timestampProperty.GetValue(value);

            Assert.AreEqual(DateTimeKind.Utc, timestamp.Kind, "Timestamp should have UTC kind");
        }

        /// <summary>
        /// Tests that Predict handles various exception types from the service with 500 status.
        /// Input: Service throws ArgumentException, InvalidOperationException, or IOException.
        /// Expected: 500 status code with generic error message for all exception types.
        /// </summary>
        [TestMethod]
        [DataRow(typeof(ArgumentException))]
        [DataRow(typeof(InvalidOperationException))]
        [DataRow(typeof(IOException))]
        public async Task Predict_ServiceThrowsVariousExceptions_Returns500Status(Type exceptionType)
        {
            // Arrange
            var mockGestureService = new Mock<IGestureRecognitionService>();
            var mockLogger = new Mock<ILogger<GestureController>>();
            var controller = new GestureController(mockGestureService.Object, mockLogger.Object);
            var imageData = new byte[1024];
            var mockImage = CreateMockFormFile(imageData, "image/jpeg");
            var exception = (Exception)Activator.CreateInstance(exceptionType, "Error message")!;
            mockGestureService.Setup(x => x.PredictGestureAsync(It.IsAny<byte[]>(), It.IsAny<CancellationToken>()))
                .ThrowsAsync(exception);

            // Act
            var result = await controller.Predict(mockImage.Object, CancellationToken.None);

            // Assert
            Assert.IsNotNull(result);
            Assert.IsInstanceOfType(result.Result, typeof(ObjectResult));
            var objectResult = (ObjectResult)result.Result!;
            Assert.AreEqual(500, objectResult.StatusCode);
            var response = objectResult.Value as GesturePredictionResponseDto;
            Assert.IsNotNull(response);
            Assert.AreEqual("error", response.Status);
            Assert.AreEqual("An error occurred during gesture prediction", response.Message);
            Assert.IsNull(response.Data);
        }

        /// <summary>
        /// Tests that Predict processes an image just under the 5MB limit successfully.
        /// Input: Image with 5MB - 1 byte.
        /// Expected: Successful processing and Ok result.
        /// </summary>
        [TestMethod]
        public async Task Predict_ImageJustUnderMaxSize_ProcessesSuccessfully()
        {
            // Arrange
            var mockGestureService = new Mock<IGestureRecognitionService>();
            var mockLogger = new Mock<ILogger<GestureController>>();
            var controller = new GestureController(mockGestureService.Object, mockLogger.Object);
            var imageData = new byte[5 * 1024 * 1024 - 1];
            var mockImage = CreateMockFormFile(imageData, "image/jpeg");
            var expectedResponse = new GesturePredictionResponseDto { Status = "success", Message = "Prediction successful", Data = null };
            mockGestureService.Setup(x => x.PredictGestureAsync(It.IsAny<byte[]>(), It.IsAny<CancellationToken>()))
                .ReturnsAsync(expectedResponse);

            // Act
            var result = await controller.Predict(mockImage.Object, CancellationToken.None);

            // Assert
            Assert.IsNotNull(result);
            Assert.IsInstanceOfType(result.Result, typeof(OkObjectResult));
            var okResult = (OkObjectResult)result.Result!;
            Assert.AreEqual(expectedResponse, okResult.Value);
        }
    }
}