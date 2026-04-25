using System;
using System.Diagnostics;
using System.Linq;
using System.Threading;
using System.Threading.Tasks;

using Microsoft.Extensions.Logging;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using Moq;
using SignLanguageApi.Dtos;
using SignLanguageApi.Services;

namespace SignLanguageApi.Services.UnitTests
{
    /// <summary>
    /// Unit tests for the GestureRecognitionService class.
    /// </summary>
    [TestClass]
    public class GestureRecognitionServiceTests
    {
        /// <summary>
        /// Tests that the constructor successfully creates an instance when provided with a valid logger.
        /// Input: Valid mocked ILogger instance.
        /// Expected: Instance is created without throwing an exception.
        /// </summary>
        [TestMethod]
        public void GestureRecognitionService_WithValidLogger_CreatesInstanceSuccessfully()
        {
            // Arrange
            var mockLogger = new Mock<ILogger<GestureRecognitionService>>();

            // Act
            var service = new GestureRecognitionService(mockLogger.Object);

            // Assert
            Assert.IsNotNull(service);
        }

        /// <summary>
        /// Tests that the constructor accepts a null logger without throwing an exception.
        /// Input: Null logger reference.
        /// Expected: Instance is created without throwing an exception (no null validation in constructor).
        /// Note: This exposes a potential bug - methods using _logger may throw NullReferenceException.
        /// </summary>
        [TestMethod]
        public void GestureRecognitionService_WithNullLogger_CreatesInstanceWithoutThrowingException()
        {
            // Arrange
            ILogger<GestureRecognitionService>? logger = null;

            // Act
            var service = new GestureRecognitionService(logger!);

            // Assert
            Assert.IsNotNull(service);
        }

        /// <summary>
        /// Tests PredictGestureAsync with null image data.
        /// Expects an error response indicating invalid image data.
        /// </summary>
        [TestMethod]
        public async Task PredictGestureAsync_NullImageData_ReturnsErrorResponse()
        {
            // Arrange
            var mockLogger = new Mock<ILogger<GestureRecognitionService>>();
            var service = new GestureRecognitionService(mockLogger.Object);

            // Act
            var result = await service.PredictGestureAsync(null!, CancellationToken.None);

            // Assert
            Assert.IsNotNull(result);
            Assert.AreEqual("error", result.Status);
            Assert.AreEqual("Invalid image data. Image must be between 100 bytes and 5MB.", result.Message);
            Assert.IsNull(result.Data);
        }

        /// <summary>
        /// Tests PredictGestureAsync with empty image data.
        /// Expects an error response indicating invalid image data.
        /// </summary>
        [TestMethod]
        public async Task PredictGestureAsync_EmptyImageData_ReturnsErrorResponse()
        {
            // Arrange
            var mockLogger = new Mock<ILogger<GestureRecognitionService>>();
            var service = new GestureRecognitionService(mockLogger.Object);
            var imageData = Array.Empty<byte>();

            // Act
            var result = await service.PredictGestureAsync(imageData, CancellationToken.None);

            // Assert
            Assert.IsNotNull(result);
            Assert.AreEqual("error", result.Status);
            Assert.AreEqual("Invalid image data. Image must be between 100 bytes and 5MB.", result.Message);
            Assert.IsNull(result.Data);
        }

        /// <summary>
        /// Tests PredictGestureAsync with image data that is too small (below MinImageSize of 100 bytes).
        /// Expects an error response indicating invalid image data.
        /// </summary>
        [TestMethod]
        [DataRow(1)]
        [DataRow(50)]
        [DataRow(99)]
        public async Task PredictGestureAsync_ImageDataTooSmall_ReturnsErrorResponse(int size)
        {
            // Arrange
            var mockLogger = new Mock<ILogger<GestureRecognitionService>>();
            var service = new GestureRecognitionService(mockLogger.Object);
            var imageData = new byte[size];
            imageData[0] = 0xFF;
            if (size > 1)
                imageData[1] = 0xD8;

            // Act
            var result = await service.PredictGestureAsync(imageData, CancellationToken.None);

            // Assert
            Assert.IsNotNull(result);
            Assert.AreEqual("error", result.Status);
            Assert.AreEqual("Invalid image data. Image must be between 100 bytes and 5MB.", result.Message);
            Assert.IsNull(result.Data);
        }

        /// <summary>
        /// Tests PredictGestureAsync with image data that is too large (above MaxImageSize of 5MB).
        /// Expects an error response indicating invalid image data.
        /// </summary>
        [TestMethod]
        [DataRow(5_000_001)]
        [DataRow(10_000_000)]
        public async Task PredictGestureAsync_ImageDataTooLarge_ReturnsErrorResponse(int size)
        {
            // Arrange
            var mockLogger = new Mock<ILogger<GestureRecognitionService>>();
            var service = new GestureRecognitionService(mockLogger.Object);
            var imageData = new byte[size];
            imageData[0] = 0xFF;
            imageData[1] = 0xD8;

            // Act
            var result = await service.PredictGestureAsync(imageData, CancellationToken.None);

            // Assert
            Assert.IsNotNull(result);
            Assert.AreEqual("error", result.Status);
            Assert.AreEqual("Invalid image data. Image must be between 100 bytes and 5MB.", result.Message);
            Assert.IsNull(result.Data);
        }

        /// <summary>
        /// Tests PredictGestureAsync with image data that does not have a valid JPEG header.
        /// Expects an error response indicating invalid image data.
        /// </summary>
        [TestMethod]
        [DataRow((byte)0x00, (byte)0x00)]
        [DataRow((byte)0xFF, (byte)0x00)]
        [DataRow((byte)0x00, (byte)0xD8)]
        [DataRow((byte)0x89, (byte)0x50)] // PNG header
        public async Task PredictGestureAsync_InvalidJpegHeader_ReturnsErrorResponse(byte byte1, byte byte2)
        {
            // Arrange
            var mockLogger = new Mock<ILogger<GestureRecognitionService>>();
            var service = new GestureRecognitionService(mockLogger.Object);
            var imageData = new byte[100];
            imageData[0] = byte1;
            imageData[1] = byte2;

            // Act
            var result = await service.PredictGestureAsync(imageData, CancellationToken.None);

            // Assert
            Assert.IsNotNull(result);
            Assert.AreEqual("error", result.Status);
            Assert.AreEqual("Invalid image data. Image must be between 100 bytes and 5MB.", result.Message);
            Assert.IsNull(result.Data);
        }

        /// <summary>
        /// Tests PredictGestureAsync with valid image data (minimum valid size with JPEG header).
        /// Expects a successful response or low confidence response with appropriate data.
        /// </summary>
        [TestMethod]
        public async Task PredictGestureAsync_ValidMinimumImageData_ReturnsResponse()
        {
            // Arrange
            var mockLogger = new Mock<ILogger<GestureRecognitionService>>();
            var service = new GestureRecognitionService(mockLogger.Object);
            var imageData = new byte[100];
            imageData[0] = 0xFF;
            imageData[1] = 0xD8;

            // Act
            var result = await service.PredictGestureAsync(imageData, CancellationToken.None);

            // Assert
            Assert.IsNotNull(result);
            Assert.IsTrue(result.Status == "success" || result.Status == "low_confidence" || result.Status == "error");
            Assert.IsFalse(string.IsNullOrEmpty(result.Message));
        }

        /// <summary>
        /// Tests PredictGestureAsync with valid image data at maximum allowed size.
        /// Expects a successful response or low confidence response with appropriate data.
        /// </summary>
        [TestMethod]
        public async Task PredictGestureAsync_ValidMaximumImageData_ReturnsResponse()
        {
            // Arrange
            var mockLogger = new Mock<ILogger<GestureRecognitionService>>();
            var service = new GestureRecognitionService(mockLogger.Object);
            var imageData = new byte[5_000_000];
            imageData[0] = 0xFF;
            imageData[1] = 0xD8;

            // Act
            var result = await service.PredictGestureAsync(imageData, CancellationToken.None);

            // Assert
            Assert.IsNotNull(result);
            Assert.IsTrue(result.Status == "success" || result.Status == "low_confidence" || result.Status == "error");
            Assert.IsFalse(string.IsNullOrEmpty(result.Message));
        }

        /// <summary>
        /// Tests PredictGestureAsync with a cancelled cancellation token.
        /// Expects an error response indicating the request was cancelled.
        /// </summary>
        [TestMethod]
        public async Task PredictGestureAsync_CancelledToken_ReturnsErrorResponse()
        {
            // Arrange
            var mockLogger = new Mock<ILogger<GestureRecognitionService>>();
            var service = new GestureRecognitionService(mockLogger.Object);
            var imageData = new byte[100];
            imageData[0] = 0xFF;
            imageData[1] = 0xD8;
            var cts = new CancellationTokenSource();
            cts.Cancel();

            // Act
            var result = await service.PredictGestureAsync(imageData, cts.Token);

            // Assert
            Assert.IsNotNull(result);
            Assert.AreEqual("error", result.Status);
            Assert.AreEqual("Request was cancelled", result.Message);
            Assert.IsNull(result.Data);
        }

        /// <summary>
        /// Tests PredictGestureAsync with valid image data and default cancellation token.
        /// Expects logging of information message about gesture detection completion.
        /// </summary>
        [TestMethod]
        public async Task PredictGestureAsync_ValidImageData_LogsInformation()
        {
            // Arrange
            var mockLogger = new Mock<ILogger<GestureRecognitionService>>();
            var service = new GestureRecognitionService(mockLogger.Object);
            var imageData = new byte[500];
            imageData[0] = 0xFF;
            imageData[1] = 0xD8;

            // Act
            var result = await service.PredictGestureAsync(imageData, CancellationToken.None);

            // Assert
            mockLogger.Verify(
                x => x.Log(
                    LogLevel.Information,
                    It.IsAny<EventId>(),
                    It.Is<It.IsAnyType>((v, t) => v.ToString()!.Contains("Gesture detection completed")),
                    It.IsAny<Exception>(),
                    It.IsAny<Func<It.IsAnyType, Exception?, string>>()),
                Times.Once);
        }

        /// <summary>
        /// Tests PredictGestureAsync with a cancelled token.
        /// Expects logging of a warning message about cancellation.
        /// </summary>
        [TestMethod]
        public async Task PredictGestureAsync_CancelledToken_LogsWarning()
        {
            // Arrange
            var mockLogger = new Mock<ILogger<GestureRecognitionService>>();
            var service = new GestureRecognitionService(mockLogger.Object);
            var imageData = new byte[100];
            imageData[0] = 0xFF;
            imageData[1] = 0xD8;
            var cts = new CancellationTokenSource();
            cts.Cancel();

            // Act
            var result = await service.PredictGestureAsync(imageData, cts.Token);

            // Assert
            mockLogger.Verify(
                x => x.Log(
                    LogLevel.Warning,
                    It.IsAny<EventId>(),
                    It.Is<It.IsAnyType>((v, t) => v.ToString()!.Contains("Gesture detection request was cancelled")),
                    It.IsAny<Exception>(),
                    It.IsAny<Func<It.IsAnyType, Exception?, string>>()),
                Times.Once);
        }

        /// <summary>
        /// Tests PredictGestureAsync with null image data.
        /// Expects logging of a warning message about null or empty image data.
        /// </summary>
        [TestMethod]
        public async Task PredictGestureAsync_NullImageData_LogsWarning()
        {
            // Arrange
            var mockLogger = new Mock<ILogger<GestureRecognitionService>>();
            var service = new GestureRecognitionService(mockLogger.Object);

            // Act
            var result = await service.PredictGestureAsync(null!, CancellationToken.None);

            // Assert
            mockLogger.Verify(
                x => x.Log(
                    LogLevel.Warning,
                    It.IsAny<EventId>(),
                    It.Is<It.IsAnyType>((v, t) => v.ToString()!.Contains("Image data is null or empty")),
                    It.IsAny<Exception>(),
                    It.IsAny<Func<It.IsAnyType, Exception?, string>>()),
                Times.Once);
        }

        /// <summary>
        /// Tests PredictGestureAsync with valid image data.
        /// Expects the response to contain valid status values.
        /// </summary>
        [TestMethod]
        public async Task PredictGestureAsync_ValidImageData_ReturnsValidStatus()
        {
            // Arrange
            var mockLogger = new Mock<ILogger<GestureRecognitionService>>();
            var service = new GestureRecognitionService(mockLogger.Object);
            var imageData = new byte[1000];
            imageData[0] = 0xFF;
            imageData[1] = 0xD8;

            // Act
            var result = await service.PredictGestureAsync(imageData, CancellationToken.None);

            // Assert
            Assert.IsNotNull(result);
            Assert.IsNotNull(result.Status);
            Assert.IsTrue(
                result.Status == "success" ||
                result.Status == "low_confidence" ||
                result.Status == "error",
                $"Unexpected status: {result.Status}");
        }

        /// <summary>
        /// Tests PredictGestureAsync with valid image data.
        /// Expects the response message to not be null or empty.
        /// </summary>
        [TestMethod]
        public async Task PredictGestureAsync_ValidImageData_ReturnsNonEmptyMessage()
        {
            // Arrange
            var mockLogger = new Mock<ILogger<GestureRecognitionService>>();
            var service = new GestureRecognitionService(mockLogger.Object);
            var imageData = new byte[200];
            imageData[0] = 0xFF;
            imageData[1] = 0xD8;

            // Act
            var result = await service.PredictGestureAsync(imageData, CancellationToken.None);

            // Assert
            Assert.IsNotNull(result);
            Assert.IsFalse(string.IsNullOrWhiteSpace(result.Message));
        }

        /// <summary>
        /// Tests PredictGestureAsync response structure.
        /// Expects all status types to have appropriate Data property values.
        /// </summary>
        [TestMethod]
        public async Task PredictGestureAsync_ErrorStatus_HasNullData()
        {
            // Arrange
            var mockLogger = new Mock<ILogger<GestureRecognitionService>>();
            var service = new GestureRecognitionService(mockLogger.Object);
            var imageData = new byte[50]; // Invalid size

            // Act
            var result = await service.PredictGestureAsync(imageData, CancellationToken.None);

            // Assert
            Assert.IsNotNull(result);
            Assert.AreEqual("error", result.Status);
            Assert.IsNull(result.Data);
        }

        /// <summary>
        /// Tests PredictGestureAsync with image data at the boundary (exactly 100 bytes).
        /// Expects the method to process the valid image data.
        /// </summary>
        [TestMethod]
        public async Task PredictGestureAsync_ImageDataExactlyMinSize_ProcessesSuccessfully()
        {
            // Arrange
            var mockLogger = new Mock<ILogger<GestureRecognitionService>>();
            var service = new GestureRecognitionService(mockLogger.Object);
            var imageData = new byte[100];
            imageData[0] = 0xFF;
            imageData[1] = 0xD8;

            // Act
            var result = await service.PredictGestureAsync(imageData, CancellationToken.None);

            // Assert
            Assert.IsNotNull(result);
            Assert.AreNotEqual("error", result.Status, "Should not return error for valid minimum size");
        }

        /// <summary>
        /// Tests PredictGestureAsync with image data at the upper boundary (exactly 5MB).
        /// Expects the method to process the valid image data.
        /// </summary>
        [TestMethod]
        public async Task PredictGestureAsync_ImageDataExactlyMaxSize_ProcessesSuccessfully()
        {
            // Arrange
            var mockLogger = new Mock<ILogger<GestureRecognitionService>>();
            var service = new GestureRecognitionService(mockLogger.Object);
            var imageData = new byte[5_000_000];
            imageData[0] = 0xFF;
            imageData[1] = 0xD8;

            // Act
            var result = await service.PredictGestureAsync(imageData, CancellationToken.None);

            // Assert
            Assert.IsNotNull(result);
            Assert.AreNotEqual("error", result.Status, "Should not return error for valid maximum size");
        }

        /// <summary>
        /// Tests PredictGestureAsync to ensure it returns a Task that completes successfully.
        /// Validates async behavior with valid inputs.
        /// </summary>
        [TestMethod]
        public async Task PredictGestureAsync_ValidImageData_CompletesAsynchronously()
        {
            // Arrange
            var mockLogger = new Mock<ILogger<GestureRecognitionService>>();
            var service = new GestureRecognitionService(mockLogger.Object);
            var imageData = new byte[150];
            imageData[0] = 0xFF;
            imageData[1] = 0xD8;

            // Act
            var task = service.PredictGestureAsync(imageData, CancellationToken.None);
            var result = await task;

            // Assert
            Assert.IsTrue(task.IsCompleted);
            Assert.IsNotNull(result);
        }

        /// <summary>
        /// Tests PredictGestureAsync with different valid image sizes.
        /// Expects all valid sizes to be processed without validation errors.
        /// </summary>
        [TestMethod]
        [DataRow(100)]
        [DataRow(500)]
        [DataRow(1000)]
        [DataRow(100_000)]
        [DataRow(1_000_000)]
        [DataRow(5_000_000)]
        public async Task PredictGestureAsync_VariousValidImageSizes_ProcessesSuccessfully(int size)
        {
            // Arrange
            var mockLogger = new Mock<ILogger<GestureRecognitionService>>();
            var service = new GestureRecognitionService(mockLogger.Object);
            var imageData = new byte[size];
            imageData[0] = 0xFF;
            imageData[1] = 0xD8;

            // Act
            var result = await service.PredictGestureAsync(imageData, CancellationToken.None);

            // Assert
            Assert.IsNotNull(result);
            Assert.IsTrue(result.Status == "success" || result.Status == "low_confidence" || result.Status == "error");
            Assert.AreNotEqual("Invalid image data. Image must be between 100 bytes and 5MB.", result.Message);
        }

        /// <summary>
        /// Tests that ValidateImageData returns false when imageData is null.
        /// </summary>
        [TestMethod]
        public void ValidateImageData_NullImageData_ReturnsFalse()
        {
            // Arrange
            var mockLogger = new Mock<ILogger<GestureRecognitionService>>();
            var service = new GestureRecognitionService(mockLogger.Object);
            byte[]? imageData = null;

            // Act
            var result = service.ValidateImageData(imageData!);

            // Assert
            Assert.IsFalse(result);
            mockLogger.Verify(
                x => x.Log(
                    LogLevel.Warning,
                    It.IsAny<EventId>(),
                    It.Is<It.IsAnyType>((v, t) => v.ToString()!.Contains("Image data is null or empty")),
                    It.IsAny<Exception>(),
                    It.IsAny<Func<It.IsAnyType, Exception?, string>>()),
                Times.Once);
        }

        /// <summary>
        /// Tests that ValidateImageData returns false when imageData is an empty array.
        /// </summary>
        [TestMethod]
        public void ValidateImageData_EmptyImageData_ReturnsFalse()
        {
            // Arrange
            var mockLogger = new Mock<ILogger<GestureRecognitionService>>();
            var service = new GestureRecognitionService(mockLogger.Object);
            var imageData = new byte[0];

            // Act
            var result = service.ValidateImageData(imageData);

            // Assert
            Assert.IsFalse(result);
            mockLogger.Verify(
                x => x.Log(
                    LogLevel.Warning,
                    It.IsAny<EventId>(),
                    It.Is<It.IsAnyType>((v, t) => v.ToString()!.Contains("Image data is null or empty")),
                    It.IsAny<Exception>(),
                    It.IsAny<Func<It.IsAnyType, Exception?, string>>()),
                Times.Once);
        }

        /// <summary>
        /// Tests that ValidateImageData returns false when imageData size is below the minimum threshold.
        /// </summary>
        /// <param name="size">The size of the image data in bytes.</param>
        [TestMethod]
        [DataRow(1)]
        [DataRow(50)]
        [DataRow(99)]
        public void ValidateImageData_ImageDataBelowMinimumSize_ReturnsFalse(int size)
        {
            // Arrange
            var mockLogger = new Mock<ILogger<GestureRecognitionService>>();
            var service = new GestureRecognitionService(mockLogger.Object);
            var imageData = new byte[size];

            // Act
            var result = service.ValidateImageData(imageData);

            // Assert
            Assert.IsFalse(result);
            mockLogger.Verify(
                x => x.Log(
                    LogLevel.Warning,
                    It.IsAny<EventId>(),
                    It.Is<It.IsAnyType>((v, t) => v.ToString()!.Contains("outside valid range")),
                    It.IsAny<Exception>(),
                    It.IsAny<Func<It.IsAnyType, Exception?, string>>()),
                Times.Once);
        }

        /// <summary>
        /// Tests that ValidateImageData returns false when imageData size exceeds the maximum threshold.
        /// </summary>
        /// <param name="size">The size of the image data in bytes.</param>
        [TestMethod]
        [DataRow(5_000_001)]
        [DataRow(6_000_000)]
        [DataRow(10_000_000)]
        public void ValidateImageData_ImageDataAboveMaximumSize_ReturnsFalse(int size)
        {
            // Arrange
            var mockLogger = new Mock<ILogger<GestureRecognitionService>>();
            var service = new GestureRecognitionService(mockLogger.Object);
            var imageData = new byte[size];

            // Act
            var result = service.ValidateImageData(imageData);

            // Assert
            Assert.IsFalse(result);
            mockLogger.Verify(
                x => x.Log(
                    LogLevel.Warning,
                    It.IsAny<EventId>(),
                    It.Is<It.IsAnyType>((v, t) => v.ToString()!.Contains("outside valid range")),
                    It.IsAny<Exception>(),
                    It.IsAny<Func<It.IsAnyType, Exception?, string>>()),
                Times.Once);
        }

        /// <summary>
        /// Tests that ValidateImageData returns false when imageData does not have a valid JPEG header.
        /// </summary>
        /// <param name="firstByte">The first byte of the image data.</param>
        /// <param name="secondByte">The second byte of the image data.</param>
        [TestMethod]
        [DataRow((byte)0x00, (byte)0x00)]
        [DataRow((byte)0xFF, (byte)0x00)]
        [DataRow((byte)0x00, (byte)0xD8)]
        [DataRow((byte)0xFF, (byte)0xD9)]
        [DataRow((byte)0xFE, (byte)0xD8)]
        public void ValidateImageData_InvalidJpegHeader_ReturnsFalse(byte firstByte, byte secondByte)
        {
            // Arrange
            var mockLogger = new Mock<ILogger<GestureRecognitionService>>();
            var service = new GestureRecognitionService(mockLogger.Object);
            var imageData = new byte[100];
            imageData[0] = firstByte;
            imageData[1] = secondByte;

            // Act
            var result = service.ValidateImageData(imageData);

            // Assert
            Assert.IsFalse(result);
            mockLogger.Verify(
                x => x.Log(
                    LogLevel.Warning,
                    It.IsAny<EventId>(),
                    It.Is<It.IsAnyType>((v, t) => v.ToString()!.Contains("valid JPEG header")),
                    It.IsAny<Exception>(),
                    It.IsAny<Func<It.IsAnyType, Exception?, string>>()),
                Times.Once);
        }

        /// <summary>
        /// Tests that ValidateImageData returns true when imageData has valid size and JPEG header.
        /// </summary>
        /// <param name="size">The size of the image data in bytes.</param>
        [TestMethod]
        [DataRow(100)]
        [DataRow(1000)]
        [DataRow(500_000)]
        [DataRow(5_000_000)]
        public void ValidateImageData_ValidImageDataWithCorrectJpegHeader_ReturnsTrue(int size)
        {
            // Arrange
            var mockLogger = new Mock<ILogger<GestureRecognitionService>>();
            var service = new GestureRecognitionService(mockLogger.Object);
            var imageData = new byte[size];
            imageData[0] = 0xFF;
            imageData[1] = 0xD8;

            // Act
            var result = service.ValidateImageData(imageData);

            // Assert
            Assert.IsTrue(result);
            mockLogger.Verify(
                x => x.Log(
                    LogLevel.Warning,
                    It.IsAny<EventId>(),
                    It.IsAny<It.IsAnyType>(),
                    It.IsAny<Exception>(),
                    It.IsAny<Func<It.IsAnyType, Exception?, string>>()),
                Times.Never);
        }

        /// <summary>
        /// Tests that ValidateImageData returns true for minimum valid size with correct JPEG header.
        /// </summary>
        [TestMethod]
        public void ValidateImageData_MinimumValidSizeWithValidHeader_ReturnsTrue()
        {
            // Arrange
            var mockLogger = new Mock<ILogger<GestureRecognitionService>>();
            var service = new GestureRecognitionService(mockLogger.Object);
            var imageData = new byte[100];
            imageData[0] = 0xFF;
            imageData[1] = 0xD8;

            // Act
            var result = service.ValidateImageData(imageData);

            // Assert
            Assert.IsTrue(result);
        }

        /// <summary>
        /// Tests that ValidateImageData returns true for maximum valid size with correct JPEG header.
        /// </summary>
        [TestMethod]
        public void ValidateImageData_MaximumValidSizeWithValidHeader_ReturnsTrue()
        {
            // Arrange
            var mockLogger = new Mock<ILogger<GestureRecognitionService>>();
            var service = new GestureRecognitionService(mockLogger.Object);
            var imageData = new byte[5_000_000];
            imageData[0] = 0xFF;
            imageData[1] = 0xD8;

            // Act
            var result = service.ValidateImageData(imageData);

            // Assert
            Assert.IsTrue(result);
        }

        /// <summary>
        /// Tests that ValidateImageData correctly validates first byte of JPEG header.
        /// Input: Byte arrays with valid size, correct second byte (0xD8), but incorrect first byte.
        /// Expected: Returns false and logs warning about invalid JPEG header.
        /// </summary>
        /// <param name="firstByte">The first byte to test (should fail if not 0xFF).</param>
        [TestMethod]
        [DataRow((byte)0x00)]
        [DataRow((byte)0xFE)]
        [DataRow((byte)0x01)]
        [DataRow((byte)0x89)]
        public void ValidateImageData_InvalidFirstByteOfJpegHeader_ReturnsFalse(byte firstByte)
        {
            // Arrange
            var mockLogger = new Mock<ILogger<GestureRecognitionService>>();
            var service = new GestureRecognitionService(mockLogger.Object);
            var imageData = new byte[100];
            imageData[0] = firstByte;
            imageData[1] = 0xD8;

            // Act
            var result = service.ValidateImageData(imageData);

            // Assert
            Assert.IsFalse(result);
            mockLogger.Verify(
                x => x.Log(
                    LogLevel.Warning,
                    It.IsAny<EventId>(),
                    It.Is<It.IsAnyType>((v, t) => v.ToString()!.Contains("Image does not have valid JPEG header")),
                    It.IsAny<Exception>(),
                    It.IsAny<Func<It.IsAnyType, Exception?, string>>()),
                Times.Once);
        }

        /// <summary>
        /// Tests that ValidateImageData correctly validates second byte of JPEG header.
        /// Input: Byte arrays with valid size, correct first byte (0xFF), but incorrect second byte.
        /// Expected: Returns false and logs warning about invalid JPEG header.
        /// </summary>
        /// <param name="secondByte">The second byte to test (should fail if not 0xD8).</param>
        [TestMethod]
        [DataRow((byte)0x00)]
        [DataRow((byte)0xD9)]
        [DataRow((byte)0x01)]
        [DataRow((byte)0x50)]
        public void ValidateImageData_InvalidSecondByteOfJpegHeader_ReturnsFalse(byte secondByte)
        {
            // Arrange
            var mockLogger = new Mock<ILogger<GestureRecognitionService>>();
            var service = new GestureRecognitionService(mockLogger.Object);
            var imageData = new byte[100];
            imageData[0] = 0xFF;
            imageData[1] = secondByte;

            // Act
            var result = service.ValidateImageData(imageData);

            // Assert
            Assert.IsFalse(result);
            mockLogger.Verify(
                x => x.Log(
                    LogLevel.Warning,
                    It.IsAny<EventId>(),
                    It.Is<It.IsAnyType>((v, t) => v.ToString()!.Contains("Image does not have valid JPEG header")),
                    It.IsAny<Exception>(),
                    It.IsAny<Func<It.IsAnyType, Exception?, string>>()),
                Times.Once);
        }

        /// <summary>
        /// Tests that ValidateImageData validates size before checking JPEG header.
        /// Input: Byte array with size below minimum but with valid JPEG header.
        /// Expected: Returns false and logs warning about size (header check should not be reached).
        /// </summary>
        [TestMethod]
        public void ValidateImageData_BelowMinSizeWithValidHeader_FailsOnSizeCheck()
        {
            // Arrange
            var mockLogger = new Mock<ILogger<GestureRecognitionService>>();
            var service = new GestureRecognitionService(mockLogger.Object);
            var imageData = new byte[99];
            imageData[0] = 0xFF;
            imageData[1] = 0xD8;

            // Act
            var result = service.ValidateImageData(imageData);

            // Assert
            Assert.IsFalse(result);
            mockLogger.Verify(
                x => x.Log(
                    LogLevel.Warning,
                    It.IsAny<EventId>(),
                    It.Is<It.IsAnyType>((v, t) => v.ToString()!.Contains("outside valid range")),
                    It.IsAny<Exception>(),
                    It.IsAny<Func<It.IsAnyType, Exception?, string>>()),
                Times.Once);
            mockLogger.Verify(
                x => x.Log(
                    LogLevel.Warning,
                    It.IsAny<EventId>(),
                    It.Is<It.IsAnyType>((v, t) => v.ToString()!.Contains("JPEG header")),
                    It.IsAny<Exception>(),
                    It.IsAny<Func<It.IsAnyType, Exception?, string>>()),
                Times.Never);
        }

        /// <summary>
        /// Tests that ValidateImageData validates size before checking JPEG header.
        /// Input: Byte array with size above maximum but with valid JPEG header.
        /// Expected: Returns false and logs warning about size (header check should not be reached).
        /// </summary>
        [TestMethod]
        public void ValidateImageData_AboveMaxSizeWithValidHeader_FailsOnSizeCheck()
        {
            // Arrange
            var mockLogger = new Mock<ILogger<GestureRecognitionService>>();
            var service = new GestureRecognitionService(mockLogger.Object);
            var imageData = new byte[5_000_001];
            imageData[0] = 0xFF;
            imageData[1] = 0xD8;

            // Act
            var result = service.ValidateImageData(imageData);

            // Assert
            Assert.IsFalse(result);
            mockLogger.Verify(
                x => x.Log(
                    LogLevel.Warning,
                    It.IsAny<EventId>(),
                    It.Is<It.IsAnyType>((v, t) => v.ToString()!.Contains("outside valid range")),
                    It.IsAny<Exception>(),
                    It.IsAny<Func<It.IsAnyType, Exception?, string>>()),
                Times.Once);
            mockLogger.Verify(
                x => x.Log(
                    LogLevel.Warning,
                    It.IsAny<EventId>(),
                    It.Is<It.IsAnyType>((v, t) => v.ToString()!.Contains("JPEG header")),
                    It.IsAny<Exception>(),
                    It.IsAny<Func<It.IsAnyType, Exception?, string>>()),
                Times.Never);
        }
    }
}