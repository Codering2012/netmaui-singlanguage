using System;
using System.IO;
using System.Net;
using System.Text;
using System.Text.Json;
using System.Threading.Tasks;

using Microsoft.AspNetCore.Builder;
using Microsoft.AspNetCore.Http;
using Microsoft.Extensions.Logging;
using Microsoft.Extensions.Primitives;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using Moq;
using SignLanguageApi;
using SignLanguageApi.Middleware;
using SignLanguageApi.Services;

namespace SignLanguageApi.Middleware.UnitTests;

/// <summary>
/// Unit tests for the <see cref="TokenBlacklistMiddlewareExtensions"/> class.
/// </summary>
[TestClass]
public class TokenBlacklistMiddlewareExtensionsTests
{
    /// <summary>
    /// Tests that UseTokenBlacklistMiddleware returns a non-null IApplicationBuilder when given a valid builder.
    /// </summary>
    [TestMethod]
    public void UseTokenBlacklistMiddleware_ValidBuilder_ReturnsNonNullApplicationBuilder()
    {
        // Arrange
        var mockBuilder = new Mock<IApplicationBuilder>();
        mockBuilder.Setup(b => b.Use(It.IsAny<Func<Microsoft.AspNetCore.Http.RequestDelegate, Microsoft.AspNetCore.Http.RequestDelegate>>()))
                   .Returns(mockBuilder.Object);

        // Act
        var result = TokenBlacklistMiddlewareExtensions.UseTokenBlacklistMiddleware(mockBuilder.Object);

        // Assert
        Assert.IsNotNull(result);
    }

    /// <summary>
    /// Tests that UseTokenBlacklistMiddleware returns an IApplicationBuilder instance when given a valid builder.
    /// </summary>
    [TestMethod]
    public void UseTokenBlacklistMiddleware_ValidBuilder_ReturnsIApplicationBuilder()
    {
        // Arrange
        var mockBuilder = new Mock<IApplicationBuilder>();
        mockBuilder.Setup(b => b.Use(It.IsAny<Func<Microsoft.AspNetCore.Http.RequestDelegate, Microsoft.AspNetCore.Http.RequestDelegate>>()))
                   .Returns(mockBuilder.Object);

        // Act
        var result = TokenBlacklistMiddlewareExtensions.UseTokenBlacklistMiddleware(mockBuilder.Object);

        // Assert
        Assert.IsInstanceOfType<IApplicationBuilder>(result);
    }

    /// <summary>
    /// Tests that UseTokenBlacklistMiddleware calls the Use method on the builder to register middleware.
    /// Input: Valid mocked IApplicationBuilder instance.
    /// Expected: The Use method is called exactly once on the builder.
    /// </summary>
    [TestMethod]
    public void UseTokenBlacklistMiddleware_ValidBuilder_CallsUseMethod()
    {
        // Arrange
        Mock<IApplicationBuilder> mockBuilder = new Mock<IApplicationBuilder>();
        mockBuilder.Setup(b => b.Use(It.IsAny<Func<RequestDelegate, RequestDelegate>>()))
                   .Returns(mockBuilder.Object);

        // Act
        TokenBlacklistMiddlewareExtensions.UseTokenBlacklistMiddleware(mockBuilder.Object);

        // Assert
        mockBuilder.Verify(b => b.Use(It.IsAny<Func<RequestDelegate, RequestDelegate>>()), Times.Once);
    }
}



/// <summary>
/// Unit tests for the TokenBlacklistMiddleware class.
/// </summary>
[TestClass]
public class TokenBlacklistMiddlewareTests
{
    /// <summary>
    /// Tests that InvokeAsync calls the next middleware when no Authorization header is present.
    /// </summary>
    [TestMethod]
    public async Task InvokeAsync_NoAuthorizationHeader_CallsNextMiddleware()
    {
        // Arrange
        var mockNext = new Mock<RequestDelegate>();
        var mockLogger = new Mock<ILogger<TokenBlacklistMiddleware>>();
        var mockTokenService = new Mock<ITokenBlacklistService>();
        var mockContext = CreateMockHttpContext(authorizationHeader: null);

        var middleware = new TokenBlacklistMiddleware(mockNext.Object, mockLogger.Object);

        // Act
        await middleware.InvokeAsync(mockContext.Object, mockTokenService.Object);

        // Assert
        mockNext.Verify(next => next(mockContext.Object), Times.Once);
        mockTokenService.Verify(service => service.IsTokenBlacklistedAsync(It.IsAny<string>()), Times.Never);
    }

    /// <summary>
    /// Tests that InvokeAsync calls the next middleware when Authorization header is empty.
    /// </summary>
    [TestMethod]
    public async Task InvokeAsync_EmptyAuthorizationHeader_CallsNextMiddleware()
    {
        // Arrange
        var mockNext = new Mock<RequestDelegate>();
        var mockLogger = new Mock<ILogger<TokenBlacklistMiddleware>>();
        var mockTokenService = new Mock<ITokenBlacklistService>();
        var mockContext = CreateMockHttpContext(authorizationHeader: string.Empty);

        var middleware = new TokenBlacklistMiddleware(mockNext.Object, mockLogger.Object);

        // Act
        await middleware.InvokeAsync(mockContext.Object, mockTokenService.Object);

        // Assert
        mockNext.Verify(next => next(mockContext.Object), Times.Once);
        mockTokenService.Verify(service => service.IsTokenBlacklistedAsync(It.IsAny<string>()), Times.Never);
    }

    /// <summary>
    /// Tests that InvokeAsync calls the next middleware when Authorization header doesn't start with "Bearer ".
    /// </summary>
    /// <param name="authHeader">The authorization header value that doesn't start with "Bearer ".</param>
    [TestMethod]
    [DataRow("Basic dXNlcjpwYXNz")]
    [DataRow("bearer token123")]
    [DataRow("BEARER token123")]
    [DataRow("Token abc")]
    [DataRow("BearerToken")]
    public async Task InvokeAsync_AuthorizationHeaderNotBearer_CallsNextMiddleware(string authHeader)
    {
        // Arrange
        var mockNext = new Mock<RequestDelegate>();
        var mockLogger = new Mock<ILogger<TokenBlacklistMiddleware>>();
        var mockTokenService = new Mock<ITokenBlacklistService>();
        var mockContext = CreateMockHttpContext(authorizationHeader: authHeader);

        var middleware = new TokenBlacklistMiddleware(mockNext.Object, mockLogger.Object);

        // Act
        await middleware.InvokeAsync(mockContext.Object, mockTokenService.Object);

        // Assert
        mockNext.Verify(next => next(mockContext.Object), Times.Once);
        mockTokenService.Verify(service => service.IsTokenBlacklistedAsync(It.IsAny<string>()), Times.Never);
    }

    /// <summary>
    /// Tests that InvokeAsync calls the next middleware when token is not blacklisted.
    /// </summary>
    [TestMethod]
    public async Task InvokeAsync_TokenNotBlacklisted_CallsNextMiddleware()
    {
        // Arrange
        var mockNext = new Mock<RequestDelegate>();
        var mockLogger = new Mock<ILogger<TokenBlacklistMiddleware>>();
        var mockTokenService = new Mock<ITokenBlacklistService>();
        var mockContext = CreateMockHttpContext(authorizationHeader: "Bearer validtoken123");

        mockTokenService
            .Setup(service => service.IsTokenBlacklistedAsync("validtoken123"))
            .ReturnsAsync(false);

        var middleware = new TokenBlacklistMiddleware(mockNext.Object, mockLogger.Object);

        // Act
        await middleware.InvokeAsync(mockContext.Object, mockTokenService.Object);

        // Assert
        mockNext.Verify(next => next(mockContext.Object), Times.Once);
        mockTokenService.Verify(service => service.IsTokenBlacklistedAsync("validtoken123"), Times.Once);
    }

    /// <summary>
    /// Tests that InvokeAsync returns 401 Unauthorized when token is blacklisted.
    /// </summary>
    [TestMethod]
    public async Task InvokeAsync_TokenBlacklisted_Returns401Unauthorized()
    {
        // Arrange
        var mockNext = new Mock<RequestDelegate>();
        var mockLogger = new Mock<ILogger<TokenBlacklistMiddleware>>();
        var mockTokenService = new Mock<ITokenBlacklistService>();

        // Use DefaultHttpContext instead of mocking for proper WriteAsJsonAsync support
        var context = new DefaultHttpContext();
        context.Request.Headers["Authorization"] = "Bearer blacklistedtoken";
        context.Response.Body = new MemoryStream();

        mockTokenService
            .Setup(service => service.IsTokenBlacklistedAsync("blacklistedtoken"))
            .ReturnsAsync(true);

        var middleware = new TokenBlacklistMiddleware(mockNext.Object, mockLogger.Object);

        // Act
        await middleware.InvokeAsync(context, mockTokenService.Object);

        // Assert
        mockNext.Verify(next => next(It.IsAny<HttpContext>()), Times.Never);
        Assert.AreEqual(StatusCodes.Status401Unauthorized, context.Response.StatusCode);
        Assert.AreEqual("application/json; charset=utf-8", context.Response.ContentType);
    }

    /// <summary>
    /// Tests that InvokeAsync logs a warning when a blacklisted token is used with a known IP address.
    /// </summary>
    [TestMethod]
    public async Task InvokeAsync_TokenBlacklistedWithKnownIP_LogsWarning()
    {
        // Arrange
        var mockNext = new Mock<RequestDelegate>();
        var mockLogger = new Mock<ILogger<TokenBlacklistMiddleware>>();
        var mockTokenService = new Mock<ITokenBlacklistService>();
        var ipAddress = IPAddress.Parse("192.168.1.1");

        // Use DefaultHttpContext instead of mocking for proper WriteAsJsonAsync support
        var context = new DefaultHttpContext();
        context.Request.Headers["Authorization"] = "Bearer blacklistedtoken";
        context.Response.Body = new MemoryStream();
        context.Connection.RemoteIpAddress = ipAddress;

        mockTokenService
            .Setup(service => service.IsTokenBlacklistedAsync("blacklistedtoken"))
            .ReturnsAsync(true);

        var middleware = new TokenBlacklistMiddleware(mockNext.Object, mockLogger.Object);

        // Act
        await middleware.InvokeAsync(context, mockTokenService.Object);

        // Assert
        mockLogger.Verify(
            logger => logger.Log(
                LogLevel.Warning,
                It.IsAny<EventId>(),
                It.Is<It.IsAnyType>((v, t) => v.ToString()!.Contains("Attempt to use blacklisted token")),
                It.IsAny<Exception>(),
                It.IsAny<Func<It.IsAnyType, Exception?, string>>()),
            Times.Once);
    }

    /// <summary>
    /// Tests that InvokeAsync logs a warning with "Unknown" IP when RemoteIpAddress is null.
    /// </summary>
    [TestMethod]
    public async Task InvokeAsync_TokenBlacklistedWithNullIP_LogsWarningWithUnknown()
    {
        // Arrange
        var mockNext = new Mock<RequestDelegate>();
        var mockLogger = new Mock<ILogger<TokenBlacklistMiddleware>>();
        var mockTokenService = new Mock<ITokenBlacklistService>();

        // Use DefaultHttpContext instead of mocking for proper WriteAsJsonAsync support
        var context = new DefaultHttpContext();
        context.Request.Headers["Authorization"] = "Bearer blacklistedtoken";
        context.Response.Body = new MemoryStream();

        mockTokenService
            .Setup(service => service.IsTokenBlacklistedAsync("blacklistedtoken"))
            .ReturnsAsync(true);

        var middleware = new TokenBlacklistMiddleware(mockNext.Object, mockLogger.Object);

        // Act
        await middleware.InvokeAsync(context, mockTokenService.Object);

        // Assert
        mockLogger.Verify(
            logger => logger.Log(
                LogLevel.Warning,
                It.IsAny<EventId>(),
                It.Is<It.IsAnyType>((v, t) => v.ToString()!.Contains("Attempt to use blacklisted token")),
                It.IsAny<Exception>(),
                It.IsAny<Func<It.IsAnyType, Exception?, string>>()),
            Times.Once);
    }

    /// <summary>
    /// Tests that InvokeAsync trims whitespace from the token before checking if it's blacklisted.
    /// </summary>
    /// <param name="authHeader">The authorization header with various whitespace patterns.</param>
    /// <param name="expectedToken">The expected token after trimming.</param>
    [TestMethod]
    [DataRow("Bearer   tokenWithLeadingSpaces", "tokenWithLeadingSpaces")]
    [DataRow("Bearer tokenWithTrailingSpaces   ", "tokenWithTrailingSpaces")]
    [DataRow("Bearer   tokenWithBothSpaces   ", "tokenWithBothSpaces")]
    [DataRow("Bearer \t\ttokenWithTabs\t\t", "tokenWithTabs")]
    public async Task InvokeAsync_TokenWithWhitespace_TrimsAndChecksBlacklist(string authHeader, string expectedToken)
    {
        // Arrange
        var mockNext = new Mock<RequestDelegate>();
        var mockLogger = new Mock<ILogger<TokenBlacklistMiddleware>>();
        var mockTokenService = new Mock<ITokenBlacklistService>();
        var mockContext = CreateMockHttpContext(authorizationHeader: authHeader);

        mockTokenService
            .Setup(service => service.IsTokenBlacklistedAsync(expectedToken))
            .ReturnsAsync(false);

        var middleware = new TokenBlacklistMiddleware(mockNext.Object, mockLogger.Object);

        // Act
        await middleware.InvokeAsync(mockContext.Object, mockTokenService.Object);

        // Assert
        mockTokenService.Verify(service => service.IsTokenBlacklistedAsync(expectedToken), Times.Once);
        mockNext.Verify(next => next(mockContext.Object), Times.Once);
    }

    /// <summary>
    /// Tests that InvokeAsync handles Bearer token with only whitespace after "Bearer " prefix.
    /// </summary>
    [TestMethod]
    [DataRow("Bearer ")]
    [DataRow("Bearer    ")]
    [DataRow("Bearer \t")]
    public async Task InvokeAsync_BearerWithOnlyWhitespace_ChecksEmptyToken(string authHeader)
    {
        // Arrange
        var mockNext = new Mock<RequestDelegate>();
        var mockLogger = new Mock<ILogger<TokenBlacklistMiddleware>>();
        var mockTokenService = new Mock<ITokenBlacklistService>();
        var mockContext = CreateMockHttpContext(authorizationHeader: authHeader);

        mockTokenService
            .Setup(service => service.IsTokenBlacklistedAsync(string.Empty))
            .ReturnsAsync(false);

        var middleware = new TokenBlacklistMiddleware(mockNext.Object, mockLogger.Object);

        // Act
        await middleware.InvokeAsync(mockContext.Object, mockTokenService.Object);

        // Assert
        mockTokenService.Verify(service => service.IsTokenBlacklistedAsync(string.Empty), Times.Once);
    }

    /// <summary>
    /// Tests that InvokeAsync handles various valid token formats.
    /// </summary>
    /// <param name="token">The token value to test.</param>
    [TestMethod]
    [DataRow("eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJzdWIiOiIxMjM0NTY3ODkwIn0.dozjgNryP4J3jVmNHl0w5N_XgL0n3I9PlFUP0THsR8U")]
    [DataRow("simple-token-123")]
    [DataRow("token_with_underscores")]
    [DataRow("TOKEN-WITH-CAPS-123")]
    [DataRow("a")]
    [DataRow("verylongtokenverylongtokenverylongtokenverylongtokenverylongtokenverylongtokenverylongtokenverylongtokenverylongtokenverylongtoken")]
    public async Task InvokeAsync_VariousTokenFormats_ChecksBlacklist(string token)
    {
        // Arrange
        var mockNext = new Mock<RequestDelegate>();
        var mockLogger = new Mock<ILogger<TokenBlacklistMiddleware>>();
        var mockTokenService = new Mock<ITokenBlacklistService>();
        var mockContext = CreateMockHttpContext(authorizationHeader: $"Bearer {token}");

        mockTokenService
            .Setup(service => service.IsTokenBlacklistedAsync(token))
            .ReturnsAsync(false);

        var middleware = new TokenBlacklistMiddleware(mockNext.Object, mockLogger.Object);

        // Act
        await middleware.InvokeAsync(mockContext.Object, mockTokenService.Object);

        // Assert
        mockTokenService.Verify(service => service.IsTokenBlacklistedAsync(token), Times.Once);
        mockNext.Verify(next => next(mockContext.Object), Times.Once);
    }

    /// <summary>
    /// Tests that InvokeAsync writes the correct JSON error message when token is blacklisted.
    /// </summary>
    [TestMethod]
    public async Task InvokeAsync_TokenBlacklisted_WritesCorrectJsonResponse()
    {
        // Arrange
        var mockNext = new Mock<RequestDelegate>();
        var mockLogger = new Mock<ILogger<TokenBlacklistMiddleware>>();
        var mockTokenService = new Mock<ITokenBlacklistService>();

        // Use DefaultHttpContext instead of mocking for proper WriteAsJsonAsync support
        var context = new DefaultHttpContext();
        context.Request.Headers["Authorization"] = "Bearer blacklistedtoken";
        context.Response.Body = new MemoryStream();

        mockTokenService
            .Setup(service => service.IsTokenBlacklistedAsync("blacklistedtoken"))
            .ReturnsAsync(true);

        var middleware = new TokenBlacklistMiddleware(mockNext.Object, mockLogger.Object);

        // Act
        await middleware.InvokeAsync(context, mockTokenService.Object);

        // Assert
        context.Response.Body.Position = 0;
        using var reader = new StreamReader(context.Response.Body);
        var responseText = await reader.ReadToEndAsync();
        Assert.IsTrue(responseText.Contains("Token is no longer valid"));
        Assert.IsTrue(responseText.Contains("Please log in again"));
    }

    /// <summary>
    /// Helper method to create a mock HttpContext with specified parameters.
    /// </summary>
    /// <param name="authorizationHeader">The Authorization header value.</param>
    /// <param name="remoteIpAddress">The remote IP address.</param>
    /// <param name="responseBody">The response body stream.</param>
    /// <returns>A mock HttpContext.</returns>
    private static Mock<HttpContext> CreateMockHttpContext(
        string? authorizationHeader = null,
        IPAddress? remoteIpAddress = null,
        MemoryStream? responseBody = null)
    {
        var mockContext = new Mock<HttpContext>();
        var mockRequest = new Mock<HttpRequest>();
        var mockResponse = new Mock<HttpResponse>();
        var mockConnection = new Mock<ConnectionInfo>();
        var mockHeaders = new Mock<IHeaderDictionary>();

        // Setup headers
        var headerValue = authorizationHeader != null
            ? new StringValues(authorizationHeader)
            : StringValues.Empty;
        mockHeaders.Setup(h => h["Authorization"]).Returns(headerValue);
        mockRequest.Setup(r => r.Headers).Returns(mockHeaders.Object);

        // Setup connection
        mockConnection.Setup(c => c.RemoteIpAddress).Returns(remoteIpAddress);

        // Setup response
        mockResponse.SetupProperty(r => r.StatusCode);
        mockResponse.SetupProperty(r => r.ContentType);

        if (responseBody != null)
        {
            mockResponse.Setup(r => r.Body).Returns(responseBody);
        }

        // Setup context
        mockContext.Setup(c => c.Request).Returns(mockRequest.Object);
        mockContext.Setup(c => c.Response).Returns(mockResponse.Object);
        mockContext.Setup(c => c.Connection).Returns(mockConnection.Object);

        return mockContext;
    }

    /// <summary>
    /// Tests that the constructor successfully creates an instance when provided with valid non-null parameters.
    /// </summary>
    [TestMethod]
    public void Constructor_ValidParameters_CreatesInstance()
    {
        // Arrange
        RequestDelegate next = context => Task.CompletedTask;
        var loggerMock = new Mock<ILogger<TokenBlacklistMiddleware>>();

        // Act
        var middleware = new TokenBlacklistMiddleware(next, loggerMock.Object);

        // Assert
        Assert.IsNotNull(middleware);
    }

    /// <summary>
    /// Tests that the constructor accepts a null RequestDelegate parameter without throwing an exception.
    /// This tests an edge case where the nullability contract is violated.
    /// </summary>
    [TestMethod]
    public void Constructor_NullNextParameter_CreatesInstance()
    {
        // Arrange
        RequestDelegate? next = null;
        var loggerMock = new Mock<ILogger<TokenBlacklistMiddleware>>();

        // Act
        var middleware = new TokenBlacklistMiddleware(next!, loggerMock.Object);

        // Assert
        Assert.IsNotNull(middleware);
    }

    /// <summary>
    /// Tests that the constructor accepts a null ILogger parameter without throwing an exception.
    /// This tests an edge case where the nullability contract is violated.
    /// </summary>
    [TestMethod]
    public void Constructor_NullLoggerParameter_CreatesInstance()
    {
        // Arrange
        RequestDelegate next = context => Task.CompletedTask;
        ILogger<TokenBlacklistMiddleware>? logger = null;

        // Act
        var middleware = new TokenBlacklistMiddleware(next, logger!);

        // Assert
        Assert.IsNotNull(middleware);
    }

    /// <summary>
    /// Tests that the constructor accepts both null parameters without throwing an exception.
    /// This tests an extreme edge case where all nullability contracts are violated.
    /// </summary>
    [TestMethod]
    public void Constructor_BothParametersNull_CreatesInstance()
    {
        // Arrange
        RequestDelegate? next = null;
        ILogger<TokenBlacklistMiddleware>? logger = null;

        // Act
        var middleware = new TokenBlacklistMiddleware(next!, logger!);

        // Assert
        Assert.IsNotNull(middleware);
    }

    /// <summary>
    /// Tests that InvokeAsync does not call next middleware when token is blacklisted.
    /// </summary>
    [TestMethod]
    public async Task InvokeAsync_TokenBlacklisted_DoesNotCallNextMiddleware()
    {
        // Arrange
        var mockNext = new Mock<RequestDelegate>();
        var mockLogger = new Mock<ILogger<TokenBlacklistMiddleware>>();
        var mockTokenService = new Mock<ITokenBlacklistService>();
        var responseBody = new MemoryStream();
        var mockContext = CreateMockHttpContext(
            authorizationHeader: "Bearer blacklistedtoken",
            responseBody: responseBody);

        mockTokenService.Setup(s => s.IsTokenBlacklistedAsync("blacklistedtoken")).ReturnsAsync(true);

        var middleware = new TokenBlacklistMiddleware(mockNext.Object, mockLogger.Object);

        // Act
        await middleware.InvokeAsync(mockContext.Object, mockTokenService.Object);

        // Assert
        mockNext.Verify(next => next(It.IsAny<HttpContext>()), Times.Never);
    }

    /// <summary>
    /// Tests that InvokeAsync does not log when token is not blacklisted.
    /// </summary>
    [TestMethod]
    public async Task InvokeAsync_TokenNotBlacklisted_DoesNotLog()
    {
        // Arrange
        var mockNext = new Mock<RequestDelegate>();
        var mockLogger = new Mock<ILogger<TokenBlacklistMiddleware>>();
        var mockTokenService = new Mock<ITokenBlacklistService>();
        var mockContext = CreateMockHttpContext(authorizationHeader: "Bearer validtoken");

        mockTokenService.Setup(s => s.IsTokenBlacklistedAsync("validtoken")).ReturnsAsync(false);

        var middleware = new TokenBlacklistMiddleware(mockNext.Object, mockLogger.Object);

        // Act
        await middleware.InvokeAsync(mockContext.Object, mockTokenService.Object);

        // Assert
        mockLogger.Verify(
            logger => logger.Log(
                It.IsAny<LogLevel>(),
                It.IsAny<EventId>(),
                It.IsAny<It.IsAnyType>(),
                It.IsAny<Exception>(),
                It.IsAny<Func<It.IsAnyType, Exception?, string>>()),
            Times.Never);
    }

    /// <summary>
    /// Tests that InvokeAsync does not log when no valid Bearer token is present.
    /// </summary>
    [TestMethod]
    public async Task InvokeAsync_NoValidBearerToken_DoesNotLog()
    {
        // Arrange
        var mockNext = new Mock<RequestDelegate>();
        var mockLogger = new Mock<ILogger<TokenBlacklistMiddleware>>();
        var mockTokenService = new Mock<ITokenBlacklistService>();
        var mockContext = CreateMockHttpContext(authorizationHeader: "Basic token");

        var middleware = new TokenBlacklistMiddleware(mockNext.Object, mockLogger.Object);

        // Act
        await middleware.InvokeAsync(mockContext.Object, mockTokenService.Object);

        // Assert
        mockLogger.Verify(
            logger => logger.Log(
                It.IsAny<LogLevel>(),
                It.IsAny<EventId>(),
                It.IsAny<It.IsAnyType>(),
                It.IsAny<Exception>(),
                It.IsAny<Func<It.IsAnyType, Exception?, string>>()),
            Times.Never);
    }

    /// <summary>
    /// Tests that InvokeAsync handles whitespace-only Authorization header correctly.
    /// </summary>
    [TestMethod]
    [DataRow("   ")]
    [DataRow("\t\t")]
    [DataRow("\n")]
    public async Task InvokeAsync_WhitespaceOnlyAuthorizationHeader_CallsNextMiddleware(string authHeader)
    {
        // Arrange
        var mockNext = new Mock<RequestDelegate>();
        var mockLogger = new Mock<ILogger<TokenBlacklistMiddleware>>();
        var mockTokenService = new Mock<ITokenBlacklistService>();
        var mockContext = CreateMockHttpContext(authorizationHeader: authHeader);

        var middleware = new TokenBlacklistMiddleware(mockNext.Object, mockLogger.Object);

        // Act
        await middleware.InvokeAsync(mockContext.Object, mockTokenService.Object);

        // Assert
        mockNext.Verify(next => next(mockContext.Object), Times.Once);
        mockTokenService.Verify(service => service.IsTokenBlacklistedAsync(It.IsAny<string>()), Times.Never);
    }

    /// <summary>
    /// Tests that InvokeAsync handles tokens with special characters correctly.
    /// </summary>
    /// <param name="token">The token with special characters.</param>
    [TestMethod]
    [DataRow("token@#$%")]
    [DataRow("token!@#$%^&*()")]
    [DataRow("token.with.dots")]
    [DataRow("token/with/slashes")]
    [DataRow("token\\with\\backslashes")]
    public async Task InvokeAsync_TokenWithSpecialCharacters_ChecksBlacklist(string token)
    {
        // Arrange
        var mockNext = new Mock<RequestDelegate>();
        var mockLogger = new Mock<ILogger<TokenBlacklistMiddleware>>();
        var mockTokenService = new Mock<ITokenBlacklistService>();
        var mockContext = CreateMockHttpContext(authorizationHeader: $"Bearer {token}");

        mockTokenService.Setup(s => s.IsTokenBlacklistedAsync(token)).ReturnsAsync(false);

        var middleware = new TokenBlacklistMiddleware(mockNext.Object, mockLogger.Object);

        // Act
        await middleware.InvokeAsync(mockContext.Object, mockTokenService.Object);

        // Assert
        mockTokenService.Verify(service => service.IsTokenBlacklistedAsync(token), Times.Once);
        mockNext.Verify(next => next(mockContext.Object), Times.Once);
    }

    /// <summary>
    /// Tests that InvokeAsync handles different IP address formats when logging blacklisted token attempts.
    /// </summary>
    /// <param name="ipAddressString">The IP address string to test.</param>
    [TestMethod]
    [DataRow("127.0.0.1")]
    [DataRow("0.0.0.0")]
    [DataRow("255.255.255.255")]
    [DataRow("::1")]
    [DataRow("2001:0db8:85a3:0000:0000:8a2e:0370:7334")]
    public async Task InvokeAsync_TokenBlacklistedWithVariousIPFormats_LogsWarning(string ipAddressString)
    {
        // Arrange
        var mockNext = new Mock<RequestDelegate>();
        var mockLogger = new Mock<ILogger<TokenBlacklistMiddleware>>();
        var mockTokenService = new Mock<ITokenBlacklistService>();
        var ipAddress = IPAddress.Parse(ipAddressString);
        var responseBody = new MemoryStream();
        var mockContext = CreateMockHttpContext(
            authorizationHeader: "Bearer blacklistedtoken",
            remoteIpAddress: ipAddress,
            responseBody: responseBody);

        mockTokenService.Setup(s => s.IsTokenBlacklistedAsync("blacklistedtoken")).ReturnsAsync(true);

        var middleware = new TokenBlacklistMiddleware(mockNext.Object, mockLogger.Object);

        // Act
        await middleware.InvokeAsync(mockContext.Object, mockTokenService.Object);

        // Assert
        mockLogger.Verify(
            logger => logger.Log(
                LogLevel.Warning,
                It.IsAny<EventId>(),
                It.Is<It.IsAnyType>((v, t) => v.ToString()!.Contains(ipAddress.ToString())),
                It.IsAny<Exception>(),
                It.IsAny<Func<It.IsAnyType, Exception?, string>>()),
            Times.Once);
    }
}