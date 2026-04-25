using System;
using System.Collections.Generic;
using System.IO;
using System.Net;
using System.Threading.Tasks;

using Microsoft.AspNetCore.Builder;
using Microsoft.AspNetCore.Http;
using Microsoft.Extensions.Logging;
using Microsoft.Extensions.Primitives;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using Moq;
using SignLanguageApi.Middleware;

namespace SignLanguageApi.Middleware.UnitTests;

/// <summary>
/// Unit tests for the <see cref="RateLimitMiddleware"/> class.
/// </summary>
[TestClass]
public class RateLimitMiddlewareTests
{
    /// <summary>
    /// Tests that the constructor successfully creates an instance when provided with valid non-null parameters.
    /// Input: Valid RequestDelegate and ILogger instances.
    /// Expected: Constructor completes without throwing an exception.
    /// </summary>
    [TestMethod]
    public void Constructor_WithValidParameters_CompletesSuccessfully()
    {
        // Arrange
        RequestDelegate next = context => Task.CompletedTask;
        var mockLogger = new Mock<ILogger<RateLimitMiddleware>>();

        // Act
        var middleware = new RateLimitMiddleware(next, mockLogger.Object);

        // Assert
        Assert.IsNotNull(middleware);
    }

    /// <summary>
    /// Tests that the constructor accepts a null RequestDelegate parameter.
    /// Input: Null RequestDelegate.
    /// Expected: Constructor completes without throwing (no runtime validation for nullable reference types).
    /// </summary>
    [TestMethod]
    public void Constructor_WithNullNext_CompletesSuccessfully()
    {
        // Arrange
        RequestDelegate? next = null;
        var mockLogger = new Mock<ILogger<RateLimitMiddleware>>();

        // Act
        var middleware = new RateLimitMiddleware(next!, mockLogger.Object);

        // Assert
        Assert.IsNotNull(middleware);
    }

    /// <summary>
    /// Tests that the constructor accepts a null ILogger parameter.
    /// Input: Null ILogger.
    /// Expected: Constructor completes without throwing (no runtime validation for nullable reference types).
    /// </summary>
    [TestMethod]
    public void Constructor_WithNullLogger_CompletesSuccessfully()
    {
        // Arrange
        RequestDelegate next = context => Task.CompletedTask;
        ILogger<RateLimitMiddleware>? logger = null;

        // Act
        var middleware = new RateLimitMiddleware(next, logger!);

        // Assert
        Assert.IsNotNull(middleware);
    }

    /// <summary>
    /// Tests that the constructor accepts both null parameters.
    /// Input: Null RequestDelegate and null ILogger.
    /// Expected: Constructor completes without throwing (no runtime validation for nullable reference types).
    /// </summary>
    [TestMethod]
    public void Constructor_WithBothParametersNull_CompletesSuccessfully()
    {
        // Arrange
        RequestDelegate? next = null;
        ILogger<RateLimitMiddleware>? logger = null;

        // Act
        var middleware = new RateLimitMiddleware(next!, logger!);

        // Assert
        Assert.IsNotNull(middleware);
    }

    /// <summary>
    /// Tests that InvokeAsync bypasses rate limiting when the request path does not contain auth endpoints.
    /// Expects the next middleware to be invoked without any rate limiting logic applied.
    /// </summary>
    /// <param name="path">The request path to test.</param>
    [TestMethod]
    [DataRow("/api/users")]
    [DataRow("/api/products")]
    [DataRow("/")]
    [DataRow("/api/auth")]
    [DataRow("/authlogin")]
    [DataRow("/authentication/login")]
    public async Task InvokeAsync_PathDoesNotContainAuthEndpoints_BypassesRateLimiting(string path)
    {
        // Arrange
        var nextMock = new Mock<RequestDelegate>();
        var loggerMock = new Mock<ILogger<RateLimitMiddleware>>();
        var middleware = new RateLimitMiddleware(nextMock.Object, loggerMock.Object);

        var contextMock = new Mock<HttpContext>();
        var requestMock = new Mock<HttpRequest>();
        var responseMock = new Mock<HttpResponse>();
        var connectionMock = new Mock<ConnectionInfo>();
        var headersMock = new Mock<IHeaderDictionary>();

        // Setup path
        requestMock.Setup(r => r.Path).Returns(new PathString(path));

        // Setup headers to return false for any TryGetValue call
        headersMock.Setup(h => h.TryGetValue(It.IsAny<string>(), out It.Ref<StringValues>.IsAny))
            .Returns(false);

        requestMock.Setup(r => r.Headers).Returns(headersMock.Object);

        // Setup connection with remote IP
        var ipAddress = IPAddress.Parse("192.168.1.2");
        connectionMock.Setup(c => c.RemoteIpAddress).Returns(ipAddress);

        // Setup response (no need to mock WriteAsJsonAsync as it won't be called for these paths)
        responseMock.SetupProperty(r => r.StatusCode);
        responseMock.SetupProperty(r => r.ContentType);

        // Wire up the context
        contextMock.Setup(c => c.Request).Returns(requestMock.Object);
        contextMock.Setup(c => c.Response).Returns(responseMock.Object);
        contextMock.Setup(c => c.Connection).Returns(connectionMock.Object);

        // Act
        await middleware.InvokeAsync(contextMock.Object);

        // Assert
        nextMock.Verify(next => next(It.IsAny<HttpContext>()), Times.Once);
    }

    /// <summary>
    /// Tests that InvokeAsync returns 429 Too Many Requests when the rate limit is exceeded.
    /// After 5 requests, expects the 6th request to return 429 status code and not invoke next middleware.
    /// </summary>
    [TestMethod]
    public async Task InvokeAsync_ExceedsRateLimit_Returns429TooManyRequests()
    {
        // Arrange
        var nextMock = new Mock<RequestDelegate>();
        var loggerMock = new Mock<ILogger<RateLimitMiddleware>>();
        var middleware = new RateLimitMiddleware(nextMock.Object, loggerMock.Object);

        var ipAddress = $"10.3.0.{Guid.NewGuid().GetHashCode() & 0xFF}";
        var path = "/api/auth/login";

        // Make 5 requests to reach the limit
        for (int i = 0; i < 5; i++)
        {
            var context = new DefaultHttpContext();
            context.Request.Path = new PathString(path);
            context.Connection.RemoteIpAddress = IPAddress.Parse(ipAddress);

            await middleware.InvokeAsync(context);
        }

        // Act - Make the 6th request that should be rate limited
        var rateLimitedContext = new DefaultHttpContext();
        rateLimitedContext.Request.Path = new PathString(path);
        rateLimitedContext.Connection.RemoteIpAddress = IPAddress.Parse(ipAddress);
        rateLimitedContext.Response.Body = new System.IO.MemoryStream();

        await middleware.InvokeAsync(rateLimitedContext);

        // Assert
        Assert.AreEqual(StatusCodes.Status429TooManyRequests, rateLimitedContext.Response.StatusCode);
        Assert.IsTrue(rateLimitedContext.Response.ContentType?.StartsWith("application/json") == true);
        nextMock.Verify(next => next(It.IsAny<HttpContext>()), Times.Exactly(5)); // Only first 5 calls
    }

    /// <summary>
    /// Tests that InvokeAsync writes a JSON response with error message and retry information when rate limited.
    /// Expects the response to contain a message and retryAfterSeconds field.
    /// </summary>
    [TestMethod]
    public async Task InvokeAsync_ExceedsRateLimit_WritesJsonResponse()
    {
        // Arrange
        var nextMock = new Mock<RequestDelegate>();
        var loggerMock = new Mock<ILogger<RateLimitMiddleware>>();
        var middleware = new RateLimitMiddleware(nextMock.Object, loggerMock.Object);

        var ipAddress = $"10.5.0.{Guid.NewGuid().GetHashCode() & 0xFF}";
        var path = "/api/auth/login";

        // Make 5 requests to reach the limit
        for (int i = 0; i < 5; i++)
        {
            var tempContext = new DefaultHttpContext();
            tempContext.Request.Path = new PathString(path);
            tempContext.Connection.RemoteIpAddress = IPAddress.Parse(ipAddress);
            tempContext.Response.Body = new MemoryStream();

            await middleware.InvokeAsync(tempContext);
        }

        // Act - Make the 6th request that should be rate limited
        var context = new DefaultHttpContext();
        var responseBody = new MemoryStream();
        context.Request.Path = new PathString(path);
        context.Connection.RemoteIpAddress = IPAddress.Parse(ipAddress);
        context.Response.Body = responseBody;

        await middleware.InvokeAsync(context);

        // Assert
        responseBody.Position = 0;
        var reader = new StreamReader(responseBody);
        var responseJson = await reader.ReadToEndAsync();

        Assert.IsFalse(string.IsNullOrEmpty(responseJson));

        var capturedResponse = System.Text.Json.JsonSerializer.Deserialize<System.Text.Json.JsonElement>(responseJson);

        var message = capturedResponse.GetProperty("message").GetString();
        var retryAfterSeconds = capturedResponse.GetProperty("retryAfterSeconds").GetInt32();

        Assert.AreEqual("Too many requests. Please try again later.", message);
        Assert.IsTrue(retryAfterSeconds > 0);
    }

    /// <summary>
    /// Tests that InvokeAsync allows the 5th request before returning 429 on the 6th.
    /// Expects exactly 5 requests to be allowed through rate limiting.
    /// </summary>
    [TestMethod]
    public async Task InvokeAsync_FifthRequest_IsAllowed()
    {
        // Arrange
        var nextMock = new Mock<RequestDelegate>();
        var loggerMock = new Mock<ILogger<RateLimitMiddleware>>();
        var middleware = new RateLimitMiddleware(nextMock.Object, loggerMock.Object);

        var ipAddress = $"10.6.0.{Guid.NewGuid().GetHashCode() & 0xFF}";
        var path = "/api/auth/login";

        // Act - Make exactly 5 requests
        for (int i = 0; i < 5; i++)
        {
            var contextMock = CreateHttpContextMock(path, ipAddress);
            await middleware.InvokeAsync(contextMock.Object);
        }

        // Assert - All 5 should have been allowed
        nextMock.Verify(next => next(It.IsAny<HttpContext>()), Times.Exactly(5));
    }

    /// <summary>
    /// Tests that InvokeAsync falls back to RemoteIpAddress when no proxy headers are present.
    /// Expects rate limiting to track requests by the connection's remote IP address.
    /// </summary>
    [TestMethod]
    public async Task InvokeAsync_WithoutProxyHeaders_UsesRemoteIpAddress()
    {
        // Arrange
        var nextMock = new Mock<RequestDelegate>();
        var loggerMock = new Mock<ILogger<RateLimitMiddleware>>();
        var middleware = new RateLimitMiddleware(nextMock.Object, loggerMock.Object);

        var remoteIp = $"172.16.0.{Guid.NewGuid().GetHashCode() & 0xFF}";
        var path = "/api/auth/login";

        // Act - Make 5 requests without proxy headers
        for (int i = 0; i < 5; i++)
        {
            var contextMock = CreateHttpContextMock(path, remoteIp);
            await middleware.InvokeAsync(contextMock.Object);
        }

        // Make 6th request that should be rate limited
        var rateLimitedContextMock = CreateHttpContextMock(path, remoteIp);
        var responseMock = Mock.Get(rateLimitedContextMock.Object.Response);

        await middleware.InvokeAsync(rateLimitedContextMock.Object);

        // Assert - Should be rate limited based on remote IP
        responseMock.VerifySet(r => r.StatusCode = StatusCodes.Status429TooManyRequests, Times.Once);
    }

    /// <summary>
    /// Tests that InvokeAsync handles empty string path by bypassing rate limiting.
    /// Expects the next middleware to be invoked without rate limiting applied.
    /// </summary>
    [TestMethod]
    public async Task InvokeAsync_EmptyPath_BypassesRateLimiting()
    {
        // Arrange
        var nextMock = new Mock<RequestDelegate>();
        var loggerMock = new Mock<ILogger<RateLimitMiddleware>>();
        var middleware = new RateLimitMiddleware(nextMock.Object, loggerMock.Object);

        var contextMock = new Mock<HttpContext>();
        var requestMock = new Mock<HttpRequest>();
        var responseMock = new Mock<HttpResponse>();
        var connectionMock = new Mock<ConnectionInfo>();
        var headersMock = new Mock<IHeaderDictionary>();

        // Setup path
        requestMock.Setup(r => r.Path).Returns(new PathString(string.Empty));

        // Setup headers
        headersMock.Setup(h => h.TryGetValue(It.IsAny<string>(), out It.Ref<StringValues>.IsAny))
            .Returns(false);

        requestMock.Setup(r => r.Headers).Returns(headersMock.Object);

        // Setup connection with remote IP
        IPAddress? ipAddress = IPAddress.Parse("192.168.1.10");
        connectionMock.Setup(c => c.RemoteIpAddress).Returns(ipAddress);

        // Wire up the context
        contextMock.Setup(c => c.Request).Returns(requestMock.Object);
        contextMock.Setup(c => c.Response).Returns(responseMock.Object);
        contextMock.Setup(c => c.Connection).Returns(connectionMock.Object);

        // Act
        await middleware.InvokeAsync(contextMock.Object);

        // Assert
        nextMock.Verify(next => next(It.IsAny<HttpContext>()), Times.Once);
    }

    /// <summary>
    /// Helper method to create a mocked HttpContext with specified path and IP address.
    /// </summary>
    /// <param name="path">The request path.</param>
    /// <param name="remoteIp">The remote IP address.</param>
    /// <param name="forwardedFor">Optional X-Forwarded-For header value.</param>
    /// <returns>A mocked HttpContext.</returns>
    private Mock<HttpContext> CreateHttpContextMock(string? path, string remoteIp, string? forwardedFor = null)
    {
        var contextMock = new Mock<HttpContext>();
        var requestMock = new Mock<HttpRequest>();
        var responseMock = new Mock<HttpResponse>();
        var connectionMock = new Mock<ConnectionInfo>();
        var headersMock = new Mock<IHeaderDictionary>();

        // Setup path
        requestMock.Setup(r => r.Path).Returns(new PathString(path));

        // Setup headers
        if (!string.IsNullOrEmpty(forwardedFor))
        {
            headersMock.Setup(h => h.TryGetValue("X-Forwarded-For", out It.Ref<StringValues>.IsAny))
                .Returns(new HeaderTryGetValueCallback((string key, out StringValues value) =>
                {
                    if (key == "X-Forwarded-For")
                    {
                        value = new StringValues(forwardedFor);
                        return true;
                    }
                    value = default;
                    return false;
                }));
        }
        else
        {
            headersMock.Setup(h => h.TryGetValue(It.IsAny<string>(), out It.Ref<StringValues>.IsAny))
                .Returns(false);
        }

        requestMock.Setup(r => r.Headers).Returns(headersMock.Object);

        // Setup connection with remote IP
        IPAddress? ipAddress = IPAddress.TryParse(remoteIp, out var parsedIp) ? parsedIp : null;
        connectionMock.Setup(c => c.RemoteIpAddress).Returns(ipAddress);

        // Setup response
        responseMock.SetupProperty(r => r.StatusCode);
        responseMock.SetupProperty(r => r.ContentType);
        responseMock.Setup(r => r.WriteAsJsonAsync(It.IsAny<object>(), It.IsAny<System.Threading.CancellationToken>()))
            .Returns(Task.CompletedTask);

        // Wire up the context
        contextMock.Setup(c => c.Request).Returns(requestMock.Object);
        contextMock.Setup(c => c.Response).Returns(responseMock.Object);
        contextMock.Setup(c => c.Connection).Returns(connectionMock.Object);

        return contextMock;
    }

    /// <summary>
    /// Helper method to create a mocked HttpContext with X-Real-IP header.
    /// </summary>
    /// <param name="path">The request path.</param>
    /// <param name="remoteIp">The remote IP address.</param>
    /// <param name="realIp">The X-Real-IP header value.</param>
    /// <returns>A mocked HttpContext.</returns>
    private Mock<HttpContext> CreateHttpContextMockWithRealIp(string path, string remoteIp, string realIp)
    {
        var contextMock = new Mock<HttpContext>();
        var requestMock = new Mock<HttpRequest>();
        var responseMock = new Mock<HttpResponse>();
        var connectionMock = new Mock<ConnectionInfo>();
        var headersMock = new Mock<IHeaderDictionary>();

        // Setup path
        requestMock.Setup(r => r.Path).Returns(new PathString(path));

        // Setup headers with X-Real-IP
        headersMock.Setup(h => h.TryGetValue("X-Real-IP", out It.Ref<StringValues>.IsAny))
            .Returns(new HeaderTryGetValueCallback((string key, out StringValues value) =>
            {
                if (key == "X-Real-IP")
                {
                    value = new StringValues(realIp);
                    return true;
                }
                value = default;
                return false;
            }));

        headersMock.Setup(h => h.TryGetValue("X-Forwarded-For", out It.Ref<StringValues>.IsAny))
            .Returns(false);

        requestMock.Setup(r => r.Headers).Returns(headersMock.Object);

        // Setup connection with remote IP
        IPAddress? ipAddress = IPAddress.TryParse(remoteIp, out var parsedIp) ? parsedIp : null;
        connectionMock.Setup(c => c.RemoteIpAddress).Returns(ipAddress);

        // Setup response
        responseMock.SetupProperty(r => r.StatusCode);
        responseMock.SetupProperty(r => r.ContentType);
        responseMock.Setup(r => r.WriteAsJsonAsync(It.IsAny<object>(), It.IsAny<System.Threading.CancellationToken>()))
            .Returns(Task.CompletedTask);

        // Wire up the context
        contextMock.Setup(c => c.Request).Returns(requestMock.Object);
        contextMock.Setup(c => c.Response).Returns(responseMock.Object);
        contextMock.Setup(c => c.Connection).Returns(connectionMock.Object);

        return contextMock;
    }

    /// <summary>
    /// Delegate for mocking IHeaderDictionary.TryGetValue behavior.
    /// </summary>
    private delegate bool HeaderTryGetValueCallback(string key, out StringValues value);

    /// <summary>
    /// Tests that InvokeAsync applies rate limiting to auth/register endpoint.
    /// Input: Path containing "auth/register".
    /// Expected: Rate limiting is applied and 6th request returns 429.
    /// </summary>
    [TestMethod]
    public async Task InvokeAsync_AuthRegisterEndpoint_AppliesRateLimiting()
    {
        // Arrange
        var nextMock = new Mock<RequestDelegate>();
        var loggerMock = new Mock<ILogger<RateLimitMiddleware>>();
        var middleware = new RateLimitMiddleware(nextMock.Object, loggerMock.Object);

        var ipAddress = $"10.20.0.{Guid.NewGuid().GetHashCode() & 0xFF}";
        var path = "/api/auth/register";

        // Make 5 requests to reach the limit
        for (int i = 0; i < 5; i++)
        {
            var context = new DefaultHttpContext();
            context.Request.Path = new PathString(path);
            context.Connection.RemoteIpAddress = IPAddress.Parse(ipAddress);
            context.Response.Body = new MemoryStream();

            await middleware.InvokeAsync(context);
        }

        // Act - Make the 6th request that should be rate limited
        var rateLimitedContext = new DefaultHttpContext();
        rateLimitedContext.Request.Path = new PathString(path);
        rateLimitedContext.Connection.RemoteIpAddress = IPAddress.Parse(ipAddress);
        rateLimitedContext.Response.Body = new MemoryStream();

        await middleware.InvokeAsync(rateLimitedContext);

        // Assert
        Assert.AreEqual(StatusCodes.Status429TooManyRequests, rateLimitedContext.Response.StatusCode);
        nextMock.Verify(next => next(It.IsAny<HttpContext>()), Times.Exactly(5));
    }

    /// <summary>
    /// Tests that InvokeAsync applies rate limiting regardless of path casing.
    /// Input: Paths with various case combinations containing auth/login or auth/register.
    /// Expected: Rate limiting is applied to all variations.
    /// </summary>
    /// <param name="path">The request path with different casing.</param>
    [TestMethod]
    [DataRow("/api/AUTH/LOGIN")]
    [DataRow("/api/Auth/Login")]
    [DataRow("/api/AuTh/LoGiN")]
    [DataRow("/API/AUTH/REGISTER")]
    [DataRow("/api/Auth/Register")]
    public async Task InvokeAsync_AuthEndpointsWithVariousCasing_AppliesRateLimiting(string path)
    {
        // Arrange
        var nextMock = new Mock<RequestDelegate>();
        var loggerMock = new Mock<ILogger<RateLimitMiddleware>>();
        var middleware = new RateLimitMiddleware(nextMock.Object, loggerMock.Object);

        var ipAddress = $"10.21.{Guid.NewGuid().GetHashCode() & 0xFF}.{Guid.NewGuid().GetHashCode() & 0xFF}";

        // Make 5 requests
        for (int i = 0; i < 5; i++)
        {
            var context = new DefaultHttpContext();
            context.Request.Path = new PathString(path);
            context.Connection.RemoteIpAddress = IPAddress.Parse(ipAddress);
            context.Response.Body = new MemoryStream();

            await middleware.InvokeAsync(context);
        }

        // Act - 6th request
        var rateLimitedContext = new DefaultHttpContext();
        rateLimitedContext.Request.Path = new PathString(path);
        rateLimitedContext.Connection.RemoteIpAddress = IPAddress.Parse(ipAddress);
        rateLimitedContext.Response.Body = new MemoryStream();

        await middleware.InvokeAsync(rateLimitedContext);

        // Assert
        Assert.AreEqual(StatusCodes.Status429TooManyRequests, rateLimitedContext.Response.StatusCode);
    }

    /// <summary>
    /// Tests that InvokeAsync bypasses rate limiting when path value is null.
    /// Input: HttpContext with null path value.
    /// Expected: Next middleware is invoked without rate limiting.
    /// </summary>
    [TestMethod]
    public async Task InvokeAsync_NullPathValue_BypassesRateLimiting()
    {
        // Arrange
        var nextMock = new Mock<RequestDelegate>();
        var loggerMock = new Mock<ILogger<RateLimitMiddleware>>();
        var middleware = new RateLimitMiddleware(nextMock.Object, loggerMock.Object);

        var contextMock = new Mock<HttpContext>();
        var requestMock = new Mock<HttpRequest>();
        var responseMock = new Mock<HttpResponse>();
        var connectionMock = new Mock<ConnectionInfo>();

        requestMock.Setup(r => r.Path).Returns(new PathString());
        contextMock.Setup(c => c.Request).Returns(requestMock.Object);
        contextMock.Setup(c => c.Response).Returns(responseMock.Object);
        contextMock.Setup(c => c.Connection).Returns(connectionMock.Object);

        // Act
        await middleware.InvokeAsync(contextMock.Object);

        // Assert
        nextMock.Verify(next => next(It.IsAny<HttpContext>()), Times.Once);
    }

    /// <summary>
    /// Tests that InvokeAsync bypasses rate limiting for whitespace-only paths.
    /// Input: Paths containing only whitespace characters.
    /// Expected: Next middleware is invoked without rate limiting.
    /// </summary>
    /// <param name="path">The whitespace path to test.</param>
    [TestMethod]
    [DataRow(" ")]
    [DataRow("   ")]
    [DataRow("\t")]
    [DataRow("\n")]
    public async Task InvokeAsync_WhitespaceOnlyPath_BypassesRateLimiting(string path)
    {
        // Arrange
        var nextMock = new Mock<RequestDelegate>();
        var loggerMock = new Mock<ILogger<RateLimitMiddleware>>();
        var middleware = new RateLimitMiddleware(nextMock.Object, loggerMock.Object);

        var context = new DefaultHttpContext();
        context.Request.Path = new PathString(path);
        context.Connection.RemoteIpAddress = IPAddress.Parse("192.168.1.100");

        // Act
        await middleware.InvokeAsync(context);

        // Assert
        nextMock.Verify(next => next(It.IsAny<HttpContext>()), Times.Once);
    }

    /// <summary>
    /// Tests that InvokeAsync tracks different IP addresses independently.
    /// Input: Multiple requests from different IP addresses to auth endpoints.
    /// Expected: Each IP has its own rate limit counter.
    /// </summary>
    [TestMethod]
    public async Task InvokeAsync_DifferentIpAddresses_TrackedIndependently()
    {
        // Arrange
        var nextMock = new Mock<RequestDelegate>();
        var loggerMock = new Mock<ILogger<RateLimitMiddleware>>();
        var middleware = new RateLimitMiddleware(nextMock.Object, loggerMock.Object);

        var ip1 = $"10.30.0.{Guid.NewGuid().GetHashCode() & 0xFF}";
        var ip2 = $"10.31.0.{Guid.NewGuid().GetHashCode() & 0xFF}";
        var path = "/api/auth/login";

        // Make 5 requests from IP1
        for (int i = 0; i < 5; i++)
        {
            var context = new DefaultHttpContext();
            context.Request.Path = new PathString(path);
            context.Connection.RemoteIpAddress = IPAddress.Parse(ip1);
            context.Response.Body = new MemoryStream();
            await middleware.InvokeAsync(context);
        }

        // Act - Make 1 request from IP2 (should be allowed)
        var ip2Context = new DefaultHttpContext();
        ip2Context.Request.Path = new PathString(path);
        ip2Context.Connection.RemoteIpAddress = IPAddress.Parse(ip2);
        ip2Context.Response.Body = new MemoryStream();
        await middleware.InvokeAsync(ip2Context);

        // Assert
        Assert.AreNotEqual(StatusCodes.Status429TooManyRequests, ip2Context.Response.StatusCode);
        nextMock.Verify(next => next(It.IsAny<HttpContext>()), Times.Exactly(6));
    }

    /// <summary>
    /// Tests that InvokeAsync uses X-Forwarded-For header when present.
    /// Input: HttpContext with X-Forwarded-For header.
    /// Expected: Rate limiting uses IP from X-Forwarded-For header.
    /// </summary>
    [TestMethod]
    public async Task InvokeAsync_WithXForwardedForHeader_UsesHeaderIp()
    {
        // Arrange
        var nextMock = new Mock<RequestDelegate>();
        var loggerMock = new Mock<ILogger<RateLimitMiddleware>>();
        var middleware = new RateLimitMiddleware(nextMock.Object, loggerMock.Object);

        var forwardedIp = $"10.40.0.{Guid.NewGuid().GetHashCode() & 0xFF}";
        var path = "/api/auth/login";

        // Make 5 requests with X-Forwarded-For header
        for (int i = 0; i < 5; i++)
        {
            var contextMock = CreateHttpContextMock(path, "192.168.1.1", forwardedIp);
            await middleware.InvokeAsync(contextMock.Object);
        }

        // Act - 6th request should be rate limited
        var rateLimitedContextMock = CreateHttpContextMock(path, "192.168.1.1", forwardedIp);
        var memoryStream = new MemoryStream();
        rateLimitedContextMock.Setup(c => c.Response.Body).Returns(memoryStream);

        await middleware.InvokeAsync(rateLimitedContextMock.Object);

        // Assert
        rateLimitedContextMock.VerifySet(c => c.Response.StatusCode = StatusCodes.Status429TooManyRequests);
    }

    /// <summary>
    /// Tests that InvokeAsync uses X-Real-IP header when X-Forwarded-For is not present.
    /// Input: HttpContext with X-Real-IP header but no X-Forwarded-For.
    /// Expected: Rate limiting uses IP from X-Real-IP header.
    /// </summary>
    [TestMethod]
    public async Task InvokeAsync_WithXRealIpHeader_UsesHeaderIp()
    {
        // Arrange
        var nextMock = new Mock<RequestDelegate>();
        var loggerMock = new Mock<ILogger<RateLimitMiddleware>>();
        var middleware = new RateLimitMiddleware(nextMock.Object, loggerMock.Object);

        var realIp = $"10.50.0.{Guid.NewGuid().GetHashCode() & 0xFF}";
        var path = "/api/auth/login";

        // Make 5 requests with X-Real-IP header
        for (int i = 0; i < 5; i++)
        {
            var contextMock = CreateHttpContextMockWithRealIp(path, "192.168.1.1", realIp);
            await middleware.InvokeAsync(contextMock.Object);
        }

        // Act - 6th request should be rate limited
        var rateLimitedContextMock = CreateHttpContextMockWithRealIp(path, "192.168.1.1", realIp);
        var memoryStream = new MemoryStream();
        rateLimitedContextMock.Setup(c => c.Response.Body).Returns(memoryStream);

        await middleware.InvokeAsync(rateLimitedContextMock.Object);

        // Assert
        rateLimitedContextMock.VerifySet(c => c.Response.StatusCode = StatusCodes.Status429TooManyRequests);
    }

    /// <summary>
    /// Tests that InvokeAsync logs a warning when rate limit is exceeded.
    /// Input: 6 requests from same IP to auth endpoint.
    /// Expected: Warning is logged with IP address and request count.
    /// </summary>
    [TestMethod]
    public async Task InvokeAsync_ExceedsRateLimit_LogsWarning()
    {
        // Arrange
        var nextMock = new Mock<RequestDelegate>();
        var loggerMock = new Mock<ILogger<RateLimitMiddleware>>();
        var middleware = new RateLimitMiddleware(nextMock.Object, loggerMock.Object);

        var ipAddress = $"10.60.0.{Guid.NewGuid().GetHashCode() & 0xFF}";
        var path = "/api/auth/login";

        // Make 5 requests
        for (int i = 0; i < 5; i++)
        {
            var context = new DefaultHttpContext();
            context.Request.Path = new PathString(path);
            context.Connection.RemoteIpAddress = IPAddress.Parse(ipAddress);
            context.Response.Body = new MemoryStream();
            await middleware.InvokeAsync(context);
        }

        // Act - 6th request triggers warning
        var rateLimitedContext = new DefaultHttpContext();
        rateLimitedContext.Request.Path = new PathString(path);
        rateLimitedContext.Connection.RemoteIpAddress = IPAddress.Parse(ipAddress);
        rateLimitedContext.Response.Body = new MemoryStream();

        await middleware.InvokeAsync(rateLimitedContext);

        // Assert
        loggerMock.Verify(
            x => x.Log(
                LogLevel.Warning,
                It.IsAny<EventId>(),
                It.Is<It.IsAnyType>((v, t) => v.ToString()!.Contains("Rate limit exceeded")),
                It.IsAny<Exception>(),
                It.IsAny<Func<It.IsAnyType, Exception?, string>>()),
            Times.Once);
    }

    /// <summary>
    /// Tests that InvokeAsync calculates retryAfterSeconds correctly when rate limited.
    /// Input: 6th request after rate limit exceeded.
    /// Expected: retryAfterSeconds is positive and represents remaining window time.
    /// </summary>
    [TestMethod]
    public async Task InvokeAsync_RateLimited_ReturnsValidRetryAfterSeconds()
    {
        // Arrange
        var nextMock = new Mock<RequestDelegate>();
        var loggerMock = new Mock<ILogger<RateLimitMiddleware>>();
        var middleware = new RateLimitMiddleware(nextMock.Object, loggerMock.Object);

        var ipAddress = $"10.70.0.{Guid.NewGuid().GetHashCode() & 0xFF}";
        var path = "/api/auth/login";

        // Make 5 requests
        for (int i = 0; i < 5; i++)
        {
            var context = new DefaultHttpContext();
            context.Request.Path = new PathString(path);
            context.Connection.RemoteIpAddress = IPAddress.Parse(ipAddress);
            context.Response.Body = new MemoryStream();
            await middleware.InvokeAsync(context);
        }

        // Act - 6th request
        var rateLimitedContext = new DefaultHttpContext();
        var responseBody = new MemoryStream();
        rateLimitedContext.Request.Path = new PathString(path);
        rateLimitedContext.Connection.RemoteIpAddress = IPAddress.Parse(ipAddress);
        rateLimitedContext.Response.Body = responseBody;

        await middleware.InvokeAsync(rateLimitedContext);

        // Assert
        responseBody.Position = 0;
        var reader = new StreamReader(responseBody);
        var responseJson = await reader.ReadToEndAsync();

        var capturedResponse = System.Text.Json.JsonSerializer.Deserialize<System.Text.Json.JsonElement>(responseJson);
        var retryAfterSeconds = capturedResponse.GetProperty("retryAfterSeconds").GetInt32();

        Assert.IsTrue(retryAfterSeconds > 0);
        Assert.IsTrue(retryAfterSeconds <= 15 * 60); // Should not exceed 15 minutes
    }

    /// <summary>
    /// Tests that InvokeAsync bypasses rate limiting for very long paths that don't contain auth endpoints.
    /// Input: Very long path (1000 characters) not containing auth keywords.
    /// Expected: Next middleware is invoked without rate limiting.
    /// </summary>
    [TestMethod]
    public async Task InvokeAsync_VeryLongPathWithoutAuth_BypassesRateLimiting()
    {
        // Arrange
        var nextMock = new Mock<RequestDelegate>();
        var loggerMock = new Mock<ILogger<RateLimitMiddleware>>();
        var middleware = new RateLimitMiddleware(nextMock.Object, loggerMock.Object);

        var longPath = "/api/" + new string('a', 1000);
        var context = new DefaultHttpContext();
        context.Request.Path = new PathString(longPath);
        context.Connection.RemoteIpAddress = IPAddress.Parse("192.168.1.200");

        // Act
        await middleware.InvokeAsync(context);

        // Assert
        nextMock.Verify(next => next(It.IsAny<HttpContext>()), Times.Once);
    }

    /// <summary>
    /// Tests that InvokeAsync applies rate limiting to very long paths containing auth endpoints.
    /// Input: Very long path containing "auth/login".
    /// Expected: Rate limiting is applied.
    /// </summary>
    [TestMethod]
    public async Task InvokeAsync_VeryLongPathWithAuth_AppliesRateLimiting()
    {
        // Arrange
        var nextMock = new Mock<RequestDelegate>();
        var loggerMock = new Mock<ILogger<RateLimitMiddleware>>();
        var middleware = new RateLimitMiddleware(nextMock.Object, loggerMock.Object);

        var ipAddress = $"10.80.0.{Guid.NewGuid().GetHashCode() & 0xFF}";
        var longPath = "/api/auth/login/" + new string('x', 1000);

        // Make 5 requests
        for (int i = 0; i < 5; i++)
        {
            var context = new DefaultHttpContext();
            context.Request.Path = new PathString(longPath);
            context.Connection.RemoteIpAddress = IPAddress.Parse(ipAddress);
            context.Response.Body = new MemoryStream();
            await middleware.InvokeAsync(context);
        }

        // Act - 6th request
        var rateLimitedContext = new DefaultHttpContext();
        rateLimitedContext.Request.Path = new PathString(longPath);
        rateLimitedContext.Connection.RemoteIpAddress = IPAddress.Parse(ipAddress);
        rateLimitedContext.Response.Body = new MemoryStream();

        await middleware.InvokeAsync(rateLimitedContext);

        // Assert
        Assert.AreEqual(StatusCodes.Status429TooManyRequests, rateLimitedContext.Response.StatusCode);
    }

    /// <summary>
    /// Tests that InvokeAsync bypasses rate limiting for paths with special characters but no auth keywords.
    /// Input: Path with special characters like !, @, #, $, %, etc.
    /// Expected: Next middleware is invoked without rate limiting.
    /// </summary>
    [TestMethod]
    [DataRow("/api/test!@#$%")]
    [DataRow("/api/test^&*()")]
    [DataRow("/api/test-_=+")]
    public async Task InvokeAsync_PathWithSpecialCharactersWithoutAuth_BypassesRateLimiting(string path)
    {
        // Arrange
        var nextMock = new Mock<RequestDelegate>();
        var loggerMock = new Mock<ILogger<RateLimitMiddleware>>();
        var middleware = new RateLimitMiddleware(nextMock.Object, loggerMock.Object);

        var context = new DefaultHttpContext();
        context.Request.Path = new PathString(path);
        context.Connection.RemoteIpAddress = IPAddress.Parse("192.168.1.250");

        // Act
        await middleware.InvokeAsync(context);

        // Assert
        nextMock.Verify(next => next(It.IsAny<HttpContext>()), Times.Once);
    }

    /// <summary>
    /// Tests that InvokeAsync sets correct Content-Type header when rate limited.
    /// Input: 6th request after exceeding rate limit.
    /// Expected: Content-Type is set to "application/json".
    /// </summary>
    [TestMethod]
    public async Task InvokeAsync_ExceedsRateLimit_SetsContentTypeToJson()
    {
        // Arrange
        var nextMock = new Mock<RequestDelegate>();
        var loggerMock = new Mock<ILogger<RateLimitMiddleware>>();
        var middleware = new RateLimitMiddleware(nextMock.Object, loggerMock.Object);

        var ipAddress = $"10.90.0.{Guid.NewGuid().GetHashCode() & 0xFF}";
        var path = "/api/auth/login";

        // Make 5 requests
        for (int i = 0; i < 5; i++)
        {
            var context = new DefaultHttpContext();
            context.Request.Path = new PathString(path);
            context.Connection.RemoteIpAddress = IPAddress.Parse(ipAddress);
            context.Response.Body = new MemoryStream();
            await middleware.InvokeAsync(context);
        }

        // Act - 6th request
        var rateLimitedContext = new DefaultHttpContext();
        rateLimitedContext.Request.Path = new PathString(path);
        rateLimitedContext.Connection.RemoteIpAddress = IPAddress.Parse(ipAddress);
        rateLimitedContext.Response.Body = new MemoryStream();

        await middleware.InvokeAsync(rateLimitedContext);

        // Assert
        Assert.IsTrue(rateLimitedContext.Response.ContentType?.StartsWith("application/json") == true);
    }

}



/// <summary>
/// Unit tests for the <see cref="RateLimitMiddlewareExtensions"/> class.
/// </summary>
[TestClass]
public class RateLimitMiddlewareExtensionsTests
{
    /// <summary>
    /// Tests that UseRateLimitMiddleware returns a non-null IApplicationBuilder when called with a valid builder.
    /// </summary>
    [TestMethod]
    public void UseRateLimitMiddleware_ValidBuilder_ReturnsNonNullApplicationBuilder()
    {
        // Arrange
        var mockBuilder = new Mock<IApplicationBuilder>();
        mockBuilder
            .Setup(b => b.Use(It.IsAny<Func<Microsoft.AspNetCore.Http.RequestDelegate, Microsoft.AspNetCore.Http.RequestDelegate>>()))
            .Returns(mockBuilder.Object);

        // Act
        var result = RateLimitMiddlewareExtensions.UseRateLimitMiddleware(mockBuilder.Object);

        // Assert
        Assert.IsNotNull(result);
    }

    /// <summary>
    /// Tests that UseRateLimitMiddleware returns an IApplicationBuilder instance when called with a valid builder.
    /// </summary>
    [TestMethod]
    public void UseRateLimitMiddleware_ValidBuilder_ReturnsIApplicationBuilder()
    {
        // Arrange
        var mockBuilder = new Mock<IApplicationBuilder>();
        mockBuilder
            .Setup(b => b.Use(It.IsAny<Func<Microsoft.AspNetCore.Http.RequestDelegate, Microsoft.AspNetCore.Http.RequestDelegate>>()))
            .Returns(mockBuilder.Object);

        // Act
        var result = RateLimitMiddlewareExtensions.UseRateLimitMiddleware(mockBuilder.Object);

        // Assert
        Assert.IsInstanceOfType(result, typeof(IApplicationBuilder));
    }

    /// <summary>
    /// Tests that UseRateLimitMiddleware calls UseMiddleware on the provided builder.
    /// Input: Valid mocked IApplicationBuilder.
    /// Expected: UseMiddleware (via Use) is called exactly once on the builder.
    /// </summary>
    [TestMethod]
    public void UseRateLimitMiddleware_ValidBuilder_CallsUseMiddleware()
    {
        // Arrange
        Mock<IApplicationBuilder> mockBuilder = new Mock<IApplicationBuilder>();
        mockBuilder
            .Setup(b => b.Use(It.IsAny<Func<RequestDelegate, RequestDelegate>>()))
            .Returns(mockBuilder.Object);

        // Act
        RateLimitMiddlewareExtensions.UseRateLimitMiddleware(mockBuilder.Object);

        // Assert
        mockBuilder.Verify(
            b => b.Use(It.IsAny<Func<RequestDelegate, RequestDelegate>>()),
            Times.Once);
    }

    /// <summary>
    /// Tests that UseRateLimitMiddleware returns the same builder instance returned by UseMiddleware.
    /// Input: Valid mocked IApplicationBuilder.
    /// Expected: The result is the same instance as returned by the mocked Use method.
    /// </summary>
    [TestMethod]
    public void UseRateLimitMiddleware_ValidBuilder_ReturnsSameBuilderInstance()
    {
        // Arrange
        Mock<IApplicationBuilder> mockBuilder = new Mock<IApplicationBuilder>();
        mockBuilder
            .Setup(b => b.Use(It.IsAny<Func<RequestDelegate, RequestDelegate>>()))
            .Returns(mockBuilder.Object);

        // Act
        IApplicationBuilder result = RateLimitMiddlewareExtensions.UseRateLimitMiddleware(mockBuilder.Object);

        // Assert
        Assert.AreSame(mockBuilder.Object, result);
    }
}