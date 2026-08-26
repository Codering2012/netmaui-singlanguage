using System;
using System.IO;
using System.Net;
using System.Threading.Tasks;
using Microsoft.AspNetCore.Http;
using Microsoft.Extensions.Logging;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using Moq;
using SignLanguageApi.Middleware;

namespace SignLanguageApi.UnitTests.Middleware
{
    [TestClass]
    public class RateLimitAndSecurityMiddlewareTests
    {
        [TestMethod]
        public async Task SecurityHeadersMiddleware_AppendsStandardSecurityHeaders()
        {
            var context = new DefaultHttpContext();
            var mockLogger = new Mock<ILogger<SecurityHeadersMiddleware>>();

            RequestDelegate next = (ctx) => Task.CompletedTask;
            var middleware = new SecurityHeadersMiddleware(next);

            await middleware.InvokeAsync(context);

            Assert.IsTrue(context.Response.Headers.ContainsKey("X-Content-Type-Options"));
            Assert.AreEqual("nosniff", context.Response.Headers["X-Content-Type-Options"].ToString());
            Assert.IsTrue(context.Response.Headers.ContainsKey("X-Frame-Options"));
            Assert.AreEqual("DENY", context.Response.Headers["X-Frame-Options"].ToString());
        }

        [TestMethod]
        public async Task RateLimitMiddleware_NormalEndpoint_PassesThrough()
        {
            var context = new DefaultHttpContext();
            context.Request.Path = "/api/learn/data";
            context.Connection.RemoteIpAddress = IPAddress.Parse("127.0.0.1");

            var mockLogger = new Mock<ILogger<RateLimitMiddleware>>();
            bool nextCalled = false;
            RequestDelegate next = (ctx) => { nextCalled = true; return Task.CompletedTask; };

            var middleware = new RateLimitMiddleware(next, mockLogger.Object);
            await middleware.InvokeAsync(context);

            Assert.IsTrue(nextCalled);
        }

        [TestMethod]
        public async Task RateLimitMiddleware_XForwardedForHeader_ExtractsClientIp()
        {
            var context = new DefaultHttpContext();
            context.Request.Path = "/api/auth/login";
            context.Request.Headers["X-Forwarded-For"] = "203.0.113.195, 70.41.3.18";
            context.Response.Body = new MemoryStream();

            var mockLogger = new Mock<ILogger<RateLimitMiddleware>>();
            RequestDelegate next = (ctx) => Task.CompletedTask;

            var middleware = new RateLimitMiddleware(next, mockLogger.Object);
            await middleware.InvokeAsync(context);

            Assert.AreEqual(200, context.Response.StatusCode);
        }

        [TestMethod]
        public async Task RateLimitMiddleware_ExceedsLimit_Returns429TooManyRequests()
        {
            var mockLogger = new Mock<ILogger<RateLimitMiddleware>>();
            RequestDelegate next = (ctx) => Task.CompletedTask;

            var middleware = new RateLimitMiddleware(next, mockLogger.Object);

            for (int i = 0; i < 6; i++)
            {
                var context = new DefaultHttpContext();
                context.Request.Path = "/api/auth/login";
                context.Connection.RemoteIpAddress = IPAddress.Parse("198.51.100.44");
                context.Response.Body = new MemoryStream();

                await middleware.InvokeAsync(context);

                if (i >= 5)
                {
                    Assert.AreEqual(StatusCodes.Status429TooManyRequests, context.Response.StatusCode);
                }
            }
        }
    }
}
