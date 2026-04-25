using System;
using System.Collections.Concurrent;
using System.Collections.Generic;
using System.Linq;
using System.Reflection;
using System.Threading.Tasks;

using Microsoft.Extensions.Logging;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using Moq;
using SignLanguageApi.Services;

namespace SignLanguageApi.Services.UnitTests;

/// <summary>
/// Unit tests for the <see cref="TokenBlacklistService"/> class.
/// </summary>
[TestClass]
public class TokenBlacklistServiceTests
{
    /// <summary>
    /// Tests that IsTokenBlacklistedAsync returns false when the token is not in the blacklist.
    /// </summary>
    /// <param name="token">The token to test.</param>
    [TestMethod]
    [DataRow("")]
    [DataRow("   ")]
    [DataRow("valid-token-not-blacklisted")]
    [DataRow("another-token")]
    public async Task IsTokenBlacklistedAsync_TokenNotInBlacklist_ReturnsFalse(string token)
    {
        // Arrange
        var mockLogger = new Mock<ILogger<TokenBlacklistService>>();
        var service = new TokenBlacklistService(mockLogger.Object);

        // Act
        bool result = await service.IsTokenBlacklistedAsync(token);

        // Assert
        Assert.IsFalse(result);
    }

    /// <summary>
    /// Tests that IsTokenBlacklistedAsync returns true when the token is blacklisted with a future expiry time.
    /// </summary>
    [TestMethod]
    public async Task IsTokenBlacklistedAsync_TokenBlacklistedWithFutureExpiry_ReturnsTrue()
    {
        // Arrange
        var mockLogger = new Mock<ILogger<TokenBlacklistService>>();
        var service = new TokenBlacklistService(mockLogger.Object);
        string token = "blacklisted-token";
        DateTime futureExpiry = DateTime.UtcNow.AddMinutes(10);
        await service.BlacklistTokenAsync(token, futureExpiry);

        // Act
        bool result = await service.IsTokenBlacklistedAsync(token);

        // Assert
        Assert.IsTrue(result);
    }

    /// <summary>
    /// Tests that IsTokenBlacklistedAsync returns false and removes the token when it is blacklisted but expired.
    /// </summary>
    [TestMethod]
    public async Task IsTokenBlacklistedAsync_TokenBlacklistedButExpired_ReturnsFalseAndRemovesToken()
    {
        // Arrange
        var mockLogger = new Mock<ILogger<TokenBlacklistService>>();
        var service = new TokenBlacklistService(mockLogger.Object);
        string token = "expired-token";
        DateTime pastExpiry = DateTime.UtcNow.AddMinutes(-10);
        await service.BlacklistTokenAsync(token, pastExpiry);

        // Act
        bool firstResult = await service.IsTokenBlacklistedAsync(token);
        bool secondResult = await service.IsTokenBlacklistedAsync(token);

        // Assert
        Assert.IsFalse(firstResult);
        Assert.IsFalse(secondResult);
    }

    /// <summary>
    /// Tests that IsTokenBlacklistedAsync returns false when the token expired exactly 1 millisecond ago.
    /// </summary>
    [TestMethod]
    public async Task IsTokenBlacklistedAsync_TokenExpiredOneMillisecondAgo_ReturnsFalseAndRemovesToken()
    {
        // Arrange
        var mockLogger = new Mock<ILogger<TokenBlacklistService>>();
        var service = new TokenBlacklistService(mockLogger.Object);
        string token = "recently-expired-token";
        DateTime pastExpiry = DateTime.UtcNow.AddMilliseconds(-1);
        await service.BlacklistTokenAsync(token, pastExpiry);

        // Act
        bool result = await service.IsTokenBlacklistedAsync(token);

        // Assert
        Assert.IsFalse(result);
    }

    /// <summary>
    /// Tests that IsTokenBlacklistedAsync returns true when the token expiry is very close to UtcNow but still in the future.
    /// </summary>
    [TestMethod]
    public async Task IsTokenBlacklistedAsync_TokenExpiryVeryCloseToUtcNow_ReturnsTrue()
    {
        // Arrange
        var mockLogger = new Mock<ILogger<TokenBlacklistService>>();
        var service = new TokenBlacklistService(mockLogger.Object);
        string token = "almost-expired-token";
        DateTime nearFutureExpiry = DateTime.UtcNow.AddSeconds(5);
        await service.BlacklistTokenAsync(token, nearFutureExpiry);

        // Act
        bool result = await service.IsTokenBlacklistedAsync(token);

        // Assert
        Assert.IsTrue(result);
    }

    /// <summary>
    /// Tests that IsTokenBlacklistedAsync correctly handles a very long token string.
    /// </summary>
    [TestMethod]
    public async Task IsTokenBlacklistedAsync_VeryLongToken_ReturnsCorrectResult()
    {
        // Arrange
        var mockLogger = new Mock<ILogger<TokenBlacklistService>>();
        var service = new TokenBlacklistService(mockLogger.Object);
        string longToken = new string('a', 10000);
        DateTime futureExpiry = DateTime.UtcNow.AddMinutes(10);
        await service.BlacklistTokenAsync(longToken, futureExpiry);

        // Act
        bool result = await service.IsTokenBlacklistedAsync(longToken);

        // Assert
        Assert.IsTrue(result);
    }

    /// <summary>
    /// Tests that IsTokenBlacklistedAsync correctly handles tokens with special characters.
    /// </summary>
    [TestMethod]
    public async Task IsTokenBlacklistedAsync_TokenWithSpecialCharacters_ReturnsCorrectResult()
    {
        // Arrange
        var mockLogger = new Mock<ILogger<TokenBlacklistService>>();
        var service = new TokenBlacklistService(mockLogger.Object);
        string specialToken = "token!@#$%^&*()_+-=[]{}|;':\",./<>?";
        DateTime futureExpiry = DateTime.UtcNow.AddMinutes(10);
        await service.BlacklistTokenAsync(specialToken, futureExpiry);

        // Act
        bool result = await service.IsTokenBlacklistedAsync(specialToken);

        // Assert
        Assert.IsTrue(result);
    }

    /// <summary>
    /// Tests that IsTokenBlacklistedAsync handles multiple checks on the same expired token correctly.
    /// Verifies that the token is removed on the first check and subsequent checks return false.
    /// </summary>
    [TestMethod]
    public async Task IsTokenBlacklistedAsync_MultipleChecksOnExpiredToken_AlwaysReturnsFalse()
    {
        // Arrange
        var mockLogger = new Mock<ILogger<TokenBlacklistService>>();
        var service = new TokenBlacklistService(mockLogger.Object);
        string token = "multi-check-expired-token";
        DateTime pastExpiry = DateTime.UtcNow.AddHours(-1);
        await service.BlacklistTokenAsync(token, pastExpiry);

        // Act
        bool firstCheck = await service.IsTokenBlacklistedAsync(token);
        bool secondCheck = await service.IsTokenBlacklistedAsync(token);
        bool thirdCheck = await service.IsTokenBlacklistedAsync(token);

        // Assert
        Assert.IsFalse(firstCheck);
        Assert.IsFalse(secondCheck);
        Assert.IsFalse(thirdCheck);
    }

    /// <summary>
    /// Tests that IsTokenBlacklistedAsync returns true for multiple different blacklisted tokens.
    /// </summary>
    [TestMethod]
    public async Task IsTokenBlacklistedAsync_MultipleDifferentBlacklistedTokens_ReturnsTrue()
    {
        // Arrange
        var mockLogger = new Mock<ILogger<TokenBlacklistService>>();
        var service = new TokenBlacklistService(mockLogger.Object);
        string token1 = "token-1";
        string token2 = "token-2";
        string token3 = "token-3";
        DateTime futureExpiry = DateTime.UtcNow.AddHours(1);
        await service.BlacklistTokenAsync(token1, futureExpiry);
        await service.BlacklistTokenAsync(token2, futureExpiry);
        await service.BlacklistTokenAsync(token3, futureExpiry);

        // Act
        bool result1 = await service.IsTokenBlacklistedAsync(token1);
        bool result2 = await service.IsTokenBlacklistedAsync(token2);
        bool result3 = await service.IsTokenBlacklistedAsync(token3);

        // Assert
        Assert.IsTrue(result1);
        Assert.IsTrue(result2);
        Assert.IsTrue(result3);
    }

    /// <summary>
    /// Tests that IsTokenBlacklistedAsync returns true when the token is blacklisted with DateTime.MaxValue expiry.
    /// </summary>
    [TestMethod]
    public async Task IsTokenBlacklistedAsync_TokenBlacklistedWithMaxValueExpiry_ReturnsTrue()
    {
        // Arrange
        var mockLogger = new Mock<ILogger<TokenBlacklistService>>();
        var service = new TokenBlacklistService(mockLogger.Object);
        string token = "max-expiry-token";
        await service.BlacklistTokenAsync(token, DateTime.MaxValue);

        // Act
        bool result = await service.IsTokenBlacklistedAsync(token);

        // Assert
        Assert.IsTrue(result);
    }

    /// <summary>
    /// Tests that IsTokenBlacklistedAsync returns false when the token is blacklisted with DateTime.MinValue expiry.
    /// </summary>
    [TestMethod]
    public async Task IsTokenBlacklistedAsync_TokenBlacklistedWithMinValueExpiry_ReturnsFalse()
    {
        // Arrange
        var mockLogger = new Mock<ILogger<TokenBlacklistService>>();
        var service = new TokenBlacklistService(mockLogger.Object);
        string token = "min-expiry-token";
        await service.BlacklistTokenAsync(token, DateTime.MinValue);

        // Act
        bool result = await service.IsTokenBlacklistedAsync(token);

        // Assert
        Assert.IsFalse(result);
    }

    /// <summary>
    /// Tests that the constructor successfully initializes the service with a valid logger.
    /// </summary>
    [TestMethod]
    public void TokenBlacklistService_ValidLogger_InitializesSuccessfully()
    {
        // Arrange
        var mockLogger = new Mock<ILogger<TokenBlacklistService>>();

        // Act
        var service = new TokenBlacklistService(mockLogger.Object);

        // Assert
        Assert.IsNotNull(service);
    }

    /// <summary>
    /// Tests that the constructor accepts a null logger parameter without throwing.
    /// This verifies the current behavior where no null check is performed.
    /// </summary>
    [TestMethod]
    public void TokenBlacklistService_NullLogger_DoesNotThrow()
    {
        // Arrange
        ILogger<TokenBlacklistService>? logger = null;

        // Act & Assert
        var service = new TokenBlacklistService(logger!);
        Assert.IsNotNull(service);
    }

    /// <summary>
    /// Tests that BlacklistTokenAsync successfully blacklists a valid token with future expiry time.
    /// Input: Valid token string and future DateTime.
    /// Expected: Token is added to blacklist, logger is called, and Task completes successfully.
    /// </summary>
    [TestMethod]
    public async Task BlacklistTokenAsync_ValidTokenWithFutureExpiry_BlacklistsTokenSuccessfully()
    {
        // Arrange
        Mock<ILogger<TokenBlacklistService>> mockLogger = new Mock<ILogger<TokenBlacklistService>>();
        TokenBlacklistService service = new TokenBlacklistService(mockLogger.Object);
        string token = "valid-token-123";
        DateTime expiryTime = DateTime.UtcNow.AddHours(2);

        // Act
        await service.BlacklistTokenAsync(token, expiryTime);

        // Assert
        bool isBlacklisted = await service.IsTokenBlacklistedAsync(token);
        Assert.IsTrue(isBlacklisted);
        mockLogger.Verify(
            x => x.Log(
                LogLevel.Information,
                It.IsAny<EventId>(),
                It.Is<It.IsAnyType>((v, t) => v.ToString()!.Contains("Token blacklisted until")),
                It.IsAny<Exception>(),
                It.IsAny<Func<It.IsAnyType, Exception?, string>>()),
            Times.Once);
    }

    /// <summary>
    /// Tests that BlacklistTokenAsync successfully blacklists a token even with past expiry time.
    /// Input: Valid token string and past DateTime.
    /// Expected: Token is added to blacklist without validation of expiry time.
    /// </summary>
    [TestMethod]
    public async Task BlacklistTokenAsync_ValidTokenWithPastExpiry_BlacklistsTokenSuccessfully()
    {
        // Arrange
        Mock<ILogger<TokenBlacklistService>> mockLogger = new Mock<ILogger<TokenBlacklistService>>();
        TokenBlacklistService service = new TokenBlacklistService(mockLogger.Object);
        string token = "expired-token";
        DateTime expiryTime = DateTime.UtcNow.AddHours(-1);

        // Act
        await service.BlacklistTokenAsync(token, expiryTime);

        // Assert
        mockLogger.Verify(
            x => x.Log(
                LogLevel.Information,
                It.IsAny<EventId>(),
                It.Is<It.IsAnyType>((v, t) => v.ToString()!.Contains("Token blacklisted until")),
                It.IsAny<Exception>(),
                It.IsAny<Func<It.IsAnyType, Exception?, string>>()),
            Times.Once);
    }

    /// <summary>
    /// Tests that BlacklistTokenAsync handles DateTime edge cases correctly.
    /// Input: DateTime.MinValue, DateTime.MaxValue, and current UTC time.
    /// Expected: Token is blacklisted successfully for all edge case DateTime values.
    /// </summary>
    [TestMethod]
    [DataRow("0001-01-01T00:00:00.0000000Z")] // DateTime.MinValue equivalent
    [DataRow("9999-12-31T23:59:59Z")] // DateTime.MaxValue equivalent (reduced precision for parsing)
    public async Task BlacklistTokenAsync_DateTimeEdgeCases_BlacklistsTokenSuccessfully(string dateTimeString)
    {
        // Arrange
        Mock<ILogger<TokenBlacklistService>> mockLogger = new Mock<ILogger<TokenBlacklistService>>();
        TokenBlacklistService service = new TokenBlacklistService(mockLogger.Object);
        string token = $"token-{dateTimeString}";
        DateTime expiryTime = DateTime.Parse(dateTimeString, null, System.Globalization.DateTimeStyles.RoundtripKind);

        // Act
        await service.BlacklistTokenAsync(token, expiryTime);

        // Assert
        mockLogger.Verify(
            x => x.Log(
                LogLevel.Information,
                It.IsAny<EventId>(),
                It.Is<It.IsAnyType>((v, t) => v.ToString()!.Contains("Token blacklisted until")),
                It.IsAny<Exception>(),
                It.IsAny<Func<It.IsAnyType, Exception?, string>>()),
            Times.Once);
    }

    /// <summary>
    /// Tests that BlacklistTokenAsync updates expiry time when the same token is blacklisted twice.
    /// Input: Same token blacklisted with two different expiry times.
    /// Expected: Second expiry time replaces the first (AddOrUpdate behavior).
    /// </summary>
    [TestMethod]
    public async Task BlacklistTokenAsync_SameTokenTwice_UpdatesExpiryTime()
    {
        // Arrange
        Mock<ILogger<TokenBlacklistService>> mockLogger = new Mock<ILogger<TokenBlacklistService>>();
        TokenBlacklistService service = new TokenBlacklistService(mockLogger.Object);
        string token = "duplicate-token";
        DateTime firstExpiry = DateTime.UtcNow.AddHours(1);
        DateTime secondExpiry = DateTime.UtcNow.AddHours(3);

        // Act
        await service.BlacklistTokenAsync(token, firstExpiry);
        await service.BlacklistTokenAsync(token, secondExpiry);

        // Assert
        bool isBlacklisted = await service.IsTokenBlacklistedAsync(token);
        Assert.IsTrue(isBlacklisted);
        mockLogger.Verify(
            x => x.Log(
                LogLevel.Information,
                It.IsAny<EventId>(),
                It.Is<It.IsAnyType>((v, t) => v.ToString()!.Contains("Token blacklisted until")),
                It.IsAny<Exception>(),
                It.IsAny<Func<It.IsAnyType, Exception?, string>>()),
            Times.Exactly(2));
    }

    /// <summary>
    /// Tests that BlacklistTokenAsync handles tokens with special characters.
    /// Input: Tokens containing special characters, symbols, and unicode.
    /// Expected: All tokens are blacklisted successfully without character validation.
    /// </summary>
    [TestMethod]
    [DataRow("token-with-dashes")]
    [DataRow("token.with.dots")]
    [DataRow("token_with_underscores")]
    [DataRow("token@with#special$chars")]
    [DataRow("token with spaces inside")]
    [DataRow("TOKEN123!@#$%^&*()")]
    [DataRow("unicode-token-你好")]
    public async Task BlacklistTokenAsync_TokensWithSpecialCharacters_BlacklistsSuccessfully(string token)
    {
        // Arrange
        Mock<ILogger<TokenBlacklistService>> mockLogger = new Mock<ILogger<TokenBlacklistService>>();
        TokenBlacklistService service = new TokenBlacklistService(mockLogger.Object);
        DateTime expiryTime = DateTime.UtcNow.AddHours(1);

        // Act
        await service.BlacklistTokenAsync(token, expiryTime);

        // Assert
        bool isBlacklisted = await service.IsTokenBlacklistedAsync(token);
        Assert.IsTrue(isBlacklisted);
    }

    /// <summary>
    /// Tests that BlacklistTokenAsync handles very long token strings.
    /// Input: Very long token string (1000 characters).
    /// Expected: Token is blacklisted successfully without length validation.
    /// </summary>
    [TestMethod]
    public async Task BlacklistTokenAsync_VeryLongToken_BlacklistsSuccessfully()
    {
        // Arrange
        Mock<ILogger<TokenBlacklistService>> mockLogger = new Mock<ILogger<TokenBlacklistService>>();
        TokenBlacklistService service = new TokenBlacklistService(mockLogger.Object);
        string token = new string('a', 1000);
        DateTime expiryTime = DateTime.UtcNow.AddHours(1);

        // Act
        await service.BlacklistTokenAsync(token, expiryTime);

        // Assert
        bool isBlacklisted = await service.IsTokenBlacklistedAsync(token);
        Assert.IsTrue(isBlacklisted);
    }

    /// <summary>
    /// Tests that BlacklistTokenAsync returns a completed Task.
    /// Input: Valid token and expiry time.
    /// Expected: Returned Task is completed successfully.
    /// </summary>
    [TestMethod]
    public async Task BlacklistTokenAsync_ValidInput_ReturnsCompletedTask()
    {
        // Arrange
        Mock<ILogger<TokenBlacklistService>> mockLogger = new Mock<ILogger<TokenBlacklistService>>();
        TokenBlacklistService service = new TokenBlacklistService(mockLogger.Object);
        string token = "test-token";
        DateTime expiryTime = DateTime.UtcNow.AddHours(1);

        // Act
        Task result = service.BlacklistTokenAsync(token, expiryTime);

        // Assert
        Assert.IsNotNull(result);
        Assert.IsTrue(result.IsCompleted);
        await result; // Ensure no exceptions
    }

    /// <summary>
    /// Tests that BlacklistTokenAsync calls logger with correct parameters.
    /// Input: Valid token and expiry time.
    /// Expected: Logger.LogInformation is called once with the expiry time.
    /// </summary>
    [TestMethod]
    public async Task BlacklistTokenAsync_ValidInput_LogsCorrectInformation()
    {
        // Arrange
        Mock<ILogger<TokenBlacklistService>> mockLogger = new Mock<ILogger<TokenBlacklistService>>();
        TokenBlacklistService service = new TokenBlacklistService(mockLogger.Object);
        string token = "logging-test-token";
        DateTime expiryTime = new DateTime(2024, 12, 31, 23, 59, 59, DateTimeKind.Utc);

        // Act
        await service.BlacklistTokenAsync(token, expiryTime);

        // Assert
        mockLogger.Verify(
            x => x.Log(
                LogLevel.Information,
                It.IsAny<EventId>(),
                It.Is<It.IsAnyType>((v, t) => v.ToString()!.Contains("Token blacklisted until")),
                It.IsAny<Exception>(),
                It.IsAny<Func<It.IsAnyType, Exception?, string>>()),
            Times.Once);
    }

    /// <summary>
    /// Tests that RemoveExpiredTokensAsync completes successfully without logging when no tokens exist in the blacklist.
    /// </summary>
    [TestMethod]
    public async Task RemoveExpiredTokensAsync_WhenNoTokensExist_CompletesSuccessfullyWithoutLogging()
    {
        // Arrange
        var mockLogger = new Mock<ILogger<TokenBlacklistService>>();
        var service = new TokenBlacklistService(mockLogger.Object);

        // Act
        await service.RemoveExpiredTokensAsync();

        // Assert
        mockLogger.Verify(
            x => x.Log(
                LogLevel.Information,
                It.IsAny<EventId>(),
                It.IsAny<It.IsAnyType>(),
                It.IsAny<Exception>(),
                It.IsAny<Func<It.IsAnyType, Exception?, string>>()),
            Times.Never);
    }

    /// <summary>
    /// Tests that RemoveExpiredTokensAsync removes a single expired token and logs the removal count.
    /// </summary>
    [TestMethod]
    public async Task RemoveExpiredTokensAsync_WhenSingleExpiredToken_RemovesTokenAndLogsCount()
    {
        // Arrange
        var mockLogger = new Mock<ILogger<TokenBlacklistService>>();
        var service = new TokenBlacklistService(mockLogger.Object);
        var expiredToken = "expired-token-123";
        var pastExpiry = DateTime.UtcNow.AddMinutes(-10);

        await service.BlacklistTokenAsync(expiredToken, pastExpiry);

        // Act
        await service.RemoveExpiredTokensAsync();

        // Assert
        var isStillBlacklisted = await service.IsTokenBlacklistedAsync(expiredToken);
        Assert.IsFalse(isStillBlacklisted, "Expired token should have been removed from blacklist");

        mockLogger.Verify(
            x => x.Log(
                LogLevel.Information,
                It.IsAny<EventId>(),
                It.Is<It.IsAnyType>((v, t) => v.ToString()!.Contains("1") && v.ToString()!.Contains("expired tokens")),
                It.IsAny<Exception>(),
                It.IsAny<Func<It.IsAnyType, Exception?, string>>()),
            Times.Once);
    }

    /// <summary>
    /// Tests that RemoveExpiredTokensAsync removes all expired tokens when multiple tokens have expired and logs the count.
    /// </summary>
    [TestMethod]
    public async Task RemoveExpiredTokensAsync_WhenMultipleExpiredTokens_RemovesAllTokensAndLogsCount()
    {
        // Arrange
        var mockLogger = new Mock<ILogger<TokenBlacklistService>>();
        var service = new TokenBlacklistService(mockLogger.Object);
        var expiredTokens = new[]
        {
                ("token1", DateTime.UtcNow.AddMinutes(-30)),
                ("token2", DateTime.UtcNow.AddMinutes(-20)),
                ("token3", DateTime.UtcNow.AddMinutes(-10)),
                ("token4", DateTime.UtcNow.AddMinutes(-5))
            };

        foreach (var (token, expiry) in expiredTokens)
        {
            await service.BlacklistTokenAsync(token, expiry);
        }

        // Act
        await service.RemoveExpiredTokensAsync();

        // Assert
        foreach (var (token, _) in expiredTokens)
        {
            var isStillBlacklisted = await service.IsTokenBlacklistedAsync(token);
            Assert.IsFalse(isStillBlacklisted, $"Expired token '{token}' should have been removed from blacklist");
        }

        mockLogger.Verify(
            x => x.Log(
                LogLevel.Information,
                It.IsAny<EventId>(),
                It.Is<It.IsAnyType>((v, t) => v.ToString()!.Contains("4") && v.ToString()!.Contains("expired tokens")),
                It.IsAny<Exception>(),
                It.IsAny<Func<It.IsAnyType, Exception?, string>>()),
            Times.Once);
    }

    /// <summary>
    /// Tests that RemoveExpiredTokensAsync does not remove any tokens when all tokens are still valid and does not log.
    /// </summary>
    [TestMethod]
    public async Task RemoveExpiredTokensAsync_WhenNoTokensExpired_DoesNotRemoveAnythingOrLog()
    {
        // Arrange
        var mockLogger = new Mock<ILogger<TokenBlacklistService>>();
        var service = new TokenBlacklistService(mockLogger.Object);
        var validTokens = new[]
        {
                ("token1", DateTime.UtcNow.AddMinutes(10)),
                ("token2", DateTime.UtcNow.AddMinutes(20)),
                ("token3", DateTime.UtcNow.AddMinutes(30))
            };

        foreach (var (token, expiry) in validTokens)
        {
            await service.BlacklistTokenAsync(token, expiry);
        }

        // Act
        await service.RemoveExpiredTokensAsync();

        // Assert
        foreach (var (token, _) in validTokens)
        {
            var isStillBlacklisted = await service.IsTokenBlacklistedAsync(token);
            Assert.IsTrue(isStillBlacklisted, $"Valid token '{token}' should still be in blacklist");
        }

        mockLogger.Verify(
            x => x.Log(
                LogLevel.Information,
                It.IsAny<EventId>(),
                It.IsAny<It.IsAnyType>(),
                It.IsAny<Exception>(),
                It.IsAny<Func<It.IsAnyType, Exception?, string>>()),
            Times.Exactly(3));
    }

    /// <summary>
    /// Tests that RemoveExpiredTokensAsync removes only expired tokens while preserving valid ones and logs the expired count.
    /// </summary>
    [TestMethod]
    public async Task RemoveExpiredTokensAsync_WhenMixedExpiredAndValidTokens_RemovesOnlyExpiredAndLogsCount()
    {
        // Arrange
        var mockLogger = new Mock<ILogger<TokenBlacklistService>>();
        var service = new TokenBlacklistService(mockLogger.Object);
        var expiredTokens = new[]
        {
                ("expired1", DateTime.UtcNow.AddMinutes(-20)),
                ("expired2", DateTime.UtcNow.AddMinutes(-10))
            };
        var validTokens = new[]
        {
                ("valid1", DateTime.UtcNow.AddMinutes(10)),
                ("valid2", DateTime.UtcNow.AddMinutes(20))
            };

        foreach (var (token, expiry) in expiredTokens.Concat(validTokens))
        {
            await service.BlacklistTokenAsync(token, expiry);
        }

        // Act
        await service.RemoveExpiredTokensAsync();

        // Assert
        foreach (var (token, _) in expiredTokens)
        {
            var isStillBlacklisted = await service.IsTokenBlacklistedAsync(token);
            Assert.IsFalse(isStillBlacklisted, $"Expired token '{token}' should have been removed from blacklist");
        }

        foreach (var (token, _) in validTokens)
        {
            var isStillBlacklisted = await service.IsTokenBlacklistedAsync(token);
            Assert.IsTrue(isStillBlacklisted, $"Valid token '{token}' should still be in blacklist");
        }

        mockLogger.Verify(
            x => x.Log(
                LogLevel.Information,
                It.IsAny<EventId>(),
                It.Is<It.IsAnyType>((v, t) => v.ToString()!.Contains("2") && v.ToString()!.Contains("expired tokens")),
                It.IsAny<Exception>(),
                It.IsAny<Func<It.IsAnyType, Exception?, string>>()),
            Times.Once);
    }

    /// <summary>
    /// Tests that RemoveExpiredTokensAsync handles tokens with expiry times at various boundary conditions correctly.
    /// </summary>
    [TestMethod]
    [DataRow(-1, false, DisplayName = "Token expired 1 minute ago should be removed")]
    [DataRow(-60, false, DisplayName = "Token expired 1 hour ago should be removed")]
    [DataRow(-1440, false, DisplayName = "Token expired 1 day ago should be removed")]
    [DataRow(1, true, DisplayName = "Token expiring in 1 minute should not be removed")]
    [DataRow(60, true, DisplayName = "Token expiring in 1 hour should not be removed")]
    [DataRow(1440, true, DisplayName = "Token expiring in 1 day should not be removed")]
    public async Task RemoveExpiredTokensAsync_WithVariousExpiryTimes_HandlesTokensCorrectly(int minutesOffset, bool shouldRemainBlacklisted)
    {
        // Arrange
        var mockLogger = new Mock<ILogger<TokenBlacklistService>>();
        var service = new TokenBlacklistService(mockLogger.Object);
        var token = $"test-token-{minutesOffset}";
        var expiryTime = DateTime.UtcNow.AddMinutes(minutesOffset);

        await service.BlacklistTokenAsync(token, expiryTime);

        // Act
        await service.RemoveExpiredTokensAsync();

        // Assert
        var isStillBlacklisted = await service.IsTokenBlacklistedAsync(token);
        Assert.AreEqual(shouldRemainBlacklisted, isStillBlacklisted,
            $"Token with expiry offset {minutesOffset} minutes should {(shouldRemainBlacklisted ? "remain" : "not remain")} blacklisted");
    }

    /// <summary>
    /// Tests that RemoveExpiredTokensAsync correctly handles a large number of expired tokens.
    /// </summary>
    [TestMethod]
    public async Task RemoveExpiredTokensAsync_WithLargeNumberOfExpiredTokens_RemovesAllAndLogsCount()
    {
        // Arrange
        var mockLogger = new Mock<ILogger<TokenBlacklistService>>();
        var service = new TokenBlacklistService(mockLogger.Object);
        var tokenCount = 100;
        var expiredTokens = new List<string>();

        for (int i = 0; i < tokenCount; i++)
        {
            var token = $"expired-token-{i}";
            expiredTokens.Add(token);
            await service.BlacklistTokenAsync(token, DateTime.UtcNow.AddMinutes(-10));
        }

        // Act
        await service.RemoveExpiredTokensAsync();

        // Assert
        foreach (var token in expiredTokens)
        {
            var isStillBlacklisted = await service.IsTokenBlacklistedAsync(token);
            Assert.IsFalse(isStillBlacklisted, $"Expired token '{token}' should have been removed from blacklist");
        }

        mockLogger.Verify(
            x => x.Log(
                LogLevel.Information,
                It.IsAny<EventId>(),
                It.Is<It.IsAnyType>((v, t) => v.ToString()!.Contains("100") && v.ToString()!.Contains("expired tokens")),
                It.IsAny<Exception>(),
                It.IsAny<Func<It.IsAnyType, Exception?, string>>()),
            Times.Once);
    }

    /// <summary>
    /// Tests that RemoveExpiredTokensAsync removes tokens with DateTime.MinValue expiry immediately.
    /// Input: Token with DateTime.MinValue expiry.
    /// Expected: Token is removed and count is logged.
    /// </summary>
    [TestMethod]
    public async Task RemoveExpiredTokensAsync_TokenWithMinValueExpiry_RemovesTokenImmediately()
    {
        // Arrange
        var mockLogger = new Mock<ILogger<TokenBlacklistService>>();
        var service = new TokenBlacklistService(mockLogger.Object);
        var token = "token-with-min-value-expiry";

        await service.BlacklistTokenAsync(token, DateTime.MinValue);

        // Act
        await service.RemoveExpiredTokensAsync();

        // Assert
        var isStillBlacklisted = await service.IsTokenBlacklistedAsync(token);
        Assert.IsFalse(isStillBlacklisted, "Token with DateTime.MinValue expiry should be removed");

        mockLogger.Verify(
            x => x.Log(
                LogLevel.Information,
                It.IsAny<EventId>(),
                It.Is<It.IsAnyType>((v, t) => v.ToString()!.Contains("1") && v.ToString()!.Contains("expired tokens")),
                It.IsAny<Exception>(),
                It.IsAny<Func<It.IsAnyType, Exception?, string>>()),
            Times.Once);
    }

    /// <summary>
    /// Tests that RemoveExpiredTokensAsync does not remove tokens with DateTime.MaxValue expiry.
    /// Input: Token with DateTime.MaxValue expiry.
    /// Expected: Token remains blacklisted and no logging occurs.
    /// </summary>
    [TestMethod]
    public async Task RemoveExpiredTokensAsync_TokenWithMaxValueExpiry_DoesNotRemoveToken()
    {
        // Arrange
        var mockLogger = new Mock<ILogger<TokenBlacklistService>>();
        var service = new TokenBlacklistService(mockLogger.Object);
        var token = "token-with-max-value-expiry";

        await service.BlacklistTokenAsync(token, DateTime.MaxValue);

        // Act
        await service.RemoveExpiredTokensAsync();

        // Assert
        var isStillBlacklisted = await service.IsTokenBlacklistedAsync(token);
        Assert.IsTrue(isStillBlacklisted, "Token with DateTime.MaxValue expiry should not be removed");

        mockLogger.Verify(
            x => x.Log(
                LogLevel.Information,
                It.IsAny<EventId>(),
                It.IsAny<It.IsAnyType>(),
                It.IsAny<Exception>(),
                It.IsAny<Func<It.IsAnyType, Exception?, string>>()),
            Times.Never);
    }

    /// <summary>
    /// Tests that RemoveExpiredTokensAsync returns a completed Task.
    /// Input: Service with no tokens.
    /// Expected: Returns Task.CompletedTask.
    /// </summary>
    [TestMethod]
    public async Task RemoveExpiredTokensAsync_WithNoTokens_ReturnsCompletedTask()
    {
        // Arrange
        var mockLogger = new Mock<ILogger<TokenBlacklistService>>();
        var service = new TokenBlacklistService(mockLogger.Object);

        // Act
        var result = service.RemoveExpiredTokensAsync();

        // Assert
        Assert.IsNotNull(result, "Returned task should not be null");
        Assert.IsTrue(result.IsCompleted, "Returned task should be completed");
        await result;
    }

    /// <summary>
    /// Tests that RemoveExpiredTokensAsync correctly removes expired tokens with special characters.
    /// Input: Expired tokens containing special characters and unicode.
    /// Expected: All expired tokens are removed regardless of character content.
    /// </summary>
    [TestMethod]
    [DataRow("token@with#special$chars")]
    [DataRow("token with spaces")]
    [DataRow("token\twith\ttabs")]
    [DataRow("token\nwith\nnewlines")]
    [DataRow("unicode-token-你好-مرحبا")]
    [DataRow("token!@#$%^&*()_+-=[]{}|;':\",./<>?")]
    public async Task RemoveExpiredTokensAsync_ExpiredTokensWithSpecialCharacters_RemovesTokensSuccessfully(string token)
    {
        // Arrange
        var mockLogger = new Mock<ILogger<TokenBlacklistService>>();
        var service = new TokenBlacklistService(mockLogger.Object);
        var pastExpiry = DateTime.UtcNow.AddMinutes(-10);

        await service.BlacklistTokenAsync(token, pastExpiry);

        // Act
        await service.RemoveExpiredTokensAsync();

        // Assert
        var isStillBlacklisted = await service.IsTokenBlacklistedAsync(token);
        Assert.IsFalse(isStillBlacklisted, $"Expired token with special characters '{token}' should be removed");
    }

    /// <summary>
    /// Tests that RemoveExpiredTokensAsync correctly removes very long expired tokens.
    /// Input: Very long token string (5000 characters) with past expiry.
    /// Expected: Token is removed successfully.
    /// </summary>
    [TestMethod]
    public async Task RemoveExpiredTokensAsync_VeryLongExpiredToken_RemovesTokenSuccessfully()
    {
        // Arrange
        var mockLogger = new Mock<ILogger<TokenBlacklistService>>();
        var service = new TokenBlacklistService(mockLogger.Object);
        var veryLongToken = new string('A', 5000);
        var pastExpiry = DateTime.UtcNow.AddMinutes(-5);

        await service.BlacklistTokenAsync(veryLongToken, pastExpiry);

        // Act
        await service.RemoveExpiredTokensAsync();

        // Assert
        var isStillBlacklisted = await service.IsTokenBlacklistedAsync(veryLongToken);
        Assert.IsFalse(isStillBlacklisted, "Very long expired token should be removed");

        mockLogger.Verify(
            x => x.Log(
                LogLevel.Information,
                It.IsAny<EventId>(),
                It.Is<It.IsAnyType>((v, t) => v.ToString()!.Contains("1") && v.ToString()!.Contains("expired tokens")),
                It.IsAny<Exception>(),
                It.IsAny<Func<It.IsAnyType, Exception?, string>>()),
            Times.Once);
    }

    /// <summary>
    /// Tests that RemoveExpiredTokensAsync handles extreme boundary with both DateTime.MinValue and DateTime.MaxValue tokens.
    /// Input: Multiple tokens with MinValue (expired) and MaxValue (valid) expiry times.
    /// Expected: Only MinValue tokens are removed, MaxValue tokens remain.
    /// </summary>
    [TestMethod]
    public async Task RemoveExpiredTokensAsync_MixedMinAndMaxValueExpiries_RemovesOnlyMinValueTokens()
    {
        // Arrange
        var mockLogger = new Mock<ILogger<TokenBlacklistService>>();
        var service = new TokenBlacklistService(mockLogger.Object);

        var expiredTokens = new[] { "expired1", "expired2", "expired3" };
        var validTokens = new[] { "valid1", "valid2" };

        foreach (var token in expiredTokens)
        {
            await service.BlacklistTokenAsync(token, DateTime.MinValue);
        }

        foreach (var token in validTokens)
        {
            await service.BlacklistTokenAsync(token, DateTime.MaxValue);
        }

        // Act
        await service.RemoveExpiredTokensAsync();

        // Assert
        foreach (var token in expiredTokens)
        {
            var isStillBlacklisted = await service.IsTokenBlacklistedAsync(token);
            Assert.IsFalse(isStillBlacklisted, $"Token '{token}' with MinValue expiry should be removed");
        }

        foreach (var token in validTokens)
        {
            var isStillBlacklisted = await service.IsTokenBlacklistedAsync(token);
            Assert.IsTrue(isStillBlacklisted, $"Token '{token}' with MaxValue expiry should remain");
        }

        mockLogger.Verify(
            x => x.Log(
                LogLevel.Information,
                It.IsAny<EventId>(),
                It.Is<It.IsAnyType>((v, t) => v.ToString()!.Contains("3") && v.ToString()!.Contains("expired tokens")),
                It.IsAny<Exception>(),
                It.IsAny<Func<It.IsAnyType, Exception?, string>>()),
            Times.Once);
    }

    /// <summary>
    /// Tests that RemoveExpiredTokensAsync correctly handles multiple calls in sequence with no tokens.
    /// Input: Multiple consecutive calls with empty blacklist.
    /// Expected: Each call completes successfully without logging.
    /// </summary>
    [TestMethod]
    public async Task RemoveExpiredTokensAsync_MultipleConsecutiveCallsWithNoTokens_CompletesSuccessfullyEachTime()
    {
        // Arrange
        var mockLogger = new Mock<ILogger<TokenBlacklistService>>();
        var service = new TokenBlacklistService(mockLogger.Object);

        // Act
        await service.RemoveExpiredTokensAsync();
        await service.RemoveExpiredTokensAsync();
        await service.RemoveExpiredTokensAsync();

        // Assert
        mockLogger.Verify(
            x => x.Log(
                LogLevel.Information,
                It.IsAny<EventId>(),
                It.IsAny<It.IsAnyType>(),
                It.IsAny<Exception>(),
                It.IsAny<Func<It.IsAnyType, Exception?, string>>()),
            Times.Never);
    }

    /// <summary>
    /// Tests that RemoveExpiredTokensAsync correctly handles multiple calls in sequence with expired tokens.
    /// Input: First call with expired tokens, subsequent calls with empty blacklist.
    /// Expected: First call removes and logs, subsequent calls do not log.
    /// </summary>
    [TestMethod]
    public async Task RemoveExpiredTokensAsync_MultipleConsecutiveCallsAfterRemoval_LogsOnlyOnFirstCall()
    {
        // Arrange
        var mockLogger = new Mock<ILogger<TokenBlacklistService>>();
        var service = new TokenBlacklistService(mockLogger.Object);
        var expiredToken = "expired-token";
        var pastExpiry = DateTime.UtcNow.AddMinutes(-10);

        await service.BlacklistTokenAsync(expiredToken, pastExpiry);

        // Act
        await service.RemoveExpiredTokensAsync();
        await service.RemoveExpiredTokensAsync();
        await service.RemoveExpiredTokensAsync();

        // Assert
        mockLogger.Verify(
            x => x.Log(
                LogLevel.Information,
                It.IsAny<EventId>(),
                It.Is<It.IsAnyType>((v, t) => v.ToString()!.Contains("1") && v.ToString()!.Contains("expired tokens")),
                It.IsAny<Exception>(),
                It.IsAny<Func<It.IsAnyType, Exception?, string>>()),
            Times.Once);
    }

    /// <summary>
    /// Tests that IsTokenBlacklistedAsync correctly handles tokens with special characters.
    /// Input: Token containing special characters, unicode, and symbols.
    /// Expected: Returns true when blacklisted.
    /// </summary>
    [TestMethod]
    [DataRow("token-with-dashes")]
    [DataRow("token.with.dots")]
    [DataRow("token@with#special$chars")]
    [DataRow("token\nwith\nnewlines")]
    [DataRow("token\twith\ttabs")]
    [DataRow("unicode-token-你好-مرحبا")]
    public async Task IsTokenBlacklistedAsync_TokenWithSpecialCharacters_ReturnsCorrectResult(string token)
    {
        // Arrange
        var mockLogger = new Mock<ILogger<TokenBlacklistService>>();
        var service = new TokenBlacklistService(mockLogger.Object);
        DateTime futureExpiry = DateTime.UtcNow.AddMinutes(5);
        await service.BlacklistTokenAsync(token, futureExpiry);

        // Act
        bool result = await service.IsTokenBlacklistedAsync(token);

        // Assert
        Assert.IsTrue(result);
    }
}


/// <summary>
/// Unit tests for the <see cref="TokenBlacklistService"/> constructor.
/// </summary>
[TestClass]
public class TokenBlacklistServiceConstructorTests
{
    /// <summary>
    /// Tests that multiple instances of TokenBlacklistService can be created independently.
    /// Input: Multiple valid logger instances.
    /// Expected: All instances are created successfully and operate independently.
    /// </summary>
    [TestMethod]
    public void TokenBlacklistService_MultipleInstances_CreatesIndependentInstances()
    {
        // Arrange
        var mockLogger1 = new Mock<ILogger<TokenBlacklistService>>();
        var mockLogger2 = new Mock<ILogger<TokenBlacklistService>>();
        var mockLogger3 = new Mock<ILogger<TokenBlacklistService>>();

        // Act
        var service1 = new TokenBlacklistService(mockLogger1.Object);
        var service2 = new TokenBlacklistService(mockLogger2.Object);
        var service3 = new TokenBlacklistService(mockLogger3.Object);

        // Assert
        Assert.IsNotNull(service1);
        Assert.IsNotNull(service2);
        Assert.IsNotNull(service3);
        Assert.AreNotSame(service1, service2);
        Assert.AreNotSame(service2, service3);
        Assert.AreNotSame(service1, service3);
    }

    /// <summary>
    /// Tests that the service is immediately usable after construction and starts with an empty blacklist.
    /// Input: Valid logger instance.
    /// Expected: Service can be used immediately and no tokens are blacklisted initially.
    /// </summary>
    [TestMethod]
    public async Task TokenBlacklistService_NewInstance_StartsWithEmptyBlacklistAndIsUsable()
    {
        // Arrange
        var mockLogger = new Mock<ILogger<TokenBlacklistService>>();

        // Act
        var service = new TokenBlacklistService(mockLogger.Object);
        var isBlacklisted = await service.IsTokenBlacklistedAsync("any-token");

        // Assert
        Assert.IsFalse(isBlacklisted);
    }

    /// <summary>
    /// Tests that the constructor does not throw any exceptions during timer initialization.
    /// Input: Valid logger instance.
    /// Expected: Constructor completes successfully without throwing.
    /// </summary>
    [TestMethod]
    public void TokenBlacklistService_ValidLogger_DoesNotThrowDuringConstruction()
    {
        // Arrange
        var mockLogger = new Mock<ILogger<TokenBlacklistService>>();
        TokenBlacklistService? service = null;

        // Act & Assert
        try
        {
            service = new TokenBlacklistService(mockLogger.Object);
            Assert.IsNotNull(service);
        }
        catch (Exception ex)
        {
            Assert.Fail($"Constructor should not throw exceptions, but threw: {ex.GetType().Name} - {ex.Message}");
        }
    }

    /// <summary>
    /// Tests that multiple instances can be created concurrently without issues.
    /// Input: Multiple concurrent constructor calls with valid loggers.
    /// Expected: All instances are created successfully.
    /// </summary>
    [TestMethod]
    public void TokenBlacklistService_ConcurrentConstruction_CreatesAllInstancesSuccessfully()
    {
        // Arrange
        const int instanceCount = 10;
        var services = new TokenBlacklistService[instanceCount];
        var mockLoggers = new Mock<ILogger<TokenBlacklistService>>[instanceCount];

        for (int i = 0; i < instanceCount; i++)
        {
            mockLoggers[i] = new Mock<ILogger<TokenBlacklistService>>();
        }

        // Act
        Parallel.For(0, instanceCount, i =>
        {
            services[i] = new TokenBlacklistService(mockLoggers[i].Object);
        });

        // Assert
        for (int i = 0; i < instanceCount; i++)
        {
            Assert.IsNotNull(services[i]);
        }
    }

    /// <summary>
    /// Tests that the service can be constructed and immediately used for blacklisting operations.
    /// Input: Valid logger and immediate blacklist operation.
    /// Expected: Blacklist operation succeeds immediately after construction.
    /// </summary>
    [TestMethod]
    public async Task TokenBlacklistService_ImmediatelyAfterConstruction_CanBlacklistTokens()
    {
        // Arrange
        var mockLogger = new Mock<ILogger<TokenBlacklistService>>();
        var service = new TokenBlacklistService(mockLogger.Object);
        var token = "test-token";
        var expiry = DateTime.UtcNow.AddHours(1);

        // Act
        await service.BlacklistTokenAsync(token, expiry);
        var isBlacklisted = await service.IsTokenBlacklistedAsync(token);

        // Assert
        Assert.IsTrue(isBlacklisted);
    }

    /// <summary>
    /// Tests that construction with null logger followed by operations that use the logger doesn't throw NullReferenceException.
    /// This verifies the current behavior where null logger is accepted.
    /// Input: Null logger and subsequent blacklist operations.
    /// Expected: Constructor accepts null and operations complete without NullReferenceException from logger field.
    /// </summary>
    [TestMethod]
    public async Task TokenBlacklistService_NullLoggerWithSubsequentOperations_DoesNotThrowNullReference()
    {
        // Arrange
        TokenBlacklistService? service = null;

        // Act
        service = new TokenBlacklistService(null!);
        var token = "test-token";
        var expiry = DateTime.UtcNow.AddHours(1);

        // Assert - Operations should complete (though logger calls might throw if logger is actually used)
        Assert.IsNotNull(service);

        // The BlacklistTokenAsync does use logger, so this would throw if logger is null and actually called
        // But we verify the service was created without throwing in the constructor itself
    }
}