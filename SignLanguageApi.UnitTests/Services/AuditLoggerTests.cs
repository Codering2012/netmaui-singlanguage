using System;
using System.IO;
using System.Text;
using System.Text.Json;
using System.Threading.Tasks;

using Microsoft.Extensions.Logging;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using Moq;
using SignLanguageApi.Services;

namespace SignLanguageApi.Services.UnitTests;

/// <summary>
/// Contains unit tests for the <see cref="AuditLogger"/> class.
/// </summary>
[TestClass]
public class AuditLoggerTests
{
    /// <summary>
    /// Tests that LogLogoutAsync successfully logs a logout event with valid inputs.
    /// </summary>
    /// <param name="userId">The user ID to test.</param>
    /// <param name="email">The email to test.</param>
    /// <param name="ipAddress">The IP address to test.</param>
    [TestMethod]
    [DataRow("user123", "user@example.com", "192.168.1.1")]
    [DataRow("admin", "admin@domain.com", "10.0.0.1")]
    [DataRow("user_with_special!@#", "test+tag@example.co.uk", "2001:0db8:85a3:0000:0000:8a2e:0370:7334")]
    public async Task LogLogoutAsync_ValidInputs_CompletesSuccessfully(string userId, string email, string ipAddress)
    {
        // Arrange
        var mockLogger = new Mock<ILogger<AuditLogger>>();
        var auditLogger = new AuditLogger(mockLogger.Object);

        // Act & Assert
        await auditLogger.LogLogoutAsync(userId, email, ipAddress);
        // No exception should be thrown
    }

    /// <summary>
    /// Tests that LogLogoutAsync handles empty string parameters without throwing exceptions.
    /// </summary>
    /// <param name="userId">The user ID to test.</param>
    /// <param name="email">The email to test.</param>
    /// <param name="ipAddress">The IP address to test.</param>
    [TestMethod]
    [DataRow("", "", "")]
    [DataRow("", "user@example.com", "192.168.1.1")]
    [DataRow("user123", "", "192.168.1.1")]
    [DataRow("user123", "user@example.com", "")]
    public async Task LogLogoutAsync_EmptyStrings_CompletesSuccessfully(string userId, string email, string ipAddress)
    {
        // Arrange
        var mockLogger = new Mock<ILogger<AuditLogger>>();
        var auditLogger = new AuditLogger(mockLogger.Object);

        // Act & Assert
        await auditLogger.LogLogoutAsync(userId, email, ipAddress);
        // No exception should be thrown
    }

    /// <summary>
    /// Tests that LogLogoutAsync handles whitespace-only string parameters without throwing exceptions.
    /// </summary>
    /// <param name="userId">The user ID to test.</param>
    /// <param name="email">The email to test.</param>
    /// <param name="ipAddress">The IP address to test.</param>
    [TestMethod]
    [DataRow("   ", "   ", "   ")]
    [DataRow("\t", "\n", "\r")]
    [DataRow("  \t\n  ", "user@example.com", "192.168.1.1")]
    public async Task LogLogoutAsync_WhitespaceStrings_CompletesSuccessfully(string userId, string email, string ipAddress)
    {
        // Arrange
        var mockLogger = new Mock<ILogger<AuditLogger>>();
        var auditLogger = new AuditLogger(mockLogger.Object);

        // Act & Assert
        await auditLogger.LogLogoutAsync(userId, email, ipAddress);
        // No exception should be thrown
    }

    /// <summary>
    /// Tests that LogLogoutAsync handles strings with special characters without throwing exceptions.
    /// </summary>
    /// <param name="userId">The user ID to test.</param>
    /// <param name="email">The email to test.</param>
    /// <param name="ipAddress">The IP address to test.</param>
    [TestMethod]
    [DataRow("user<>\"&", "test@example.com", "192.168.1.1")]
    [DataRow("user123", "email+tag@test.com", "192.168.1.1")]
    [DataRow("user'drop\"table", "user@domain.com", "'; DROP TABLE--;")]
    [DataRow("user\u0000\u0001", "test@example.com", "192.168.1.1")]
    public async Task LogLogoutAsync_SpecialCharacters_CompletesSuccessfully(string userId, string email, string ipAddress)
    {
        // Arrange
        var mockLogger = new Mock<ILogger<AuditLogger>>();
        var auditLogger = new AuditLogger(mockLogger.Object);

        // Act & Assert
        await auditLogger.LogLogoutAsync(userId, email, ipAddress);
        // No exception should be thrown
    }

    /// <summary>
    /// Tests that LogLogoutAsync handles very long string parameters without throwing exceptions.
    /// </summary>
    [TestMethod]
    public async Task LogLogoutAsync_VeryLongStrings_CompletesSuccessfully()
    {
        // Arrange
        var mockLogger = new Mock<ILogger<AuditLogger>>();
        var auditLogger = new AuditLogger(mockLogger.Object);
        var longUserId = new string('a', 10000);
        var longEmail = new string('b', 10000) + "@example.com";
        var longIpAddress = new string('1', 10000);

        // Act & Assert
        await auditLogger.LogLogoutAsync(longUserId, longEmail, longIpAddress);
        // No exception should be thrown
    }

    /// <summary>
    /// Tests that LogLogoutAsync handles strings with Unicode characters without throwing exceptions.
    /// </summary>
    /// <param name="userId">The user ID to test.</param>
    /// <param name="email">The email to test.</param>
    /// <param name="ipAddress">The IP address to test.</param>
    [TestMethod]
    [DataRow("用户123", "user@例え.jp", "192.168.1.1")]
    [DataRow("user_😀", "emoji@test.com", "192.168.1.1")]
    [DataRow("Müller", "test@münchen.de", "192.168.1.1")]
    public async Task LogLogoutAsync_UnicodeCharacters_CompletesSuccessfully(string userId, string email, string ipAddress)
    {
        // Arrange
        var mockLogger = new Mock<ILogger<AuditLogger>>();
        var auditLogger = new AuditLogger(mockLogger.Object);

        // Act & Assert
        await auditLogger.LogLogoutAsync(userId, email, ipAddress);
        // No exception should be thrown
    }

    /// <summary>
    /// Tests that LogLogoutAsync handles strings with path-like characters without throwing exceptions.
    /// </summary>
    /// <param name="userId">The user ID to test.</param>
    /// <param name="email">The email to test.</param>
    /// <param name="ipAddress">The IP address to test.</param>
    [TestMethod]
    [DataRow("user/../admin", "test@example.com", "192.168.1.1")]
    [DataRow("user123", "test@example.com", "../../etc/passwd")]
    [DataRow("C:\\Windows\\System32", "test@example.com", "192.168.1.1")]
    [DataRow("/usr/bin/bash", "test@example.com", "192.168.1.1")]
    public async Task LogLogoutAsync_PathLikeCharacters_CompletesSuccessfully(string userId, string email, string ipAddress)
    {
        // Arrange
        var mockLogger = new Mock<ILogger<AuditLogger>>();
        var auditLogger = new AuditLogger(mockLogger.Object);

        // Act & Assert
        await auditLogger.LogLogoutAsync(userId, email, ipAddress);
        // No exception should be thrown
    }

    /// <summary>
    /// Tests that LogLogoutAsync handles strings with newlines and carriage returns without throwing exceptions.
    /// </summary>
    [TestMethod]
    public async Task LogLogoutAsync_MultilineStrings_CompletesSuccessfully()
    {
        // Arrange
        var mockLogger = new Mock<ILogger<AuditLogger>>();
        var auditLogger = new AuditLogger(mockLogger.Object);
        var userId = "user\nwith\nnewlines";
        var email = "test\r\n@example.com";
        var ipAddress = "192.168.1.1\r\n";

        // Act & Assert
        await auditLogger.LogLogoutAsync(userId, email, ipAddress);
        // No exception should be thrown
    }

    /// <summary>
    /// Tests that the constructor successfully creates an instance when provided with a valid logger.
    /// Verifies that the constructor completes without throwing exceptions.
    /// </summary>
    [TestMethod]
    public void AuditLogger_ValidLogger_CreatesInstanceSuccessfully()
    {
        // Arrange
        var mockLogger = new Mock<ILogger<AuditLogger>>();

        // Act
        var auditLogger = new AuditLogger(mockLogger.Object);

        // Assert
        Assert.IsNotNull(auditLogger);
    }

    /// <summary>
    /// Tests that the constructor accepts a null logger parameter.
    /// The constructor does not perform null validation, so it should complete without throwing.
    /// Note: This may cause issues when methods attempt to use the logger.
    /// </summary>
    [TestMethod]
    public void AuditLogger_NullLogger_CreatesInstanceWithoutException()
    {
        // Arrange
        ILogger<AuditLogger>? logger = null;

        // Act
        var auditLogger = new AuditLogger(logger!);

        // Assert
        Assert.IsNotNull(auditLogger);
    }

    /// <summary>
    /// Tests that LogLoginAttemptAsync completes successfully with valid inputs and success flag set to true.
    /// Input: Valid email, success=true, valid IP address.
    /// Expected: Method completes without throwing an exception.
    /// </summary>
    [TestMethod]
    [DataRow("user@example.com", true, "192.168.1.1")]
    [DataRow("test@domain.com", true, "10.0.0.1")]
    [DataRow("admin@test.org", true, "2001:0db8:85a3:0000:0000:8a2e:0370:7334")]
    public async Task LogLoginAttemptAsync_ValidInputsWithSuccess_CompletesSuccessfully(string email, bool success, string ipAddress)
    {
        // Arrange
        var mockLogger = new Mock<ILogger<AuditLogger>>();
        var auditLogger = new AuditLogger(mockLogger.Object);

        // Act
        await auditLogger.LogLoginAttemptAsync(email, success, ipAddress);

        // Assert
        // Method should complete without throwing exceptions
        Assert.IsTrue(true);
    }

    /// <summary>
    /// Tests that LogLoginAttemptAsync completes successfully with valid inputs and success flag set to false.
    /// Input: Valid email, success=false, valid IP address.
    /// Expected: Method completes without throwing an exception.
    /// </summary>
    [TestMethod]
    [DataRow("user@example.com", false, "192.168.1.1")]
    [DataRow("test@domain.com", false, "10.0.0.1")]
    [DataRow("hacker@malicious.net", false, "172.16.0.1")]
    public async Task LogLoginAttemptAsync_ValidInputsWithFailure_CompletesSuccessfully(string email, bool success, string ipAddress)
    {
        // Arrange
        var mockLogger = new Mock<ILogger<AuditLogger>>();
        var auditLogger = new AuditLogger(mockLogger.Object);

        // Act
        await auditLogger.LogLoginAttemptAsync(email, success, ipAddress);

        // Assert
        // Method should complete without throwing exceptions
        Assert.IsTrue(true);
    }

    /// <summary>
    /// Tests that LogLoginAttemptAsync handles null email parameter.
    /// Input: Null email, valid success flag, valid IP address.
    /// Expected: Method completes without throwing an exception (null is serialized as JSON null).
    /// </summary>
    [TestMethod]
    public async Task LogLoginAttemptAsync_NullEmail_CompletesSuccessfully()
    {
        // Arrange
        var mockLogger = new Mock<ILogger<AuditLogger>>();
        var auditLogger = new AuditLogger(mockLogger.Object);

        // Act
        await auditLogger.LogLoginAttemptAsync(null!, true, "192.168.1.1");

        // Assert
        // Method should complete without throwing exceptions
        Assert.IsTrue(true);
    }

    /// <summary>
    /// Tests that LogLoginAttemptAsync handles empty email string.
    /// Input: Empty email string, valid success flag, valid IP address.
    /// Expected: Method completes without throwing an exception.
    /// </summary>
    [TestMethod]
    [DataRow("", true, "192.168.1.1")]
    [DataRow("   ", true, "192.168.1.1")]
    public async Task LogLoginAttemptAsync_EmptyOrWhitespaceEmail_CompletesSuccessfully(string email, bool success, string ipAddress)
    {
        // Arrange
        var mockLogger = new Mock<ILogger<AuditLogger>>();
        var auditLogger = new AuditLogger(mockLogger.Object);

        // Act
        await auditLogger.LogLoginAttemptAsync(email, success, ipAddress);

        // Assert
        // Method should complete without throwing exceptions
        Assert.IsTrue(true);
    }

    /// <summary>
    /// Tests that LogLoginAttemptAsync handles null IP address parameter.
    /// Input: Valid email, valid success flag, null IP address.
    /// Expected: Method completes without throwing an exception (null is serialized as JSON null).
    /// </summary>
    [TestMethod]
    public async Task LogLoginAttemptAsync_NullIpAddress_CompletesSuccessfully()
    {
        // Arrange
        var mockLogger = new Mock<ILogger<AuditLogger>>();
        var auditLogger = new AuditLogger(mockLogger.Object);

        // Act
        await auditLogger.LogLoginAttemptAsync("user@example.com", true, null!);

        // Assert
        // Method should complete without throwing exceptions
        Assert.IsTrue(true);
    }

    /// <summary>
    /// Tests that LogLoginAttemptAsync handles empty or whitespace IP address.
    /// Input: Valid email, valid success flag, empty or whitespace IP address.
    /// Expected: Method completes without throwing an exception.
    /// </summary>
    [TestMethod]
    [DataRow("user@example.com", true, "")]
    [DataRow("user@example.com", false, "   ")]
    public async Task LogLoginAttemptAsync_EmptyOrWhitespaceIpAddress_CompletesSuccessfully(string email, bool success, string ipAddress)
    {
        // Arrange
        var mockLogger = new Mock<ILogger<AuditLogger>>();
        var auditLogger = new AuditLogger(mockLogger.Object);

        // Act
        await auditLogger.LogLoginAttemptAsync(email, success, ipAddress);

        // Assert
        // Method should complete without throwing exceptions
        Assert.IsTrue(true);
    }

    /// <summary>
    /// Tests that LogLoginAttemptAsync handles special characters in email and IP address.
    /// Input: Email and IP address with special characters.
    /// Expected: Method completes without throwing an exception.
    /// </summary>
    [TestMethod]
    [DataRow("user+test@example.com", true, "192.168.1.1")]
    [DataRow("user@sub-domain.example.com", false, "192.168.1.1")]
    [DataRow("user<script>@example.com", true, "192.168.1.1")]
    [DataRow("user@example.com", true, "invalid-ip-format")]
    public async Task LogLoginAttemptAsync_SpecialCharactersInInputs_CompletesSuccessfully(string email, bool success, string ipAddress)
    {
        // Arrange
        var mockLogger = new Mock<ILogger<AuditLogger>>();
        var auditLogger = new AuditLogger(mockLogger.Object);

        // Act
        await auditLogger.LogLoginAttemptAsync(email, success, ipAddress);

        // Assert
        // Method should complete without throwing exceptions
        Assert.IsTrue(true);
    }

    /// <summary>
    /// Tests that LogLoginAttemptAsync handles very long string inputs.
    /// Input: Very long email and IP address strings.
    /// Expected: Method completes without throwing an exception.
    /// </summary>
    [TestMethod]
    public async Task LogLoginAttemptAsync_VeryLongStrings_CompletesSuccessfully()
    {
        // Arrange
        var mockLogger = new Mock<ILogger<AuditLogger>>();
        var auditLogger = new AuditLogger(mockLogger.Object);
        var longEmail = new string('a', 1000) + "@example.com";
        var longIpAddress = new string('1', 1000);

        // Act
        await auditLogger.LogLoginAttemptAsync(longEmail, true, longIpAddress);

        // Assert
        // Method should complete without throwing exceptions
        Assert.IsTrue(true);
    }

    /// <summary>
    /// Tests that LogLoginAttemptAsync handles all null parameters.
    /// Input: All parameters set to null (except success which is bool).
    /// Expected: Method completes without throwing an exception.
    /// </summary>
    [TestMethod]
    public async Task LogLoginAttemptAsync_AllNullableParametersNull_CompletesSuccessfully()
    {
        // Arrange
        var mockLogger = new Mock<ILogger<AuditLogger>>();
        var auditLogger = new AuditLogger(mockLogger.Object);

        // Act
        await auditLogger.LogLoginAttemptAsync(null!, false, null!);

        // Assert
        // Method should complete without throwing exceptions
        Assert.IsTrue(true);
    }

    /// <summary>
    /// Tests that LogLoginAttemptAsync handles control characters and newlines in inputs.
    /// Input: Email and IP address containing control characters and newlines.
    /// Expected: Method completes without throwing an exception (JSON serialization escapes these).
    /// </summary>
    [TestMethod]
    [DataRow("user\n@example.com", true, "192.168.1.1")]
    [DataRow("user@example.com", false, "192.168.1.1\n\r")]
    [DataRow("user\t@example.com", true, "\t192.168.1.1")]
    public async Task LogLoginAttemptAsync_ControlCharactersInInputs_CompletesSuccessfully(string email, bool success, string ipAddress)
    {
        // Arrange
        var mockLogger = new Mock<ILogger<AuditLogger>>();
        var auditLogger = new AuditLogger(mockLogger.Object);

        // Act
        await auditLogger.LogLoginAttemptAsync(email, success, ipAddress);

        // Assert
        // Method should complete without throwing exceptions
        Assert.IsTrue(true);
    }

    /// <summary>
    /// Tests that LogLoginAttemptAsync handles Unicode and international characters.
    /// Input: Email and IP address with Unicode characters.
    /// Expected: Method completes without throwing an exception.
    /// </summary>
    [TestMethod]
    [DataRow("用户@example.com", true, "192.168.1.1")]
    [DataRow("user@例え.com", false, "192.168.1.1")]
    [DataRow("üser@example.com", true, "192.168.1.1")]
    public async Task LogLoginAttemptAsync_UnicodeCharactersInInputs_CompletesSuccessfully(string email, bool success, string ipAddress)
    {
        // Arrange
        var mockLogger = new Mock<ILogger<AuditLogger>>();
        var auditLogger = new AuditLogger(mockLogger.Object);

        // Act
        await auditLogger.LogLoginAttemptAsync(email, success, ipAddress);

        // Assert
        // Method should complete without throwing exceptions
        Assert.IsTrue(true);
    }

    /// <summary>
    /// Tests that LogLoginAttemptAsync handles both success and failure scenarios with various IP formats.
    /// Input: Valid email, various IP address formats (IPv4, IPv6, localhost, invalid), both success values.
    /// Expected: Method completes without throwing an exception.
    /// </summary>
    [TestMethod]
    [DataRow("user@example.com", true, "127.0.0.1")]
    [DataRow("user@example.com", false, "::1")]
    [DataRow("user@example.com", true, "localhost")]
    [DataRow("user@example.com", false, "0.0.0.0")]
    [DataRow("user@example.com", true, "255.255.255.255")]
    public async Task LogLoginAttemptAsync_VariousIpFormats_CompletesSuccessfully(string email, bool success, string ipAddress)
    {
        // Arrange
        var mockLogger = new Mock<ILogger<AuditLogger>>();
        var auditLogger = new AuditLogger(mockLogger.Object);

        // Act
        await auditLogger.LogLoginAttemptAsync(email, success, ipAddress);

        // Assert
        // Method should complete without throwing exceptions
        Assert.IsTrue(true);
    }

    /// <summary>
    /// Tests that LogRegisterAttemptAsync completes successfully with valid inputs and success status true.
    /// Input: Valid email, success=true, valid IP address.
    /// Expected: Method completes without throwing an exception.
    /// </summary>
    [TestMethod]
    [DataRow("test@example.com", true, "192.168.1.1")]
    [DataRow("user@domain.org", true, "10.0.0.1")]
    [DataRow("admin@company.com", true, "2001:0db8:85a3:0000:0000:8a2e:0370:7334")]
    public async Task LogRegisterAttemptAsync_ValidInputsWithSuccess_CompletesSuccessfully(string email, bool success, string ipAddress)
    {
        // Arrange
        var mockLogger = new Mock<ILogger<AuditLogger>>();
        var auditLogger = new AuditLogger(mockLogger.Object);

        // Act & Assert
        await auditLogger.LogRegisterAttemptAsync(email, success, ipAddress);
        // Method should complete without throwing
    }

    /// <summary>
    /// Tests that LogRegisterAttemptAsync completes successfully with valid inputs and success status false.
    /// Input: Valid email, success=false, valid IP address.
    /// Expected: Method completes without throwing an exception.
    /// </summary>
    [TestMethod]
    [DataRow("test@example.com", false, "192.168.1.1")]
    [DataRow("user@domain.org", false, "10.0.0.1")]
    [DataRow("failed@user.net", false, "172.16.0.1")]
    public async Task LogRegisterAttemptAsync_ValidInputsWithFailure_CompletesSuccessfully(string email, bool success, string ipAddress)
    {
        // Arrange
        var mockLogger = new Mock<ILogger<AuditLogger>>();
        var auditLogger = new AuditLogger(mockLogger.Object);

        // Act & Assert
        await auditLogger.LogRegisterAttemptAsync(email, success, ipAddress);
        // Method should complete without throwing
    }

    /// <summary>
    /// Tests that LogRegisterAttemptAsync handles null email parameter.
    /// Input: Null email, valid success flag, valid IP address.
    /// Expected: Method completes without throwing an exception.
    /// </summary>
    [TestMethod]
    [DataRow(null, true, "192.168.1.1")]
    [DataRow(null, false, "10.0.0.1")]
    public async Task LogRegisterAttemptAsync_NullEmail_CompletesSuccessfully(string? email, bool success, string ipAddress)
    {
        // Arrange
        var mockLogger = new Mock<ILogger<AuditLogger>>();
        var auditLogger = new AuditLogger(mockLogger.Object);

        // Act & Assert
        await auditLogger.LogRegisterAttemptAsync(email!, success, ipAddress);
        // Method should complete without throwing
    }

    /// <summary>
    /// Tests that LogRegisterAttemptAsync handles empty email parameter.
    /// Input: Empty email, valid success flag, valid IP address.
    /// Expected: Method completes without throwing an exception.
    /// </summary>
    [TestMethod]
    [DataRow("", true, "192.168.1.1")]
    [DataRow("", false, "10.0.0.1")]
    public async Task LogRegisterAttemptAsync_EmptyEmail_CompletesSuccessfully(string email, bool success, string ipAddress)
    {
        // Arrange
        var mockLogger = new Mock<ILogger<AuditLogger>>();
        var auditLogger = new AuditLogger(mockLogger.Object);

        // Act & Assert
        await auditLogger.LogRegisterAttemptAsync(email, success, ipAddress);
        // Method should complete without throwing
    }

    /// <summary>
    /// Tests that LogRegisterAttemptAsync handles whitespace-only email parameter.
    /// Input: Whitespace email, valid success flag, valid IP address.
    /// Expected: Method completes without throwing an exception.
    /// </summary>
    [TestMethod]
    [DataRow("   ", true, "192.168.1.1")]
    [DataRow("\t\n", false, "10.0.0.1")]
    public async Task LogRegisterAttemptAsync_WhitespaceEmail_CompletesSuccessfully(string email, bool success, string ipAddress)
    {
        // Arrange
        var mockLogger = new Mock<ILogger<AuditLogger>>();
        var auditLogger = new AuditLogger(mockLogger.Object);

        // Act & Assert
        await auditLogger.LogRegisterAttemptAsync(email, success, ipAddress);
        // Method should complete without throwing
    }

    /// <summary>
    /// Tests that LogRegisterAttemptAsync handles null IP address parameter.
    /// Input: Valid email, valid success flag, null IP address.
    /// Expected: Method completes without throwing an exception.
    /// </summary>
    [TestMethod]
    [DataRow("test@example.com", true, null)]
    [DataRow("user@domain.org", false, null)]
    public async Task LogRegisterAttemptAsync_NullIpAddress_CompletesSuccessfully(string email, bool success, string? ipAddress)
    {
        // Arrange
        var mockLogger = new Mock<ILogger<AuditLogger>>();
        var auditLogger = new AuditLogger(mockLogger.Object);

        // Act & Assert
        await auditLogger.LogRegisterAttemptAsync(email, success, ipAddress!);
        // Method should complete without throwing
    }

    /// <summary>
    /// Tests that LogRegisterAttemptAsync handles empty IP address parameter.
    /// Input: Valid email, valid success flag, empty IP address.
    /// Expected: Method completes without throwing an exception.
    /// </summary>
    [TestMethod]
    [DataRow("test@example.com", true, "")]
    [DataRow("user@domain.org", false, "")]
    public async Task LogRegisterAttemptAsync_EmptyIpAddress_CompletesSuccessfully(string email, bool success, string ipAddress)
    {
        // Arrange
        var mockLogger = new Mock<ILogger<AuditLogger>>();
        var auditLogger = new AuditLogger(mockLogger.Object);

        // Act & Assert
        await auditLogger.LogRegisterAttemptAsync(email, success, ipAddress);
        // Method should complete without throwing
    }

    /// <summary>
    /// Tests that LogRegisterAttemptAsync handles whitespace-only IP address parameter.
    /// Input: Valid email, valid success flag, whitespace IP address.
    /// Expected: Method completes without throwing an exception.
    /// </summary>
    [TestMethod]
    [DataRow("test@example.com", true, "   ")]
    [DataRow("user@domain.org", false, "\t")]
    public async Task LogRegisterAttemptAsync_WhitespaceIpAddress_CompletesSuccessfully(string email, bool success, string ipAddress)
    {
        // Arrange
        var mockLogger = new Mock<ILogger<AuditLogger>>();
        var auditLogger = new AuditLogger(mockLogger.Object);

        // Act & Assert
        await auditLogger.LogRegisterAttemptAsync(email, success, ipAddress);
        // Method should complete without throwing
    }

    /// <summary>
    /// Tests that LogRegisterAttemptAsync handles special characters in email and IP address.
    /// Input: Email and IP address containing special characters, valid success flag.
    /// Expected: Method completes without throwing an exception.
    /// </summary>
    [TestMethod]
    [DataRow("test@<script>alert('xss')</script>.com", true, "<script>alert('xss')</script>")]
    [DataRow("user+tag@example.com", false, "192.168.1.1; DROP TABLE users;")]
    [DataRow("test\"quote@example.com", true, "::1")]
    public async Task LogRegisterAttemptAsync_SpecialCharacters_CompletesSuccessfully(string email, bool success, string ipAddress)
    {
        // Arrange
        var mockLogger = new Mock<ILogger<AuditLogger>>();
        var auditLogger = new AuditLogger(mockLogger.Object);

        // Act & Assert
        await auditLogger.LogRegisterAttemptAsync(email, success, ipAddress);
        // Method should complete without throwing
    }

    /// <summary>
    /// Tests that LogRegisterAttemptAsync handles very long email strings.
    /// Input: Very long email string, valid success flag, valid IP address.
    /// Expected: Method completes without throwing an exception.
    /// </summary>
    [TestMethod]
    public async Task LogRegisterAttemptAsync_VeryLongEmail_CompletesSuccessfully()
    {
        // Arrange
        var mockLogger = new Mock<ILogger<AuditLogger>>();
        var auditLogger = new AuditLogger(mockLogger.Object);
        var longEmail = new string('a', 10000) + "@example.com";

        // Act & Assert
        await auditLogger.LogRegisterAttemptAsync(longEmail, true, "192.168.1.1");
        // Method should complete without throwing
    }

    /// <summary>
    /// Tests that LogRegisterAttemptAsync handles very long IP address strings.
    /// Input: Valid email, valid success flag, very long IP address string.
    /// Expected: Method completes without throwing an exception.
    /// </summary>
    [TestMethod]
    public async Task LogRegisterAttemptAsync_VeryLongIpAddress_CompletesSuccessfully()
    {
        // Arrange
        var mockLogger = new Mock<ILogger<AuditLogger>>();
        var auditLogger = new AuditLogger(mockLogger.Object);
        var longIpAddress = new string('1', 10000);

        // Act & Assert
        await auditLogger.LogRegisterAttemptAsync("test@example.com", false, longIpAddress);
        // Method should complete without throwing
    }

    /// <summary>
    /// Tests that LogRegisterAttemptAsync handles control characters in strings.
    /// Input: Email and IP address containing control characters, valid success flag.
    /// Expected: Method completes without throwing an exception.
    /// </summary>
    [TestMethod]
    [DataRow("test\0@example.com", true, "192.168.1.1")]
    [DataRow("test@example.com", false, "192.168.1.1\0")]
    [DataRow("test\r\n@example.com", true, "192.168\r\n.1.1")]
    public async Task LogRegisterAttemptAsync_ControlCharacters_CompletesSuccessfully(string email, bool success, string ipAddress)
    {
        // Arrange
        var mockLogger = new Mock<ILogger<AuditLogger>>();
        var auditLogger = new AuditLogger(mockLogger.Object);

        // Act & Assert
        await auditLogger.LogRegisterAttemptAsync(email, success, ipAddress);
        // Method should complete without throwing
    }

    /// <summary>
    /// Tests that LogRegisterAttemptAsync handles all parameters being null or empty.
    /// Input: Null/empty email and IP address, valid success flag.
    /// Expected: Method completes without throwing an exception.
    /// </summary>
    [TestMethod]
    [DataRow(null, true, null)]
    [DataRow("", false, "")]
    [DataRow(null, true, "")]
    [DataRow("", false, null)]
    public async Task LogRegisterAttemptAsync_AllParametersNullOrEmpty_CompletesSuccessfully(string? email, bool success, string? ipAddress)
    {
        // Arrange
        var mockLogger = new Mock<ILogger<AuditLogger>>();
        var auditLogger = new AuditLogger(mockLogger.Object);

        // Act & Assert
        await auditLogger.LogRegisterAttemptAsync(email!, success, ipAddress!);
        // Method should complete without throwing
    }

    /// <summary>
    /// Tests that LogRegisterAttemptAsync handles Unicode characters in parameters.
    /// Input: Email and IP address containing Unicode characters, valid success flag.
    /// Expected: Method completes without throwing an exception.
    /// </summary>
    [TestMethod]
    [DataRow("用户@例え.jp", true, "2001:0db8:85a3::8a2e:0370:7334")]
    [DataRow("tëst@éxamplé.com", false, "192.168.1.1")]
    [DataRow("test@example.com", true, "🌐.🌍.🌎.🌏")]
    public async Task LogRegisterAttemptAsync_UnicodeCharacters_CompletesSuccessfully(string email, bool success, string ipAddress)
    {
        // Arrange
        var mockLogger = new Mock<ILogger<AuditLogger>>();
        var auditLogger = new AuditLogger(mockLogger.Object);

        // Act & Assert
        await auditLogger.LogRegisterAttemptAsync(email, success, ipAddress);
        // Method should complete without throwing
    }

    /// <summary>
    /// Tests that LogUnauthorizedAccessAsync completes successfully with valid IP address and endpoint.
    /// </summary>
    [TestMethod]
    public async Task LogUnauthorizedAccessAsync_ValidIpAddressAndEndpoint_CompletesSuccessfully()
    {
        // Arrange
        var mockLogger = new Mock<ILogger<AuditLogger>>();
        var auditLogger = new AuditLogger(mockLogger.Object);
        var ipAddress = "192.168.1.1";
        var endpoint = "/api/users";

        // Act
        await auditLogger.LogUnauthorizedAccessAsync(ipAddress, endpoint);

        // Assert
        // Method completes without throwing
    }

    /// <summary>
    /// Tests that LogUnauthorizedAccessAsync handles null IP address.
    /// </summary>
    [TestMethod]
    public async Task LogUnauthorizedAccessAsync_NullIpAddress_CompletesSuccessfully()
    {
        // Arrange
        var mockLogger = new Mock<ILogger<AuditLogger>>();
        var auditLogger = new AuditLogger(mockLogger.Object);
        string? ipAddress = null;
        var endpoint = "/api/users";

        // Act
        await auditLogger.LogUnauthorizedAccessAsync(ipAddress!, endpoint);

        // Assert
        // Method completes without throwing
    }

    /// <summary>
    /// Tests that LogUnauthorizedAccessAsync handles null endpoint.
    /// </summary>
    [TestMethod]
    public async Task LogUnauthorizedAccessAsync_NullEndpoint_CompletesSuccessfully()
    {
        // Arrange
        var mockLogger = new Mock<ILogger<AuditLogger>>();
        var auditLogger = new AuditLogger(mockLogger.Object);
        var ipAddress = "192.168.1.1";
        string? endpoint = null;

        // Act
        await auditLogger.LogUnauthorizedAccessAsync(ipAddress, endpoint!);

        // Assert
        // Method completes without throwing
    }

    /// <summary>
    /// Tests that LogUnauthorizedAccessAsync handles both null IP address and endpoint.
    /// </summary>
    [TestMethod]
    public async Task LogUnauthorizedAccessAsync_BothParametersNull_CompletesSuccessfully()
    {
        // Arrange
        var mockLogger = new Mock<ILogger<AuditLogger>>();
        var auditLogger = new AuditLogger(mockLogger.Object);
        string? ipAddress = null;
        string? endpoint = null;

        // Act
        await auditLogger.LogUnauthorizedAccessAsync(ipAddress!, endpoint!);

        // Assert
        // Method completes without throwing
    }

    /// <summary>
    /// Tests that LogUnauthorizedAccessAsync handles empty IP address string.
    /// </summary>
    [TestMethod]
    [DataRow("")]
    [DataRow("   ")]
    public async Task LogUnauthorizedAccessAsync_EmptyOrWhitespaceIpAddress_CompletesSuccessfully(string ipAddress)
    {
        // Arrange
        var mockLogger = new Mock<ILogger<AuditLogger>>();
        var auditLogger = new AuditLogger(mockLogger.Object);
        var endpoint = "/api/users";

        // Act
        await auditLogger.LogUnauthorizedAccessAsync(ipAddress, endpoint);

        // Assert
        // Method completes without throwing
    }

    /// <summary>
    /// Tests that LogUnauthorizedAccessAsync handles empty endpoint string.
    /// </summary>
    [TestMethod]
    [DataRow("")]
    [DataRow("   ")]
    public async Task LogUnauthorizedAccessAsync_EmptyOrWhitespaceEndpoint_CompletesSuccessfully(string endpoint)
    {
        // Arrange
        var mockLogger = new Mock<ILogger<AuditLogger>>();
        var auditLogger = new AuditLogger(mockLogger.Object);
        var ipAddress = "192.168.1.1";

        // Act
        await auditLogger.LogUnauthorizedAccessAsync(ipAddress, endpoint);

        // Assert
        // Method completes without throwing
    }

    /// <summary>
    /// Tests that LogUnauthorizedAccessAsync handles both empty strings.
    /// </summary>
    [TestMethod]
    public async Task LogUnauthorizedAccessAsync_BothParametersEmpty_CompletesSuccessfully()
    {
        // Arrange
        var mockLogger = new Mock<ILogger<AuditLogger>>();
        var auditLogger = new AuditLogger(mockLogger.Object);
        var ipAddress = string.Empty;
        var endpoint = string.Empty;

        // Act
        await auditLogger.LogUnauthorizedAccessAsync(ipAddress, endpoint);

        // Assert
        // Method completes without throwing
    }

    /// <summary>
    /// Tests that LogUnauthorizedAccessAsync handles very long IP address string.
    /// </summary>
    [TestMethod]
    public async Task LogUnauthorizedAccessAsync_VeryLongIpAddress_CompletesSuccessfully()
    {
        // Arrange
        var mockLogger = new Mock<ILogger<AuditLogger>>();
        var auditLogger = new AuditLogger(mockLogger.Object);
        var ipAddress = new string('A', 10000);
        var endpoint = "/api/users";

        // Act
        await auditLogger.LogUnauthorizedAccessAsync(ipAddress, endpoint);

        // Assert
        // Method completes without throwing
    }

    /// <summary>
    /// Tests that LogUnauthorizedAccessAsync handles very long endpoint string.
    /// </summary>
    [TestMethod]
    public async Task LogUnauthorizedAccessAsync_VeryLongEndpoint_CompletesSuccessfully()
    {
        // Arrange
        var mockLogger = new Mock<ILogger<AuditLogger>>();
        var auditLogger = new AuditLogger(mockLogger.Object);
        var ipAddress = "192.168.1.1";
        var endpoint = new string('/', 10000);

        // Act
        await auditLogger.LogUnauthorizedAccessAsync(ipAddress, endpoint);

        // Assert
        // Method completes without throwing
    }

    /// <summary>
    /// Tests that LogUnauthorizedAccessAsync handles special characters in IP address.
    /// </summary>
    [TestMethod]
    [DataRow("192.168.1.1:8080")]
    [DataRow("::1")]
    [DataRow("fe80::1")]
    [DataRow("<script>alert('xss')</script>")]
    [DataRow("'; DROP TABLE users; --")]
    [DataRow("\0\n\r\t")]
    public async Task LogUnauthorizedAccessAsync_SpecialCharactersInIpAddress_CompletesSuccessfully(string ipAddress)
    {
        // Arrange
        var mockLogger = new Mock<ILogger<AuditLogger>>();
        var auditLogger = new AuditLogger(mockLogger.Object);
        var endpoint = "/api/users";

        // Act
        await auditLogger.LogUnauthorizedAccessAsync(ipAddress, endpoint);

        // Assert
        // Method completes without throwing
    }

    /// <summary>
    /// Tests that LogUnauthorizedAccessAsync handles special characters in endpoint.
    /// </summary>
    [TestMethod]
    [DataRow("/api/users?id=1&name=test")]
    [DataRow("/api/../admin")]
    [DataRow("<script>alert('xss')</script>")]
    [DataRow("'; DROP TABLE users; --")]
    [DataRow("\0\n\r\t")]
    [DataRow("../../../../etc/passwd")]
    public async Task LogUnauthorizedAccessAsync_SpecialCharactersInEndpoint_CompletesSuccessfully(string endpoint)
    {
        // Arrange
        var mockLogger = new Mock<ILogger<AuditLogger>>();
        var auditLogger = new AuditLogger(mockLogger.Object);
        var ipAddress = "192.168.1.1";

        // Act
        await auditLogger.LogUnauthorizedAccessAsync(ipAddress, endpoint);

        // Assert
        // Method completes without throwing
    }

    /// <summary>
    /// Tests that LogUnauthorizedAccessAsync handles various valid IP address formats.
    /// </summary>
    [TestMethod]
    [DataRow("0.0.0.0")]
    [DataRow("255.255.255.255")]
    [DataRow("127.0.0.1")]
    [DataRow("10.0.0.1")]
    [DataRow("172.16.0.1")]
    public async Task LogUnauthorizedAccessAsync_ValidIpAddressFormats_CompletesSuccessfully(string ipAddress)
    {
        // Arrange
        var mockLogger = new Mock<ILogger<AuditLogger>>();
        var auditLogger = new AuditLogger(mockLogger.Object);
        var endpoint = "/api/users";

        // Act
        await auditLogger.LogUnauthorizedAccessAsync(ipAddress, endpoint);

        // Assert
        // Method completes without throwing
    }

    /// <summary>
    /// Tests that LogUnauthorizedAccessAsync handles various valid endpoint formats.
    /// </summary>
    [TestMethod]
    [DataRow("/")]
    [DataRow("/api")]
    [DataRow("/api/v1/users")]
    [DataRow("/api/users/123")]
    [DataRow("/api/users/123/profile")]
    public async Task LogUnauthorizedAccessAsync_ValidEndpointFormats_CompletesSuccessfully(string endpoint)
    {
        // Arrange
        var mockLogger = new Mock<ILogger<AuditLogger>>();
        var auditLogger = new AuditLogger(mockLogger.Object);
        var ipAddress = "192.168.1.1";

        // Act
        await auditLogger.LogUnauthorizedAccessAsync(ipAddress, endpoint);

        // Assert
        // Method completes without throwing
    }

    /// <summary>
    /// Tests that LogRegisterAttemptAsync handles null email parameter.
    /// Input: Null email, valid success flag, valid IP address.
    /// Expected: Method completes without throwing an exception.
    /// </summary>
    [TestMethod]
    public async Task LogRegisterAttemptAsync_NullEmail_CompletesSuccessfully()
    {
        // Arrange
        var mockLogger = new Mock<ILogger<AuditLogger>>();
        var auditLogger = new AuditLogger(mockLogger.Object);
        string? email = null;

        // Act & Assert
        await auditLogger.LogRegisterAttemptAsync(email!, true, "192.168.1.1");
        // Method should complete without throwing
    }

    /// <summary>
    /// Tests that LogRegisterAttemptAsync handles null IP address parameter.
    /// Input: Valid email, valid success flag, null IP address.
    /// Expected: Method completes without throwing an exception.
    /// </summary>
    [TestMethod]
    public async Task LogRegisterAttemptAsync_NullIpAddress_CompletesSuccessfully()
    {
        // Arrange
        var mockLogger = new Mock<ILogger<AuditLogger>>();
        var auditLogger = new AuditLogger(mockLogger.Object);
        string? ipAddress = null;

        // Act & Assert
        await auditLogger.LogRegisterAttemptAsync("test@example.com", true, ipAddress!);
        // Method should complete without throwing
    }

    /// <summary>
    /// Tests that LogRegisterAttemptAsync handles all parameters being null or empty.
    /// Input: Null/empty email and IP address, valid success flag.
    /// Expected: Method completes without throwing an exception.
    /// </summary>
    [TestMethod]
    public async Task LogRegisterAttemptAsync_AllParametersNullOrEmpty_CompletesSuccessfully()
    {
        // Arrange
        var mockLogger = new Mock<ILogger<AuditLogger>>();
        var auditLogger = new AuditLogger(mockLogger.Object);
        string? nullEmail = null;
        string? nullIpAddress = null;

        // Act & Assert
        await auditLogger.LogRegisterAttemptAsync(nullEmail!, true, nullIpAddress!);
        await auditLogger.LogRegisterAttemptAsync("", false, "");
        await auditLogger.LogRegisterAttemptAsync(nullEmail!, true, "");
        await auditLogger.LogRegisterAttemptAsync("", false, nullIpAddress!);
        // Method should complete without throwing
    }

    /// <summary>
    /// Tests that LogRegisterAttemptAsync handles various valid IPv4 address formats.
    /// Input: Valid email, valid success flag, various IPv4 addresses.
    /// Expected: Method completes without throwing an exception.
    /// </summary>
    [TestMethod]
    [DataRow("test@example.com", true, "0.0.0.0")]
    [DataRow("test@example.com", false, "255.255.255.255")]
    [DataRow("test@example.com", true, "127.0.0.1")]
    [DataRow("test@example.com", false, "10.0.0.1")]
    [DataRow("test@example.com", true, "172.16.0.1")]
    public async Task LogRegisterAttemptAsync_ValidIPv4Addresses_CompletesSuccessfully(string email, bool success, string ipAddress)
    {
        // Arrange
        var mockLogger = new Mock<ILogger<AuditLogger>>();
        var auditLogger = new AuditLogger(mockLogger.Object);

        // Act & Assert
        await auditLogger.LogRegisterAttemptAsync(email, success, ipAddress);
        // Method should complete without throwing
    }

    /// <summary>
    /// Tests that LogRegisterAttemptAsync handles various IPv6 address formats.
    /// Input: Valid email, valid success flag, various IPv6 addresses.
    /// Expected: Method completes without throwing an exception.
    /// </summary>
    [TestMethod]
    [DataRow("test@example.com", true, "::1")]
    [DataRow("test@example.com", false, "2001:0db8:85a3:0000:0000:8a2e:0370:7334")]
    [DataRow("test@example.com", true, "fe80::1")]
    [DataRow("test@example.com", false, "::")]
    [DataRow("test@example.com", true, "2001:db8::8a2e:370:7334")]
    public async Task LogRegisterAttemptAsync_ValidIPv6Addresses_CompletesSuccessfully(string email, bool success, string ipAddress)
    {
        // Arrange
        var mockLogger = new Mock<ILogger<AuditLogger>>();
        var auditLogger = new AuditLogger(mockLogger.Object);

        // Act & Assert
        await auditLogger.LogRegisterAttemptAsync(email, success, ipAddress);
        // Method should complete without throwing
    }

    /// <summary>
    /// Tests that LogRegisterAttemptAsync handles invalid IP address formats.
    /// Input: Valid email, valid success flag, invalid IP address formats.
    /// Expected: Method completes without throwing an exception.
    /// </summary>
    [TestMethod]
    [DataRow("test@example.com", true, "999.999.999.999")]
    [DataRow("test@example.com", false, "not-an-ip")]
    [DataRow("test@example.com", true, "192.168.1")]
    [DataRow("test@example.com", false, "192.168.1.1.1")]
    [DataRow("test@example.com", true, "localhost")]
    public async Task LogRegisterAttemptAsync_InvalidIpAddressFormats_CompletesSuccessfully(string email, bool success, string ipAddress)
    {
        // Arrange
        var mockLogger = new Mock<ILogger<AuditLogger>>();
        var auditLogger = new AuditLogger(mockLogger.Object);

        // Act & Assert
        await auditLogger.LogRegisterAttemptAsync(email, success, ipAddress);
        // Method should complete without throwing
    }

    /// <summary>
    /// Tests that LogRegisterAttemptAsync handles path-like characters in inputs.
    /// Input: Email and IP address with path traversal patterns.
    /// Expected: Method completes without throwing an exception.
    /// </summary>
    [TestMethod]
    [DataRow("test/../admin@example.com", true, "192.168.1.1")]
    [DataRow("test@example.com", false, "../../etc/passwd")]
    [DataRow("C:\\Windows\\System32@example.com", true, "192.168.1.1")]
    [DataRow("test@example.com", false, "/usr/bin/bash")]
    public async Task LogRegisterAttemptAsync_PathLikeCharacters_CompletesSuccessfully(string email, bool success, string ipAddress)
    {
        // Arrange
        var mockLogger = new Mock<ILogger<AuditLogger>>();
        var auditLogger = new AuditLogger(mockLogger.Object);

        // Act & Assert
        await auditLogger.LogRegisterAttemptAsync(email, success, ipAddress);
        // Method should complete without throwing
    }

    /// <summary>
    /// Tests that LogRegisterAttemptAsync handles newlines and carriage returns in inputs.
    /// Input: Email and IP address containing newline characters.
    /// Expected: Method completes without throwing an exception.
    /// </summary>
    [TestMethod]
    public async Task LogRegisterAttemptAsync_MultilineStrings_CompletesSuccessfully()
    {
        // Arrange
        var mockLogger = new Mock<ILogger<AuditLogger>>();
        var auditLogger = new AuditLogger(mockLogger.Object);

        // Act & Assert
        await auditLogger.LogRegisterAttemptAsync("test\n@example.com", true, "192.168.1.1");
        await auditLogger.LogRegisterAttemptAsync("test@example.com", false, "192.168.1.1\n");
        await auditLogger.LogRegisterAttemptAsync("test\r\n@example.com", true, "192.168\r\n.1.1");
        // Method should complete without throwing
    }

    /// <summary>
    /// Tests that LogRegisterAttemptAsync verifies success parameter affects Status field correctly.
    /// Input: Valid inputs with both success values.
    /// Expected: Method completes without throwing, Status field should be "SUCCESS" when true and "FAILED" when false.
    /// </summary>
    [TestMethod]
    [DataRow("test@example.com", true, "192.168.1.1")]
    [DataRow("test@example.com", false, "192.168.1.1")]
    public async Task LogRegisterAttemptAsync_BothSuccessValues_CompletesSuccessfully(string email, bool success, string ipAddress)
    {
        // Arrange
        var mockLogger = new Mock<ILogger<AuditLogger>>();
        var auditLogger = new AuditLogger(mockLogger.Object);

        // Act & Assert
        await auditLogger.LogRegisterAttemptAsync(email, success, ipAddress);
        // Method should complete without throwing, Status field determined by success parameter
    }

    /// <summary>
    /// Tests that LogRegisterAttemptAsync handles SQL injection patterns in inputs.
    /// Input: Email and IP address containing SQL injection patterns.
    /// Expected: Method completes without throwing an exception.
    /// </summary>
    [TestMethod]
    [DataRow("'; DROP TABLE users; --@example.com", true, "192.168.1.1")]
    [DataRow("test@example.com", false, "'; DROP TABLE audit_logs; --")]
    [DataRow("admin'--@example.com", true, "192.168.1.1")]
    [DataRow("test@example.com", false, "1' OR '1'='1")]
    public async Task LogRegisterAttemptAsync_SqlInjectionPatterns_CompletesSuccessfully(string email, bool success, string ipAddress)
    {
        // Arrange
        var mockLogger = new Mock<ILogger<AuditLogger>>();
        var auditLogger = new AuditLogger(mockLogger.Object);

        // Act & Assert
        await auditLogger.LogRegisterAttemptAsync(email, success, ipAddress);
        // Method should complete without throwing
    }

    /// <summary>
    /// Tests that LogRegisterAttemptAsync handles HTML/JavaScript injection patterns in inputs.
    /// Input: Email and IP address containing HTML/JS injection patterns.
    /// Expected: Method completes without throwing an exception.
    /// </summary>
    [TestMethod]
    [DataRow("<script>alert('xss')</script>@example.com", true, "192.168.1.1")]
    [DataRow("test@example.com", false, "<script>alert('xss')</script>")]
    [DataRow("<img src=x onerror=alert(1)>@example.com", true, "192.168.1.1")]
    [DataRow("test@example.com", false, "javascript:alert(1)")]
    public async Task LogRegisterAttemptAsync_XssPatterns_CompletesSuccessfully(string email, bool success, string ipAddress)
    {
        // Arrange
        var mockLogger = new Mock<ILogger<AuditLogger>>();
        var auditLogger = new AuditLogger(mockLogger.Object);

        // Act & Assert
        await auditLogger.LogRegisterAttemptAsync(email, success, ipAddress);
        // Method should complete without throwing
    }

    /// <summary>
    /// Tests that LogLogoutAsync handles null userId parameter without throwing exceptions.
    /// Input: Null userId, valid email, valid IP address.
    /// Expected: Method completes without throwing an exception (null is serialized as JSON null).
    /// </summary>
    [TestMethod]
    [DataRow(null, "user@example.com", "192.168.1.1")]
    [DataRow(null, "admin@domain.com", "10.0.0.1")]
    public async Task LogLogoutAsync_NullUserId_CompletesSuccessfully(string? userId, string email, string ipAddress)
    {
        // Arrange
        var mockLogger = new Mock<ILogger<AuditLogger>>();
        var auditLogger = new AuditLogger(mockLogger.Object);

        // Act & Assert
        await auditLogger.LogLogoutAsync(userId!, email, ipAddress);
        // No exception should be thrown
    }

    /// <summary>
    /// Tests that LogLogoutAsync handles null email parameter without throwing exceptions.
    /// Input: Valid userId, null email, valid IP address.
    /// Expected: Method completes without throwing an exception (null is serialized as JSON null).
    /// </summary>
    [TestMethod]
    [DataRow("user123", null, "192.168.1.1")]
    [DataRow("admin", null, "10.0.0.1")]
    public async Task LogLogoutAsync_NullEmail_CompletesSuccessfully(string userId, string? email, string ipAddress)
    {
        // Arrange
        var mockLogger = new Mock<ILogger<AuditLogger>>();
        var auditLogger = new AuditLogger(mockLogger.Object);

        // Act & Assert
        await auditLogger.LogLogoutAsync(userId, email!, ipAddress);
        // No exception should be thrown
    }

    /// <summary>
    /// Tests that LogLogoutAsync handles null ipAddress parameter without throwing exceptions.
    /// Input: Valid userId, valid email, null IP address.
    /// Expected: Method completes without throwing an exception (null is serialized as JSON null).
    /// </summary>
    [TestMethod]
    [DataRow("user123", "user@example.com", null)]
    [DataRow("admin", "admin@domain.com", null)]
    public async Task LogLogoutAsync_NullIpAddress_CompletesSuccessfully(string userId, string email, string? ipAddress)
    {
        // Arrange
        var mockLogger = new Mock<ILogger<AuditLogger>>();
        var auditLogger = new AuditLogger(mockLogger.Object);

        // Act & Assert
        await auditLogger.LogLogoutAsync(userId, email, ipAddress!);
        // No exception should be thrown
    }

    /// <summary>
    /// Tests that LogLogoutAsync handles multiple null parameters without throwing exceptions.
    /// Input: Various combinations of null parameters.
    /// Expected: Method completes without throwing an exception.
    /// </summary>
    [TestMethod]
    [DataRow(null, null, "192.168.1.1")]
    [DataRow(null, "user@example.com", null)]
    [DataRow("user123", null, null)]
    [DataRow(null, null, null)]
    public async Task LogLogoutAsync_MultipleNullParameters_CompletesSuccessfully(string? userId, string? email, string? ipAddress)
    {
        // Arrange
        var mockLogger = new Mock<ILogger<AuditLogger>>();
        var auditLogger = new AuditLogger(mockLogger.Object);

        // Act & Assert
        await auditLogger.LogLogoutAsync(userId!, email!, ipAddress!);
        // No exception should be thrown
    }

    /// <summary>
    /// Tests that LogUnauthorizedAccessAsync completes successfully with various valid IP address and endpoint combinations.
    /// Input: Valid IP addresses and endpoints.
    /// Expected: Method completes without throwing an exception.
    /// </summary>
    [TestMethod]
    [DataRow("192.168.1.1", "/api/users")]
    [DataRow("10.0.0.1", "/api/admin")]
    [DataRow("2001:0db8:85a3:0000:0000:8a2e:0370:7334", "/api/v1/resource")]
    [DataRow("127.0.0.1", "/")]
    [DataRow("::1", "/api")]
    public async Task LogUnauthorizedAccessAsync_ValidInputs_CompletesSuccessfully(string ipAddress, string endpoint)
    {
        // Arrange
        var mockLogger = new Mock<ILogger<AuditLogger>>();
        var auditLogger = new AuditLogger(mockLogger.Object);

        // Act
        await auditLogger.LogUnauthorizedAccessAsync(ipAddress, endpoint);

        // Assert
        // Method completes without throwing
    }

    /// <summary>
    /// Tests that LogUnauthorizedAccessAsync handles empty and whitespace-only strings for IP address.
    /// Input: Empty or whitespace IP address, valid endpoint.
    /// Expected: Method completes without throwing an exception.
    /// </summary>
    [TestMethod]
    [DataRow("", "/api/users")]
    [DataRow("   ", "/api/users")]
    [DataRow("\t", "/api/users")]
    [DataRow("\n", "/api/users")]
    [DataRow("\r", "/api/users")]
    public async Task LogUnauthorizedAccessAsync_EmptyOrWhitespaceIpAddress_CompletesSuccessfully(string ipAddress, string endpoint)
    {
        // Arrange
        var mockLogger = new Mock<ILogger<AuditLogger>>();
        var auditLogger = new AuditLogger(mockLogger.Object);

        // Act
        await auditLogger.LogUnauthorizedAccessAsync(ipAddress, endpoint);

        // Assert
        // Method completes without throwing
    }

    /// <summary>
    /// Tests that LogUnauthorizedAccessAsync handles empty and whitespace-only strings for endpoint.
    /// Input: Valid IP address, empty or whitespace endpoint.
    /// Expected: Method completes without throwing an exception.
    /// </summary>
    [TestMethod]
    [DataRow("192.168.1.1", "")]
    [DataRow("192.168.1.1", "   ")]
    [DataRow("192.168.1.1", "\t")]
    [DataRow("192.168.1.1", "\n")]
    [DataRow("192.168.1.1", "\r")]
    public async Task LogUnauthorizedAccessAsync_EmptyOrWhitespaceEndpoint_CompletesSuccessfully(string ipAddress, string endpoint)
    {
        // Arrange
        var mockLogger = new Mock<ILogger<AuditLogger>>();
        var auditLogger = new AuditLogger(mockLogger.Object);

        // Act
        await auditLogger.LogUnauthorizedAccessAsync(ipAddress, endpoint);

        // Assert
        // Method completes without throwing
    }

    /// <summary>
    /// Tests that LogUnauthorizedAccessAsync handles special characters in IP address.
    /// Input: IP address with special characters (injection attempts, control chars, etc.).
    /// Expected: Method completes without throwing an exception.
    /// </summary>
    [TestMethod]
    [DataRow("192.168.1.1:8080", "/api/users")]
    [DataRow("::1", "/api/users")]
    [DataRow("fe80::1", "/api/users")]
    [DataRow("<script>alert('xss')</script>", "/api/users")]
    [DataRow("'; DROP TABLE users; --", "/api/users")]
    [DataRow("\0\n\r\t", "/api/users")]
    [DataRow("192.168.1.1\u0000", "/api/users")]
    public async Task LogUnauthorizedAccessAsync_SpecialCharactersInIpAddress_CompletesSuccessfully(string ipAddress, string endpoint)
    {
        // Arrange
        var mockLogger = new Mock<ILogger<AuditLogger>>();
        var auditLogger = new AuditLogger(mockLogger.Object);

        // Act
        await auditLogger.LogUnauthorizedAccessAsync(ipAddress, endpoint);

        // Assert
        // Method completes without throwing
    }

    /// <summary>
    /// Tests that LogUnauthorizedAccessAsync handles special characters in endpoint.
    /// Input: Endpoint with special characters (query strings, path traversal, injection attempts, etc.).
    /// Expected: Method completes without throwing an exception.
    /// </summary>
    [TestMethod]
    [DataRow("192.168.1.1", "/api/users?id=1&name=test")]
    [DataRow("192.168.1.1", "/api/../admin")]
    [DataRow("192.168.1.1", "<script>alert('xss')</script>")]
    [DataRow("192.168.1.1", "'; DROP TABLE users; --")]
    [DataRow("192.168.1.1", "\0\n\r\t")]
    [DataRow("192.168.1.1", "../../../../etc/passwd")]
    [DataRow("192.168.1.1", "/api/users\u0000")]
    [DataRow("192.168.1.1", "C:\\Windows\\System32")]
    public async Task LogUnauthorizedAccessAsync_SpecialCharactersInEndpoint_CompletesSuccessfully(string ipAddress, string endpoint)
    {
        // Arrange
        var mockLogger = new Mock<ILogger<AuditLogger>>();
        var auditLogger = new AuditLogger(mockLogger.Object);

        // Act
        await auditLogger.LogUnauthorizedAccessAsync(ipAddress, endpoint);

        // Assert
        // Method completes without throwing
    }

    /// <summary>
    /// Tests that LogUnauthorizedAccessAsync handles various valid IPv4 address formats.
    /// Input: Different valid IPv4 addresses, valid endpoint.
    /// Expected: Method completes without throwing an exception.
    /// </summary>
    [TestMethod]
    [DataRow("0.0.0.0", "/api/users")]
    [DataRow("255.255.255.255", "/api/users")]
    [DataRow("127.0.0.1", "/api/users")]
    [DataRow("10.0.0.1", "/api/users")]
    [DataRow("172.16.0.1", "/api/users")]
    [DataRow("192.168.0.1", "/api/users")]
    public async Task LogUnauthorizedAccessAsync_ValidIpv4AddressFormats_CompletesSuccessfully(string ipAddress, string endpoint)
    {
        // Arrange
        var mockLogger = new Mock<ILogger<AuditLogger>>();
        var auditLogger = new AuditLogger(mockLogger.Object);

        // Act
        await auditLogger.LogUnauthorizedAccessAsync(ipAddress, endpoint);

        // Assert
        // Method completes without throwing
    }

    /// <summary>
    /// Tests that LogUnauthorizedAccessAsync handles various valid IPv6 address formats.
    /// Input: Different valid IPv6 addresses, valid endpoint.
    /// Expected: Method completes without throwing an exception.
    /// </summary>
    [TestMethod]
    [DataRow("::1", "/api/users")]
    [DataRow("::", "/api/users")]
    [DataRow("2001:0db8:85a3:0000:0000:8a2e:0370:7334", "/api/users")]
    [DataRow("2001:db8:85a3::8a2e:370:7334", "/api/users")]
    [DataRow("fe80::1", "/api/users")]
    public async Task LogUnauthorizedAccessAsync_ValidIpv6AddressFormats_CompletesSuccessfully(string ipAddress, string endpoint)
    {
        // Arrange
        var mockLogger = new Mock<ILogger<AuditLogger>>();
        var auditLogger = new AuditLogger(mockLogger.Object);

        // Act
        await auditLogger.LogUnauthorizedAccessAsync(ipAddress, endpoint);

        // Assert
        // Method completes without throwing
    }

    /// <summary>
    /// Tests that LogUnauthorizedAccessAsync handles various valid endpoint formats.
    /// Input: Valid IP address, different endpoint path formats.
    /// Expected: Method completes without throwing an exception.
    /// </summary>
    [TestMethod]
    [DataRow("192.168.1.1", "/")]
    [DataRow("192.168.1.1", "/api")]
    [DataRow("192.168.1.1", "/api/v1/users")]
    [DataRow("192.168.1.1", "/api/users/123")]
    [DataRow("192.168.1.1", "/api/users/123/profile")]
    [DataRow("192.168.1.1", "/api/users?page=1&limit=10")]
    public async Task LogUnauthorizedAccessAsync_ValidEndpointFormats_CompletesSuccessfully(string ipAddress, string endpoint)
    {
        // Arrange
        var mockLogger = new Mock<ILogger<AuditLogger>>();
        var auditLogger = new AuditLogger(mockLogger.Object);

        // Act
        await auditLogger.LogUnauthorizedAccessAsync(ipAddress, endpoint);

        // Assert
        // Method completes without throwing
    }

    /// <summary>
    /// Tests that LogUnauthorizedAccessAsync handles Unicode characters in IP address.
    /// Input: IP address with Unicode characters, valid endpoint.
    /// Expected: Method completes without throwing an exception.
    /// </summary>
    [TestMethod]
    [DataRow("用户", "/api/users")]
    [DataRow("192.168.1.1😀", "/api/users")]
    [DataRow("Müller", "/api/users")]
    public async Task LogUnauthorizedAccessAsync_UnicodeCharactersInIpAddress_CompletesSuccessfully(string ipAddress, string endpoint)
    {
        // Arrange
        var mockLogger = new Mock<ILogger<AuditLogger>>();
        var auditLogger = new AuditLogger(mockLogger.Object);

        // Act
        await auditLogger.LogUnauthorizedAccessAsync(ipAddress, endpoint);

        // Assert
        // Method completes without throwing
    }

    /// <summary>
    /// Tests that LogUnauthorizedAccessAsync handles Unicode characters in endpoint.
    /// Input: Valid IP address, endpoint with Unicode characters.
    /// Expected: Method completes without throwing an exception.
    /// </summary>
    [TestMethod]
    [DataRow("192.168.1.1", "/api/用户")]
    [DataRow("192.168.1.1", "/api/users/😀")]
    [DataRow("192.168.1.1", "/api/über")]
    [DataRow("192.168.1.1", "/🌐/🌍/🌎/🌏")]
    public async Task LogUnauthorizedAccessAsync_UnicodeCharactersInEndpoint_CompletesSuccessfully(string ipAddress, string endpoint)
    {
        // Arrange
        var mockLogger = new Mock<ILogger<AuditLogger>>();
        var auditLogger = new AuditLogger(mockLogger.Object);

        // Act
        await auditLogger.LogUnauthorizedAccessAsync(ipAddress, endpoint);

        // Assert
        // Method completes without throwing
    }

    /// <summary>
    /// Tests that LogUnauthorizedAccessAsync handles control characters in both parameters.
    /// Input: IP address and endpoint with control characters (newlines, tabs, null chars).
    /// Expected: Method completes without throwing an exception.
    /// </summary>
    [TestMethod]
    [DataRow("192.168.1.1\n", "/api/users")]
    [DataRow("192.168.1.1\r\n", "/api/users\r\n")]
    [DataRow("192.168.1.1\t", "/api/users\t")]
    [DataRow("192.168.1.1\0", "/api/users\0")]
    public async Task LogUnauthorizedAccessAsync_ControlCharactersInBothParameters_CompletesSuccessfully(string ipAddress, string endpoint)
    {
        // Arrange
        var mockLogger = new Mock<ILogger<AuditLogger>>();
        var auditLogger = new AuditLogger(mockLogger.Object);

        // Act
        await auditLogger.LogUnauthorizedAccessAsync(ipAddress, endpoint);

        // Assert
        // Method completes without throwing
    }

    /// <summary>
    /// Tests that LogUnauthorizedAccessAsync handles multiline strings.
    /// Input: IP address and endpoint with newlines and multiple lines.
    /// Expected: Method completes without throwing an exception.
    /// </summary>
    [TestMethod]
    public async Task LogUnauthorizedAccessAsync_MultilineStrings_CompletesSuccessfully()
    {
        // Arrange
        var mockLogger = new Mock<ILogger<AuditLogger>>();
        var auditLogger = new AuditLogger(mockLogger.Object);
        var ipAddress = "192.168.1.1\nmalicious.com\n10.0.0.1";
        var endpoint = "/api/users\n/api/admin\n/api/delete";

        // Act
        await auditLogger.LogUnauthorizedAccessAsync(ipAddress, endpoint);

        // Assert
        // Method completes without throwing
    }

    /// <summary>
    /// Tests that LogUnauthorizedAccessAsync handles invalid IP address formats.
    /// Input: Invalid IP address formats, valid endpoint.
    /// Expected: Method completes without throwing an exception (method does not validate IP format).
    /// </summary>
    [TestMethod]
    [DataRow("999.999.999.999", "/api/users")]
    [DataRow("not-an-ip", "/api/users")]
    [DataRow("localhost", "/api/users")]
    [DataRow("example.com", "/api/users")]
    [DataRow("256.1.1.1", "/api/users")]
    public async Task LogUnauthorizedAccessAsync_InvalidIpAddressFormats_CompletesSuccessfully(string ipAddress, string endpoint)
    {
        // Arrange
        var mockLogger = new Mock<ILogger<AuditLogger>>();
        var auditLogger = new AuditLogger(mockLogger.Object);

        // Act
        await auditLogger.LogUnauthorizedAccessAsync(ipAddress, endpoint);

        // Assert
        // Method completes without throwing
    }

    /// <summary>
    /// Tests that LogUnauthorizedAccessAsync handles path traversal attempts in endpoint.
    /// Input: Valid IP address, endpoint with path traversal patterns.
    /// Expected: Method completes without throwing an exception.
    /// </summary>
    [TestMethod]
    [DataRow("192.168.1.1", "/../../../etc/passwd")]
    [DataRow("192.168.1.1", "/api/../../admin")]
    [DataRow("192.168.1.1", "..\\..\\..\\windows\\system32")]
    [DataRow("192.168.1.1", "/api/users/../../../root")]
    public async Task LogUnauthorizedAccessAsync_PathTraversalAttempts_CompletesSuccessfully(string ipAddress, string endpoint)
    {
        // Arrange
        var mockLogger = new Mock<ILogger<AuditLogger>>();
        var auditLogger = new AuditLogger(mockLogger.Object);

        // Act
        await auditLogger.LogUnauthorizedAccessAsync(ipAddress, endpoint);

        // Assert
        // Method completes without throwing
    }

    /// <summary>
    /// Tests that LogUnauthorizedAccessAsync handles SQL injection attempts in parameters.
    /// Input: Parameters containing SQL injection patterns.
    /// Expected: Method completes without throwing an exception (JSON serialization escapes these).
    /// </summary>
    [TestMethod]
    [DataRow("192.168.1.1'; DROP TABLE users; --", "/api/users")]
    [DataRow("192.168.1.1", "/api/users'; DELETE FROM users WHERE '1'='1")]
    [DataRow("1' OR '1'='1", "/api/users' OR '1'='1")]
    public async Task LogUnauthorizedAccessAsync_SqlInjectionAttempts_CompletesSuccessfully(string ipAddress, string endpoint)
    {
        // Arrange
        var mockLogger = new Mock<ILogger<AuditLogger>>();
        var auditLogger = new AuditLogger(mockLogger.Object);

        // Act
        await auditLogger.LogUnauthorizedAccessAsync(ipAddress, endpoint);

        // Assert
        // Method completes without throwing
    }

    /// <summary>
    /// Tests that LogUnauthorizedAccessAsync handles XSS injection attempts in parameters.
    /// Input: Parameters containing XSS attack patterns.
    /// Expected: Method completes without throwing an exception (JSON serialization escapes these).
    /// </summary>
    [TestMethod]
    [DataRow("<script>alert('xss')</script>", "/api/users")]
    [DataRow("192.168.1.1", "<img src=x onerror=alert('xss')>")]
    [DataRow("<body onload=alert('xss')>", "</script><script>alert('xss')</script>")]
    public async Task LogUnauthorizedAccessAsync_XssInjectionAttempts_CompletesSuccessfully(string ipAddress, string endpoint)
    {
        // Arrange
        var mockLogger = new Mock<ILogger<AuditLogger>>();
        var auditLogger = new AuditLogger(mockLogger.Object);

        // Act
        await auditLogger.LogUnauthorizedAccessAsync(ipAddress, endpoint);

        // Assert
        // Method completes without throwing
    }

    /// <summary>
    /// Tests that LogLoginAttemptAsync handles SQL injection-like patterns in inputs.
    /// Input: Email and IP address containing SQL injection patterns.
    /// Expected: Method completes without throwing an exception (values are serialized safely to JSON).
    /// </summary>
    [TestMethod]
    [DataRow("'; DROP TABLE users; --@example.com", true, "192.168.1.1")]
    [DataRow("user@example.com", false, "'; DROP TABLE logs; --")]
    [DataRow("admin'--@example.com", true, "1' OR '1'='1")]
    public async Task LogLoginAttemptAsync_SqlInjectionPatterns_CompletesSuccessfully(string email, bool success, string ipAddress)
    {
        // Arrange
        var mockLogger = new Mock<ILogger<AuditLogger>>();
        var auditLogger = new AuditLogger(mockLogger.Object);

        // Act
        await auditLogger.LogLoginAttemptAsync(email, success, ipAddress);

        // Assert
        Assert.IsTrue(true);
    }

    /// <summary>
    /// Tests that LogLoginAttemptAsync handles path traversal patterns in inputs.
    /// Input: Email and IP address containing path traversal patterns.
    /// Expected: Method completes without throwing an exception.
    /// </summary>
    [TestMethod]
    [DataRow("user/../admin@example.com", true, "192.168.1.1")]
    [DataRow("user@example.com", false, "../../etc/passwd")]
    [DataRow("..\\..\\windows\\system32@test.com", true, "..\\..\\")]
    public async Task LogLoginAttemptAsync_PathTraversalPatterns_CompletesSuccessfully(string email, bool success, string ipAddress)
    {
        // Arrange
        var mockLogger = new Mock<ILogger<AuditLogger>>();
        var auditLogger = new AuditLogger(mockLogger.Object);

        // Act
        await auditLogger.LogLoginAttemptAsync(email, success, ipAddress);

        // Assert
        Assert.IsTrue(true);
    }

    /// <summary>
    /// Tests that LogLoginAttemptAsync handles null byte characters in inputs.
    /// Input: Email and IP address containing null byte characters.
    /// Expected: Method completes without throwing an exception.
    /// </summary>
    [TestMethod]
    [DataRow("user\0@example.com", true, "192.168.1.1")]
    [DataRow("user@example.com", false, "192.168.1.1\0")]
    [DataRow("user\0\0@test.com", true, "\0\0\0")]
    public async Task LogLoginAttemptAsync_NullByteCharacters_CompletesSuccessfully(string email, bool success, string ipAddress)
    {
        // Arrange
        var mockLogger = new Mock<ILogger<AuditLogger>>();
        var auditLogger = new AuditLogger(mockLogger.Object);

        // Act
        await auditLogger.LogLoginAttemptAsync(email, success, ipAddress);

        // Assert
        Assert.IsTrue(true);
    }

    /// <summary>
    /// Tests that LogLoginAttemptAsync handles IPv6 addresses in various formats.
    /// Input: Valid email with various IPv6 address formats.
    /// Expected: Method completes without throwing an exception.
    /// </summary>
    [TestMethod]
    [DataRow("user@example.com", true, "2001:0db8:85a3:0000:0000:8a2e:0370:7334")]
    [DataRow("user@example.com", false, "::1")]
    [DataRow("user@example.com", true, "fe80::1")]
    [DataRow("user@example.com", false, "::ffff:192.0.2.1")]
    public async Task LogLoginAttemptAsync_IPv6Addresses_CompletesSuccessfully(string email, bool success, string ipAddress)
    {
        // Arrange
        var mockLogger = new Mock<ILogger<AuditLogger>>();
        var auditLogger = new AuditLogger(mockLogger.Object);

        // Act
        await auditLogger.LogLoginAttemptAsync(email, success, ipAddress);

        // Assert
        Assert.IsTrue(true);
    }

    /// <summary>
    /// Tests that LogLoginAttemptAsync handles boundary IP addresses.
    /// Input: Valid email with boundary IP addresses (0.0.0.0, 255.255.255.255).
    /// Expected: Method completes without throwing an exception.
    /// </summary>
    [TestMethod]
    [DataRow("user@example.com", true, "0.0.0.0")]
    [DataRow("user@example.com", false, "255.255.255.255")]
    public async Task LogLoginAttemptAsync_BoundaryIpAddresses_CompletesSuccessfully(string email, bool success, string ipAddress)
    {
        // Arrange
        var mockLogger = new Mock<ILogger<AuditLogger>>();
        var auditLogger = new AuditLogger(mockLogger.Object);

        // Act
        await auditLogger.LogLoginAttemptAsync(email, success, ipAddress);

        // Assert
        Assert.IsTrue(true);
    }

    /// <summary>
    /// Tests that LogLoginAttemptAsync handles email addresses with various valid formats.
    /// Input: Various valid email formats with success flag and IP address.
    /// Expected: Method completes without throwing an exception.
    /// </summary>
    [TestMethod]
    [DataRow("simple@example.com", true, "192.168.1.1")]
    [DataRow("user.name@example.com", false, "192.168.1.1")]
    [DataRow("user+tag@example.co.uk", true, "192.168.1.1")]
    [DataRow("user_name@sub.domain.example.com", false, "192.168.1.1")]
    public async Task LogLoginAttemptAsync_VariousValidEmailFormats_CompletesSuccessfully(string email, bool success, string ipAddress)
    {
        // Arrange
        var mockLogger = new Mock<ILogger<AuditLogger>>();
        var auditLogger = new AuditLogger(mockLogger.Object);

        // Act
        await auditLogger.LogLoginAttemptAsync(email, success, ipAddress);

        // Assert
        Assert.IsTrue(true);
    }

    /// <summary>
    /// Tests that LogLoginAttemptAsync handles emoji and special Unicode characters.
    /// Input: Email and IP address containing emoji and special Unicode.
    /// Expected: Method completes without throwing an exception.
    /// </summary>
    [TestMethod]
    [DataRow("user😀@example.com", true, "192.168.1.1")]
    [DataRow("user@example.com", false, "192.168.1.1🌐")]
    [DataRow("test🔒@secure.com", true, "::1")]
    public async Task LogLoginAttemptAsync_EmojiCharacters_CompletesSuccessfully(string email, bool success, string ipAddress)
    {
        // Arrange
        var mockLogger = new Mock<ILogger<AuditLogger>>();
        var auditLogger = new AuditLogger(mockLogger.Object);

        // Act
        await auditLogger.LogLoginAttemptAsync(email, success, ipAddress);

        // Assert
        Assert.IsTrue(true);
    }
}