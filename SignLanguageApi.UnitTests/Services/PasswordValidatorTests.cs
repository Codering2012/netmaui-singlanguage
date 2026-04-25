using System;
using System.Linq;
using System.Text;
using System.Text.RegularExpressions;

using Microsoft.Extensions.Logging;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using Moq;
using SignLanguageApi;
using SignLanguageApi.Services;

namespace SignLanguageApi.Services.UnitTests;

/// <summary>
/// Unit tests for the PasswordValidator class.
/// </summary>
[TestClass]
public class PasswordValidatorTests
{
    /// <summary>
    /// Tests that ValidatePassword returns false with appropriate error message when password is null.
    /// Input: null
    /// Expected: (false, "Password cannot be empty.")
    /// </summary>
    [TestMethod]
    public void ValidatePassword_NullPassword_ReturnsFalseWithErrorMessage()
    {
        // Arrange
        var mockLogger = new Mock<ILogger<PasswordValidator>>();
        var validator = new PasswordValidator(mockLogger.Object);

        // Act
        var result = validator.ValidatePassword(null!);

        // Assert
        Assert.IsFalse(result.IsValid);
        Assert.AreEqual("Password cannot be empty.", result.ErrorMessage);
    }

    /// <summary>
    /// Tests that ValidatePassword returns false with appropriate error message for various empty or whitespace-only passwords.
    /// Input: Empty string, whitespace strings
    /// Expected: (false, "Password cannot be empty.")
    /// </summary>
    [TestMethod]
    [DataRow("")]
    [DataRow(" ")]
    [DataRow("   ")]
    [DataRow("\t")]
    [DataRow("\n")]
    [DataRow("\r\n")]
    [DataRow("  \t  \n  ")]
    public void ValidatePassword_EmptyOrWhitespacePassword_ReturnsFalseWithErrorMessage(string password)
    {
        // Arrange
        var mockLogger = new Mock<ILogger<PasswordValidator>>();
        var validator = new PasswordValidator(mockLogger.Object);

        // Act
        var result = validator.ValidatePassword(password);

        // Assert
        Assert.IsFalse(result.IsValid);
        Assert.AreEqual("Password cannot be empty.", result.ErrorMessage);
    }

    /// <summary>
    /// Tests that ValidatePassword returns false when password is shorter than 8 characters.
    /// Input: Passwords with 1-7 characters
    /// Expected: (false, "Password must be at least 8 characters long.")
    /// </summary>
    [TestMethod]
    [DataRow("A")]
    [DataRow("Ab1!")]
    [DataRow("Ab1!xyz")]
    public void ValidatePassword_PasswordTooShort_ReturnsFalseWithErrorMessage(string password)
    {
        // Arrange
        var mockLogger = new Mock<ILogger<PasswordValidator>>();
        var validator = new PasswordValidator(mockLogger.Object);

        // Act
        var result = validator.ValidatePassword(password);

        // Assert
        Assert.IsFalse(result.IsValid);
        Assert.AreEqual("Password must be at least 8 characters long.", result.ErrorMessage);
    }

    /// <summary>
    /// Tests that ValidatePassword returns false when password exceeds 128 characters.
    /// Input: Password with 129 characters
    /// Expected: (false, "Password must not exceed 128 characters.")
    /// </summary>
    [TestMethod]
    public void ValidatePassword_PasswordTooLong_ReturnsFalseWithErrorMessage()
    {
        // Arrange
        var mockLogger = new Mock<ILogger<PasswordValidator>>();
        var validator = new PasswordValidator(mockLogger.Object);
        var password = new string('A', 129);

        // Act
        var result = validator.ValidatePassword(password);

        // Assert
        Assert.IsFalse(result.IsValid);
        Assert.AreEqual("Password must not exceed 128 characters.", result.ErrorMessage);
    }

    /// <summary>
    /// Tests that ValidatePassword returns false when password exceeds maximum length significantly.
    /// Input: Password with 200 characters
    /// Expected: (false, "Password must not exceed 128 characters.")
    /// </summary>
    [TestMethod]
    public void ValidatePassword_PasswordExceedsMaximumLengthSignificantly_ReturnsFalseWithErrorMessage()
    {
        // Arrange
        var mockLogger = new Mock<ILogger<PasswordValidator>>();
        var validator = new PasswordValidator(mockLogger.Object);
        var password = new string('a', 200);

        // Act
        var result = validator.ValidatePassword(password);

        // Assert
        Assert.IsFalse(result.IsValid);
        Assert.AreEqual("Password must not exceed 128 characters.", result.ErrorMessage);
    }

    /// <summary>
    /// Tests that ValidatePassword returns false when password lacks an uppercase letter.
    /// Input: Password with length 8+, but no uppercase letters
    /// Expected: (false, "Password must contain at least one uppercase letter.")
    /// </summary>
    [TestMethod]
    public void ValidatePassword_PasswordWithoutUppercase_ReturnsFalseWithErrorMessage()
    {
        // Arrange
        var mockLogger = new Mock<ILogger<PasswordValidator>>();
        var validator = new PasswordValidator(mockLogger.Object);
        var password = "abcdefg1!";

        // Act
        var result = validator.ValidatePassword(password);

        // Assert
        Assert.IsFalse(result.IsValid);
        Assert.AreEqual("Password must contain at least one uppercase letter.", result.ErrorMessage);
    }

    /// <summary>
    /// Tests that ValidatePassword returns false when password lacks a lowercase letter.
    /// Input: Password with length 8+, uppercase, but no lowercase letters
    /// Expected: (false, "Password must contain at least one lowercase letter.")
    /// </summary>
    [TestMethod]
    public void ValidatePassword_PasswordWithoutLowercase_ReturnsFalseWithErrorMessage()
    {
        // Arrange
        var mockLogger = new Mock<ILogger<PasswordValidator>>();
        var validator = new PasswordValidator(mockLogger.Object);
        var password = "ABCDEFG1!";

        // Act
        var result = validator.ValidatePassword(password);

        // Assert
        Assert.IsFalse(result.IsValid);
        Assert.AreEqual("Password must contain at least one lowercase letter.", result.ErrorMessage);
    }

    /// <summary>
    /// Tests that ValidatePassword returns false when password lacks a digit.
    /// Input: Password with length 8+, uppercase, lowercase, but no digits
    /// Expected: (false, "Password must contain at least one digit.")
    /// </summary>
    [TestMethod]
    public void ValidatePassword_PasswordWithoutDigit_ReturnsFalseWithErrorMessage()
    {
        // Arrange
        var mockLogger = new Mock<ILogger<PasswordValidator>>();
        var validator = new PasswordValidator(mockLogger.Object);
        var password = "Abcdefg!";

        // Act
        var result = validator.ValidatePassword(password);

        // Assert
        Assert.IsFalse(result.IsValid);
        Assert.AreEqual("Password must contain at least one digit.", result.ErrorMessage);
    }

    /// <summary>
    /// Tests that ValidatePassword returns false when password lacks a special character.
    /// Input: Password with length 8+, uppercase, lowercase, digit, but no special characters
    /// Expected: (false, "Password must contain at least one special character (!@#$%^&*...).")
    /// </summary>
    [TestMethod]
    public void ValidatePassword_PasswordWithoutSpecialCharacter_ReturnsFalseWithErrorMessage()
    {
        // Arrange
        var mockLogger = new Mock<ILogger<PasswordValidator>>();
        var validator = new PasswordValidator(mockLogger.Object);
        var password = "Abcdefg1";

        // Act
        var result = validator.ValidatePassword(password);

        // Assert
        Assert.IsFalse(result.IsValid);
        Assert.AreEqual("Password must contain at least one special character (!@#$%^&*...).", result.ErrorMessage);
    }

    /// <summary>
    /// Tests that ValidatePassword returns true for a valid password meeting all requirements.
    /// Input: Valid password with 8 characters, uppercase, lowercase, digit, and special character
    /// Expected: (true, "Password is valid.")
    /// </summary>
    [TestMethod]
    public void ValidatePassword_ValidPasswordWithMinimumLength_ReturnsTrueWithSuccessMessage()
    {
        // Arrange
        var mockLogger = new Mock<ILogger<PasswordValidator>>();
        var validator = new PasswordValidator(mockLogger.Object);
        var password = "Abcdef1!";

        // Act
        var result = validator.ValidatePassword(password);

        // Assert
        Assert.IsTrue(result.IsValid);
        Assert.AreEqual("Password is valid.", result.ErrorMessage);
    }

    /// <summary>
    /// Tests that ValidatePassword returns true for a valid password with exactly 128 characters.
    /// Input: Valid password with exactly 128 characters meeting all requirements
    /// Expected: (true, "Password is valid.")
    /// </summary>
    [TestMethod]
    public void ValidatePassword_ValidPasswordWithMaximumLength_ReturnsTrueWithSuccessMessage()
    {
        // Arrange
        var mockLogger = new Mock<ILogger<PasswordValidator>>();
        var validator = new PasswordValidator(mockLogger.Object);
        var passwordBuilder = new StringBuilder();
        passwordBuilder.Append("Aa1!");
        passwordBuilder.Append(new string('b', 124));
        var password = passwordBuilder.ToString();

        // Act
        var result = validator.ValidatePassword(password);

        // Assert
        Assert.IsTrue(result.IsValid);
        Assert.AreEqual("Password is valid.", result.ErrorMessage);
        Assert.AreEqual(128, password.Length);
    }

    /// <summary>
    /// Tests that ValidatePassword accepts passwords with each individual special character.
    /// Input: Valid passwords containing different special characters
    /// Expected: (true, "Password is valid.")
    /// </summary>
    [TestMethod]
    [DataRow("Abcdef1!")]
    [DataRow("Abcdef1@")]
    [DataRow("Abcdef1#")]
    [DataRow("Abcdef1$")]
    [DataRow("Abcdef1%")]
    [DataRow("Abcdef1^")]
    [DataRow("Abcdef1&")]
    [DataRow("Abcdef1*")]
    [DataRow("Abcdef1(")]
    [DataRow("Abcdef1)")]
    [DataRow("Abcdef1_")]
    [DataRow("Abcdef1+")]
    [DataRow("Abcdef1-")]
    [DataRow("Abcdef1=")]
    [DataRow("Abcdef1[")]
    [DataRow("Abcdef1]")]
    [DataRow("Abcdef1{")]
    [DataRow("Abcdef1}")]
    [DataRow("Abcdef1;")]
    [DataRow("Abcdef1'")]
    [DataRow("Abcdef1:")]
    [DataRow("Abcdef1\"")]
    [DataRow("Abcdef1\\")]
    [DataRow("Abcdef1|")]
    [DataRow("Abcdef1,")]
    [DataRow("Abcdef1.")]
    [DataRow("Abcdef1<")]
    [DataRow("Abcdef1>")]
    [DataRow("Abcdef1/")]
    [DataRow("Abcdef1?")]
    public void ValidatePassword_ValidPasswordWithVariousSpecialCharacters_ReturnsTrueWithSuccessMessage(string password)
    {
        // Arrange
        var mockLogger = new Mock<ILogger<PasswordValidator>>();
        var validator = new PasswordValidator(mockLogger.Object);

        // Act
        var result = validator.ValidatePassword(password);

        // Assert
        Assert.IsTrue(result.IsValid);
        Assert.AreEqual("Password is valid.", result.ErrorMessage);
    }

    /// <summary>
    /// Tests that ValidatePassword returns true for a complex valid password with multiple special characters.
    /// Input: Valid password with multiple special characters and mixed case
    /// Expected: (true, "Password is valid.")
    /// </summary>
    [TestMethod]
    public void ValidatePassword_ValidPasswordWithMultipleSpecialCharacters_ReturnsTrueWithSuccessMessage()
    {
        // Arrange
        var mockLogger = new Mock<ILogger<PasswordValidator>>();
        var validator = new PasswordValidator(mockLogger.Object);
        var password = "MyP@ssw0rd!2023#";

        // Act
        var result = validator.ValidatePassword(password);

        // Assert
        Assert.IsTrue(result.IsValid);
        Assert.AreEqual("Password is valid.", result.ErrorMessage);
    }

    /// <summary>
    /// Tests that ValidatePassword logs information message when password validation is successful.
    /// Input: Valid password
    /// Expected: Logger.LogInformation is called with "Password validation successful"
    /// </summary>
    [TestMethod]
    public void ValidatePassword_ValidPassword_LogsInformationMessage()
    {
        // Arrange
        var mockLogger = new Mock<ILogger<PasswordValidator>>();
        var validator = new PasswordValidator(mockLogger.Object);
        var password = "Abcdef1!";

        // Act
        var result = validator.ValidatePassword(password);

        // Assert
        Assert.IsTrue(result.IsValid);
        mockLogger.Verify(
            x => x.Log(
                LogLevel.Information,
                It.IsAny<EventId>(),
                It.Is<It.IsAnyType>((v, t) => v.ToString() == "Password validation successful"),
                It.IsAny<Exception>(),
                It.IsAny<Func<It.IsAnyType, Exception?, string>>()),
            Times.Once);
    }

    /// <summary>
    /// Tests that ValidatePassword does not log when password validation fails.
    /// Input: Invalid password (missing special character)
    /// Expected: Logger.LogInformation is never called
    /// </summary>
    [TestMethod]
    public void ValidatePassword_InvalidPassword_DoesNotLogInformation()
    {
        // Arrange
        var mockLogger = new Mock<ILogger<PasswordValidator>>();
        var validator = new PasswordValidator(mockLogger.Object);
        var password = "Abcdefg1";

        // Act
        var result = validator.ValidatePassword(password);

        // Assert
        Assert.IsFalse(result.IsValid);
        mockLogger.Verify(
            x => x.Log(
                It.IsAny<LogLevel>(),
                It.IsAny<EventId>(),
                It.IsAny<It.IsAnyType>(),
                It.IsAny<Exception>(),
                It.IsAny<Func<It.IsAnyType, Exception?, string>>()),
            Times.Never);
    }

    /// <summary>
    /// Tests that ValidatePassword returns the first validation error encountered when multiple violations exist.
    /// Input: Password with multiple violations (too short, missing requirements)
    /// Expected: (false, "Password must be at least 8 characters long.") - first error in validation order
    /// </summary>
    [TestMethod]
    public void ValidatePassword_PasswordWithMultipleViolations_ReturnsFirstError()
    {
        // Arrange
        var mockLogger = new Mock<ILogger<PasswordValidator>>();
        var validator = new PasswordValidator(mockLogger.Object);
        var password = "abc";

        // Act
        var result = validator.ValidatePassword(password);

        // Assert
        Assert.IsFalse(result.IsValid);
        Assert.AreEqual("Password must be at least 8 characters long.", result.ErrorMessage);
    }

    /// <summary>
    /// Tests that ValidatePassword with exactly 8 characters returns appropriate error for missing requirements.
    /// Input: 8-character password missing uppercase
    /// Expected: (false, "Password must contain at least one uppercase letter.")
    /// </summary>
    [TestMethod]
    public void ValidatePassword_PasswordWithExactly8CharactersMissingUppercase_ReturnsFalseWithErrorMessage()
    {
        // Arrange
        var mockLogger = new Mock<ILogger<PasswordValidator>>();
        var validator = new PasswordValidator(mockLogger.Object);
        var password = "abcdef1!";

        // Act
        var result = validator.ValidatePassword(password);

        // Assert
        Assert.IsFalse(result.IsValid);
        Assert.AreEqual("Password must contain at least one uppercase letter.", result.ErrorMessage);
    }

    /// <summary>
    /// Tests that ValidatePassword with exactly 128 characters returns appropriate error for missing requirements.
    /// Input: 128-character password missing special character
    /// Expected: (false, "Password must contain at least one special character (!@#$%^&*...).")
    /// </summary>
    [TestMethod]
    public void ValidatePassword_PasswordWithExactly128CharactersMissingSpecialChar_ReturnsFalseWithErrorMessage()
    {
        // Arrange
        var mockLogger = new Mock<ILogger<PasswordValidator>>();
        var validator = new PasswordValidator(mockLogger.Object);
        var passwordBuilder = new StringBuilder();
        passwordBuilder.Append("Aa1");
        passwordBuilder.Append(new string('b', 125));
        var password = passwordBuilder.ToString();

        // Act
        var result = validator.ValidatePassword(password);

        // Assert
        Assert.IsFalse(result.IsValid);
        Assert.AreEqual("Password must contain at least one special character (!@#$%^&*...).", result.ErrorMessage);
        Assert.AreEqual(128, password.Length);
    }

    /// <summary>
    /// Tests that ValidatePassword rejects passwords with characters that are not in the allowed special character set.
    /// Input: Password with character not in special character regex (e.g., ~)
    /// Expected: (false, "Password must contain at least one special character (!@#$%^&*...).")
    /// </summary>
    [TestMethod]
    public void ValidatePassword_PasswordWithNonAllowedSpecialCharacter_ReturnsFalseWithErrorMessage()
    {
        // Arrange
        var mockLogger = new Mock<ILogger<PasswordValidator>>();
        var validator = new PasswordValidator(mockLogger.Object);
        var password = "Abcdefg1~";

        // Act
        var result = validator.ValidatePassword(password);

        // Assert
        Assert.IsFalse(result.IsValid);
        Assert.AreEqual("Password must contain at least one special character (!@#$%^&*...).", result.ErrorMessage);
    }

    /// <summary>
    /// Tests that ValidatePassword handles passwords with Unicode characters appropriately.
    /// Input: Password with Unicode characters (non-ASCII)
    /// Expected: Validation based on character requirements
    /// </summary>
    [TestMethod]
    public void ValidatePassword_PasswordWithUnicodeCharacters_ReturnsFalseWithErrorMessage()
    {
        // Arrange
        var mockLogger = new Mock<ILogger<PasswordValidator>>();
        var validator = new PasswordValidator(mockLogger.Object);
        var password = "Abcdef1€";

        // Act
        var result = validator.ValidatePassword(password);

        // Assert
        Assert.IsFalse(result.IsValid);
        Assert.AreEqual("Password must contain at least one special character (!@#$%^&*...).", result.ErrorMessage);
    }

    /// <summary>
    /// Tests that ValidatePassword correctly validates passwords with numbers at different positions.
    /// Input: Valid passwords with digits at start, middle, and end
    /// Expected: (true, "Password is valid.")
    /// </summary>
    [TestMethod]
    [DataRow("1Abcdef!")]
    [DataRow("Ab1cdef!")]
    [DataRow("Abcdef1!")]
    public void ValidatePassword_ValidPasswordWithDigitAtVariousPositions_ReturnsTrueWithSuccessMessage(string password)
    {
        // Arrange
        var mockLogger = new Mock<ILogger<PasswordValidator>>();
        var validator = new PasswordValidator(mockLogger.Object);

        // Act
        var result = validator.ValidatePassword(password);

        // Assert
        Assert.IsTrue(result.IsValid);
        Assert.AreEqual("Password is valid.", result.ErrorMessage);
    }

    /// <summary>
    /// Tests that ValidatePassword correctly validates passwords with multiple digits.
    /// Input: Valid password with multiple digits
    /// Expected: (true, "Password is valid.")
    /// </summary>
    [TestMethod]
    public void ValidatePassword_ValidPasswordWithMultipleDigits_ReturnsTrueWithSuccessMessage()
    {
        // Arrange
        var mockLogger = new Mock<ILogger<PasswordValidator>>();
        var validator = new PasswordValidator(mockLogger.Object);
        var password = "Abc123def!";

        // Act
        var result = validator.ValidatePassword(password);

        // Assert
        Assert.IsTrue(result.IsValid);
        Assert.AreEqual("Password is valid.", result.ErrorMessage);
    }

    /// <summary>
    /// Tests that the constructor successfully creates an instance when provided with a valid logger.
    /// </summary>
    [TestMethod]
    public void Constructor_WithValidLogger_CreatesInstance()
    {
        // Arrange
        var mockLogger = new Mock<ILogger<PasswordValidator>>();

        // Act
        var validator = new PasswordValidator(mockLogger.Object);

        // Assert
        Assert.IsNotNull(validator);
    }

    /// <summary>
    /// Tests that the constructor accepts a null logger parameter without throwing an exception.
    /// This documents the current behavior where no null validation is performed.
    /// </summary>
    [TestMethod]
    public void Constructor_WithNullLogger_DoesNotThrowException()
    {
        // Arrange
        ILogger<PasswordValidator>? logger = null;

        // Act & Assert
        var validator = new PasswordValidator(logger!);
        Assert.IsNotNull(validator);
    }
}