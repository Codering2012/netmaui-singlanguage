using System;
using System.IdentityModel.Tokens.Jwt;
using System.Linq;
using System.Security;
using System.Security.Claims;
using System.Security.Cryptography;
using System.Text;

using BCrypt.Net;
using Microsoft.Extensions.Configuration;
using Microsoft.IdentityModel.Tokens;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using Moq;
using SignLanguageApi.Data;
using SignLanguageApi.Services;

namespace SignLanguageApi.Services.UnitTests
{
    /// <summary>
    /// Unit tests for the <see cref="AuthService"/> class.
    /// </summary>
    [TestClass]
    public class AuthServiceTests
    {
        /// <summary>
        /// Tests that GenerateRefreshToken returns a non-null and non-empty string.
        /// This verifies the basic functionality of the method.
        /// Expected result: The returned string should not be null or empty.
        /// </summary>
        [TestMethod]
        public void GenerateRefreshToken_WhenCalled_ReturnsNonNullNonEmptyString()
        {
            // Arrange
            var mockConfiguration = new Mock<IConfiguration>();
            var authService = new AuthService(mockConfiguration.Object);

            // Act
            string result = authService.GenerateRefreshToken();

            // Assert
            Assert.IsNotNull(result);
            Assert.IsFalse(string.IsNullOrEmpty(result));
        }

        /// <summary>
        /// Tests that GenerateRefreshToken returns a Base64 string of expected length.
        /// Since the method generates 32 random bytes and encodes them as Base64,
        /// the result should always be 44 characters long.
        /// Expected result: The returned string should be exactly 44 characters.
        /// </summary>
        [TestMethod]
        public void GenerateRefreshToken_WhenCalled_ReturnsStringOfExpectedLength()
        {
            // Arrange
            var mockConfiguration = new Mock<IConfiguration>();
            var authService = new AuthService(mockConfiguration.Object);

            // Act
            string result = authService.GenerateRefreshToken();

            // Assert
            Assert.AreEqual(44, result.Length);
        }

        /// <summary>
        /// Tests that GenerateRefreshToken returns a valid Base64 string.
        /// This verifies that the string can be successfully converted back to bytes.
        /// Expected result: The string should be valid Base64 and decode without exceptions.
        /// </summary>
        [TestMethod]
        public void GenerateRefreshToken_WhenCalled_ReturnsValidBase64String()
        {
            // Arrange
            var mockConfiguration = new Mock<IConfiguration>();
            var authService = new AuthService(mockConfiguration.Object);

            // Act
            string result = authService.GenerateRefreshToken();

            // Assert
            byte[] decodedBytes = Convert.FromBase64String(result);
            Assert.IsNotNull(decodedBytes);
            Assert.AreEqual(32, decodedBytes.Length);
        }

        /// <summary>
        /// Tests that GenerateRefreshToken produces unique tokens on multiple invocations.
        /// Since the method uses cryptographic random number generation, each call should
        /// produce a different token with extremely high probability.
        /// Expected result: Multiple calls should return different strings.
        /// </summary>
        [TestMethod]
        public void GenerateRefreshToken_WhenCalledMultipleTimes_ProducesDifferentTokens()
        {
            // Arrange
            var mockConfiguration = new Mock<IConfiguration>();
            var authService = new AuthService(mockConfiguration.Object);

            // Act
            string token1 = authService.GenerateRefreshToken();
            string token2 = authService.GenerateRefreshToken();
            string token3 = authService.GenerateRefreshToken();

            // Assert
            Assert.AreNotEqual(token1, token2);
            Assert.AreNotEqual(token2, token3);
            Assert.AreNotEqual(token1, token3);
        }

        /// <summary>
        /// Tests that GenerateRefreshToken does not throw any exceptions.
        /// This verifies that the method completes successfully under normal conditions.
        /// Expected result: No exception should be thrown.
        /// </summary>
        [TestMethod]
        public void GenerateRefreshToken_WhenCalled_DoesNotThrowException()
        {
            // Arrange
            var mockConfiguration = new Mock<IConfiguration>();
            var authService = new AuthService(mockConfiguration.Object);

            // Act & Assert
            try
            {
                string result = authService.GenerateRefreshToken();
                Assert.IsNotNull(result);
            }
            catch (Exception ex)
            {
                Assert.Fail($"Expected no exception, but got: {ex.Message}");
            }
        }

        /// <summary>
        /// Tests that GenerateJwtToken successfully generates a valid JWT token with valid configuration and user.
        /// Verifies that the token contains the expected claims (NameIdentifier and Email).
        /// </summary>
        [TestMethod]
        public void GenerateJwtToken_ValidConfigurationAndUser_GeneratesValidToken()
        {
            // Arrange
            Mock<IConfiguration> configurationMock = new Mock<IConfiguration>();
            string secretKey = "this-is-a-very-secure-secret-key-with-at-least-32-characters";
            string issuer = "test-issuer";
            string audience = "test-audience";
            configurationMock.Setup(c => c["Jwt:SecretKey"]).Returns(secretKey);
            configurationMock.Setup(c => c["Jwt:Issuer"]).Returns(issuer);
            configurationMock.Setup(c => c["Jwt:Audience"]).Returns(audience);
            AuthService authService = new AuthService(configurationMock.Object);
            User user = new User { Id = "user123", Email = "test@example.com" };

            // Act
            string token = authService.GenerateJwtToken(user);

            // Assert
            Assert.IsNotNull(token);
            Assert.IsFalse(string.IsNullOrEmpty(token));

            JwtSecurityTokenHandler tokenHandler = new JwtSecurityTokenHandler();
            JwtSecurityToken jwtToken = tokenHandler.ReadJwtToken(token);

            Assert.AreEqual(issuer, jwtToken.Issuer);
            Assert.IsTrue(jwtToken.Audiences.Contains(audience));

            Claim? nameIdentifierClaim = jwtToken.Claims.FirstOrDefault(c => c.Type == ClaimTypes.NameIdentifier);
            Assert.IsNotNull(nameIdentifierClaim);
            Assert.AreEqual("user123", nameIdentifierClaim.Value);

            Claim? emailClaim = jwtToken.Claims.FirstOrDefault(c => c.Type == ClaimTypes.Email);
            Assert.IsNotNull(emailClaim);
            Assert.AreEqual("test@example.com", emailClaim.Value);

            Assert.IsNotNull(jwtToken.ValidTo);
            TimeSpan expirationDifference = jwtToken.ValidTo - DateTime.UtcNow;
            Assert.IsTrue(expirationDifference.TotalMinutes >= 59 && expirationDifference.TotalMinutes <= 61);
        }

        /// <summary>
        /// Tests that GenerateJwtToken successfully generates a token when the user has an empty Id.
        /// Verifies that the empty Id is correctly embedded in the NameIdentifier claim.
        /// </summary>
        [TestMethod]
        public void GenerateJwtToken_UserWithEmptyId_GeneratesTokenWithEmptyClaim()
        {
            // Arrange
            Mock<IConfiguration> configurationMock = new Mock<IConfiguration>();
            configurationMock.Setup(c => c["Jwt:SecretKey"]).Returns("this-is-a-very-secure-secret-key-with-at-least-32-characters");
            configurationMock.Setup(c => c["Jwt:Issuer"]).Returns("test-issuer");
            configurationMock.Setup(c => c["Jwt:Audience"]).Returns("test-audience");
            AuthService authService = new AuthService(configurationMock.Object);
            User user = new User { Id = string.Empty, Email = "test@example.com" };

            // Act
            string token = authService.GenerateJwtToken(user);

            // Assert
            Assert.IsNotNull(token);
            JwtSecurityTokenHandler tokenHandler = new JwtSecurityTokenHandler();
            JwtSecurityToken jwtToken = tokenHandler.ReadJwtToken(token);
            Claim? nameIdentifierClaim = jwtToken.Claims.FirstOrDefault(c => c.Type == ClaimTypes.NameIdentifier);
            Assert.IsNotNull(nameIdentifierClaim);
            Assert.AreEqual(string.Empty, nameIdentifierClaim.Value);
        }

        /// <summary>
        /// Tests that GenerateJwtToken successfully generates a token when the user has an empty Email.
        /// Verifies that the empty Email is correctly embedded in the Email claim.
        /// </summary>
        [TestMethod]
        public void GenerateJwtToken_UserWithEmptyEmail_GeneratesTokenWithEmptyClaim()
        {
            // Arrange
            Mock<IConfiguration> configurationMock = new Mock<IConfiguration>();
            configurationMock.Setup(c => c["Jwt:SecretKey"]).Returns("this-is-a-very-secure-secret-key-with-at-least-32-characters");
            configurationMock.Setup(c => c["Jwt:Issuer"]).Returns("test-issuer");
            configurationMock.Setup(c => c["Jwt:Audience"]).Returns("test-audience");
            AuthService authService = new AuthService(configurationMock.Object);
            User user = new User { Id = "user123", Email = string.Empty };

            // Act
            string token = authService.GenerateJwtToken(user);

            // Assert
            Assert.IsNotNull(token);
            JwtSecurityTokenHandler tokenHandler = new JwtSecurityTokenHandler();
            JwtSecurityToken jwtToken = tokenHandler.ReadJwtToken(token);
            Claim? emailClaim = jwtToken.Claims.FirstOrDefault(c => c.Type == ClaimTypes.Email);
            Assert.IsNotNull(emailClaim);
            Assert.AreEqual(string.Empty, emailClaim.Value);
        }

        /// <summary>
        /// Tests that GenerateJwtToken successfully generates a token when the user has special characters in Id and Email.
        /// Verifies that special characters are correctly preserved in the token claims.
        /// </summary>
        [TestMethod]
        public void GenerateJwtToken_UserWithSpecialCharacters_GeneratesValidToken()
        {
            // Arrange
            Mock<IConfiguration> configurationMock = new Mock<IConfiguration>();
            configurationMock.Setup(c => c["Jwt:SecretKey"]).Returns("this-is-a-very-secure-secret-key-with-at-least-32-characters");
            configurationMock.Setup(c => c["Jwt:Issuer"]).Returns("test-issuer");
            configurationMock.Setup(c => c["Jwt:Audience"]).Returns("test-audience");
            AuthService authService = new AuthService(configurationMock.Object);
            User user = new User { Id = "user-123!@#$%", Email = "test+tag@example.com" };

            // Act
            string token = authService.GenerateJwtToken(user);

            // Assert
            Assert.IsNotNull(token);
            JwtSecurityTokenHandler tokenHandler = new JwtSecurityTokenHandler();
            JwtSecurityToken jwtToken = tokenHandler.ReadJwtToken(token);
            Claim? nameIdentifierClaim = jwtToken.Claims.FirstOrDefault(c => c.Type == ClaimTypes.NameIdentifier);
            Assert.IsNotNull(nameIdentifierClaim);
            Assert.AreEqual("user-123!@#$%", nameIdentifierClaim.Value);
            Claim? emailClaim = jwtToken.Claims.FirstOrDefault(c => c.Type == ClaimTypes.Email);
            Assert.IsNotNull(emailClaim);
            Assert.AreEqual("test+tag@example.com", emailClaim.Value);
        }

        /// <summary>
        /// Tests that GenerateJwtToken generates different tokens for different users.
        /// Verifies that each user's unique information is embedded correctly.
        /// </summary>
        [TestMethod]
        public void GenerateJwtToken_DifferentUsers_GeneratesDifferentTokens()
        {
            // Arrange
            Mock<IConfiguration> configurationMock = new Mock<IConfiguration>();
            configurationMock.Setup(c => c["Jwt:SecretKey"]).Returns("this-is-a-very-secure-secret-key-with-at-least-32-characters");
            configurationMock.Setup(c => c["Jwt:Issuer"]).Returns("test-issuer");
            configurationMock.Setup(c => c["Jwt:Audience"]).Returns("test-audience");
            AuthService authService = new AuthService(configurationMock.Object);
            User user1 = new User { Id = "user1", Email = "user1@example.com" };
            User user2 = new User { Id = "user2", Email = "user2@example.com" };

            // Act
            string token1 = authService.GenerateJwtToken(user1);
            string token2 = authService.GenerateJwtToken(user2);

            // Assert
            Assert.AreNotEqual(token1, token2);
            JwtSecurityTokenHandler tokenHandler = new JwtSecurityTokenHandler();
            JwtSecurityToken jwtToken1 = tokenHandler.ReadJwtToken(token1);
            JwtSecurityToken jwtToken2 = tokenHandler.ReadJwtToken(token2);
            Assert.AreEqual("user1", jwtToken1.Claims.First(c => c.Type == ClaimTypes.NameIdentifier).Value);
            Assert.AreEqual("user2", jwtToken2.Claims.First(c => c.Type == ClaimTypes.NameIdentifier).Value);
        }

        /// <summary>
        /// Tests that VerifyPassword returns true when the password matches the hash.
        /// </summary>
        [TestMethod]
        public void VerifyPassword_MatchingPasswordAndHash_ReturnsTrue()
        {
            // Arrange
            var mockConfiguration = new Mock<IConfiguration>();
            var authService = new AuthService(mockConfiguration.Object);
            var password = "TestPassword123!";
            var hash = BCrypt.Net.BCrypt.HashPassword(password);

            // Act
            var result = authService.VerifyPassword(password, hash);

            // Assert
            Assert.IsTrue(result);
        }

        /// <summary>
        /// Tests that VerifyPassword returns false when the password does not match the hash.
        /// </summary>
        [TestMethod]
        public void VerifyPassword_NonMatchingPasswordAndHash_ReturnsFalse()
        {
            // Arrange
            var mockConfiguration = new Mock<IConfiguration>();
            var authService = new AuthService(mockConfiguration.Object);
            var correctPassword = "CorrectPassword123!";
            var wrongPassword = "WrongPassword456!";
            var hash = BCrypt.Net.BCrypt.HashPassword(correctPassword);

            // Act
            var result = authService.VerifyPassword(wrongPassword, hash);

            // Assert
            Assert.IsFalse(result);
        }

        /// <summary>
        /// Tests that VerifyPassword returns false when an empty password is verified against a valid hash.
        /// </summary>
        [TestMethod]
        public void VerifyPassword_EmptyPassword_ReturnsFalse()
        {
            // Arrange
            var mockConfiguration = new Mock<IConfiguration>();
            var authService = new AuthService(mockConfiguration.Object);
            var hash = BCrypt.Net.BCrypt.HashPassword("TestPassword");

            // Act
            var result = authService.VerifyPassword(string.Empty, hash);

            // Assert
            Assert.IsFalse(result);
        }

        /// <summary>
        /// Tests that VerifyPassword correctly verifies an empty password when the hash was created from an empty password.
        /// </summary>
        [TestMethod]
        public void VerifyPassword_EmptyPasswordWithMatchingHash_ReturnsTrue()
        {
            // Arrange
            var mockConfiguration = new Mock<IConfiguration>();
            var authService = new AuthService(mockConfiguration.Object);
            var password = string.Empty;
            var hash = BCrypt.Net.BCrypt.HashPassword(password);

            // Act
            var result = authService.VerifyPassword(password, hash);

            // Assert
            Assert.IsTrue(result);
        }

        /// <summary>
        /// Tests that VerifyPassword correctly verifies passwords with special characters.
        /// </summary>
        [TestMethod]
        [DataRow("P@ssw0rd!")]
        [DataRow("Test#123$%^")]
        [DataRow("パスワード")]
        [DataRow("Pass\nWord")]
        [DataRow("Pass\tWord")]
        public void VerifyPassword_PasswordWithSpecialCharacters_ReturnsTrue(string password)
        {
            // Arrange
            var mockConfiguration = new Mock<IConfiguration>();
            var authService = new AuthService(mockConfiguration.Object);
            var hash = BCrypt.Net.BCrypt.HashPassword(password);

            // Act
            var result = authService.VerifyPassword(password, hash);

            // Assert
            Assert.IsTrue(result);
        }

        /// <summary>
        /// Tests that VerifyPassword is case-sensitive and returns false when password case differs.
        /// </summary>
        [TestMethod]
        public void VerifyPassword_DifferentCase_ReturnsFalse()
        {
            // Arrange
            var mockConfiguration = new Mock<IConfiguration>();
            var authService = new AuthService(mockConfiguration.Object);
            var password = "TestPassword";
            var hash = BCrypt.Net.BCrypt.HashPassword(password);

            // Act
            var result = authService.VerifyPassword("TESTPASSWORD", hash);

            // Assert
            Assert.IsFalse(result);
        }

        /// <summary>
        /// Tests that VerifyPassword correctly handles very long passwords.
        /// </summary>
        [TestMethod]
        public void VerifyPassword_VeryLongPassword_ReturnsTrue()
        {
            // Arrange
            var mockConfiguration = new Mock<IConfiguration>();
            var authService = new AuthService(mockConfiguration.Object);
            var password = new string('a', 1000);
            var hash = BCrypt.Net.BCrypt.HashPassword(password);

            // Act
            var result = authService.VerifyPassword(password, hash);

            // Assert
            Assert.IsTrue(result);
        }

        /// <summary>
        /// Tests that VerifyPassword returns false when whitespace-only password is verified against a different hash.
        /// </summary>
        [TestMethod]
        public void VerifyPassword_WhitespacePassword_ReturnsFalse()
        {
            // Arrange
            var mockConfiguration = new Mock<IConfiguration>();
            var authService = new AuthService(mockConfiguration.Object);
            var hash = BCrypt.Net.BCrypt.HashPassword("TestPassword");

            // Act
            var result = authService.VerifyPassword("   ", hash);

            // Assert
            Assert.IsFalse(result);
        }

        /// <summary>
        /// Tests that VerifyPassword correctly verifies whitespace-only password when hash matches.
        /// </summary>
        [TestMethod]
        public void VerifyPassword_WhitespacePasswordWithMatchingHash_ReturnsTrue()
        {
            // Arrange
            var mockConfiguration = new Mock<IConfiguration>();
            var authService = new AuthService(mockConfiguration.Object);
            var password = "   ";
            var hash = BCrypt.Net.BCrypt.HashPassword(password);

            // Act
            var result = authService.VerifyPassword(password, hash);

            // Assert
            Assert.IsTrue(result);
        }

        /// <summary>
        /// Tests that HashPassword returns a non-null and non-empty hash for valid password inputs.
        /// </summary>
        /// <param name="password">The password to hash.</param>
        [TestMethod]
        [DataRow("Password123")]
        [DataRow("")]
        [DataRow("   ")]
        [DataRow("p")]
        [DataRow("VeryLongPasswordWithLotsOfCharacters1234567890!@#$%^&*()_+-=[]{}|;:',.<>?/~`")]
        [DataRow("パスワード")]
        [DataRow("password\nwith\nnewlines")]
        [DataRow("password\twith\ttabs")]
        [DataRow("!@#$%^&*()_+-=[]{}|;:',.<>?/~`")]
        public void HashPassword_ValidPasswordInputs_ReturnsNonEmptyHash(string password)
        {
            // Arrange
            var mockConfiguration = new Mock<IConfiguration>();
            var authService = new AuthService(mockConfiguration.Object);

            // Act
            var result = authService.HashPassword(password);

            // Assert
            Assert.IsNotNull(result);
            Assert.IsFalse(string.IsNullOrEmpty(result));
        }

        /// <summary>
        /// Tests that HashPassword with an extremely long password returns a valid hash.
        /// Verifies the method can handle passwords with thousands of characters.
        /// </summary>
        [TestMethod]
        public void HashPassword_ExtremelyLongPassword_ReturnsNonEmptyHash()
        {
            // Arrange
            var mockConfiguration = new Mock<IConfiguration>();
            var authService = new AuthService(mockConfiguration.Object);
            var longPassword = new string('a', 10000);

            // Act
            var result = authService.HashPassword(longPassword);

            // Assert
            Assert.IsNotNull(result);
            Assert.IsFalse(string.IsNullOrEmpty(result));
        }

        /// <summary>
        /// Tests that HashPassword generates different hashes for the same password on multiple calls.
        /// This verifies that BCrypt properly uses random salts.
        /// </summary>
        [TestMethod]
        public void HashPassword_SamePasswordMultipleCalls_GeneratesDifferentHashes()
        {
            // Arrange
            var mockConfiguration = new Mock<IConfiguration>();
            var authService = new AuthService(mockConfiguration.Object);
            var password = "TestPassword123";

            // Act
            var hash1 = authService.HashPassword(password);
            var hash2 = authService.HashPassword(password);

            // Assert
            Assert.IsNotNull(hash1);
            Assert.IsNotNull(hash2);
            Assert.AreNotEqual(hash1, hash2, "BCrypt should generate different hashes for the same password due to random salts.");
        }

        /// <summary>
        /// Tests that HashPassword with special control characters returns a valid hash.
        /// Verifies handling of edge case string inputs with control characters.
        /// </summary>
        [TestMethod]
        public void HashPassword_PasswordWithControlCharacters_ReturnsNonEmptyHash()
        {
            // Arrange
            var mockConfiguration = new Mock<IConfiguration>();
            var authService = new AuthService(mockConfiguration.Object);
            var passwordWithControlChars = "pass\0word\u0001\u001F";

            // Act
            var result = authService.HashPassword(passwordWithControlChars);

            // Assert
            Assert.IsNotNull(result);
            Assert.IsFalse(string.IsNullOrEmpty(result));
        }

        /// <summary>
        /// Tests that HashPassword with maximum length string at int.MaxValue boundary behavior.
        /// Note: This test uses a very large string to test boundary conditions.
        /// </summary>
        [TestMethod]
        public void HashPassword_VeryLargePassword_ReturnsNonEmptyHash()
        {
            // Arrange
            var mockConfiguration = new Mock<IConfiguration>();
            var authService = new AuthService(mockConfiguration.Object);
            var largePassword = new string('x', 100000);

            // Act
            var result = authService.HashPassword(largePassword);

            // Assert
            Assert.IsNotNull(result);
            Assert.IsFalse(string.IsNullOrEmpty(result));
        }

        /// <summary>
        /// Tests that the AuthService constructor successfully initializes when provided with a valid IConfiguration instance.
        /// Input: A valid mocked IConfiguration.
        /// Expected: Constructor completes without throwing an exception and creates a valid instance.
        /// </summary>
        [TestMethod]
        public void AuthService_WithValidConfiguration_InitializesSuccessfully()
        {
            // Arrange
            var mockConfiguration = new Mock<IConfiguration>();

            // Act
            var authService = new AuthService(mockConfiguration.Object);

            // Assert
            Assert.IsNotNull(authService);
        }

        /// <summary>
        /// Tests that the AuthService constructor accepts null configuration parameter.
        /// Since the constructor does not perform null validation, it should complete without throwing.
        /// Input: null IConfiguration.
        /// Expected: Constructor completes without throwing an exception.
        /// </summary>
        [TestMethod]
        public void AuthService_WithNullConfiguration_InitializesWithoutException()
        {
            // Arrange
            IConfiguration? nullConfiguration = null;

            // Act
            var authService = new AuthService(nullConfiguration!);

            // Assert
            Assert.IsNotNull(authService);
        }

        /// <summary>
        /// Tests that VerifyPassword correctly verifies a password with null characters.
        /// Input: Password containing null character (\0) with matching hash.
        /// Expected: Returns true when password matches the hash.
        /// </summary>
        [TestMethod]
        public void VerifyPassword_PasswordWithNullCharacter_ReturnsTrue()
        {
            // Arrange
            var mockConfiguration = new Mock<IConfiguration>();
            var authService = new AuthService(mockConfiguration.Object);
            var password = "Test\0Password";
            var hash = BCrypt.Net.BCrypt.HashPassword(password);

            // Act
            var result = authService.VerifyPassword(password, hash);

            // Assert
            Assert.IsTrue(result);
        }

        /// <summary>
        /// Tests that VerifyPassword correctly handles passwords with various Unicode characters.
        /// Input: Passwords with emojis, different scripts, and special Unicode characters.
        /// Expected: Returns true when password matches the hash.
        /// </summary>
        [TestMethod]
        [DataRow("😀🔒🎉")]
        [DataRow("Пароль")]
        [DataRow("密码")]
        [DataRow("كلمة المرور")]
        [DataRow("𝕡𝕒𝕤𝕤𝕨𝕠𝕣𝕕")]
        public void VerifyPassword_PasswordWithUnicodeCharacters_ReturnsTrue(string password)
        {
            // Arrange
            var mockConfiguration = new Mock<IConfiguration>();
            var authService = new AuthService(mockConfiguration.Object);
            var hash = BCrypt.Net.BCrypt.HashPassword(password);

            // Act
            var result = authService.VerifyPassword(password, hash);

            // Assert
            Assert.IsTrue(result);
        }

        /// <summary>
        /// Tests that VerifyPassword returns false when password has trailing whitespace that doesn't match the hash.
        /// Input: Password with trailing spaces vs password without trailing spaces.
        /// Expected: Returns false as passwords are different.
        /// </summary>
        [TestMethod]
        public void VerifyPassword_PasswordWithTrailingWhitespace_ReturnsFalse()
        {
            // Arrange
            var mockConfiguration = new Mock<IConfiguration>();
            var authService = new AuthService(mockConfiguration.Object);
            var password = "TestPassword";
            var passwordWithSpaces = "TestPassword   ";
            var hash = BCrypt.Net.BCrypt.HashPassword(password);

            // Act
            var result = authService.VerifyPassword(passwordWithSpaces, hash);

            // Assert
            Assert.IsFalse(result);
        }

        /// <summary>
        /// Tests that VerifyPassword returns false when password has leading whitespace that doesn't match the hash.
        /// Input: Password with leading spaces vs password without leading spaces.
        /// Expected: Returns false as passwords are different.
        /// </summary>
        [TestMethod]
        public void VerifyPassword_PasswordWithLeadingWhitespace_ReturnsFalse()
        {
            // Arrange
            var mockConfiguration = new Mock<IConfiguration>();
            var authService = new AuthService(mockConfiguration.Object);
            var password = "TestPassword";
            var passwordWithSpaces = "   TestPassword";
            var hash = BCrypt.Net.BCrypt.HashPassword(password);

            // Act
            var result = authService.VerifyPassword(passwordWithSpaces, hash);

            // Assert
            Assert.IsFalse(result);
        }

        /// <summary>
        /// Tests that VerifyPassword correctly verifies password containing only special characters.
        /// Input: Password made entirely of special characters with matching hash.
        /// Expected: Returns true when password matches the hash.
        /// </summary>
        [TestMethod]
        public void VerifyPassword_PasswordWithOnlySpecialCharacters_ReturnsTrue()
        {
            // Arrange
            var mockConfiguration = new Mock<IConfiguration>();
            var authService = new AuthService(mockConfiguration.Object);
            var password = "!@#$%^&*()_+-=[]{}|;:',.<>?/~`";
            var hash = BCrypt.Net.BCrypt.HashPassword(password);

            // Act
            var result = authService.VerifyPassword(password, hash);

            // Assert
            Assert.IsTrue(result);
        }

        /// <summary>
        /// Tests that VerifyPassword correctly handles single character passwords.
        /// Input: Single character password with matching hash.
        /// Expected: Returns true when password matches the hash.
        /// </summary>
        [TestMethod]
        [DataRow("a")]
        [DataRow("Z")]
        [DataRow("1")]
        [DataRow("!")]
        [DataRow(" ")]
        public void VerifyPassword_SingleCharacterPassword_ReturnsTrue(string password)
        {
            // Arrange
            var mockConfiguration = new Mock<IConfiguration>();
            var authService = new AuthService(mockConfiguration.Object);
            var hash = BCrypt.Net.BCrypt.HashPassword(password);

            // Act
            var result = authService.VerifyPassword(password, hash);

            // Assert
            Assert.IsTrue(result);
        }

        /// <summary>
        /// Tests that VerifyPassword returns false when verifying against a hash created with a different work factor.
        /// Input: Password verified against hash of a different password (to test cross-compatibility).
        /// Expected: Returns false when passwords don't match regardless of work factor.
        /// </summary>
        [TestMethod]
        public void VerifyPassword_DifferentPasswordDifferentWorkFactor_ReturnsFalse()
        {
            // Arrange
            var mockConfiguration = new Mock<IConfiguration>();
            var authService = new AuthService(mockConfiguration.Object);
            var password1 = "Password1";
            var password2 = "Password2";
            var hash = BCrypt.Net.BCrypt.HashPassword(password1, 12);

            // Act
            var result = authService.VerifyPassword(password2, hash);

            // Assert
            Assert.IsFalse(result);
        }

        /// <summary>
        /// Tests that VerifyPassword correctly verifies password when hash was created with different work factors.
        /// Input: Same password verified against hashes created with different work factors (4, 10, 12).
        /// Expected: Returns true as BCrypt should handle different work factors transparently.
        /// </summary>
        [TestMethod]
        [DataRow(4)]
        [DataRow(10)]
        [DataRow(12)]
        public void VerifyPassword_SamePasswordDifferentWorkFactor_ReturnsTrue(int workFactor)
        {
            // Arrange
            var mockConfiguration = new Mock<IConfiguration>();
            var authService = new AuthService(mockConfiguration.Object);
            var password = "TestPassword123";
            var hash = BCrypt.Net.BCrypt.HashPassword(password, workFactor);

            // Act
            var result = authService.VerifyPassword(password, hash);

            // Assert
            Assert.IsTrue(result);
        }

        /// <summary>
        /// Tests that GenerateJwtToken successfully generates a token when user has whitespace-only Id.
        /// Input: User with whitespace-only Id.
        /// Expected: Valid token with whitespace Id in the NameIdentifier claim.
        /// </summary>
        [TestMethod]
        public void GenerateJwtToken_UserWithWhitespaceId_GeneratesValidToken()
        {
            // Arrange
            Mock<IConfiguration> configurationMock = new Mock<IConfiguration>();
            configurationMock.Setup(c => c["Jwt:SecretKey"]).Returns("this-is-a-very-secure-secret-key-with-at-least-32-characters");
            configurationMock.Setup(c => c["Jwt:Issuer"]).Returns("test-issuer");
            configurationMock.Setup(c => c["Jwt:Audience"]).Returns("test-audience");
            AuthService authService = new AuthService(configurationMock.Object);
            User user = new User { Id = "   ", Email = "test@example.com" };

            // Act
            string token = authService.GenerateJwtToken(user);

            // Assert
            Assert.IsNotNull(token);
            JwtSecurityTokenHandler tokenHandler = new JwtSecurityTokenHandler();
            JwtSecurityToken jwtToken = tokenHandler.ReadJwtToken(token);
            Claim? nameIdentifierClaim = jwtToken.Claims.FirstOrDefault(c => c.Type == ClaimTypes.NameIdentifier);
            Assert.IsNotNull(nameIdentifierClaim);
            Assert.AreEqual("   ", nameIdentifierClaim.Value);
        }

        /// <summary>
        /// Tests that GenerateJwtToken successfully generates a token when user has whitespace-only Email.
        /// Input: User with whitespace-only Email.
        /// Expected: Valid token with whitespace Email in the Email claim.
        /// </summary>
        [TestMethod]
        public void GenerateJwtToken_UserWithWhitespaceEmail_GeneratesValidToken()
        {
            // Arrange
            Mock<IConfiguration> configurationMock = new Mock<IConfiguration>();
            configurationMock.Setup(c => c["Jwt:SecretKey"]).Returns("this-is-a-very-secure-secret-key-with-at-least-32-characters");
            configurationMock.Setup(c => c["Jwt:Issuer"]).Returns("test-issuer");
            configurationMock.Setup(c => c["Jwt:Audience"]).Returns("test-audience");
            AuthService authService = new AuthService(configurationMock.Object);
            User user = new User { Id = "user123", Email = "   " };

            // Act
            string token = authService.GenerateJwtToken(user);

            // Assert
            Assert.IsNotNull(token);
            JwtSecurityTokenHandler tokenHandler = new JwtSecurityTokenHandler();
            JwtSecurityToken jwtToken = tokenHandler.ReadJwtToken(token);
            Claim? emailClaim = jwtToken.Claims.FirstOrDefault(c => c.Type == ClaimTypes.Email);
            Assert.IsNotNull(emailClaim);
            Assert.AreEqual("   ", emailClaim.Value);
        }

        /// <summary>
        /// Tests that GenerateJwtToken successfully generates a token when user has very long Id.
        /// Input: User with Id of 10,000 characters.
        /// Expected: Valid token with long Id correctly embedded in the NameIdentifier claim.
        /// </summary>
        [TestMethod]
        public void GenerateJwtToken_UserWithVeryLongId_GeneratesValidToken()
        {
            // Arrange
            Mock<IConfiguration> configurationMock = new Mock<IConfiguration>();
            configurationMock.Setup(c => c["Jwt:SecretKey"]).Returns("this-is-a-very-secure-secret-key-with-at-least-32-characters");
            configurationMock.Setup(c => c["Jwt:Issuer"]).Returns("test-issuer");
            configurationMock.Setup(c => c["Jwt:Audience"]).Returns("test-audience");
            AuthService authService = new AuthService(configurationMock.Object);
            string veryLongId = new string('a', 10000);
            User user = new User { Id = veryLongId, Email = "test@example.com" };

            // Act
            string token = authService.GenerateJwtToken(user);

            // Assert
            Assert.IsNotNull(token);
            JwtSecurityTokenHandler tokenHandler = new JwtSecurityTokenHandler();
            JwtSecurityToken jwtToken = tokenHandler.ReadJwtToken(token);
            Claim? nameIdentifierClaim = jwtToken.Claims.FirstOrDefault(c => c.Type == ClaimTypes.NameIdentifier);
            Assert.IsNotNull(nameIdentifierClaim);
            Assert.AreEqual(veryLongId, nameIdentifierClaim.Value);
        }

        /// <summary>
        /// Tests that GenerateJwtToken successfully generates a token when user has very long Email.
        /// Input: User with Email of 10,000 characters.
        /// Expected: Valid token with long Email correctly embedded in the Email claim.
        /// </summary>
        [TestMethod]
        public void GenerateJwtToken_UserWithVeryLongEmail_GeneratesValidToken()
        {
            // Arrange
            Mock<IConfiguration> configurationMock = new Mock<IConfiguration>();
            configurationMock.Setup(c => c["Jwt:SecretKey"]).Returns("this-is-a-very-secure-secret-key-with-at-least-32-characters");
            configurationMock.Setup(c => c["Jwt:Issuer"]).Returns("test-issuer");
            configurationMock.Setup(c => c["Jwt:Audience"]).Returns("test-audience");
            AuthService authService = new AuthService(configurationMock.Object);
            string veryLongEmail = new string('b', 10000) + "@example.com";
            User user = new User { Id = "user123", Email = veryLongEmail };

            // Act
            string token = authService.GenerateJwtToken(user);

            // Assert
            Assert.IsNotNull(token);
            JwtSecurityTokenHandler tokenHandler = new JwtSecurityTokenHandler();
            JwtSecurityToken jwtToken = tokenHandler.ReadJwtToken(token);
            Claim? emailClaim = jwtToken.Claims.FirstOrDefault(c => c.Type == ClaimTypes.Email);
            Assert.IsNotNull(emailClaim);
            Assert.AreEqual(veryLongEmail, emailClaim.Value);
        }

        /// <summary>
        /// Tests that GenerateJwtToken successfully generates a token when user has control characters in Id and Email.
        /// Input: User with control characters (newlines, tabs, etc.) in Id and Email.
        /// Expected: Valid token with control characters correctly embedded in claims.
        /// </summary>
        [TestMethod]
        [DataRow("user\n123", "test\n@example.com")]
        [DataRow("user\t123", "test\t@example.com")]
        [DataRow("user\r\n123", "test\r\n@example.com")]
        [DataRow("user\0123", "test\0@example.com")]
        public void GenerateJwtToken_UserWithControlCharacters_GeneratesValidToken(string userId, string userEmail)
        {
            // Arrange
            Mock<IConfiguration> configurationMock = new Mock<IConfiguration>();
            configurationMock.Setup(c => c["Jwt:SecretKey"]).Returns("this-is-a-very-secure-secret-key-with-at-least-32-characters");
            configurationMock.Setup(c => c["Jwt:Issuer"]).Returns("test-issuer");
            configurationMock.Setup(c => c["Jwt:Audience"]).Returns("test-audience");
            AuthService authService = new AuthService(configurationMock.Object);
            User user = new User { Id = userId, Email = userEmail };

            // Act
            string token = authService.GenerateJwtToken(user);

            // Assert
            Assert.IsNotNull(token);
            JwtSecurityTokenHandler tokenHandler = new JwtSecurityTokenHandler();
            JwtSecurityToken jwtToken = tokenHandler.ReadJwtToken(token);
            Claim? nameIdentifierClaim = jwtToken.Claims.FirstOrDefault(c => c.Type == ClaimTypes.NameIdentifier);
            Assert.IsNotNull(nameIdentifierClaim);
            Assert.AreEqual(userId, nameIdentifierClaim.Value);
            Claim? emailClaim = jwtToken.Claims.FirstOrDefault(c => c.Type == ClaimTypes.Email);
            Assert.IsNotNull(emailClaim);
            Assert.AreEqual(userEmail, emailClaim.Value);
        }

        /// <summary>
        /// Tests that GenerateJwtToken successfully generates a token with whitespace-only SecretKey.
        /// Input: Configuration with whitespace-only SecretKey (not caught by string.IsNullOrEmpty).
        /// Expected: Token is generated successfully (may fail in token validation but method doesn't validate).
        /// </summary>
        [TestMethod]
        public void GenerateJwtToken_WhitespaceSecretKey_GeneratesToken()
        {
            // Arrange
            Mock<IConfiguration> configurationMock = new Mock<IConfiguration>();
            configurationMock.Setup(c => c["Jwt:SecretKey"]).Returns("                                                  ");
            configurationMock.Setup(c => c["Jwt:Issuer"]).Returns("test-issuer");
            configurationMock.Setup(c => c["Jwt:Audience"]).Returns("test-audience");
            AuthService authService = new AuthService(configurationMock.Object);
            User user = new User { Id = "user123", Email = "test@example.com" };

            // Act
            string token = authService.GenerateJwtToken(user);

            // Assert
            Assert.IsNotNull(token);
            Assert.IsFalse(string.IsNullOrEmpty(token));
        }

        /// <summary>
        /// Tests that GenerateJwtToken successfully generates a token with very long SecretKey.
        /// Input: Configuration with SecretKey of 10,000 characters.
        /// Expected: Valid token is generated successfully.
        /// </summary>
        [TestMethod]
        public void GenerateJwtToken_VeryLongSecretKey_GeneratesValidToken()
        {
            // Arrange
            Mock<IConfiguration> configurationMock = new Mock<IConfiguration>();
            string veryLongSecretKey = new string('x', 10000);
            configurationMock.Setup(c => c["Jwt:SecretKey"]).Returns(veryLongSecretKey);
            configurationMock.Setup(c => c["Jwt:Issuer"]).Returns("test-issuer");
            configurationMock.Setup(c => c["Jwt:Audience"]).Returns("test-audience");
            AuthService authService = new AuthService(configurationMock.Object);
            User user = new User { Id = "user123", Email = "test@example.com" };

            // Act
            string token = authService.GenerateJwtToken(user);

            // Assert
            Assert.IsNotNull(token);
            Assert.IsFalse(string.IsNullOrEmpty(token));
            JwtSecurityTokenHandler tokenHandler = new JwtSecurityTokenHandler();
            JwtSecurityToken jwtToken = tokenHandler.ReadJwtToken(token);
            Assert.AreEqual("test-issuer", jwtToken.Issuer);
        }

        /// <summary>
        /// Tests that GenerateJwtToken successfully generates a token with very long Issuer.
        /// Input: Configuration with Issuer of 10,000 characters.
        /// Expected: Valid token with long Issuer correctly embedded.
        /// </summary>
        [TestMethod]
        public void GenerateJwtToken_VeryLongIssuer_GeneratesValidToken()
        {
            // Arrange
            Mock<IConfiguration> configurationMock = new Mock<IConfiguration>();
            string veryLongIssuer = new string('y', 10000);
            configurationMock.Setup(c => c["Jwt:SecretKey"]).Returns("this-is-a-very-secure-secret-key-with-at-least-32-characters");
            configurationMock.Setup(c => c["Jwt:Issuer"]).Returns(veryLongIssuer);
            configurationMock.Setup(c => c["Jwt:Audience"]).Returns("test-audience");
            AuthService authService = new AuthService(configurationMock.Object);
            User user = new User { Id = "user123", Email = "test@example.com" };

            // Act
            string token = authService.GenerateJwtToken(user);

            // Assert
            Assert.IsNotNull(token);
            JwtSecurityTokenHandler tokenHandler = new JwtSecurityTokenHandler();
            JwtSecurityToken jwtToken = tokenHandler.ReadJwtToken(token);
            Assert.AreEqual(veryLongIssuer, jwtToken.Issuer);
        }

        /// <summary>
        /// Tests that GenerateJwtToken successfully generates a token with very long Audience.
        /// Input: Configuration with Audience of 10,000 characters.
        /// Expected: Valid token with long Audience correctly embedded.
        /// </summary>
        [TestMethod]
        public void GenerateJwtToken_VeryLongAudience_GeneratesValidToken()
        {
            // Arrange
            Mock<IConfiguration> configurationMock = new Mock<IConfiguration>();
            string veryLongAudience = new string('z', 10000);
            configurationMock.Setup(c => c["Jwt:SecretKey"]).Returns("this-is-a-very-secure-secret-key-with-at-least-32-characters");
            configurationMock.Setup(c => c["Jwt:Issuer"]).Returns("test-issuer");
            configurationMock.Setup(c => c["Jwt:Audience"]).Returns(veryLongAudience);
            AuthService authService = new AuthService(configurationMock.Object);
            User user = new User { Id = "user123", Email = "test@example.com" };

            // Act
            string token = authService.GenerateJwtToken(user);

            // Assert
            Assert.IsNotNull(token);
            JwtSecurityTokenHandler tokenHandler = new JwtSecurityTokenHandler();
            JwtSecurityToken jwtToken = tokenHandler.ReadJwtToken(token);
            Assert.IsTrue(jwtToken.Audiences.Contains(veryLongAudience));
        }

        /// <summary>
        /// Tests that GenerateJwtToken successfully generates a token with special characters in configuration values.
        /// Input: Configuration with special characters in Issuer and Audience.
        /// Expected: Valid token with special characters correctly embedded.
        /// </summary>
        [TestMethod]
        [DataRow("issuer@#$%^&*()", "audience!@#$%")]
        [DataRow("issuer\nwith\nnewlines", "audience\twith\ttabs")]
        [DataRow("発行者", "観客")]
        public void GenerateJwtToken_SpecialCharactersInConfiguration_GeneratesValidToken(string issuer, string audience)
        {
            // Arrange
            Mock<IConfiguration> configurationMock = new Mock<IConfiguration>();
            configurationMock.Setup(c => c["Jwt:SecretKey"]).Returns("this-is-a-very-secure-secret-key-with-at-least-32-characters");
            configurationMock.Setup(c => c["Jwt:Issuer"]).Returns(issuer);
            configurationMock.Setup(c => c["Jwt:Audience"]).Returns(audience);
            AuthService authService = new AuthService(configurationMock.Object);
            User user = new User { Id = "user123", Email = "test@example.com" };

            // Act
            string token = authService.GenerateJwtToken(user);

            // Assert
            Assert.IsNotNull(token);
            JwtSecurityTokenHandler tokenHandler = new JwtSecurityTokenHandler();
            JwtSecurityToken jwtToken = tokenHandler.ReadJwtToken(token);
            Assert.AreEqual(issuer, jwtToken.Issuer);
            Assert.IsTrue(jwtToken.Audiences.Contains(audience));
        }

        /// <summary>
        /// Tests that GenerateJwtToken generates token with correct expiration time.
        /// Input: Valid configuration and user.
        /// Expected: Token expires approximately 1 hour from the current UTC time.
        /// </summary>
        [TestMethod]
        public void GenerateJwtToken_ValidInput_TokenExpiresInOneHour()
        {
            // Arrange
            Mock<IConfiguration> configurationMock = new Mock<IConfiguration>();
            configurationMock.Setup(c => c["Jwt:SecretKey"]).Returns("this-is-a-very-secure-secret-key-with-at-least-32-characters");
            configurationMock.Setup(c => c["Jwt:Issuer"]).Returns("test-issuer");
            configurationMock.Setup(c => c["Jwt:Audience"]).Returns("test-audience");
            AuthService authService = new AuthService(configurationMock.Object);
            User user = new User { Id = "user123", Email = "test@example.com" };
            DateTime beforeGeneration = DateTime.UtcNow;

            // Act
            string token = authService.GenerateJwtToken(user);

            // Assert
            DateTime afterGeneration = DateTime.UtcNow;
            JwtSecurityTokenHandler tokenHandler = new JwtSecurityTokenHandler();
            JwtSecurityToken jwtToken = tokenHandler.ReadJwtToken(token);
            DateTime expectedExpiration = beforeGeneration.AddHours(1);
            TimeSpan timeDifference = jwtToken.ValidTo - expectedExpiration;
            Assert.IsTrue(Math.Abs(timeDifference.TotalSeconds) <= 2, $"Token expiration time differs by {timeDifference.TotalSeconds} seconds");
        }

        /// <summary>
        /// Tests that GenerateJwtToken uses HMAC SHA256 signature algorithm.
        /// Input: Valid configuration and user.
        /// Expected: Token header contains the correct signature algorithm.
        /// </summary>
        [TestMethod]
        public void GenerateJwtToken_ValidInput_UsesHmacSha256Algorithm()
        {
            // Arrange
            Mock<IConfiguration> configurationMock = new Mock<IConfiguration>();
            configurationMock.Setup(c => c["Jwt:SecretKey"]).Returns("this-is-a-very-secure-secret-key-with-at-least-32-characters");
            configurationMock.Setup(c => c["Jwt:Issuer"]).Returns("test-issuer");
            configurationMock.Setup(c => c["Jwt:Audience"]).Returns("test-audience");
            AuthService authService = new AuthService(configurationMock.Object);
            User user = new User { Id = "user123", Email = "test@example.com" };

            // Act
            string token = authService.GenerateJwtToken(user);

            // Assert
            JwtSecurityTokenHandler tokenHandler = new JwtSecurityTokenHandler();
            JwtSecurityToken jwtToken = tokenHandler.ReadJwtToken(token);
            Assert.AreEqual(SecurityAlgorithms.HmacSha256, jwtToken.SignatureAlgorithm);
        }

        /// <summary>
        /// Tests that GenerateJwtToken generates exactly two claims (NameIdentifier and Email).
        /// Input: Valid configuration and user.
        /// Expected: Token contains exactly 2 claims.
        /// </summary>
        [TestMethod]
        public void GenerateJwtToken_ValidInput_ContainsTwoClaims()
        {
            // Arrange
            Mock<IConfiguration> configurationMock = new Mock<IConfiguration>();
            configurationMock.Setup(c => c["Jwt:SecretKey"]).Returns("this-is-a-very-secure-secret-key-with-at-least-32-characters");
            configurationMock.Setup(c => c["Jwt:Issuer"]).Returns("test-issuer");
            configurationMock.Setup(c => c["Jwt:Audience"]).Returns("test-audience");
            AuthService authService = new AuthService(configurationMock.Object);
            User user = new User { Id = "user123", Email = "test@example.com" };

            // Act
            string token = authService.GenerateJwtToken(user);

            // Assert
            JwtSecurityTokenHandler tokenHandler = new JwtSecurityTokenHandler();
            JwtSecurityToken jwtToken = tokenHandler.ReadJwtToken(token);
            Assert.AreEqual(2, jwtToken.Claims.Count());
        }
    }
}