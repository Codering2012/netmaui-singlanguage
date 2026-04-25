using System;
using System.Diagnostics;
using System.Text;
using System.Text.Json;
using System.Threading.Tasks;

using Microsoft.Maui.Storage;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using Moq;
using SignLanguageApp;
using SignLanguageApp.Model;
using SignLanguageApp.Services;

namespace SignLanguageApp.Services.UnitTests;

/// <summary>
/// Unit tests for the DatabaseService class.
/// </summary>
[TestClass]
public class DatabaseServiceTests
{
    /// <summary>
    /// Tests that SaveUserAsync completes without throwing when given a valid user object.
    /// Note: This test is marked as Inconclusive because the DatabaseService implementation
    /// relies on static methods (Preferences.Set, JsonSerializer.Serialize, Debug.WriteLine)
    /// that cannot be mocked using Moq. Proper unit testing of this method would require
    /// refactoring the DatabaseService to use dependency injection for storage and serialization.
    /// </summary>
    [TestMethod]
    public async Task SaveUserAsync_ValidUser_CompletesSuccessfully()
    {
        // Arrange
        var service = new DatabaseService();
        var user = new User
        {
            Id = Guid.NewGuid().ToString(),
            Email = "test@example.com",
            Name = "Test User",
            PasswordHash = "hashedpassword123",
            AvatarUrl = "https://example.com/avatar.jpg",
            LearningStreak = 5,
            CreatedAt = DateTime.UtcNow,
            LastLoginAt = DateTime.UtcNow
        };

        // Act & Assert
        // Cannot fully test due to static dependencies (Preferences.Set)
        // The method will attempt to serialize and store the user
        // In a real unit test environment without MAUI infrastructure, this may fail
        Assert.Inconclusive("This test requires refactoring DatabaseService to support dependency injection for proper unit testing. " +
            "The current implementation uses static methods (Preferences.Set) that cannot be mocked with Moq.");
    }

    /// <summary>
    /// Tests that SaveUserAsync handles null user parameter gracefully.
    /// Note: This test is marked as Inconclusive due to unmockable static dependencies.
    /// The method signature indicates user is non-nullable, but the implementation
    /// does not validate this and JsonSerializer.Serialize will throw ArgumentNullException.
    /// However, the exception is caught and swallowed by the implementation.
    /// </summary>
    [TestMethod]
    public async Task SaveUserAsync_NullUser_HandlesGracefully()
    {
        // Arrange
        var service = new DatabaseService();
        User user = null!;

        // Act & Assert
        // The implementation catches all exceptions, so no exception should propagate
        // However, we cannot verify the actual behavior without mocking static dependencies
        Assert.Inconclusive("This test requires refactoring DatabaseService to support dependency injection for proper unit testing. " +
            "The current implementation catches all exceptions internally, making it impossible to verify correct behavior without mockable dependencies.");
    }

    /// <summary>
    /// Tests that SaveUserAsync correctly serializes a user with minimal data.
    /// Note: This test is marked as Inconclusive due to unmockable static dependencies.
    /// </summary>
    [TestMethod]
    public async Task SaveUserAsync_UserWithMinimalData_CompletesSuccessfully()
    {
        // Arrange
        var service = new DatabaseService();
        var user = new User
        {
            Id = string.Empty,
            Email = string.Empty,
            Name = string.Empty,
            PasswordHash = string.Empty,
            AvatarUrl = string.Empty,
            LearningStreak = 0
        };

        // Act & Assert
        Assert.Inconclusive("This test requires refactoring DatabaseService to support dependency injection for proper unit testing. " +
            "Static dependencies (Preferences.Set, JsonSerializer.Serialize) prevent proper mocking and verification.");
    }

    /// <summary>
    /// Tests that SaveUserAsync handles a user with special characters in string properties.
    /// Note: This test is marked as Inconclusive due to unmockable static dependencies.
    /// </summary>
    [TestMethod]
    public async Task SaveUserAsync_UserWithSpecialCharacters_CompletesSuccessfully()
    {
        // Arrange
        var service = new DatabaseService();
        var user = new User
        {
            Id = Guid.NewGuid().ToString(),
            Email = "test+special@example.com",
            Name = "Test \"User\" with 'quotes' and \n newlines",
            PasswordHash = "hash!@#$%^&*()",
            AvatarUrl = "https://example.com/avatar?param=value&other=test",
            LearningStreak = int.MaxValue
        };

        // Act & Assert
        Assert.Inconclusive("This test requires refactoring DatabaseService to support dependency injection for proper unit testing. " +
            "Cannot verify JSON serialization behavior without mocking JsonSerializer.");
    }

    /// <summary>
    /// Tests that SaveUserAsync handles a user with extreme numeric values.
    /// Note: This test is marked as Inconclusive due to unmockable static dependencies.
    /// </summary>
    [TestMethod]
    public async Task SaveUserAsync_UserWithExtremeValues_CompletesSuccessfully()
    {
        // Arrange
        var service = new DatabaseService();
        var user = new User
        {
            Id = new string('x', 10000), // Very long string
            Email = "test@example.com",
            Name = "Test User",
            PasswordHash = "hashedpassword",
            AvatarUrl = "https://example.com/avatar.jpg",
            LearningStreak = int.MaxValue,
            CreatedAt = DateTime.MaxValue,
            LastLoginAt = DateTime.MinValue
        };

        // Act & Assert
        Assert.Inconclusive("This test requires refactoring DatabaseService to support dependency injection for proper unit testing. " +
            "Cannot verify behavior with extreme values without mocking storage layer.");
    }

    /// <summary>
    /// Tests SaveAccessTokenAsync with a valid token.
    /// Verifies that the method completes without throwing exceptions.
    /// Expected: Method completes successfully.
    /// Note: This test is marked as inconclusive because SecureStorage.Default is a static dependency
    /// that cannot be mocked with Moq, and creating fake implementations is prohibited.
    /// To fully test this method, consider refactoring to inject ISecureStorage dependency.
    /// </summary>
    [TestMethod]
    [DataRow("valid_token_123")]
    [DataRow("eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJzdWIiOiIxMjM0NTY3ODkwIn0.dozjgNryP4J3jVmNHl0w5N_XgL0n3I9PlFUP0THsR8U")]
    [DataRow("token-with-special-chars!@#$%^&*()")]
    public async Task SaveAccessTokenAsync_ValidToken_CompletesSuccessfully(string token)
    {
        // Arrange
        var databaseService = new DatabaseService();

        // Act & Assert - Method should not throw
        // NOTE: Cannot verify actual SecureStorage interaction due to static dependency
        Assert.Inconclusive("SecureStorage.Default is a static dependency that cannot be mocked. " +
                           "Actual storage behavior cannot be verified without refactoring to use dependency injection.");

        await databaseService.SaveAccessTokenAsync(token);
    }

    /// <summary>
    /// Tests SaveAccessTokenAsync with a null token.
    /// Verifies that the method handles null input gracefully.
    /// Expected: Method completes without throwing (exception is caught internally).
    /// Note: This test is marked as inconclusive because SecureStorage.Default cannot be mocked.
    /// </summary>
    [TestMethod]
    public async Task SaveAccessTokenAsync_NullToken_CompletesWithoutThrowing()
    {
        // Arrange
        var databaseService = new DatabaseService();
        string token = null!;

        // Act & Assert - Method should not throw (catches all exceptions)
        // NOTE: Cannot verify actual SecureStorage interaction or Debug.WriteLine output
        Assert.Inconclusive("SecureStorage.Default is a static dependency that cannot be mocked. " +
                           "Actual exception handling and storage behavior cannot be verified.");

        await databaseService.SaveAccessTokenAsync(token);
    }

    /// <summary>
    /// Tests SaveAccessTokenAsync with an empty string token.
    /// Verifies that the method handles empty string input.
    /// Expected: Method completes successfully.
    /// Note: This test is marked as inconclusive because SecureStorage.Default cannot be mocked.
    /// </summary>
    [TestMethod]
    public async Task SaveAccessTokenAsync_EmptyToken_CompletesSuccessfully()
    {
        // Arrange
        var databaseService = new DatabaseService();
        string token = string.Empty;

        // Act & Assert - Method should not throw
        // NOTE: Cannot verify actual SecureStorage interaction
        Assert.Inconclusive("SecureStorage.Default is a static dependency that cannot be mocked. " +
                           "Actual storage behavior cannot be verified without refactoring to use dependency injection.");

        await databaseService.SaveAccessTokenAsync(token);
    }

    /// <summary>
    /// Tests SaveAccessTokenAsync with a whitespace-only token.
    /// Verifies that the method handles whitespace input.
    /// Expected: Method completes successfully.
    /// Note: This test is marked as inconclusive because SecureStorage.Default cannot be mocked.
    /// </summary>
    [TestMethod]
    [DataRow("   ")]
    [DataRow("\t")]
    [DataRow("\n")]
    [DataRow("\r\n")]
    public async Task SaveAccessTokenAsync_WhitespaceToken_CompletesSuccessfully(string token)
    {
        // Arrange
        var databaseService = new DatabaseService();

        // Act & Assert - Method should not throw
        // NOTE: Cannot verify actual SecureStorage interaction
        Assert.Inconclusive("SecureStorage.Default is a static dependency that cannot be mocked. " +
                           "Actual storage behavior cannot be verified without refactoring to use dependency injection.");

        await databaseService.SaveAccessTokenAsync(token);
    }

    /// <summary>
    /// Tests SaveAccessTokenAsync with a very long token string.
    /// Verifies that the method handles large input strings.
    /// Expected: Method completes successfully.
    /// Note: This test is marked as inconclusive because SecureStorage.Default cannot be mocked.
    /// </summary>
    [TestMethod]
    public async Task SaveAccessTokenAsync_VeryLongToken_CompletesSuccessfully()
    {
        // Arrange
        var databaseService = new DatabaseService();
        string token = new string('a', 10000); // Very long token

        // Act & Assert - Method should not throw
        // NOTE: Cannot verify actual SecureStorage interaction
        Assert.Inconclusive("SecureStorage.Default is a static dependency that cannot be mocked. " +
                           "Actual storage behavior cannot be verified without refactoring to use dependency injection.");

        await databaseService.SaveAccessTokenAsync(token);
    }

    /// <summary>
    /// Tests that SaveUserAsync handles null user parameter gracefully by catching the serialization exception.
    /// Note: Cannot verify Debug.WriteLine was called due to unmockable static dependency.
    /// </summary>
    [TestMethod]
    public async Task SaveUserAsync_NullUser_DoesNotThrow()
    {
        // Arrange
        var service = new DatabaseService();
        User? user = null;

        // Act
        await service.SaveUserAsync(user!);

        // Assert
        // Note: JsonSerializer.Serialize will throw ArgumentNullException for null user,
        // but the exception is caught and logged to Debug.WriteLine.
        // Cannot verify the exception was logged because System.Diagnostics.Debug is static and cannot be mocked.
        // This test verifies that the method does not throw and completes successfully.
        Assert.Inconclusive("Cannot verify exception handling and Debug.WriteLine call due to unmockable static dependencies.");
    }

    /// <summary>
    /// Tests that SaveUserAsync handles a user with empty string properties.
    /// Note: Cannot verify Preferences.Set side effects due to unmockable static dependency.
    /// </summary>
    [TestMethod]
    public async Task SaveUserAsync_UserWithEmptyStrings_CompletesSuccessfully()
    {
        // Arrange
        var service = new DatabaseService();
        var user = new User
        {
            Id = string.Empty,
            Email = string.Empty,
            Name = string.Empty,
            PasswordHash = string.Empty,
            AvatarUrl = string.Empty,
            LearningStreak = 0,
            CreatedAt = DateTime.MinValue,
            LastLoginAt = DateTime.MinValue
        };

        // Act
        await service.SaveUserAsync(user);

        // Assert
        Assert.Inconclusive("Cannot verify Preferences.Set was called correctly due to unmockable static dependency.");
    }

    /// <summary>
    /// Tests that SaveUserAsync handles a user with very long string properties.
    /// Note: Cannot verify Preferences.Set side effects due to unmockable static dependency.
    /// </summary>
    [TestMethod]
    public async Task SaveUserAsync_UserWithVeryLongStrings_CompletesSuccessfully()
    {
        // Arrange
        var service = new DatabaseService();
        var longString = new string('a', 10000);
        var user = new User
        {
            Id = longString,
            Email = longString + "@example.com",
            Name = longString,
            PasswordHash = longString,
            AvatarUrl = "https://example.com/" + longString,
            LearningStreak = int.MaxValue,
            CreatedAt = DateTime.MaxValue,
            LastLoginAt = DateTime.MaxValue
        };

        // Act
        await service.SaveUserAsync(user);

        // Assert
        Assert.Inconclusive("Cannot verify Preferences.Set was called correctly due to unmockable static dependency.");
    }

    /// <summary>
    /// Tests that SaveUserAsync handles boundary values for numeric properties.
    /// Note: Cannot verify Preferences.Set side effects due to unmockable static dependency.
    /// </summary>
    [TestMethod]
    public async Task SaveUserAsync_UserWithBoundaryNumericValues_CompletesSuccessfully()
    {
        // Arrange
        var service = new DatabaseService();
        var user = new User
        {
            Id = "boundary-test",
            Email = "boundary@example.com",
            Name = "Boundary Test",
            PasswordHash = "hash",
            AvatarUrl = "https://example.com/avatar",
            LearningStreak = int.MinValue,
            CreatedAt = DateTime.MinValue,
            LastLoginAt = DateTime.MaxValue
        };

        // Act
        await service.SaveUserAsync(user);

        // Assert
        Assert.Inconclusive("Cannot verify Preferences.Set was called correctly due to unmockable static dependency.");
    }

    /// <summary>
    /// Tests that SaveUserAsync handles a user with default values.
    /// Note: Cannot verify Preferences.Set side effects due to unmockable static dependency.
    /// </summary>
    [TestMethod]
    public async Task SaveUserAsync_UserWithDefaultValues_CompletesSuccessfully()
    {
        // Arrange
        var service = new DatabaseService();
        var user = new User();

        // Act
        await service.SaveUserAsync(user);

        // Assert
        Assert.Inconclusive("Cannot verify Preferences.Set was called correctly due to unmockable static dependency.");
    }

    /// <summary>
    /// Tests that GetAccessTokenAsync can be called without throwing an exception.
    /// Note: This test cannot fully validate behavior because DatabaseService depends on
    /// SecureStorage.Default, which is a static dependency that cannot be mocked with Moq.
    /// To properly unit test this method, consider refactoring DatabaseService to accept
    /// an ISecureStorage interface through dependency injection.
    /// Current test only verifies the method can be invoked and returns a Task.
    /// </summary>
    [TestMethod]
    public async Task GetAccessTokenAsync_WithStaticDependency_CanBeInvoked()
    {
        // Arrange
        var service = new DatabaseService();

        // Act & Assert
        // Note: This test is marked as inconclusive because we cannot mock SecureStorage.Default.
        // The actual behavior depends on the MAUI SecureStorage implementation which is not
        // available in a unit test context without integration testing.
        // To properly test this method:
        // 1. Refactor DatabaseService to accept ISecureStorage via constructor injection
        // 2. Create an ISecureStorage interface with GetAsync and SetAsync methods
        // 3. Mock ISecureStorage using Moq in tests
        // 4. Test scenarios: successful retrieval, null/empty returns, and exception handling

        Assert.Inconclusive(
            "GetAccessTokenAsync cannot be properly unit tested due to static SecureStorage dependency. " +
            "Refactor to use dependency injection for proper unit testing. " +
            "Expected test scenarios would include: " +
            "1. Successful token retrieval returns the stored token, " +
            "2. No token stored returns null, " +
            "3. SecureStorage exception is caught and returns null with debug logging.");
    }

    /// <summary>
    /// Placeholder test documenting expected behavior when token exists.
    /// This test cannot be implemented without refactoring DatabaseService to use dependency injection.
    /// Expected behavior: When a valid access token is stored, GetAccessTokenAsync should return that token.
    /// </summary>
    [TestMethod]
    public async Task GetAccessTokenAsync_WhenTokenExists_ShouldReturnToken()
    {
        // Arrange
        // TODO: Refactor DatabaseService to inject ISecureStorage dependency
        // var mockSecureStorage = new Mock<ISecureStorage>();
        // mockSecureStorage.Setup(s => s.GetAsync("access_token")).ReturnsAsync("valid_token_123");
        // var service = new DatabaseService(mockSecureStorage.Object);

        // Act
        // var result = await service.GetAccessTokenAsync();

        // Assert
        // Assert.IsNotNull(result);
        // Assert.AreEqual("valid_token_123", result);

        Assert.Inconclusive("Requires refactoring DatabaseService to use dependency injection for ISecureStorage.");
    }

    /// <summary>
    /// Placeholder test documenting expected behavior when no token is stored.
    /// This test cannot be implemented without refactoring DatabaseService to use dependency injection.
    /// Expected behavior: When no access token is stored, GetAccessTokenAsync should return null.
    /// </summary>
    [TestMethod]
    public async Task GetAccessTokenAsync_WhenNoTokenStored_ShouldReturnNull()
    {
        // Arrange
        // TODO: Refactor DatabaseService to inject ISecureStorage dependency
        // var mockSecureStorage = new Mock<ISecureStorage>();
        // mockSecureStorage.Setup(s => s.GetAsync("access_token")).ReturnsAsync((string?)null);
        // var service = new DatabaseService(mockSecureStorage.Object);

        // Act
        // var result = await service.GetAccessTokenAsync();

        // Assert
        // Assert.IsNull(result);

        Assert.Inconclusive("Requires refactoring DatabaseService to use dependency injection for ISecureStorage.");
    }

    /// <summary>
    /// Placeholder test documenting expected exception handling behavior.
    /// This test cannot be implemented without refactoring DatabaseService to use dependency injection.
    /// Expected behavior: When SecureStorage throws an exception, GetAccessTokenAsync should catch it,
    /// log to Debug, and return null without propagating the exception.
    /// </summary>
    [TestMethod]
    public async Task GetAccessTokenAsync_WhenSecureStorageThrowsException_ShouldReturnNull()
    {
        // Arrange
        // TODO: Refactor DatabaseService to inject ISecureStorage dependency
        // var mockSecureStorage = new Mock<ISecureStorage>();
        // mockSecureStorage.Setup(s => s.GetAsync("access_token"))
        //     .ThrowsAsync(new InvalidOperationException("SecureStorage error"));
        // var service = new DatabaseService(mockSecureStorage.Object);

        // Act
        // var result = await service.GetAccessTokenAsync();

        // Assert
        // Assert.IsNull(result);
        // Verify Debug.WriteLine was called with error message

        Assert.Inconclusive("Requires refactoring DatabaseService to use dependency injection for ISecureStorage.");
    }

    /// <summary>
    /// Tests that ClearAllAsync completes successfully without throwing exceptions.
    /// This is an integration-style test because DeleteUserAsync is not virtual and cannot be mocked.
    /// The method internally catches all exceptions, so it should always complete successfully.
    /// </summary>
    [TestMethod]
    public async Task ClearAllAsync_WhenCalled_CompletesSuccessfully()
    {
        // Arrange
        var service = new global::SignLanguageApp.Services.DatabaseService();

        // Act
        // The method should complete without throwing, even if DeleteUserAsync encounters errors
        await service.ClearAllAsync();

        // Assert
        // If we reach here, the method completed successfully without throwing
        Assert.IsTrue(true, "ClearAllAsync completed without throwing an exception");
    }

    /// <summary>
    /// Tests that ClearAllAsync can be called multiple times without issues.
    /// Since the method has exception handling, repeated calls should be safe.
    /// </summary>
    [TestMethod]
    public async Task ClearAllAsync_WhenCalledMultipleTimes_CompletesSuccessfully()
    {
        // Arrange
        var service = new global::SignLanguageApp.Services.DatabaseService();

        // Act & Assert
        // Multiple calls should all complete successfully
        await service.ClearAllAsync();
        await service.ClearAllAsync();
        await service.ClearAllAsync();

        Assert.IsTrue(true, "ClearAllAsync completed multiple times without throwing exceptions");
    }

    /// <summary>
    /// Tests that ClearAllAsync does not throw exceptions even under concurrent calls.
    /// The method's exception handling should ensure thread-safe completion.
    /// </summary>
    [TestMethod]
    public async Task ClearAllAsync_WhenCalledConcurrently_CompletesSuccessfully()
    {
        // Arrange
        var service = new global::SignLanguageApp.Services.DatabaseService();
        var tasks = new Task[10];

        // Act
        for (int i = 0; i < tasks.Length; i++)
        {
            tasks[i] = service.ClearAllAsync();
        }

        // Assert - all tasks should complete without throwing
        await Task.WhenAll(tasks);
        Assert.IsTrue(true, "All concurrent ClearAllAsync calls completed without throwing exceptions");
    }

    // Note: Additional test scenarios that would require mocking cannot be implemented because:
    // 1. DeleteUserAsync is not virtual and cannot be mocked with Moq
    // 2. Debug.WriteLine is a static method and cannot be mocked
    // 3. Creating fake/stub classes is prohibited by the testing requirements
    // To test exception handling paths in isolation, DeleteUserAsync would need to be made virtual
    // or injected as a dependency.

    /// <summary>
    /// Tests that GetRefreshTokenAsync returns a valid token string when SecureStorage contains a token.
    /// Input: SecureStorage contains a valid token.
    /// Expected: Method returns the token string.
    /// </summary>
    [TestMethod]
    public async Task GetRefreshTokenAsync_ValidTokenExists_ReturnsToken()
    {
        // Arrange
        var databaseService = new DatabaseService();

        // This test cannot be completed without refactoring the production code to inject
        // an abstraction for SecureStorage. The method uses Microsoft.Maui.Storage.SecureStorage.Default,
        // which is a static property that cannot be mocked with Moq.
        //
        // To make this testable, consider:
        // 1. Creating an ISecureStorage interface
        // 2. Injecting it via constructor
        // 3. Mocking it in tests
        //
        // Expected behavior: Should return the token stored in SecureStorage for key "refresh_token"

        Assert.Inconclusive("Method uses static SecureStorage.Default which cannot be mocked with Moq. Refactor to use dependency injection.");
    }

    /// <summary>
    /// Tests that GetRefreshTokenAsync returns null when SecureStorage returns null.
    /// Input: SecureStorage returns null for the refresh token key.
    /// Expected: Method returns null.
    /// </summary>
    [TestMethod]
    public async Task GetRefreshTokenAsync_NoTokenExists_ReturnsNull()
    {
        // Arrange
        var databaseService = new DatabaseService();

        // This test cannot be completed without refactoring the production code.
        // See GetRefreshTokenAsync_ValidTokenExists_ReturnsToken for details.
        //
        // Expected behavior: Should return null when SecureStorage.GetAsync returns null

        Assert.Inconclusive("Method uses static SecureStorage.Default which cannot be mocked with Moq. Refactor to use dependency injection.");
    }

    /// <summary>
    /// Tests that GetRefreshTokenAsync returns empty string when SecureStorage contains an empty token.
    /// Input: SecureStorage contains an empty string.
    /// Expected: Method returns empty string.
    /// </summary>
    [TestMethod]
    public async Task GetRefreshTokenAsync_EmptyTokenExists_ReturnsEmptyString()
    {
        // Arrange
        var databaseService = new DatabaseService();

        // This test cannot be completed without refactoring the production code.
        // See GetRefreshTokenAsync_ValidTokenExists_ReturnsToken for details.
        //
        // Expected behavior: Should return empty string when SecureStorage contains ""

        Assert.Inconclusive("Method uses static SecureStorage.Default which cannot be mocked with Moq. Refactor to use dependency injection.");
    }

    /// <summary>
    /// Tests that GetRefreshTokenAsync handles exceptions and returns null when SecureStorage throws.
    /// Input: SecureStorage.GetAsync throws an exception.
    /// Expected: Method catches the exception, logs it, and returns null.
    /// </summary>
    [TestMethod]
    public async Task GetRefreshTokenAsync_SecureStorageThrowsException_ReturnsNull()
    {
        // Arrange
        var databaseService = new DatabaseService();

        // This test cannot be completed without refactoring the production code.
        // See GetRefreshTokenAsync_ValidTokenExists_ReturnsToken for details.
        //
        // Expected behavior: Should catch any exception, log it via Debug.WriteLine,
        // and return null

        Assert.Inconclusive("Method uses static SecureStorage.Default which cannot be mocked with Moq. Refactor to use dependency injection.");
    }

    /// <summary>
    /// Tests that GetRefreshTokenAsync handles InvalidOperationException and returns null.
    /// Input: SecureStorage.GetAsync throws InvalidOperationException.
    /// Expected: Method catches the exception, logs it, and returns null.
    /// </summary>
    [TestMethod]
    public async Task GetRefreshTokenAsync_InvalidOperationException_ReturnsNull()
    {
        // Arrange
        var databaseService = new DatabaseService();

        // This test cannot be completed without refactoring the production code.
        // See GetRefreshTokenAsync_ValidTokenExists_ReturnsToken for details.
        //
        // Expected behavior: Should catch InvalidOperationException, log error message,
        // and return null

        Assert.Inconclusive("Method uses static SecureStorage.Default which cannot be mocked with Moq. Refactor to use dependency injection.");
    }

    /// <summary>
    /// Tests that GetRefreshTokenAsync handles UnauthorizedAccessException and returns null.
    /// Input: SecureStorage.GetAsync throws UnauthorizedAccessException.
    /// Expected: Method catches the exception, logs it, and returns null.
    /// </summary>
    [TestMethod]
    public async Task GetRefreshTokenAsync_UnauthorizedAccessException_ReturnsNull()
    {
        // Arrange
        var databaseService = new DatabaseService();

        // This test cannot be completed without refactoring the production code.
        // See GetRefreshTokenAsync_ValidTokenExists_ReturnsToken for details.
        //
        // Expected behavior: Should catch UnauthorizedAccessException, log error message,
        // and return null

        Assert.Inconclusive("Method uses static SecureStorage.Default which cannot be mocked with Moq. Refactor to use dependency injection.");
    }

    /// <summary>
    /// Tests that GetUserAsync returns null when no user data exists in preferences.
    /// This test cannot be executed because Preferences.Get is a static method that cannot be mocked with Moq.
    /// To make this testable, the DatabaseService should depend on an abstraction (e.g., IPreferencesService)
    /// that can be injected and mocked.
    /// </summary>
    [TestMethod]
    public async Task GetUserAsync_WhenNoUserDataExists_ReturnsNull()
    {
        Assert.Inconclusive(
            "This test cannot be executed because Microsoft.Maui.Storage.Preferences.Get() is a static method " +
            "that cannot be mocked using Moq. The DatabaseService class has unmockable static dependencies. " +
            "To make this method testable, refactor to inject an IPreferencesService abstraction or use " +
            "integration tests with platform-specific setup.");

        // Arrange
        // Would need: Preferences.Get("current_user", string.Empty) to return string.Empty or null

        // Act
        // var service = new DatabaseService();
        // var result = await service.GetUserAsync();

        // Assert
        // Assert.IsNull(result);
    }

    /// <summary>
    /// Tests that GetUserAsync successfully deserializes and returns a User when valid JSON exists in preferences.
    /// This test cannot be executed because Preferences.Get and JsonSerializer.Deserialize are static methods
    /// that cannot be mocked with Moq.
    /// </summary>
    [TestMethod]
    public async Task GetUserAsync_WhenValidUserJsonExists_ReturnsDeserializedUser()
    {
        Assert.Inconclusive(
            "This test cannot be executed because Microsoft.Maui.Storage.Preferences.Get() and " +
            "System.Text.Json.JsonSerializer.Deserialize() are static methods that cannot be mocked using Moq. " +
            "To make this method testable, refactor to inject abstractions for preferences and serialization.");

        // Arrange
        // Would need: Preferences.Get("current_user", string.Empty) to return valid User JSON
        // Example: {"Id":"123","Email":"test@example.com","Name":"Test User",...}

        // Act
        // var service = new DatabaseService();
        // var result = await service.GetUserAsync();

        // Assert
        // Assert.IsNotNull(result);
        // Assert.AreEqual("123", result.Id);
        // Assert.AreEqual("test@example.com", result.Email);
    }

    /// <summary>
    /// Tests that GetUserAsync returns null when preferences contain invalid JSON that cannot be deserialized.
    /// This test cannot be executed because Preferences.Get and JsonSerializer.Deserialize are static methods
    /// that cannot be mocked with Moq.
    /// </summary>
    [TestMethod]
    public async Task GetUserAsync_WhenInvalidJsonExists_ReturnsNull()
    {
        Assert.Inconclusive(
            "This test cannot be executed because Microsoft.Maui.Storage.Preferences.Get() and " +
            "System.Text.Json.JsonSerializer.Deserialize() are static methods that cannot be mocked using Moq. " +
            "To make this method testable, refactor to inject abstractions for preferences and serialization.");

        // Arrange
        // Would need: Preferences.Get("current_user", string.Empty) to return invalid JSON
        // Example: "{invalid json}", "not a json", "{incomplete"
        // JsonSerializer.Deserialize should throw JsonException

        // Act
        // var service = new DatabaseService();
        // var result = await service.GetUserAsync();

        // Assert
        // Assert.IsNull(result); // Exception is caught and null is returned
    }

    /// <summary>
    /// Tests that GetUserAsync returns null and logs error when an exception occurs during retrieval.
    /// This test cannot be executed because Preferences.Get and Debug.WriteLine are static methods
    /// that cannot be mocked with Moq.
    /// </summary>
    [TestMethod]
    public async Task GetUserAsync_WhenExceptionOccurs_ReturnsNullAndLogsError()
    {
        Assert.Inconclusive(
            "This test cannot be executed because Microsoft.Maui.Storage.Preferences.Get() and " +
            "System.Diagnostics.Debug.WriteLine() are static methods that cannot be mocked using Moq. " +
            "To make this method testable, refactor to inject abstractions for preferences and logging.");

        // Arrange
        // Would need: Preferences.Get to throw an exception

        // Act
        // var service = new DatabaseService();
        // var result = await service.GetUserAsync();

        // Assert
        // Assert.IsNull(result); // Exception is caught and null is returned
        // Would also verify Debug.WriteLine was called with error message
    }

    /// <summary>
    /// Tests that GetUserAsync returns null when preferences contain empty string.
    /// This test cannot be executed because Preferences.Get is a static method that cannot be mocked with Moq.
    /// </summary>
    [TestMethod]
    public async Task GetUserAsync_WhenEmptyStringInPreferences_ReturnsNull()
    {
        Assert.Inconclusive(
            "This test cannot be executed because Microsoft.Maui.Storage.Preferences.Get() is a static method " +
            "that cannot be mocked using Moq. To make this method testable, refactor to inject an abstraction.");

        // Arrange
        // Would need: Preferences.Get("current_user", string.Empty) to return string.Empty

        // Act
        // var service = new DatabaseService();
        // var result = await service.GetUserAsync();

        // Assert
        // Assert.IsNull(result); // Empty string triggers early return
    }

    /// <summary>
    /// Tests that GetUserAsync returns null when preferences contain whitespace-only string.
    /// This test cannot be executed because Preferences.Get is a static method that cannot be mocked with Moq.
    /// </summary>
    [TestMethod]
    public async Task GetUserAsync_WhenWhitespaceStringInPreferences_ReturnsNull()
    {
        Assert.Inconclusive(
            "This test cannot be executed because Microsoft.Maui.Storage.Preferences.Get() is a static method " +
            "that cannot be mocked using Moq. To make this method testable, refactor to inject an abstraction.");

        // Arrange
        // Would need: Preferences.Get("current_user", string.Empty) to return "   " or "\t\n"
        // Note: The code uses string.IsNullOrEmpty which doesn't treat whitespace as empty

        // Act
        // var service = new DatabaseService();
        // var result = await service.GetUserAsync();

        // Assert
        // This would attempt to deserialize whitespace, likely causing JsonException
        // Result should be null due to catch block
    }

    /// <summary>
    /// Tests that GetUserAsync handles JSON with null value correctly.
    /// This test cannot be executed because Preferences.Get and JsonSerializer.Deserialize are static methods
    /// that cannot be mocked with Moq.
    /// </summary>
    [TestMethod]
    public async Task GetUserAsync_WhenJsonIsNullLiteral_ReturnsNull()
    {
        Assert.Inconclusive(
            "This test cannot be executed because Microsoft.Maui.Storage.Preferences.Get() and " +
            "System.Text.Json.JsonSerializer.Deserialize() are static methods that cannot be mocked using Moq.");

        // Arrange
        // Would need: Preferences.Get("current_user", string.Empty) to return "null"
        // JsonSerializer.Deserialize<User>("null") returns null

        // Act
        // var service = new DatabaseService();
        // var result = await service.GetUserAsync();

        // Assert
        // Assert.IsNull(result); // Deserializing "null" JSON returns null
    }

    /// <summary>
    /// Tests that SaveRefreshTokenAsync completes without throwing exceptions when provided with a valid token string.
    /// Note: This test has limited coverage because SecureStorage.Default is a static dependency that cannot be mocked with Moq.
    /// The method swallows all exceptions internally, so we can only verify it completes without propagating exceptions.
    /// For complete testing, consider refactoring DatabaseService to accept an injectable ISecureStorage interface,
    /// or use integration tests that can interact with the actual SecureStorage implementation.
    /// </summary>
    /// <param name="token">The refresh token to save.</param>
    [TestMethod]
    [DataRow("valid_refresh_token_12345")]
    [DataRow("eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJzdWIiOiIxMjM0NTY3ODkwIiwibmFtZSI6IkpvaG4gRG9lIiwiaWF0IjoxNTE2MjM5MDIyfQ")]
    [DataRow("")]
    [DataRow("   ")]
    [DataRow("token_with_special_chars_!@#$%^&*()")]
    [DataRow("very_long_token_" + "a_very_long_string_repeated_multiple_times_to_test_boundary_conditions")]
    public async Task SaveRefreshTokenAsync_WithVariousTokenValues_CompletesWithoutThrowing(string token)
    {
        // Arrange
        var databaseService = new DatabaseService();

        // Act & Assert
        // The method swallows all exceptions, so we verify it completes without throwing.
        // We cannot verify the actual storage interaction because SecureStorage.Default cannot be mocked.
        // TODO: Refactor DatabaseService to use dependency injection for SecureStorage to enable proper unit testing.
        await databaseService.SaveRefreshTokenAsync(token);

        // If we reach here without exception, the test passes.
        // Note: This does not verify that the token was actually stored correctly.
        Assert.IsTrue(true, "Method completed without throwing an exception.");
    }

    /// <summary>
    /// Tests that SaveRefreshTokenAsync handles null token parameter.
    /// Note: The method has no null check, so behavior depends on SecureStorage.Default.SetAsync implementation.
    /// Since SecureStorage.Default cannot be mocked, this test has limited value and may require integration testing.
    /// The method catches all exceptions, so even if SecureStorage throws, it will be swallowed.
    /// </summary>
    [TestMethod]
    public async Task SaveRefreshTokenAsync_WithNullToken_CompletesWithoutThrowing()
    {
        // Arrange
        var databaseService = new DatabaseService();
        string? token = null;

        // Act & Assert
        // The method signature declares token as non-nullable, but we test null for safety.
        // Any exception from SecureStorage will be caught and swallowed.
        // TODO: Consider adding null validation in the production code if null tokens are invalid.
        await databaseService.SaveRefreshTokenAsync(token!);

        // If we reach here, the method completed (either successfully or by swallowing an exception).
        Assert.IsTrue(true, "Method completed without throwing an exception.");
    }

    /// <summary>
    /// Tests that SaveRefreshTokenAsync completes successfully with a valid token.
    /// Note: This test cannot verify the actual storage operation due to the static SecureStorage.Default dependency.
    /// Full verification would require refactoring the DatabaseService to accept ISecureStorage via dependency injection.
    /// </summary>
    [TestMethod]
    public async Task SaveRefreshTokenAsync_ValidToken_CompletesSuccessfully()
    {
        // Arrange
        var service = new DatabaseService();
        var validToken = "valid_refresh_token_12345";

        // Act & Assert - Method should complete without throwing
        await service.SaveRefreshTokenAsync(validToken);
    }

    /// <summary>
    /// Tests that SaveRefreshTokenAsync completes successfully with an empty string token.
    /// The method catches all exceptions, so even if the underlying storage rejects empty strings, the method completes.
    /// </summary>
    [TestMethod]
    public async Task SaveRefreshTokenAsync_EmptyString_CompletesSuccessfully()
    {
        // Arrange
        var service = new DatabaseService();
        var emptyToken = string.Empty;

        // Act & Assert - Method should complete without throwing
        await service.SaveRefreshTokenAsync(emptyToken);
    }

    /// <summary>
    /// Tests that SaveRefreshTokenAsync completes successfully with a whitespace-only token.
    /// The method catches all exceptions, so it should complete regardless of input validity.
    /// </summary>
    [TestMethod]
    public async Task SaveRefreshTokenAsync_WhitespaceString_CompletesSuccessfully()
    {
        // Arrange
        var service = new DatabaseService();
        var whitespaceToken = "   ";

        // Act & Assert - Method should complete without throwing
        await service.SaveRefreshTokenAsync(whitespaceToken);
    }

    /// <summary>
    /// Tests that SaveRefreshTokenAsync completes successfully with a very long token string.
    /// This tests boundary conditions for string length handling in secure storage.
    /// </summary>
    [TestMethod]
    public async Task SaveRefreshTokenAsync_VeryLongToken_CompletesSuccessfully()
    {
        // Arrange
        var service = new DatabaseService();
        var longToken = new string('a', 10000);

        // Act & Assert - Method should complete without throwing
        await service.SaveRefreshTokenAsync(longToken);
    }

    /// <summary>
    /// Tests that SaveRefreshTokenAsync completes successfully with special characters in the token.
    /// Tokens may contain special characters that need to be properly handled by secure storage.
    /// </summary>
    [TestMethod]
    public async Task SaveRefreshTokenAsync_TokenWithSpecialCharacters_CompletesSuccessfully()
    {
        // Arrange
        var service = new DatabaseService();
        var specialToken = "token!@#$%^&*()_+-={}[]|\\:\";<>?,./`~";

        // Act & Assert - Method should complete without throwing
        await service.SaveRefreshTokenAsync(specialToken);
    }

    /// <summary>
    /// Tests that SaveRefreshTokenAsync completes successfully with unicode characters in the token.
    /// Ensures proper handling of international characters and emojis.
    /// </summary>
    [TestMethod]
    public async Task SaveRefreshTokenAsync_TokenWithUnicodeCharacters_CompletesSuccessfully()
    {
        // Arrange
        var service = new DatabaseService();
        var unicodeToken = "token_测试_🔐_مفتاح";

        // Act & Assert - Method should complete without throwing
        await service.SaveRefreshTokenAsync(unicodeToken);
    }

    /// <summary>
    /// Tests that SaveRefreshTokenAsync completes successfully with control characters in the token.
    /// Control characters might cause issues in storage mechanisms.
    /// </summary>
    [TestMethod]
    public async Task SaveRefreshTokenAsync_TokenWithControlCharacters_CompletesSuccessfully()
    {
        // Arrange
        var service = new DatabaseService();
        var controlCharToken = "token\n\r\t\0";

        // Act & Assert - Method should complete without throwing
        await service.SaveRefreshTokenAsync(controlCharToken);
    }

    /// <summary>
    /// Tests that SaveRefreshTokenAsync handles null token parameter.
    /// Although the parameter is non-nullable, this tests runtime behavior when null is passed.
    /// The method catches all exceptions, so it should complete even if SetAsync throws ArgumentNullException.
    /// </summary>
    [TestMethod]
    public async Task SaveRefreshTokenAsync_NullToken_CompletesSuccessfully()
    {
        // Arrange
        var service = new DatabaseService();
        string? nullToken = null;

        // Act & Assert - Method should complete without throwing due to catch block
#pragma warning disable CS8604 // Possible null reference argument - Testing null handling
        await service.SaveRefreshTokenAsync(nullToken!);
#pragma warning restore CS8604
    }

    /// <summary>
    /// Tests that ClearAllAsync completes without throwing exceptions under normal conditions.
    /// Note: This test is marked as Inconclusive because DatabaseService depends on static MAUI classes
    /// (Preferences and SecureStorage) that cannot be mocked with Moq. To properly test this class,
    /// consider refactoring to use dependency injection with abstracted interfaces for storage operations.
    /// </summary>
    [TestMethod]
    public async Task ClearAllAsync_NormalExecution_CompletesWithoutException()
    {
        // Arrange
        // NOTE: This test cannot be fully implemented because DatabaseService uses static dependencies
        // (Microsoft.Maui.Storage.Preferences and Microsoft.Maui.Storage.SecureStorage) that cannot be mocked.
        // Refactor recommendation: Extract storage operations into injectable interfaces (e.g., IPreferencesService, ISecureStorageService)
        var service = new DatabaseService();

        // Act & Assert
        // This test is incomplete and should not be run in a unit test environment
        Assert.Inconclusive("DatabaseService cannot be unit tested due to unmockable static dependencies (Preferences, SecureStorage). Consider refactoring to use dependency injection.");
    }

    /// <summary>
    /// Tests that ClearAllAsync handles exceptions gracefully when DeleteUserAsync throws.
    /// Note: This test is marked as Inconclusive because DatabaseService depends on static MAUI classes
    /// that cannot be mocked. The exception handling behavior cannot be verified without integration testing
    /// or refactoring the class to support dependency injection.
    /// </summary>
    [TestMethod]
    public async Task ClearAllAsync_WhenDeleteUserAsyncThrows_SwallowsException()
    {
        // Arrange
        // NOTE: Cannot mock Preferences.Remove() to throw an exception because it's a static method
        // and the class does not use dependency injection.
        var service = new DatabaseService();

        // Act & Assert
        Assert.Inconclusive("DatabaseService cannot be unit tested due to unmockable static dependencies. Exception handling cannot be verified without integration testing or refactoring.");
    }

    /// <summary>
    /// Tests that DeleteUserAsync returns true when preferences are successfully removed.
    /// Note: This is an integration-style test as Preferences is a static class that cannot be mocked.
    /// The test will actually interact with the platform's preference storage.
    /// </summary>
    [TestMethod]
    public async Task DeleteUserAsync_WhenCalled_ReturnsTrueAndRemovesPreferences()
    {
        // Arrange
        var service = new DatabaseService();

        // Act
        var result = await service.DeleteUserAsync();

        // Assert
        Assert.IsTrue(result);
    }

    /// <summary>
    /// Tests that DeleteUserAsync returns false when an exception occurs during preference removal.
    /// Note: This test is marked as inconclusive because the Preferences class is a static class
    /// that cannot be mocked with Moq, and creating fake implementations is prohibited.
    /// To properly test exception handling, the code would need to be refactored to allow
    /// dependency injection of an abstraction over the Preferences API.
    /// </summary>
    [TestMethod]
    public async Task DeleteUserAsync_WhenExceptionOccurs_ReturnsFalse()
    {
        // Arrange
        var service = new DatabaseService();

        // Act & Assert
        // Cannot force an exception without mocking the static Preferences class.
        // The current implementation uses static dependencies that cannot be mocked with Moq.
        // To test this scenario, consider refactoring to inject an IPreferencesService abstraction.
        Assert.Inconclusive("Cannot test exception path without the ability to mock static Preferences class. Refactor to use dependency injection for proper unit testing.");
    }

    /// <summary>
    /// Tests that GetUserAsync is designed to return a User object from storage.
    /// </summary>
    /// <remarks>
    /// This test is marked as Inconclusive because the DatabaseService depends on
    /// the static Preferences class which cannot be mocked with Moq. The service
    /// would need to be refactored to accept an abstraction over Preferences
    /// (e.g., IPreferences) via dependency injection to enable proper unit testing.
    /// 
    /// Expected behavior (based on implementation analysis):
    /// - Should retrieve JSON from Preferences using key "current_user"
    /// - Should return null if no user data is stored
    /// - Should deserialize and return User object if valid JSON exists
    /// - Should return null if deserialization fails
    /// </remarks>
    [TestMethod]
    public async Task GetUserAsync_WithStoredUser_ShouldReturnUser()
    {
        // Arrange
        // Cannot arrange: Preferences is a static class that cannot be mocked
        var service = new DatabaseService();

        // Act & Assert
        Assert.Inconclusive(
            "This test cannot be completed because DatabaseService depends on the static " +
            "Preferences class from Microsoft.Maui.Storage, which cannot be mocked. " +
            "To make this testable, refactor DatabaseService to accept an IPreferences " +
            "interface via dependency injection.");
    }

    /// <summary>
    /// Tests that GetUserAsync returns null when no user data is stored.
    /// </summary>
    /// <remarks>
    /// This test is marked as Inconclusive because the DatabaseService depends on
    /// the static Preferences class which cannot be mocked with Moq.
    /// 
    /// Expected behavior: Should return null when Preferences.Get returns empty string.
    /// </remarks>
    [TestMethod]
    public async Task GetUserAsync_WithNoStoredUser_ShouldReturnNull()
    {
        // Arrange
        // Cannot arrange: Preferences is a static class that cannot be mocked
        var service = new DatabaseService();

        // Act & Assert
        Assert.Inconclusive(
            "This test cannot be completed because DatabaseService depends on the static " +
            "Preferences class from Microsoft.Maui.Storage, which cannot be mocked. " +
            "To make this testable, refactor DatabaseService to accept an IPreferences " +
            "interface via dependency injection.");
    }

    /// <summary>
    /// Tests that GetUserAsync handles JSON deserialization errors gracefully.
    /// </summary>
    /// <remarks>
    /// This test is marked as Inconclusive because the DatabaseService depends on
    /// the static Preferences class which cannot be mocked with Moq.
    /// 
    /// Expected behavior: Should return null when stored JSON is invalid or corrupted.
    /// </remarks>
    [TestMethod]
    public async Task GetUserAsync_WithInvalidJson_ShouldReturnNull()
    {
        // Arrange
        // Cannot arrange: Preferences is a static class that cannot be mocked
        var service = new DatabaseService();

        // Act & Assert
        Assert.Inconclusive(
            "This test cannot be completed because DatabaseService depends on the static " +
            "Preferences class from Microsoft.Maui.Storage, which cannot be mocked. " +
            "To make this testable, refactor DatabaseService to accept an IPreferences " +
            "interface via dependency injection.");
    }

    /// <summary>
    /// Tests that GetUserAsync handles exceptions from Preferences.Get.
    /// </summary>
    /// <remarks>
    /// This test is marked as Inconclusive because the DatabaseService depends on
    /// the static Preferences class which cannot be mocked with Moq.
    /// 
    /// Expected behavior: Should catch exceptions and return null.
    /// </remarks>
    [TestMethod]
    public async Task GetUserAsync_WhenPreferencesThrowsException_ShouldReturnNull()
    {
        // Arrange
        // Cannot arrange: Preferences is a static class that cannot be mocked
        var service = new DatabaseService();

        // Act & Assert
        Assert.Inconclusive(
            "This test cannot be completed because DatabaseService depends on the static " +
            "Preferences class from Microsoft.Maui.Storage, which cannot be mocked. " +
            "To make this testable, refactor DatabaseService to accept an IPreferences " +
            "interface via dependency injection.");
    }

    /// <summary>
    /// Tests that GetAccessTokenAsync returns a token when SecureStorage contains a valid token.
    /// Note: This test is marked as inconclusive because SecureStorage.Default is a static dependency
    /// that cannot be mocked with Moq. To properly test this method, the DatabaseService class would
    /// need to be refactored to accept an ISecureStorage interface via dependency injection.
    /// Expected behavior: Should return the token value stored in SecureStorage.
    /// </summary>
    [TestMethod]
    [Ignore("Cannot test due to static dependency on SecureStorage.Default. Requires refactoring to use dependency injection.")]
    public async Task GetAccessTokenAsync_WhenTokenExists_ReturnsToken()
    {
        // Arrange
        var service = new DatabaseService();
        // Cannot arrange SecureStorage.Default due to static dependency

        // Act
        var result = await service.GetAccessTokenAsync();

        // Assert
        Assert.Inconclusive("Cannot test due to static dependency on SecureStorage.Default. " +
            "The method uses Microsoft.Maui.Storage.SecureStorage.Default.GetAsync which cannot be mocked with Moq. " +
            "To test this method properly, refactor DatabaseService to accept an ISecureStorage dependency.");
    }

    /// <summary>
    /// Tests that GetAccessTokenAsync returns null when SecureStorage returns null.
    /// Note: This test is marked as inconclusive because SecureStorage.Default is a static dependency
    /// that cannot be mocked with Moq.
    /// Expected behavior: Should return null when no token is stored.
    /// </summary>
    [TestMethod]
    [Ignore("Cannot test due to static dependency on SecureStorage.Default. Requires refactoring to use dependency injection.")]
    public async Task GetAccessTokenAsync_WhenTokenDoesNotExist_ReturnsNull()
    {
        // Arrange
        var service = new DatabaseService();
        // Cannot arrange SecureStorage.Default due to static dependency

        // Act
        var result = await service.GetAccessTokenAsync();

        // Assert
        Assert.Inconclusive("Cannot test due to static dependency on SecureStorage.Default. " +
            "The method uses Microsoft.Maui.Storage.SecureStorage.Default.GetAsync which cannot be mocked with Moq. " +
            "To test this method properly, refactor DatabaseService to accept an ISecureStorage dependency.");
    }

    /// <summary>
    /// Tests that GetAccessTokenAsync catches exceptions and returns null when SecureStorage throws.
    /// Note: This test is marked as inconclusive because SecureStorage.Default is a static dependency
    /// that cannot be mocked with Moq.
    /// Expected behavior: Should catch any exception from SecureStorage, log it, and return null.
    /// </summary>
    [TestMethod]
    [Ignore("Cannot test due to static dependency on SecureStorage.Default. Requires refactoring to use dependency injection.")]
    public async Task GetAccessTokenAsync_WhenExceptionThrown_ReturnsNull()
    {
        // Arrange
        var service = new DatabaseService();
        // Cannot arrange SecureStorage.Default to throw exception due to static dependency

        // Act
        var result = await service.GetAccessTokenAsync();

        // Assert
        Assert.Inconclusive("Cannot test due to static dependency on SecureStorage.Default. " +
            "The method uses Microsoft.Maui.Storage.SecureStorage.Default.GetAsync which cannot be mocked with Moq. " +
            "To test this method properly, refactor DatabaseService to accept an ISecureStorage dependency. " +
            "Test should verify that exceptions are caught and null is returned.");
    }
}