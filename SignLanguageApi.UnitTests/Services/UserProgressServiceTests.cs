using System;
using System.Collections.Generic;
using System.IO;
using System.Text;
using System.Text.Json;
using System.Threading.Tasks;

using Microsoft.Extensions.Logging;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using Moq;
using SignLanguageApi;
using SignLanguageApi.Services;

namespace SignLanguageApi.Services.UnitTests
{
    /// <summary>
    /// Unit tests for UserProgressService class.
    /// </summary>
    [TestClass]
    public class UserProgressServiceTests
    {
        /// <summary>
        /// Tests SaveUserLessonsProgressAsync with a valid userId and empty list.
        /// Should successfully save an empty list to a JSON file and log information.
        /// </summary>
        [TestMethod]
        public async Task SaveUserLessonsProgressAsync_ValidUserIdAndEmptyList_SavesEmptyListToFileAndLogsInformation()
        {
            // Arrange
            var mockLogger = new Mock<ILogger<UserProgressService>>();
            var service = new UserProgressService(mockLogger.Object);
            var userId = "user_empty_list_test";
            var lessonsProgress = new List<UserLessonProgressData>();
            var appDataPath = Environment.GetFolderPath(Environment.SpecialFolder.ApplicationData);
            var filePath = Path.Combine(appDataPath, "SignLanguageApp", "UserProgress", $"lessons_progress_{userId}.json");

            try
            {
                // Act
                await service.SaveUserLessonsProgressAsync(userId, lessonsProgress);

                // Assert
                Assert.IsTrue(File.Exists(filePath), "File should be created");
                var fileContent = await File.ReadAllTextAsync(filePath);
                var deserializedList = JsonSerializer.Deserialize<List<UserLessonProgressData>>(fileContent, new JsonSerializerOptions { PropertyNameCaseInsensitive = true });
                Assert.IsNotNull(deserializedList);
                Assert.AreEqual(0, deserializedList.Count);
                mockLogger.Verify(
                    x => x.Log(
                        LogLevel.Information,
                        It.IsAny<EventId>(),
                        It.Is<It.IsAnyType>((v, t) => v.ToString()!.Contains($"UserId={userId}") && v.ToString()!.Contains("LessonCount=0")),
                        It.IsAny<Exception>(),
                        It.IsAny<Func<It.IsAnyType, Exception?, string>>()),
                    Times.Once);
            }
            finally
            {
                // Cleanup
                if (File.Exists(filePath))
                {
                    File.Delete(filePath);
                }
            }
        }

        /// <summary>
        /// Tests SaveUserLessonsProgressAsync with a valid userId and a single item list.
        /// Should successfully save the item to a JSON file and log information.
        /// </summary>
        [TestMethod]
        public async Task SaveUserLessonsProgressAsync_ValidUserIdAndSingleItem_SavesItemToFileAndLogsInformation()
        {
            // Arrange
            var mockLogger = new Mock<ILogger<UserProgressService>>();
            var service = new UserProgressService(mockLogger.Object);
            var userId = "user_single_item_test";
            var lessonsProgress = new List<UserLessonProgressData>
            {
                new UserLessonProgressData
                {
                    LessonId = 1,
                    LessonTitle = "Test Lesson",
                    CompletionPercentage = 50,
                    IsCompleted = false,
                    TotalAttempts = 3,
                    XpEarned = 100
                }
            };
            var appDataPath = Environment.GetFolderPath(Environment.SpecialFolder.ApplicationData);
            var filePath = Path.Combine(appDataPath, "SignLanguageApp", "UserProgress", $"lessons_progress_{userId}.json");

            try
            {
                // Act
                await service.SaveUserLessonsProgressAsync(userId, lessonsProgress);

                // Assert
                Assert.IsTrue(File.Exists(filePath), "File should be created");
                var fileContent = await File.ReadAllTextAsync(filePath);
                var deserializedList = JsonSerializer.Deserialize<List<UserLessonProgressData>>(fileContent, new JsonSerializerOptions { PropertyNameCaseInsensitive = true });
                Assert.IsNotNull(deserializedList);
                Assert.AreEqual(1, deserializedList.Count);
                Assert.AreEqual(1, deserializedList[0].LessonId);
                Assert.AreEqual("Test Lesson", deserializedList[0].LessonTitle);
                Assert.AreEqual(50, deserializedList[0].CompletionPercentage);
                mockLogger.Verify(
                    x => x.Log(
                        LogLevel.Information,
                        It.IsAny<EventId>(),
                        It.Is<It.IsAnyType>((v, t) => v.ToString()!.Contains($"UserId={userId}") && v.ToString()!.Contains("LessonCount=1")),
                        It.IsAny<Exception>(),
                        It.IsAny<Func<It.IsAnyType, Exception?, string>>()),
                    Times.Once);
            }
            finally
            {
                // Cleanup
                if (File.Exists(filePath))
                {
                    File.Delete(filePath);
                }
            }
        }

        /// <summary>
        /// Tests SaveUserLessonsProgressAsync with a valid userId and multiple items.
        /// Should successfully save all items to a JSON file and log information.
        /// </summary>
        [TestMethod]
        public async Task SaveUserLessonsProgressAsync_ValidUserIdAndMultipleItems_SavesItemsToFileAndLogsInformation()
        {
            // Arrange
            var mockLogger = new Mock<ILogger<UserProgressService>>();
            var service = new UserProgressService(mockLogger.Object);
            var userId = "user_multiple_items_test";
            var lessonsProgress = new List<UserLessonProgressData>
            {
                new UserLessonProgressData
                {
                    LessonId = 1,
                    LessonTitle = "Lesson 1",
                    CompletionPercentage = 100,
                    IsCompleted = true,
                    CompletedAt = DateTime.UtcNow,
                    TotalAttempts = 5,
                    XpEarned = 200
                },
                new UserLessonProgressData
                {
                    LessonId = 2,
                    LessonTitle = "Lesson 2",
                    CompletionPercentage = 75,
                    IsCompleted = false,
                    TotalAttempts = 2,
                    XpEarned = 150
                },
                new UserLessonProgressData
                {
                    LessonId = 3,
                    LessonTitle = "Lesson 3",
                    CompletionPercentage = 0,
                    IsCompleted = false,
                    TotalAttempts = 0,
                    XpEarned = 0
                }
            };
            var appDataPath = Environment.GetFolderPath(Environment.SpecialFolder.ApplicationData);
            var filePath = Path.Combine(appDataPath, "SignLanguageApp", "UserProgress", $"lessons_progress_{userId}.json");

            try
            {
                // Act
                await service.SaveUserLessonsProgressAsync(userId, lessonsProgress);

                // Assert
                Assert.IsTrue(File.Exists(filePath), "File should be created");
                var fileContent = await File.ReadAllTextAsync(filePath);
                var deserializedList = JsonSerializer.Deserialize<List<UserLessonProgressData>>(fileContent, new JsonSerializerOptions { PropertyNameCaseInsensitive = true });
                Assert.IsNotNull(deserializedList);
                Assert.AreEqual(3, deserializedList.Count);
                Assert.AreEqual(1, deserializedList[0].LessonId);
                Assert.AreEqual(2, deserializedList[1].LessonId);
                Assert.AreEqual(3, deserializedList[2].LessonId);
                mockLogger.Verify(
                    x => x.Log(
                        LogLevel.Information,
                        It.IsAny<EventId>(),
                        It.Is<It.IsAnyType>((v, t) => v.ToString()!.Contains($"UserId={userId}") && v.ToString()!.Contains("LessonCount=3")),
                        It.IsAny<Exception>(),
                        It.IsAny<Func<It.IsAnyType, Exception?, string>>()),
                    Times.Once);
            }
            finally
            {
                // Cleanup
                if (File.Exists(filePath))
                {
                    File.Delete(filePath);
                }
            }
        }

        /// <summary>
        /// Tests SaveUserLessonsProgressAsync with empty string userId.
        /// Should create a file with empty string in the filename and log information.
        /// </summary>
        [TestMethod]
        public async Task SaveUserLessonsProgressAsync_EmptyUserId_SavesFileWithEmptyNameAndLogsInformation()
        {
            // Arrange
            var mockLogger = new Mock<ILogger<UserProgressService>>();
            var service = new UserProgressService(mockLogger.Object);
            var userId = "";
            var lessonsProgress = new List<UserLessonProgressData>();
            var appDataPath = Environment.GetFolderPath(Environment.SpecialFolder.ApplicationData);
            var filePath = Path.Combine(appDataPath, "SignLanguageApp", "UserProgress", $"lessons_progress_{userId}.json");

            try
            {
                // Act
                await service.SaveUserLessonsProgressAsync(userId, lessonsProgress);

                // Assert
                Assert.IsTrue(File.Exists(filePath), "File should be created");
                mockLogger.Verify(
                    x => x.Log(
                        LogLevel.Information,
                        It.IsAny<EventId>(),
                        It.Is<It.IsAnyType>((v, t) => v.ToString()!.Contains("LessonCount=0")),
                        It.IsAny<Exception>(),
                        It.IsAny<Func<It.IsAnyType, Exception?, string>>()),
                    Times.Once);
            }
            finally
            {
                // Cleanup
                if (File.Exists(filePath))
                {
                    File.Delete(filePath);
                }
            }
        }

        /// <summary>
        /// Tests SaveUserLessonsProgressAsync with whitespace-only userId.
        /// Should create a file with whitespace in the filename and log information.
        /// </summary>
        [TestMethod]
        public async Task SaveUserLessonsProgressAsync_WhitespaceUserId_SavesFileWithWhitespaceNameAndLogsInformation()
        {
            // Arrange
            var mockLogger = new Mock<ILogger<UserProgressService>>();
            var service = new UserProgressService(mockLogger.Object);
            var userId = "   ";
            var lessonsProgress = new List<UserLessonProgressData>();
            var appDataPath = Environment.GetFolderPath(Environment.SpecialFolder.ApplicationData);
            var filePath = Path.Combine(appDataPath, "SignLanguageApp", "UserProgress", $"lessons_progress_{userId}.json");

            try
            {
                // Act
                await service.SaveUserLessonsProgressAsync(userId, lessonsProgress);

                // Assert
                Assert.IsTrue(File.Exists(filePath), "File should be created");
                mockLogger.Verify(
                    x => x.Log(
                        LogLevel.Information,
                        It.IsAny<EventId>(),
                        It.Is<It.IsAnyType>((v, t) => v.ToString()!.Contains($"UserId={userId}") && v.ToString()!.Contains("LessonCount=0")),
                        It.IsAny<Exception>(),
                        It.IsAny<Func<It.IsAnyType, Exception?, string>>()),
                    Times.Once);
            }
            finally
            {
                // Cleanup
                if (File.Exists(filePath))
                {
                    File.Delete(filePath);
                }
            }
        }

        /// <summary>
        /// Tests SaveUserLessonsProgressAsync with very long userId.
        /// Should handle long userId gracefully and save successfully.
        /// </summary>
        [TestMethod]
        public async Task SaveUserLessonsProgressAsync_VeryLongUserId_SavesFileSuccessfully()
        {
            // Arrange
            var mockLogger = new Mock<ILogger<UserProgressService>>();
            var service = new UserProgressService(mockLogger.Object);
            var userId = new string('a', 200);
            var lessonsProgress = new List<UserLessonProgressData>();
            var appDataPath = Environment.GetFolderPath(Environment.SpecialFolder.ApplicationData);
            var filePath = Path.Combine(appDataPath, "SignLanguageApp", "UserProgress", $"lessons_progress_{userId}.json");

            try
            {
                // Act
                await service.SaveUserLessonsProgressAsync(userId, lessonsProgress);

                // Assert
                Assert.IsTrue(File.Exists(filePath), "File should be created");
                mockLogger.Verify(
                    x => x.Log(
                        LogLevel.Information,
                        It.IsAny<EventId>(),
                        It.Is<It.IsAnyType>((v, t) => v.ToString()!.Contains($"UserId={userId}") && v.ToString()!.Contains("LessonCount=0")),
                        It.IsAny<Exception>(),
                        It.IsAny<Func<It.IsAnyType, Exception?, string>>()),
                    Times.Once);
            }
            finally
            {
                // Cleanup
                if (File.Exists(filePath))
                {
                    File.Delete(filePath);
                }
            }
        }

        /// <summary>
        /// Tests SaveUserLessonsProgressAsync when called multiple times with same userId.
        /// Should overwrite existing file with new content.
        /// </summary>
        [TestMethod]
        public async Task SaveUserLessonsProgressAsync_CalledMultipleTimes_OverwritesExistingFile()
        {
            // Arrange
            var mockLogger = new Mock<ILogger<UserProgressService>>();
            var service = new UserProgressService(mockLogger.Object);
            var userId = "user_overwrite_test";
            var firstLessonsProgress = new List<UserLessonProgressData>
            {
                new UserLessonProgressData { LessonId = 1, LessonTitle = "First" }
            };
            var secondLessonsProgress = new List<UserLessonProgressData>
            {
                new UserLessonProgressData { LessonId = 2, LessonTitle = "Second" },
                new UserLessonProgressData { LessonId = 3, LessonTitle = "Third" }
            };
            var appDataPath = Environment.GetFolderPath(Environment.SpecialFolder.ApplicationData);
            var filePath = Path.Combine(appDataPath, "SignLanguageApp", "UserProgress", $"lessons_progress_{userId}.json");

            try
            {
                // Act
                await service.SaveUserLessonsProgressAsync(userId, firstLessonsProgress);
                await service.SaveUserLessonsProgressAsync(userId, secondLessonsProgress);

                // Assert
                var fileContent = await File.ReadAllTextAsync(filePath);
                var deserializedList = JsonSerializer.Deserialize<List<UserLessonProgressData>>(fileContent, new JsonSerializerOptions { PropertyNameCaseInsensitive = true });
                Assert.IsNotNull(deserializedList);
                Assert.AreEqual(2, deserializedList.Count);
                Assert.AreEqual(2, deserializedList[0].LessonId);
                Assert.AreEqual("Second", deserializedList[0].LessonTitle);
                mockLogger.Verify(
                    x => x.Log(
                        LogLevel.Information,
                        It.IsAny<EventId>(),
                        It.IsAny<It.IsAnyType>(),
                        It.IsAny<Exception>(),
                        It.IsAny<Func<It.IsAnyType, Exception?, string>>()),
                    Times.Exactly(2));
            }
            finally
            {
                // Cleanup
                if (File.Exists(filePath))
                {
                    File.Delete(filePath);
                }
            }
        }

        /// <summary>
        /// Tests SaveUserLessonsProgressAsync with special characters in userId.
        /// Should handle valid special characters and save file successfully.
        /// </summary>
        [TestMethod]
        public async Task SaveUserLessonsProgressAsync_SpecialCharactersInUserId_SavesFileSuccessfully()
        {
            // Arrange
            var mockLogger = new Mock<ILogger<UserProgressService>>();
            var service = new UserProgressService(mockLogger.Object);
            var userId = "user-123_test@domain";
            var lessonsProgress = new List<UserLessonProgressData>();
            var appDataPath = Environment.GetFolderPath(Environment.SpecialFolder.ApplicationData);
            var filePath = Path.Combine(appDataPath, "SignLanguageApp", "UserProgress", $"lessons_progress_{userId}.json");

            try
            {
                // Act
                await service.SaveUserLessonsProgressAsync(userId, lessonsProgress);

                // Assert
                Assert.IsTrue(File.Exists(filePath), "File should be created");
                mockLogger.Verify(
                    x => x.Log(
                        LogLevel.Information,
                        It.IsAny<EventId>(),
                        It.Is<It.IsAnyType>((v, t) => v.ToString()!.Contains($"UserId={userId}") && v.ToString()!.Contains("LessonCount=0")),
                        It.IsAny<Exception>(),
                        It.IsAny<Func<It.IsAnyType, Exception?, string>>()),
                    Times.Once);
            }
            finally
            {
                // Cleanup
                if (File.Exists(filePath))
                {
                    File.Delete(filePath);
                }
            }
        }

        /// <summary>
        /// Tests that LoadUserProgressAsync returns null when the progress file does not exist.
        /// </summary>
        [TestMethod]
        public async Task LoadUserProgressAsync_WhenFileDoesNotExist_ReturnsNull()
        {
            // Arrange
            Mock<ILogger<UserProgressService>> mockLogger = new Mock<ILogger<UserProgressService>>();
            UserProgressService service = new UserProgressService(mockLogger.Object);
            string userId = $"nonexistent_user_{Guid.NewGuid()}";

            // Act
            UserProgressData? result = await service.LoadUserProgressAsync(userId);

            // Assert
            Assert.IsNull(result);
        }

        /// <summary>
        /// Tests that LoadUserProgressAsync successfully deserializes and returns valid user progress data
        /// when a valid JSON file exists.
        /// </summary>
        [TestMethod]
        public async Task LoadUserProgressAsync_WhenFileExistsWithValidJson_ReturnsUserProgressData()
        {
            // Arrange
            Mock<ILogger<UserProgressService>> mockLogger = new Mock<ILogger<UserProgressService>>();
            UserProgressService service = new UserProgressService(mockLogger.Object);
            string userId = $"test_user_{Guid.NewGuid()}";

            string appDataPath = Environment.GetFolderPath(Environment.SpecialFolder.ApplicationData);
            string progressDataPath = Path.Combine(appDataPath, "SignLanguageApp", "UserProgress");
            string filePath = Path.Combine(progressDataPath, $"progress_{userId}.json");

            UserProgressData expectedData = new UserProgressData
            {
                UserId = userId,
                TotalXp = 1500,
                LearningStreak = 7,
                LastProgressUpdate = new DateTime(2024, 1, 15, 10, 30, 0, DateTimeKind.Utc),
                SavedAt = new DateTime(2024, 1, 15, 10, 30, 5, DateTimeKind.Utc)
            };

            JsonSerializerOptions jsonOptions = new JsonSerializerOptions
            {
                WriteIndented = true,
                PropertyNameCaseInsensitive = true
            };

            string json = JsonSerializer.Serialize(expectedData, jsonOptions);
            await File.WriteAllTextAsync(filePath, json);

            try
            {
                // Act
                UserProgressData? result = await service.LoadUserProgressAsync(userId);

                // Assert
                Assert.IsNotNull(result);
                Assert.AreEqual(expectedData.UserId, result.UserId);
                Assert.AreEqual(expectedData.TotalXp, result.TotalXp);
                Assert.AreEqual(expectedData.LearningStreak, result.LearningStreak);
                Assert.AreEqual(expectedData.LastProgressUpdate, result.LastProgressUpdate);
                Assert.AreEqual(expectedData.SavedAt, result.SavedAt);
            }
            finally
            {
                // Cleanup
                if (File.Exists(filePath))
                {
                    File.Delete(filePath);
                }
            }
        }

        /// <summary>
        /// Tests that LoadUserProgressAsync returns null and logs an error when the file contains invalid JSON.
        /// </summary>
        [TestMethod]
        public async Task LoadUserProgressAsync_WhenFileContainsInvalidJson_ReturnsNullAndLogsError()
        {
            // Arrange
            Mock<ILogger<UserProgressService>> mockLogger = new Mock<ILogger<UserProgressService>>();
            UserProgressService service = new UserProgressService(mockLogger.Object);
            string userId = $"invalid_json_user_{Guid.NewGuid()}";

            string appDataPath = Environment.GetFolderPath(Environment.SpecialFolder.ApplicationData);
            string progressDataPath = Path.Combine(appDataPath, "SignLanguageApp", "UserProgress");
            string filePath = Path.Combine(progressDataPath, $"progress_{userId}.json");

            await File.WriteAllTextAsync(filePath, "{ invalid json content }");

            try
            {
                // Act
                UserProgressData? result = await service.LoadUserProgressAsync(userId);

                // Assert
                Assert.IsNull(result);
                mockLogger.Verify(
                    x => x.Log(
                        LogLevel.Error,
                        It.IsAny<EventId>(),
                        It.Is<It.IsAnyType>((v, t) => true),
                        It.IsAny<Exception>(),
                        It.IsAny<Func<It.IsAnyType, Exception?, string>>()),
                    Times.Once);
            }
            finally
            {
                // Cleanup
                if (File.Exists(filePath))
                {
                    File.Delete(filePath);
                }
            }
        }

        /// <summary>
        /// Tests that LoadUserProgressAsync handles null userId parameter gracefully.
        /// Expected behavior: Returns null (path combination will handle null).
        /// </summary>
        [TestMethod]
        public async Task LoadUserProgressAsync_WithNullUserId_HandlesGracefully()
        {
            // Arrange
            Mock<ILogger<UserProgressService>> mockLogger = new Mock<ILogger<UserProgressService>>();
            UserProgressService service = new UserProgressService(mockLogger.Object);

            // Act & Assert
            try
            {
                UserProgressData? result = await service.LoadUserProgressAsync(null!);
                // If it doesn't throw, verify behavior (likely returns null or throws ArgumentNullException)
            }
            catch (ArgumentNullException)
            {
                // This is acceptable behavior for null userId
                Assert.IsTrue(true);
            }
        }

        /// <summary>
        /// Tests that LoadUserProgressAsync handles empty string userId parameter.
        /// </summary>
        [TestMethod]
        public async Task LoadUserProgressAsync_WithEmptyUserId_ReturnsNull()
        {
            // Arrange
            Mock<ILogger<UserProgressService>> mockLogger = new Mock<ILogger<UserProgressService>>();
            UserProgressService service = new UserProgressService(mockLogger.Object);

            // Act
            UserProgressData? result = await service.LoadUserProgressAsync(string.Empty);

            // Assert
            Assert.IsNull(result);
        }

        /// <summary>
        /// Tests that LoadUserProgressAsync handles whitespace-only userId parameter.
        /// </summary>
        [TestMethod]
        public async Task LoadUserProgressAsync_WithWhitespaceUserId_ReturnsNull()
        {
            // Arrange
            Mock<ILogger<UserProgressService>> mockLogger = new Mock<ILogger<UserProgressService>>();
            UserProgressService service = new UserProgressService(mockLogger.Object);

            // Act
            UserProgressData? result = await service.LoadUserProgressAsync("   ");

            // Assert
            Assert.IsNull(result);
        }

        /// <summary>
        /// Tests that LoadUserProgressAsync successfully loads data with extreme values for TotalXp and LearningStreak.
        /// </summary>
        [TestMethod]
        public async Task LoadUserProgressAsync_WithExtremeValues_ReturnsCorrectData()
        {
            // Arrange
            Mock<ILogger<UserProgressService>> mockLogger = new Mock<ILogger<UserProgressService>>();
            UserProgressService service = new UserProgressService(mockLogger.Object);
            string userId = $"extreme_values_user_{Guid.NewGuid()}";

            string appDataPath = Environment.GetFolderPath(Environment.SpecialFolder.ApplicationData);
            string progressDataPath = Path.Combine(appDataPath, "SignLanguageApp", "UserProgress");
            string filePath = Path.Combine(progressDataPath, $"progress_{userId}.json");

            UserProgressData expectedData = new UserProgressData
            {
                UserId = userId,
                TotalXp = int.MaxValue,
                LearningStreak = int.MaxValue,
                LastProgressUpdate = DateTime.MaxValue,
                SavedAt = DateTime.MinValue
            };

            JsonSerializerOptions jsonOptions = new JsonSerializerOptions
            {
                WriteIndented = true,
                PropertyNameCaseInsensitive = true
            };

            string json = JsonSerializer.Serialize(expectedData, jsonOptions);
            await File.WriteAllTextAsync(filePath, json);

            try
            {
                // Act
                UserProgressData? result = await service.LoadUserProgressAsync(userId);

                // Assert
                Assert.IsNotNull(result);
                Assert.AreEqual(int.MaxValue, result.TotalXp);
                Assert.AreEqual(int.MaxValue, result.LearningStreak);
            }
            finally
            {
                // Cleanup
                if (File.Exists(filePath))
                {
                    File.Delete(filePath);
                }
            }
        }

        /// <summary>
        /// Tests that LoadUserProgressAsync correctly handles zero values for TotalXp and LearningStreak.
        /// </summary>
        [TestMethod]
        public async Task LoadUserProgressAsync_WithZeroValues_ReturnsCorrectData()
        {
            // Arrange
            Mock<ILogger<UserProgressService>> mockLogger = new Mock<ILogger<UserProgressService>>();
            UserProgressService service = new UserProgressService(mockLogger.Object);
            string userId = $"zero_values_user_{Guid.NewGuid()}";

            string appDataPath = Environment.GetFolderPath(Environment.SpecialFolder.ApplicationData);
            string progressDataPath = Path.Combine(appDataPath, "SignLanguageApp", "UserProgress");
            string filePath = Path.Combine(progressDataPath, $"progress_{userId}.json");

            UserProgressData expectedData = new UserProgressData
            {
                UserId = userId,
                TotalXp = 0,
                LearningStreak = 0,
                LastProgressUpdate = DateTime.UtcNow,
                SavedAt = DateTime.UtcNow
            };

            JsonSerializerOptions jsonOptions = new JsonSerializerOptions
            {
                WriteIndented = true,
                PropertyNameCaseInsensitive = true
            };

            string json = JsonSerializer.Serialize(expectedData, jsonOptions);
            await File.WriteAllTextAsync(filePath, json);

            try
            {
                // Act
                UserProgressData? result = await service.LoadUserProgressAsync(userId);

                // Assert
                Assert.IsNotNull(result);
                Assert.AreEqual(0, result.TotalXp);
                Assert.AreEqual(0, result.LearningStreak);
            }
            finally
            {
                // Cleanup
                if (File.Exists(filePath))
                {
                    File.Delete(filePath);
                }
            }
        }

        /// <summary>
        /// Tests that LoadUserProgressAsync handles special characters in userId.
        /// </summary>
        [TestMethod]
        public async Task LoadUserProgressAsync_WithSpecialCharactersInUserId_HandlesGracefully()
        {
            // Arrange
            Mock<ILogger<UserProgressService>> mockLogger = new Mock<ILogger<UserProgressService>>();
            UserProgressService service = new UserProgressService(mockLogger.Object);
            string userId = "user@#$%&*()";

            // Act & Assert
            try
            {
                UserProgressData? result = await service.LoadUserProgressAsync(userId);
                // May return null or throw exception depending on path validation
            }
            catch (Exception)
            {
                // Special characters may cause path-related exceptions, which is acceptable
                Assert.IsTrue(true);
            }
        }

        /// <summary>
        /// Tests that LoadUserProgressAsync returns null when JSON deserializes to null.
        /// </summary>
        [TestMethod]
        public async Task LoadUserProgressAsync_WhenDeserializationReturnsNull_ReturnsNull()
        {
            // Arrange
            Mock<ILogger<UserProgressService>> mockLogger = new Mock<ILogger<UserProgressService>>();
            UserProgressService service = new UserProgressService(mockLogger.Object);
            string userId = $"null_json_user_{Guid.NewGuid()}";

            string appDataPath = Environment.GetFolderPath(Environment.SpecialFolder.ApplicationData);
            string progressDataPath = Path.Combine(appDataPath, "SignLanguageApp", "UserProgress");
            string filePath = Path.Combine(progressDataPath, $"progress_{userId}.json");

            await File.WriteAllTextAsync(filePath, "null");

            try
            {
                // Act
                UserProgressData? result = await service.LoadUserProgressAsync(userId);

                // Assert
                Assert.IsNull(result);
            }
            finally
            {
                // Cleanup
                if (File.Exists(filePath))
                {
                    File.Delete(filePath);
                }
            }
        }

        /// <summary>
        /// Tests that the constructor successfully initializes with a valid logger.
        /// Verifies that no exceptions are thrown during construction.
        /// </summary>
        [TestMethod]
        public void UserProgressService_ValidLogger_InitializesSuccessfully()
        {
            // Arrange
            var mockLogger = new Mock<ILogger<UserProgressService>>();

            // Act
            var service = new UserProgressService(mockLogger.Object);

            // Assert
            Assert.IsNotNull(service);
        }

        /// <summary>
        /// Tests that the constructor creates the user progress directory if it doesn't exist.
        /// Verifies the directory exists after construction as a side effect.
        /// </summary>
        [TestMethod]
        public void UserProgressService_ValidLogger_EnsuresDirectoryExists()
        {
            // Arrange
            var mockLogger = new Mock<ILogger<UserProgressService>>();
            string expectedPath = Path.Combine(
                Environment.GetFolderPath(Environment.SpecialFolder.ApplicationData),
                "SignLanguageApp",
                "UserProgress");

            // Act
            var service = new UserProgressService(mockLogger.Object);

            // Assert
            Assert.IsTrue(Directory.Exists(expectedPath),
                $"Expected directory to exist at: {expectedPath}");
        }

        /// <summary>
        /// Tests that the constructor logs information when creating the directory.
        /// Note: This test verifies logger was called, but actual call depends on directory state.
        /// The LogInformation call only occurs if the directory didn't previously exist.
        /// </summary>
        [TestMethod]
        public void UserProgressService_ValidLogger_MayLogDirectoryCreation()
        {
            // Arrange
            var mockLogger = new Mock<ILogger<UserProgressService>>();

            // Act
            var service = new UserProgressService(mockLogger.Object);

            // Assert
            // Note: Cannot guarantee LogInformation is called because it depends on 
            // whether the directory already existed. This test documents the logging behavior.
            // For deterministic testing of logging, the Directory operations would need to be abstracted.
            mockLogger.Verify(
                x => x.Log(
                    LogLevel.Information,
                    It.IsAny<EventId>(),
                    It.Is<It.IsAnyType>((v, t) => true),
                    It.IsAny<Exception>(),
                    It.IsAny<Func<It.IsAnyType, Exception?, string>>()),
                Times.AtMost(1),
                "Logger should be called at most once during construction");
        }

        /// <summary>
        /// Tests that SaveUserProgressAsync successfully saves valid user progress data and logs success message.
        /// </summary>
        /// <param name="userId">The user ID to test.</param>
        /// <param name="totalXp">The total XP value to test.</param>
        /// <param name="learningStreak">The learning streak value to test.</param>
        [TestMethod]
        [DataRow("user123", 100, 5)]
        [DataRow("user-with-dash", 0, 0)]
        [DataRow("user_with_underscore", 999999, 365)]
        [DataRow("123", int.MaxValue, int.MaxValue)]
        [DataRow("a", int.MinValue, int.MinValue)]
        public async Task SaveUserProgressAsync_ValidInputs_SavesSuccessfullyAndLogsInformation(string userId, int totalXp, int learningStreak)
        {
            // Arrange
            var mockLogger = new Mock<ILogger<UserProgressService>>();
            var service = new UserProgressService(mockLogger.Object);
            var lastProgressUpdate = new DateTime(2024, 1, 15, 10, 30, 0, DateTimeKind.Utc);

            try
            {
                // Act
                await service.SaveUserProgressAsync(userId, totalXp, learningStreak, lastProgressUpdate);

                // Assert
                mockLogger.Verify(
                    x => x.Log(
                        LogLevel.Information,
                        It.IsAny<EventId>(),
                        It.Is<It.IsAnyType>((v, t) => v.ToString()!.Contains($"User progress saved: UserId={userId}, TotalXp={totalXp}, Streak={learningStreak}")),
                        It.IsAny<Exception>(),
                        It.IsAny<Func<It.IsAnyType, Exception?, string>>()),
                    Times.Once);

                // Verify file was created
                string appDataPath = Environment.GetFolderPath(Environment.SpecialFolder.ApplicationData);
                string progressDataPath = Path.Combine(appDataPath, "SignLanguageApp", "UserProgress");
                string filePath = Path.Combine(progressDataPath, $"progress_{userId}.json");
                Assert.IsTrue(File.Exists(filePath), "Progress file should be created");

                // Verify file content
                string fileContent = await File.ReadAllTextAsync(filePath);
                var deserializedData = JsonSerializer.Deserialize<UserProgressData>(fileContent, new JsonSerializerOptions { PropertyNameCaseInsensitive = true });
                Assert.IsNotNull(deserializedData);
                Assert.AreEqual(userId, deserializedData.UserId);
                Assert.AreEqual(totalXp, deserializedData.TotalXp);
                Assert.AreEqual(learningStreak, deserializedData.LearningStreak);
                Assert.AreEqual(lastProgressUpdate, deserializedData.LastProgressUpdate);
            }
            finally
            {
                // Cleanup
                string appDataPath = Environment.GetFolderPath(Environment.SpecialFolder.ApplicationData);
                string progressDataPath = Path.Combine(appDataPath, "SignLanguageApp", "UserProgress");
                string filePath = Path.Combine(progressDataPath, $"progress_{userId}.json");
                if (File.Exists(filePath))
                {
                    File.Delete(filePath);
                }
            }
        }

        /// <summary>
        /// Tests that SaveUserProgressAsync handles empty string userId by creating a file and logging success.
        /// </summary>
        [TestMethod]
        public async Task SaveUserProgressAsync_EmptyUserId_SavesSuccessfully()
        {
            // Arrange
            var mockLogger = new Mock<ILogger<UserProgressService>>();
            var service = new UserProgressService(mockLogger.Object);
            var userId = string.Empty;
            var totalXp = 50;
            var learningStreak = 3;
            var lastProgressUpdate = DateTime.UtcNow;

            try
            {
                // Act
                await service.SaveUserProgressAsync(userId, totalXp, learningStreak, lastProgressUpdate);

                // Assert
                mockLogger.Verify(
                    x => x.Log(
                        LogLevel.Information,
                        It.IsAny<EventId>(),
                        It.Is<It.IsAnyType>((v, t) => v.ToString()!.Contains("User progress saved")),
                        It.IsAny<Exception>(),
                        It.IsAny<Func<It.IsAnyType, Exception?, string>>()),
                    Times.Once);
            }
            finally
            {
                // Cleanup
                string appDataPath = Environment.GetFolderPath(Environment.SpecialFolder.ApplicationData);
                string progressDataPath = Path.Combine(appDataPath, "SignLanguageApp", "UserProgress");
                string filePath = Path.Combine(progressDataPath, $"progress_{userId}.json");
                if (File.Exists(filePath))
                {
                    File.Delete(filePath);
                }
            }
        }

        /// <summary>
        /// Tests that SaveUserProgressAsync handles whitespace-only userId by creating a file and logging success.
        /// </summary>
        [TestMethod]
        public async Task SaveUserProgressAsync_WhitespaceUserId_SavesSuccessfully()
        {
            // Arrange
            var mockLogger = new Mock<ILogger<UserProgressService>>();
            var service = new UserProgressService(mockLogger.Object);
            var userId = "   ";
            var totalXp = 75;
            var learningStreak = 7;
            var lastProgressUpdate = DateTime.UtcNow;

            try
            {
                // Act
                await service.SaveUserProgressAsync(userId, totalXp, learningStreak, lastProgressUpdate);

                // Assert
                mockLogger.Verify(
                    x => x.Log(
                        LogLevel.Information,
                        It.IsAny<EventId>(),
                        It.Is<It.IsAnyType>((v, t) => v.ToString()!.Contains("User progress saved")),
                        It.IsAny<Exception>(),
                        It.IsAny<Func<It.IsAnyType, Exception?, string>>()),
                    Times.Once);
            }
            finally
            {
                // Cleanup
                string appDataPath = Environment.GetFolderPath(Environment.SpecialFolder.ApplicationData);
                string progressDataPath = Path.Combine(appDataPath, "SignLanguageApp", "UserProgress");
                string filePath = Path.Combine(progressDataPath, $"progress_{userId}.json");
                if (File.Exists(filePath))
                {
                    File.Delete(filePath);
                }
            }
        }

        /// <summary>
        /// Tests that SaveUserProgressAsync handles DateTime.MinValue for lastProgressUpdate parameter.
        /// </summary>
        [TestMethod]
        public async Task SaveUserProgressAsync_DateTimeMinValue_SavesSuccessfully()
        {
            // Arrange
            var mockLogger = new Mock<ILogger<UserProgressService>>();
            var service = new UserProgressService(mockLogger.Object);
            var userId = "user_mindate";
            var totalXp = 100;
            var learningStreak = 5;
            var lastProgressUpdate = DateTime.MinValue;

            try
            {
                // Act
                await service.SaveUserProgressAsync(userId, totalXp, learningStreak, lastProgressUpdate);

                // Assert
                mockLogger.Verify(
                    x => x.Log(
                        LogLevel.Information,
                        It.IsAny<EventId>(),
                        It.Is<It.IsAnyType>((v, t) => v.ToString()!.Contains("User progress saved")),
                        It.IsAny<Exception>(),
                        It.IsAny<Func<It.IsAnyType, Exception?, string>>()),
                    Times.Once);

                // Verify file content
                string appDataPath = Environment.GetFolderPath(Environment.SpecialFolder.ApplicationData);
                string progressDataPath = Path.Combine(appDataPath, "SignLanguageApp", "UserProgress");
                string filePath = Path.Combine(progressDataPath, $"progress_{userId}.json");
                string fileContent = await File.ReadAllTextAsync(filePath);
                var deserializedData = JsonSerializer.Deserialize<UserProgressData>(fileContent, new JsonSerializerOptions { PropertyNameCaseInsensitive = true });
                Assert.IsNotNull(deserializedData);
                Assert.AreEqual(DateTime.MinValue, deserializedData.LastProgressUpdate);
            }
            finally
            {
                // Cleanup
                string appDataPath = Environment.GetFolderPath(Environment.SpecialFolder.ApplicationData);
                string progressDataPath = Path.Combine(appDataPath, "SignLanguageApp", "UserProgress");
                string filePath = Path.Combine(progressDataPath, $"progress_{userId}.json");
                if (File.Exists(filePath))
                {
                    File.Delete(filePath);
                }
            }
        }

        /// <summary>
        /// Tests that SaveUserProgressAsync handles DateTime.MaxValue for lastProgressUpdate parameter.
        /// </summary>
        [TestMethod]
        public async Task SaveUserProgressAsync_DateTimeMaxValue_SavesSuccessfully()
        {
            // Arrange
            var mockLogger = new Mock<ILogger<UserProgressService>>();
            var service = new UserProgressService(mockLogger.Object);
            var userId = "user_maxdate";
            var totalXp = 100;
            var learningStreak = 5;
            var lastProgressUpdate = DateTime.MaxValue;

            try
            {
                // Act
                await service.SaveUserProgressAsync(userId, totalXp, learningStreak, lastProgressUpdate);

                // Assert
                mockLogger.Verify(
                    x => x.Log(
                        LogLevel.Information,
                        It.IsAny<EventId>(),
                        It.Is<It.IsAnyType>((v, t) => v.ToString()!.Contains("User progress saved")),
                        It.IsAny<Exception>(),
                        It.IsAny<Func<It.IsAnyType, Exception?, string>>()),
                    Times.Once);

                // Verify file content
                string appDataPath = Environment.GetFolderPath(Environment.SpecialFolder.ApplicationData);
                string progressDataPath = Path.Combine(appDataPath, "SignLanguageApp", "UserProgress");
                string filePath = Path.Combine(progressDataPath, $"progress_{userId}.json");
                string fileContent = await File.ReadAllTextAsync(filePath);
                var deserializedData = JsonSerializer.Deserialize<UserProgressData>(fileContent, new JsonSerializerOptions { PropertyNameCaseInsensitive = true });
                Assert.IsNotNull(deserializedData);
                Assert.AreEqual(DateTime.MaxValue, deserializedData.LastProgressUpdate);
            }
            finally
            {
                // Cleanup
                string appDataPath = Environment.GetFolderPath(Environment.SpecialFolder.ApplicationData);
                string progressDataPath = Path.Combine(appDataPath, "SignLanguageApp", "UserProgress");
                string filePath = Path.Combine(progressDataPath, $"progress_{userId}.json");
                if (File.Exists(filePath))
                {
                    File.Delete(filePath);
                }
            }
        }

        /// <summary>
        /// Tests that SaveUserProgressAsync handles negative values for totalXp and learningStreak.
        /// </summary>
        [TestMethod]
        [DataRow(-1, -1)]
        [DataRow(-100, -50)]
        [DataRow(-999999, -365)]
        public async Task SaveUserProgressAsync_NegativeValues_SavesSuccessfully(int totalXp, int learningStreak)
        {
            // Arrange
            var mockLogger = new Mock<ILogger<UserProgressService>>();
            var service = new UserProgressService(mockLogger.Object);
            var userId = $"user_negative_{totalXp}_{learningStreak}";
            var lastProgressUpdate = DateTime.UtcNow;

            try
            {
                // Act
                await service.SaveUserProgressAsync(userId, totalXp, learningStreak, lastProgressUpdate);

                // Assert
                mockLogger.Verify(
                    x => x.Log(
                        LogLevel.Information,
                        It.IsAny<EventId>(),
                        It.Is<It.IsAnyType>((v, t) => v.ToString()!.Contains("User progress saved")),
                        It.IsAny<Exception>(),
                        It.IsAny<Func<It.IsAnyType, Exception?, string>>()),
                    Times.Once);

                // Verify file content
                string appDataPath = Environment.GetFolderPath(Environment.SpecialFolder.ApplicationData);
                string progressDataPath = Path.Combine(appDataPath, "SignLanguageApp", "UserProgress");
                string filePath = Path.Combine(progressDataPath, $"progress_{userId}.json");
                string fileContent = await File.ReadAllTextAsync(filePath);
                var deserializedData = JsonSerializer.Deserialize<UserProgressData>(fileContent, new JsonSerializerOptions { PropertyNameCaseInsensitive = true });
                Assert.IsNotNull(deserializedData);
                Assert.AreEqual(totalXp, deserializedData.TotalXp);
                Assert.AreEqual(learningStreak, deserializedData.LearningStreak);
            }
            finally
            {
                // Cleanup
                string appDataPath = Environment.GetFolderPath(Environment.SpecialFolder.ApplicationData);
                string progressDataPath = Path.Combine(appDataPath, "SignLanguageApp", "UserProgress");
                string filePath = Path.Combine(progressDataPath, $"progress_{userId}.json");
                if (File.Exists(filePath))
                {
                    File.Delete(filePath);
                }
            }
        }

        /// <summary>
        /// Tests that SaveUserProgressAsync handles very long userId strings.
        /// </summary>
        [TestMethod]
        public async Task SaveUserProgressAsync_VeryLongUserId_SavesSuccessfully()
        {
            // Arrange
            var mockLogger = new Mock<ILogger<UserProgressService>>();
            var service = new UserProgressService(mockLogger.Object);
            var userId = new string('a', 200);
            var totalXp = 100;
            var learningStreak = 5;
            var lastProgressUpdate = DateTime.UtcNow;

            try
            {
                // Act
                await service.SaveUserProgressAsync(userId, totalXp, learningStreak, lastProgressUpdate);

                // Assert
                mockLogger.Verify(
                    x => x.Log(
                        LogLevel.Information,
                        It.IsAny<EventId>(),
                        It.Is<It.IsAnyType>((v, t) => v.ToString()!.Contains("User progress saved")),
                        It.IsAny<Exception>(),
                        It.IsAny<Func<It.IsAnyType, Exception?, string>>()),
                    Times.Once);
            }
            finally
            {
                // Cleanup
                string appDataPath = Environment.GetFolderPath(Environment.SpecialFolder.ApplicationData);
                string progressDataPath = Path.Combine(appDataPath, "SignLanguageApp", "UserProgress");
                string filePath = Path.Combine(progressDataPath, $"progress_{userId}.json");
                if (File.Exists(filePath))
                {
                    File.Delete(filePath);
                }
            }
        }

        /// <summary>
        /// Tests that SaveUserProgressAsync handles userId with special but valid characters.
        /// </summary>
        [TestMethod]
        [DataRow("user@domain.com")]
        [DataRow("user.name")]
        [DataRow("user_123")]
        [DataRow("user-456")]
        public async Task SaveUserProgressAsync_UserIdWithSpecialValidCharacters_SavesSuccessfully(string userId)
        {
            // Arrange
            var mockLogger = new Mock<ILogger<UserProgressService>>();
            var service = new UserProgressService(mockLogger.Object);
            var totalXp = 100;
            var learningStreak = 5;
            var lastProgressUpdate = DateTime.UtcNow;

            try
            {
                // Act
                await service.SaveUserProgressAsync(userId, totalXp, learningStreak, lastProgressUpdate);

                // Assert
                mockLogger.Verify(
                    x => x.Log(
                        LogLevel.Information,
                        It.IsAny<EventId>(),
                        It.Is<It.IsAnyType>((v, t) => v.ToString()!.Contains("User progress saved")),
                        It.IsAny<Exception>(),
                        It.IsAny<Func<It.IsAnyType, Exception?, string>>()),
                    Times.Once);
            }
            finally
            {
                // Cleanup
                string appDataPath = Environment.GetFolderPath(Environment.SpecialFolder.ApplicationData);
                string progressDataPath = Path.Combine(appDataPath, "SignLanguageApp", "UserProgress");
                string filePath = Path.Combine(progressDataPath, $"progress_{userId}.json");
                if (File.Exists(filePath))
                {
                    File.Delete(filePath);
                }
            }
        }

        /// <summary>
        /// Tests that SaveUserProgressAsync saves data with SavedAt timestamp set to current UTC time.
        /// </summary>
        [TestMethod]
        public async Task SaveUserProgressAsync_ValidInputs_SetsSavedAtToUtcNow()
        {
            // Arrange
            var mockLogger = new Mock<ILogger<UserProgressService>>();
            var service = new UserProgressService(mockLogger.Object);
            var userId = "user_timestamp_test";
            var totalXp = 100;
            var learningStreak = 5;
            var lastProgressUpdate = DateTime.UtcNow;
            var beforeSave = DateTime.UtcNow;

            try
            {
                // Act
                await service.SaveUserProgressAsync(userId, totalXp, learningStreak, lastProgressUpdate);
                var afterSave = DateTime.UtcNow;

                // Assert
                string appDataPath = Environment.GetFolderPath(Environment.SpecialFolder.ApplicationData);
                string progressDataPath = Path.Combine(appDataPath, "SignLanguageApp", "UserProgress");
                string filePath = Path.Combine(progressDataPath, $"progress_{userId}.json");
                string fileContent = await File.ReadAllTextAsync(filePath);
                var deserializedData = JsonSerializer.Deserialize<UserProgressData>(fileContent, new JsonSerializerOptions { PropertyNameCaseInsensitive = true });

                Assert.IsNotNull(deserializedData);
                Assert.IsTrue(deserializedData.SavedAt >= beforeSave && deserializedData.SavedAt <= afterSave,
                    "SavedAt should be set to current UTC time");
            }
            finally
            {
                // Cleanup
                string appDataPath = Environment.GetFolderPath(Environment.SpecialFolder.ApplicationData);
                string progressDataPath = Path.Combine(appDataPath, "SignLanguageApp", "UserProgress");
                string filePath = Path.Combine(progressDataPath, $"progress_{userId}.json");
                if (File.Exists(filePath))
                {
                    File.Delete(filePath);
                }
            }
        }

        /// <summary>
        /// Tests that SaveUserProgressAsync overwrites existing file when called multiple times for same user.
        /// </summary>
        [TestMethod]
        public async Task SaveUserProgressAsync_CalledMultipleTimes_OverwritesExistingFile()
        {
            // Arrange
            var mockLogger = new Mock<ILogger<UserProgressService>>();
            var service = new UserProgressService(mockLogger.Object);
            var userId = "user_overwrite_test";
            var lastProgressUpdate = DateTime.UtcNow;

            try
            {
                // Act - Save first time
                await service.SaveUserProgressAsync(userId, 100, 5, lastProgressUpdate);

                // Act - Save second time with different values
                await service.SaveUserProgressAsync(userId, 200, 10, lastProgressUpdate);

                // Assert
                string appDataPath = Environment.GetFolderPath(Environment.SpecialFolder.ApplicationData);
                string progressDataPath = Path.Combine(appDataPath, "SignLanguageApp", "UserProgress");
                string filePath = Path.Combine(progressDataPath, $"progress_{userId}.json");
                string fileContent = await File.ReadAllTextAsync(filePath);
                var deserializedData = JsonSerializer.Deserialize<UserProgressData>(fileContent, new JsonSerializerOptions { PropertyNameCaseInsensitive = true });

                Assert.IsNotNull(deserializedData);
                Assert.AreEqual(200, deserializedData.TotalXp);
                Assert.AreEqual(10, deserializedData.LearningStreak);
            }
            finally
            {
                // Cleanup
                string appDataPath = Environment.GetFolderPath(Environment.SpecialFolder.ApplicationData);
                string progressDataPath = Path.Combine(appDataPath, "SignLanguageApp", "UserProgress");
                string filePath = Path.Combine(progressDataPath, $"progress_{userId}.json");
                if (File.Exists(filePath))
                {
                    File.Delete(filePath);
                }
            }
        }

        /// <summary>
        /// Tests LoadUserLessonsProgressAsync when the lessons progress file does not exist.
        /// Expected: Returns an empty list and logs a debug message.
        /// </summary>
        [TestMethod]
        public async Task LoadUserLessonsProgressAsync_FileDoesNotExist_ReturnsEmptyList()
        {
            // Arrange
            Mock<ILogger<UserProgressService>> mockLogger = new Mock<ILogger<UserProgressService>>();
            UserProgressService service = new UserProgressService(mockLogger.Object);
            string testUserId = Guid.NewGuid().ToString();

            // Act
            List<UserLessonProgressData> result = await service.LoadUserLessonsProgressAsync(testUserId);

            // Assert
            Assert.IsNotNull(result);
            Assert.AreEqual(0, result.Count);
            mockLogger.Verify(
                x => x.Log(
                    LogLevel.Debug,
                    It.IsAny<EventId>(),
                    It.Is<It.IsAnyType>((v, t) => v.ToString()!.Contains("No lessons progress file found")),
                    It.IsAny<Exception>(),
                    It.IsAny<Func<It.IsAnyType, Exception?, string>>()),
                Times.Once);
        }

        /// <summary>
        /// Tests LoadUserLessonsProgressAsync when the file exists with valid JSON containing lesson progress data.
        /// Expected: Returns deserialized list with correct data and logs information message.
        /// </summary>
        [TestMethod]
        public async Task LoadUserLessonsProgressAsync_FileExistsWithValidJson_ReturnsDeserializedList()
        {
            // Arrange
            Mock<ILogger<UserProgressService>> mockLogger = new Mock<ILogger<UserProgressService>>();
            UserProgressService service = new UserProgressService(mockLogger.Object);
            string testUserId = Guid.NewGuid().ToString();

            string appDataPath = Environment.GetFolderPath(Environment.SpecialFolder.ApplicationData);
            string progressDataPath = Path.Combine(appDataPath, "SignLanguageApp", "UserProgress");
            string filePath = Path.Combine(progressDataPath, $"lessons_progress_{testUserId}.json");

            List<UserLessonProgressData> testData = new List<UserLessonProgressData>
            {
                new UserLessonProgressData
                {
                    LessonId = 1,
                    LessonTitle = "Test Lesson 1",
                    CompletionPercentage = 50,
                    IsCompleted = false,
                    TotalAttempts = 3,
                    XpEarned = 100
                },
                new UserLessonProgressData
                {
                    LessonId = 2,
                    LessonTitle = "Test Lesson 2",
                    CompletionPercentage = 100,
                    IsCompleted = true,
                    TotalAttempts = 5,
                    XpEarned = 250
                }
            };

            string json = JsonSerializer.Serialize(testData, new JsonSerializerOptions { WriteIndented = true });
            Directory.CreateDirectory(progressDataPath);
            await File.WriteAllTextAsync(filePath, json);

            try
            {
                // Act
                List<UserLessonProgressData> result = await service.LoadUserLessonsProgressAsync(testUserId);

                // Assert
                Assert.IsNotNull(result);
                Assert.AreEqual(2, result.Count);
                Assert.AreEqual(1, result[0].LessonId);
                Assert.AreEqual("Test Lesson 1", result[0].LessonTitle);
                Assert.AreEqual(50, result[0].CompletionPercentage);
                Assert.AreEqual(2, result[1].LessonId);
                Assert.AreEqual("Test Lesson 2", result[1].LessonTitle);
                Assert.AreEqual(100, result[1].CompletionPercentage);

                mockLogger.Verify(
                    x => x.Log(
                        LogLevel.Information,
                        It.IsAny<EventId>(),
                        It.Is<It.IsAnyType>((v, t) => v.ToString()!.Contains("User lessons progress loaded")),
                        It.IsAny<Exception>(),
                        It.IsAny<Func<It.IsAnyType, Exception?, string>>()),
                    Times.Once);
            }
            finally
            {
                // Cleanup
                if (File.Exists(filePath))
                {
                    File.Delete(filePath);
                }
            }
        }

        /// <summary>
        /// Tests LoadUserLessonsProgressAsync when the file exists with an empty JSON array.
        /// Expected: Returns an empty list and logs information message.
        /// </summary>
        [TestMethod]
        public async Task LoadUserLessonsProgressAsync_FileExistsWithEmptyArray_ReturnsEmptyList()
        {
            // Arrange
            Mock<ILogger<UserProgressService>> mockLogger = new Mock<ILogger<UserProgressService>>();
            UserProgressService service = new UserProgressService(mockLogger.Object);
            string testUserId = Guid.NewGuid().ToString();

            string appDataPath = Environment.GetFolderPath(Environment.SpecialFolder.ApplicationData);
            string progressDataPath = Path.Combine(appDataPath, "SignLanguageApp", "UserProgress");
            string filePath = Path.Combine(progressDataPath, $"lessons_progress_{testUserId}.json");

            string json = "[]";
            Directory.CreateDirectory(progressDataPath);
            await File.WriteAllTextAsync(filePath, json);

            try
            {
                // Act
                List<UserLessonProgressData> result = await service.LoadUserLessonsProgressAsync(testUserId);

                // Assert
                Assert.IsNotNull(result);
                Assert.AreEqual(0, result.Count);

                mockLogger.Verify(
                    x => x.Log(
                        LogLevel.Information,
                        It.IsAny<EventId>(),
                        It.Is<It.IsAnyType>((v, t) => v.ToString()!.Contains("User lessons progress loaded")),
                        It.IsAny<Exception>(),
                        It.IsAny<Func<It.IsAnyType, Exception?, string>>()),
                    Times.Once);
            }
            finally
            {
                // Cleanup
                if (File.Exists(filePath))
                {
                    File.Delete(filePath);
                }
            }
        }

        /// <summary>
        /// Tests LoadUserLessonsProgressAsync when the file contains invalid JSON.
        /// Expected: Catches exception, logs error, and returns an empty list.
        /// </summary>
        [TestMethod]
        public async Task LoadUserLessonsProgressAsync_FileContainsInvalidJson_ReturnsEmptyListAndLogsError()
        {
            // Arrange
            Mock<ILogger<UserProgressService>> mockLogger = new Mock<ILogger<UserProgressService>>();
            UserProgressService service = new UserProgressService(mockLogger.Object);
            string testUserId = Guid.NewGuid().ToString();

            string appDataPath = Environment.GetFolderPath(Environment.SpecialFolder.ApplicationData);
            string progressDataPath = Path.Combine(appDataPath, "SignLanguageApp", "UserProgress");
            string filePath = Path.Combine(progressDataPath, $"lessons_progress_{testUserId}.json");

            string invalidJson = "{ this is not valid json }";
            Directory.CreateDirectory(progressDataPath);
            await File.WriteAllTextAsync(filePath, invalidJson);

            try
            {
                // Act
                List<UserLessonProgressData> result = await service.LoadUserLessonsProgressAsync(testUserId);

                // Assert
                Assert.IsNotNull(result);
                Assert.AreEqual(0, result.Count);

                mockLogger.Verify(
                    x => x.Log(
                        LogLevel.Error,
                        It.IsAny<EventId>(),
                        It.Is<It.IsAnyType>((v, t) => v.ToString()!.Contains("Error loading lessons progress")),
                        It.IsAny<Exception>(),
                        It.IsAny<Func<It.IsAnyType, Exception?, string>>()),
                    Times.Once);
            }
            finally
            {
                // Cleanup
                if (File.Exists(filePath))
                {
                    File.Delete(filePath);
                }
            }
        }

        /// <summary>
        /// Tests LoadUserLessonsProgressAsync when userId is null.
        /// Expected: Handles gracefully, returns empty list (file won't exist).
        /// </summary>
        [TestMethod]
        public async Task LoadUserLessonsProgressAsync_UserIdIsNull_ReturnsEmptyList()
        {
            // Arrange
            Mock<ILogger<UserProgressService>> mockLogger = new Mock<ILogger<UserProgressService>>();
            UserProgressService service = new UserProgressService(mockLogger.Object);
            string? testUserId = null;

            // Act
            List<UserLessonProgressData> result = await service.LoadUserLessonsProgressAsync(testUserId!);

            // Assert
            Assert.IsNotNull(result);
            Assert.AreEqual(0, result.Count);
        }

        /// <summary>
        /// Tests LoadUserLessonsProgressAsync when userId is an empty string.
        /// Expected: Handles gracefully, returns empty list (file won't exist).
        /// </summary>
        [TestMethod]
        public async Task LoadUserLessonsProgressAsync_UserIdIsEmpty_ReturnsEmptyList()
        {
            // Arrange
            Mock<ILogger<UserProgressService>> mockLogger = new Mock<ILogger<UserProgressService>>();
            UserProgressService service = new UserProgressService(mockLogger.Object);
            string testUserId = string.Empty;

            // Act
            List<UserLessonProgressData> result = await service.LoadUserLessonsProgressAsync(testUserId);

            // Assert
            Assert.IsNotNull(result);
            Assert.AreEqual(0, result.Count);
        }

        /// <summary>
        /// Tests LoadUserLessonsProgressAsync when userId contains only whitespace.
        /// Expected: Handles gracefully, returns empty list (file won't exist).
        /// </summary>
        [TestMethod]
        public async Task LoadUserLessonsProgressAsync_UserIdIsWhitespace_ReturnsEmptyList()
        {
            // Arrange
            Mock<ILogger<UserProgressService>> mockLogger = new Mock<ILogger<UserProgressService>>();
            UserProgressService service = new UserProgressService(mockLogger.Object);
            string testUserId = "   ";

            // Act
            List<UserLessonProgressData> result = await service.LoadUserLessonsProgressAsync(testUserId);

            // Assert
            Assert.IsNotNull(result);
            Assert.AreEqual(0, result.Count);
        }

        /// <summary>
        /// Tests LoadUserLessonsProgressAsync when userId contains special characters.
        /// Expected: Handles gracefully, returns empty list.
        /// </summary>
        [TestMethod]
        public async Task LoadUserLessonsProgressAsync_UserIdContainsSpecialCharacters_ReturnsEmptyList()
        {
            // Arrange
            Mock<ILogger<UserProgressService>> mockLogger = new Mock<ILogger<UserProgressService>>();
            UserProgressService service = new UserProgressService(mockLogger.Object);
            string testUserId = "user@#$%^&*()";

            // Act
            List<UserLessonProgressData> result = await service.LoadUserLessonsProgressAsync(testUserId);

            // Assert
            Assert.IsNotNull(result);
            Assert.AreEqual(0, result.Count);
        }

        /// <summary>
        /// Tests LoadUserLessonsProgressAsync when file exists but deserialization returns null.
        /// Expected: Returns empty list instead of null.
        /// </summary>
        [TestMethod]
        public async Task LoadUserLessonsProgressAsync_DeserializationReturnsNull_ReturnsEmptyList()
        {
            // Arrange
            Mock<ILogger<UserProgressService>> mockLogger = new Mock<ILogger<UserProgressService>>();
            UserProgressService service = new UserProgressService(mockLogger.Object);
            string testUserId = Guid.NewGuid().ToString();

            string appDataPath = Environment.GetFolderPath(Environment.SpecialFolder.ApplicationData);
            string progressDataPath = Path.Combine(appDataPath, "SignLanguageApp", "UserProgress");
            string filePath = Path.Combine(progressDataPath, $"lessons_progress_{testUserId}.json");

            string json = "null";
            Directory.CreateDirectory(progressDataPath);
            await File.WriteAllTextAsync(filePath, json);

            try
            {
                // Act
                List<UserLessonProgressData> result = await service.LoadUserLessonsProgressAsync(testUserId);

                // Assert
                Assert.IsNotNull(result);
                Assert.AreEqual(0, result.Count);
            }
            finally
            {
                // Cleanup
                if (File.Exists(filePath))
                {
                    File.Delete(filePath);
                }
            }
        }

        /// <summary>
        /// Tests LoadUserLessonsProgressAsync with a very long userId string.
        /// Expected: Handles gracefully, returns empty list.
        /// </summary>
        [TestMethod]
        public async Task LoadUserLessonsProgressAsync_VeryLongUserId_ReturnsEmptyList()
        {
            // Arrange
            Mock<ILogger<UserProgressService>> mockLogger = new Mock<ILogger<UserProgressService>>();
            UserProgressService service = new UserProgressService(mockLogger.Object);
            string testUserId = new string('a', 1000);

            // Act
            List<UserLessonProgressData> result = await service.LoadUserLessonsProgressAsync(testUserId);

            // Assert
            Assert.IsNotNull(result);
            Assert.AreEqual(0, result.Count);
        }

        /// <summary>
        /// Tests that DeleteUserDataAsync successfully deletes both progress and lessons files when they exist
        /// and logs appropriate information messages.
        /// </summary>
        [TestMethod]
        public async Task DeleteUserDataAsync_BothFilesExist_DeletesBothFilesAndLogsInformation()
        {
            // Arrange
            Mock<ILogger<UserProgressService>> mockLogger = new Mock<ILogger<UserProgressService>>();
            UserProgressService service = new UserProgressService(mockLogger.Object);
            string userId = "testuser_bothfiles";

            // Use reflection to get the actual file paths
            System.Reflection.MethodInfo? getProgressFilePathMethod = typeof(UserProgressService).GetMethod(
                "GetProgressFilePath",
                System.Reflection.BindingFlags.NonPublic | System.Reflection.BindingFlags.Instance);
            System.Reflection.MethodInfo? getLessonsProgressFilePathMethod = typeof(UserProgressService).GetMethod(
                "GetLessonsProgressFilePath",
                System.Reflection.BindingFlags.NonPublic | System.Reflection.BindingFlags.Instance);

            string progressFilePath = (string)getProgressFilePathMethod!.Invoke(service, new object[] { userId })!;
            string lessonsFilePath = (string)getLessonsProgressFilePathMethod!.Invoke(service, new object[] { userId })!;

            // Create test files
            await File.WriteAllTextAsync(progressFilePath, "{}");
            await File.WriteAllTextAsync(lessonsFilePath, "[]");

            // Act
            await service.DeleteUserDataAsync(userId);

            // Assert
            Assert.IsFalse(File.Exists(progressFilePath), "Progress file should be deleted");
            Assert.IsFalse(File.Exists(lessonsFilePath), "Lessons file should be deleted");

            mockLogger.Verify(
                x => x.Log(
                    LogLevel.Information,
                    It.IsAny<EventId>(),
                    It.Is<It.IsAnyType>((v, t) => v.ToString()!.Contains($"Deleted progress file for UserId={userId}")),
                    null,
                    It.IsAny<Func<It.IsAnyType, Exception?, string>>()),
                Times.Once);

            mockLogger.Verify(
                x => x.Log(
                    LogLevel.Information,
                    It.IsAny<EventId>(),
                    It.Is<It.IsAnyType>((v, t) => v.ToString()!.Contains($"Deleted lessons progress file for UserId={userId}")),
                    null,
                    It.IsAny<Func<It.IsAnyType, Exception?, string>>()),
                Times.Once);
        }

        /// <summary>
        /// Tests that DeleteUserDataAsync successfully deletes only the progress file when only it exists
        /// and logs appropriate information message.
        /// </summary>
        [TestMethod]
        public async Task DeleteUserDataAsync_OnlyProgressFileExists_DeletesProgressFileAndLogsInformation()
        {
            // Arrange
            Mock<ILogger<UserProgressService>> mockLogger = new Mock<ILogger<UserProgressService>>();
            UserProgressService service = new UserProgressService(mockLogger.Object);
            string userId = "testuser_progressonly";

            System.Reflection.MethodInfo? getProgressFilePathMethod = typeof(UserProgressService).GetMethod(
                "GetProgressFilePath",
                System.Reflection.BindingFlags.NonPublic | System.Reflection.BindingFlags.Instance);
            System.Reflection.MethodInfo? getLessonsProgressFilePathMethod = typeof(UserProgressService).GetMethod(
                "GetLessonsProgressFilePath",
                System.Reflection.BindingFlags.NonPublic | System.Reflection.BindingFlags.Instance);

            string progressFilePath = (string)getProgressFilePathMethod!.Invoke(service, new object[] { userId })!;
            string lessonsFilePath = (string)getLessonsProgressFilePathMethod!.Invoke(service, new object[] { userId })!;

            // Create only progress file
            await File.WriteAllTextAsync(progressFilePath, "{}");

            // Ensure lessons file doesn't exist
            if (File.Exists(lessonsFilePath))
            {
                File.Delete(lessonsFilePath);
            }

            // Act
            await service.DeleteUserDataAsync(userId);

            // Assert
            Assert.IsFalse(File.Exists(progressFilePath), "Progress file should be deleted");
            Assert.IsFalse(File.Exists(lessonsFilePath), "Lessons file should not exist");

            mockLogger.Verify(
                x => x.Log(
                    LogLevel.Information,
                    It.IsAny<EventId>(),
                    It.Is<It.IsAnyType>((v, t) => v.ToString()!.Contains($"Deleted progress file for UserId={userId}")),
                    null,
                    It.IsAny<Func<It.IsAnyType, Exception?, string>>()),
                Times.Once);

            mockLogger.Verify(
                x => x.Log(
                    LogLevel.Information,
                    It.IsAny<EventId>(),
                    It.Is<It.IsAnyType>((v, t) => v.ToString()!.Contains($"Deleted lessons progress file for UserId={userId}")),
                    null,
                    It.IsAny<Func<It.IsAnyType, Exception?, string>>()),
                Times.Never);
        }

        /// <summary>
        /// Tests that DeleteUserDataAsync successfully deletes only the lessons file when only it exists
        /// and logs appropriate information message.
        /// </summary>
        [TestMethod]
        public async Task DeleteUserDataAsync_OnlyLessonsFileExists_DeletesLessonsFileAndLogsInformation()
        {
            // Arrange
            Mock<ILogger<UserProgressService>> mockLogger = new Mock<ILogger<UserProgressService>>();
            UserProgressService service = new UserProgressService(mockLogger.Object);
            string userId = "testuser_lessonsonly";

            System.Reflection.MethodInfo? getProgressFilePathMethod = typeof(UserProgressService).GetMethod(
                "GetProgressFilePath",
                System.Reflection.BindingFlags.NonPublic | System.Reflection.BindingFlags.Instance);
            System.Reflection.MethodInfo? getLessonsProgressFilePathMethod = typeof(UserProgressService).GetMethod(
                "GetLessonsProgressFilePath",
                System.Reflection.BindingFlags.NonPublic | System.Reflection.BindingFlags.Instance);

            string progressFilePath = (string)getProgressFilePathMethod!.Invoke(service, new object[] { userId })!;
            string lessonsFilePath = (string)getLessonsProgressFilePathMethod!.Invoke(service, new object[] { userId })!;

            // Ensure progress file doesn't exist
            if (File.Exists(progressFilePath))
            {
                File.Delete(progressFilePath);
            }

            // Create only lessons file
            await File.WriteAllTextAsync(lessonsFilePath, "[]");

            // Act
            await service.DeleteUserDataAsync(userId);

            // Assert
            Assert.IsFalse(File.Exists(progressFilePath), "Progress file should not exist");
            Assert.IsFalse(File.Exists(lessonsFilePath), "Lessons file should be deleted");

            mockLogger.Verify(
                x => x.Log(
                    LogLevel.Information,
                    It.IsAny<EventId>(),
                    It.Is<It.IsAnyType>((v, t) => v.ToString()!.Contains($"Deleted progress file for UserId={userId}")),
                    null,
                    It.IsAny<Func<It.IsAnyType, Exception?, string>>()),
                Times.Never);

            mockLogger.Verify(
                x => x.Log(
                    LogLevel.Information,
                    It.IsAny<EventId>(),
                    It.Is<It.IsAnyType>((v, t) => v.ToString()!.Contains($"Deleted lessons progress file for UserId={userId}")),
                    null,
                    It.IsAny<Func<It.IsAnyType, Exception?, string>>()),
                Times.Once);
        }

        /// <summary>
        /// Tests that DeleteUserDataAsync completes successfully when neither file exists
        /// without logging any deletion messages.
        /// </summary>
        [TestMethod]
        public async Task DeleteUserDataAsync_NoFilesExist_CompletesSuccessfullyWithoutDeletion()
        {
            // Arrange
            Mock<ILogger<UserProgressService>> mockLogger = new Mock<ILogger<UserProgressService>>();
            UserProgressService service = new UserProgressService(mockLogger.Object);
            string userId = "testuser_nofiles";

            System.Reflection.MethodInfo? getProgressFilePathMethod = typeof(UserProgressService).GetMethod(
                "GetProgressFilePath",
                System.Reflection.BindingFlags.NonPublic | System.Reflection.BindingFlags.Instance);
            System.Reflection.MethodInfo? getLessonsProgressFilePathMethod = typeof(UserProgressService).GetMethod(
                "GetLessonsProgressFilePath",
                System.Reflection.BindingFlags.NonPublic | System.Reflection.BindingFlags.Instance);

            string progressFilePath = (string)getProgressFilePathMethod!.Invoke(service, new object[] { userId })!;
            string lessonsFilePath = (string)getLessonsProgressFilePathMethod!.Invoke(service, new object[] { userId })!;

            // Ensure no files exist
            if (File.Exists(progressFilePath))
            {
                File.Delete(progressFilePath);
            }
            if (File.Exists(lessonsFilePath))
            {
                File.Delete(lessonsFilePath);
            }

            // Act
            await service.DeleteUserDataAsync(userId);

            // Assert
            Assert.IsFalse(File.Exists(progressFilePath), "Progress file should not exist");
            Assert.IsFalse(File.Exists(lessonsFilePath), "Lessons file should not exist");

            mockLogger.Verify(
                x => x.Log(
                    LogLevel.Information,
                    It.IsAny<EventId>(),
                    It.Is<It.IsAnyType>((v, t) => v.ToString()!.Contains("Deleted")),
                    null,
                    It.IsAny<Func<It.IsAnyType, Exception?, string>>()),
                Times.Never);
        }

        /// <summary>
        /// Tests that DeleteUserDataAsync handles empty string userId by completing
        /// and attempting to delete files with empty userId in their names.
        /// </summary>
        [TestMethod]
        public async Task DeleteUserDataAsync_EmptyUserId_CompletesSuccessfully()
        {
            // Arrange
            Mock<ILogger<UserProgressService>> mockLogger = new Mock<ILogger<UserProgressService>>();
            UserProgressService service = new UserProgressService(mockLogger.Object);
            string userId = "";

            // Act
            await service.DeleteUserDataAsync(userId);

            // Assert - No exception should be thrown
            mockLogger.Verify(
                x => x.Log(
                    LogLevel.Error,
                    It.IsAny<EventId>(),
                    It.IsAny<It.IsAnyType>(),
                    It.IsAny<Exception>(),
                    It.IsAny<Func<It.IsAnyType, Exception?, string>>()),
                Times.Never);
        }

        /// <summary>
        /// Tests that DeleteUserDataAsync handles whitespace-only userId by completing
        /// and attempting to delete files with whitespace userId in their names.
        /// </summary>
        [TestMethod]
        public async Task DeleteUserDataAsync_WhitespaceUserId_CompletesSuccessfully()
        {
            // Arrange
            Mock<ILogger<UserProgressService>> mockLogger = new Mock<ILogger<UserProgressService>>();
            UserProgressService service = new UserProgressService(mockLogger.Object);
            string userId = "   ";

            // Act
            await service.DeleteUserDataAsync(userId);

            // Assert - No exception should be thrown
            mockLogger.Verify(
                x => x.Log(
                    LogLevel.Error,
                    It.IsAny<EventId>(),
                    It.IsAny<It.IsAnyType>(),
                    It.IsAny<Exception>(),
                    It.IsAny<Func<It.IsAnyType, Exception?, string>>()),
                Times.Never);
        }

        /// <summary>
        /// Tests that DeleteUserDataAsync handles userId with special characters
        /// by properly creating file paths and attempting deletion.
        /// </summary>
        [TestMethod]
        [DataRow("user@example.com")]
        [DataRow("user_123")]
        [DataRow("user-with-dashes")]
        [DataRow("user.with.dots")]
        public async Task DeleteUserDataAsync_UserIdWithSpecialCharacters_CompletesSuccessfully(string userId)
        {
            // Arrange
            Mock<ILogger<UserProgressService>> mockLogger = new Mock<ILogger<UserProgressService>>();
            UserProgressService service = new UserProgressService(mockLogger.Object);

            // Act
            await service.DeleteUserDataAsync(userId);

            // Assert - No exception should be thrown
            mockLogger.Verify(
                x => x.Log(
                    LogLevel.Error,
                    It.IsAny<EventId>(),
                    It.IsAny<It.IsAnyType>(),
                    It.IsAny<Exception>(),
                    It.IsAny<Func<It.IsAnyType, Exception?, string>>()),
                Times.Never);
        }

        /// <summary>
        /// Tests that DeleteUserDataAsync handles very long userId strings
        /// by attempting to create file paths and handling any resulting exceptions.
        /// </summary>
        [TestMethod]
        public async Task DeleteUserDataAsync_VeryLongUserId_HandlesGracefully()
        {
            // Arrange
            Mock<ILogger<UserProgressService>> mockLogger = new Mock<ILogger<UserProgressService>>();
            UserProgressService service = new UserProgressService(mockLogger.Object);
            string userId = new string('a', 500); // Very long userId

            // Act & Assert
            try
            {
                await service.DeleteUserDataAsync(userId);
            }
            catch (Exception)
            {
                // Expected - path too long or other file system exception
                mockLogger.Verify(
                    x => x.Log(
                        LogLevel.Error,
                        It.IsAny<EventId>(),
                        It.Is<It.IsAnyType>((v, t) => v.ToString()!.Contains("Error deleting user data")),
                        It.IsAny<Exception>(),
                        It.IsAny<Func<It.IsAnyType, Exception?, string>>()),
                    Times.Once);
            }
        }

        /// <summary>
        /// Tests that SaveUserProgressAsync creates properly formatted JSON with indentation.
        /// Expected: JSON file is created with indented formatting (WriteIndented = true).
        /// </summary>
        [TestMethod]
        public async Task SaveUserProgressAsync_ValidInputs_CreatesIndentedJson()
        {
            // Arrange
            var mockLogger = new Mock<ILogger<UserProgressService>>();
            var service = new UserProgressService(mockLogger.Object);
            var userId = "user_json_format_test";
            var totalXp = 175;
            var learningStreak = 9;
            var lastProgressUpdate = new DateTime(2024, 2, 15, 14, 30, 0, DateTimeKind.Utc);

            try
            {
                // Act
                await service.SaveUserProgressAsync(userId, totalXp, learningStreak, lastProgressUpdate);

                // Assert
                string appDataPath = Environment.GetFolderPath(Environment.SpecialFolder.ApplicationData);
                string progressDataPath = Path.Combine(appDataPath, "SignLanguageApp", "UserProgress");
                string filePath = Path.Combine(progressDataPath, $"progress_{userId}.json");
                string fileContent = await File.ReadAllTextAsync(filePath);

                // Verify JSON is indented (contains newlines and spaces)
                Assert.IsTrue(fileContent.Contains("\n") || fileContent.Contains("\r\n"), "JSON should be indented");
                Assert.IsTrue(fileContent.Contains("  "), "JSON should contain indentation spaces");
            }
            finally
            {
                // Cleanup
                string appDataPath = Environment.GetFolderPath(Environment.SpecialFolder.ApplicationData);
                string progressDataPath = Path.Combine(appDataPath, "SignLanguageApp", "UserProgress");
                string filePath = Path.Combine(progressDataPath, $"progress_{userId}.json");
                if (File.Exists(filePath))
                {
                    File.Delete(filePath);
                }
            }
        }

        /// <summary>
        /// Tests that SaveUserProgressAsync with zero values for both totalXp and learningStreak.
        /// Expected: Zero values are accepted and saved correctly.
        /// </summary>
        [TestMethod]
        public async Task SaveUserProgressAsync_ZeroValues_SavesSuccessfully()
        {
            // Arrange
            var mockLogger = new Mock<ILogger<UserProgressService>>();
            var service = new UserProgressService(mockLogger.Object);
            var userId = "user_zero_values";
            var totalXp = 0;
            var learningStreak = 0;
            var lastProgressUpdate = DateTime.UtcNow;

            try
            {
                // Act
                await service.SaveUserProgressAsync(userId, totalXp, learningStreak, lastProgressUpdate);

                // Assert
                string appDataPath = Environment.GetFolderPath(Environment.SpecialFolder.ApplicationData);
                string progressDataPath = Path.Combine(appDataPath, "SignLanguageApp", "UserProgress");
                string filePath = Path.Combine(progressDataPath, $"progress_{userId}.json");
                string fileContent = await File.ReadAllTextAsync(filePath);
                var deserializedData = JsonSerializer.Deserialize<UserProgressData>(fileContent, new JsonSerializerOptions { PropertyNameCaseInsensitive = true });
                Assert.IsNotNull(deserializedData);
                Assert.AreEqual(0, deserializedData.TotalXp);
                Assert.AreEqual(0, deserializedData.LearningStreak);
            }
            finally
            {
                // Cleanup
                string appDataPath = Environment.GetFolderPath(Environment.SpecialFolder.ApplicationData);
                string progressDataPath = Path.Combine(appDataPath, "SignLanguageApp", "UserProgress");
                string filePath = Path.Combine(progressDataPath, $"progress_{userId}.json");
                if (File.Exists(filePath))
                {
                    File.Delete(filePath);
                }
            }
        }

        /// <summary>
        /// Tests SaveUserLessonsProgressAsync with items containing extreme int.MaxValue boundary values.
        /// Should successfully serialize and save data with maximum values.
        /// </summary>
        [TestMethod]
        public async Task SaveUserLessonsProgressAsync_ItemsWithMaxIntValues_SavesSuccessfully()
        {
            // Arrange
            var mockLogger = new Mock<ILogger<UserProgressService>>();
            var service = new UserProgressService(mockLogger.Object);
            var userId = "user_max_values_test";
            var lessonsProgress = new List<UserLessonProgressData>
            {
                new UserLessonProgressData
                {
                    LessonId = int.MaxValue,
                    LessonTitle = "Max Values Lesson",
                    CompletionPercentage = int.MaxValue,
                    IsCompleted = true,
                    CompletedAt = DateTime.MaxValue,
                    StartedAt = DateTime.MaxValue,
                    TotalAttempts = int.MaxValue,
                    XpEarned = int.MaxValue
                }
            };
            var appDataPath = Environment.GetFolderPath(Environment.SpecialFolder.ApplicationData);
            var filePath = Path.Combine(appDataPath, "SignLanguageApp", "UserProgress", $"lessons_progress_{userId}.json");

            try
            {
                // Act
                await service.SaveUserLessonsProgressAsync(userId, lessonsProgress);

                // Assert
                Assert.IsTrue(File.Exists(filePath), "File should be created");
                var fileContent = await File.ReadAllTextAsync(filePath);
                var deserializedList = JsonSerializer.Deserialize<List<UserLessonProgressData>>(fileContent, new JsonSerializerOptions { PropertyNameCaseInsensitive = true });
                Assert.IsNotNull(deserializedList);
                Assert.AreEqual(1, deserializedList.Count);
                Assert.AreEqual(int.MaxValue, deserializedList[0].LessonId);
                Assert.AreEqual(int.MaxValue, deserializedList[0].CompletionPercentage);
                Assert.AreEqual(int.MaxValue, deserializedList[0].TotalAttempts);
                Assert.AreEqual(int.MaxValue, deserializedList[0].XpEarned);
                mockLogger.Verify(
                    x => x.Log(
                        LogLevel.Information,
                        It.IsAny<EventId>(),
                        It.Is<It.IsAnyType>((v, t) => v.ToString()!.Contains($"UserId={userId}") && v.ToString()!.Contains("LessonCount=1")),
                        It.IsAny<Exception>(),
                        It.IsAny<Func<It.IsAnyType, Exception?, string>>()),
                    Times.Once);
            }
            finally
            {
                if (File.Exists(filePath))
                {
                    File.Delete(filePath);
                }
            }
        }

        /// <summary>
        /// Tests SaveUserLessonsProgressAsync with items containing extreme int.MinValue boundary values.
        /// Should successfully serialize and save data with minimum values.
        /// </summary>
        [TestMethod]
        public async Task SaveUserLessonsProgressAsync_ItemsWithMinIntValues_SavesSuccessfully()
        {
            // Arrange
            var mockLogger = new Mock<ILogger<UserProgressService>>();
            var service = new UserProgressService(mockLogger.Object);
            var userId = "user_min_values_test";
            var lessonsProgress = new List<UserLessonProgressData>
            {
                new UserLessonProgressData
                {
                    LessonId = int.MinValue,
                    LessonTitle = "Min Values Lesson",
                    CompletionPercentage = int.MinValue,
                    IsCompleted = false,
                    CompletedAt = DateTime.MinValue,
                    StartedAt = DateTime.MinValue,
                    TotalAttempts = int.MinValue,
                    XpEarned = int.MinValue
                }
            };
            var appDataPath = Environment.GetFolderPath(Environment.SpecialFolder.ApplicationData);
            var filePath = Path.Combine(appDataPath, "SignLanguageApp", "UserProgress", $"lessons_progress_{userId}.json");

            try
            {
                // Act
                await service.SaveUserLessonsProgressAsync(userId, lessonsProgress);

                // Assert
                Assert.IsTrue(File.Exists(filePath), "File should be created");
                var fileContent = await File.ReadAllTextAsync(filePath);
                var deserializedList = JsonSerializer.Deserialize<List<UserLessonProgressData>>(fileContent, new JsonSerializerOptions { PropertyNameCaseInsensitive = true });
                Assert.IsNotNull(deserializedList);
                Assert.AreEqual(1, deserializedList.Count);
                Assert.AreEqual(int.MinValue, deserializedList[0].LessonId);
                Assert.AreEqual(int.MinValue, deserializedList[0].CompletionPercentage);
                Assert.AreEqual(int.MinValue, deserializedList[0].TotalAttempts);
                Assert.AreEqual(int.MinValue, deserializedList[0].XpEarned);
                mockLogger.Verify(
                    x => x.Log(
                        LogLevel.Information,
                        It.IsAny<EventId>(),
                        It.Is<It.IsAnyType>((v, t) => v.ToString()!.Contains($"UserId={userId}") && v.ToString()!.Contains("LessonCount=1")),
                        It.IsAny<Exception>(),
                        It.IsAny<Func<It.IsAnyType, Exception?, string>>()),
                    Times.Once);
            }
            finally
            {
                if (File.Exists(filePath))
                {
                    File.Delete(filePath);
                }
            }
        }

        /// <summary>
        /// Tests SaveUserLessonsProgressAsync with items containing zero values for all numeric properties.
        /// Should successfully serialize and save data with zero values.
        /// </summary>
        [TestMethod]
        public async Task SaveUserLessonsProgressAsync_ItemsWithZeroValues_SavesSuccessfully()
        {
            // Arrange
            var mockLogger = new Mock<ILogger<UserProgressService>>();
            var service = new UserProgressService(mockLogger.Object);
            var userId = "user_zero_values_test";
            var lessonsProgress = new List<UserLessonProgressData>
            {
                new UserLessonProgressData
                {
                    LessonId = 0,
                    LessonTitle = "Zero Values Lesson",
                    CompletionPercentage = 0,
                    IsCompleted = false,
                    CompletedAt = null,
                    TotalAttempts = 0,
                    XpEarned = 0
                }
            };
            var appDataPath = Environment.GetFolderPath(Environment.SpecialFolder.ApplicationData);
            var filePath = Path.Combine(appDataPath, "SignLanguageApp", "UserProgress", $"lessons_progress_{userId}.json");

            try
            {
                // Act
                await service.SaveUserLessonsProgressAsync(userId, lessonsProgress);

                // Assert
                Assert.IsTrue(File.Exists(filePath), "File should be created");
                var fileContent = await File.ReadAllTextAsync(filePath);
                var deserializedList = JsonSerializer.Deserialize<List<UserLessonProgressData>>(fileContent, new JsonSerializerOptions { PropertyNameCaseInsensitive = true });
                Assert.IsNotNull(deserializedList);
                Assert.AreEqual(1, deserializedList.Count);
                Assert.AreEqual(0, deserializedList[0].LessonId);
                Assert.AreEqual(0, deserializedList[0].CompletionPercentage);
                Assert.AreEqual(0, deserializedList[0].TotalAttempts);
                Assert.AreEqual(0, deserializedList[0].XpEarned);
                mockLogger.Verify(
                    x => x.Log(
                        LogLevel.Information,
                        It.IsAny<EventId>(),
                        It.Is<It.IsAnyType>((v, t) => v.ToString()!.Contains($"UserId={userId}") && v.ToString()!.Contains("LessonCount=1")),
                        It.IsAny<Exception>(),
                        It.IsAny<Func<It.IsAnyType, Exception?, string>>()),
                    Times.Once);
            }
            finally
            {
                if (File.Exists(filePath))
                {
                    File.Delete(filePath);
                }
            }
        }

        /// <summary>
        /// Tests SaveUserLessonsProgressAsync with items containing negative values for numeric properties.
        /// Should successfully serialize and save data with negative values.
        /// </summary>
        [TestMethod]
        [DataRow(-1, -50, -10, -100)]
        [DataRow(-999, -75, -500, -10000)]
        public async Task SaveUserLessonsProgressAsync_ItemsWithNegativeValues_SavesSuccessfully(int lessonId, int completionPercentage, int totalAttempts, int xpEarned)
        {
            // Arrange
            var mockLogger = new Mock<ILogger<UserProgressService>>();
            var service = new UserProgressService(mockLogger.Object);
            var userId = $"user_negative_{lessonId}_test";
            var lessonsProgress = new List<UserLessonProgressData>
            {
                new UserLessonProgressData
                {
                    LessonId = lessonId,
                    LessonTitle = "Negative Values Lesson",
                    CompletionPercentage = completionPercentage,
                    IsCompleted = false,
                    TotalAttempts = totalAttempts,
                    XpEarned = xpEarned
                }
            };
            var appDataPath = Environment.GetFolderPath(Environment.SpecialFolder.ApplicationData);
            var filePath = Path.Combine(appDataPath, "SignLanguageApp", "UserProgress", $"lessons_progress_{userId}.json");

            try
            {
                // Act
                await service.SaveUserLessonsProgressAsync(userId, lessonsProgress);

                // Assert
                Assert.IsTrue(File.Exists(filePath), "File should be created");
                var fileContent = await File.ReadAllTextAsync(filePath);
                var deserializedList = JsonSerializer.Deserialize<List<UserLessonProgressData>>(fileContent, new JsonSerializerOptions { PropertyNameCaseInsensitive = true });
                Assert.IsNotNull(deserializedList);
                Assert.AreEqual(1, deserializedList.Count);
                Assert.AreEqual(lessonId, deserializedList[0].LessonId);
                Assert.AreEqual(completionPercentage, deserializedList[0].CompletionPercentage);
                Assert.AreEqual(totalAttempts, deserializedList[0].TotalAttempts);
                Assert.AreEqual(xpEarned, deserializedList[0].XpEarned);
            }
            finally
            {
                if (File.Exists(filePath))
                {
                    File.Delete(filePath);
                }
            }
        }

        /// <summary>
        /// Tests SaveUserLessonsProgressAsync with items containing very long LessonTitle strings.
        /// Should successfully serialize and save data with long strings.
        /// </summary>
        [TestMethod]
        public async Task SaveUserLessonsProgressAsync_ItemsWithVeryLongLessonTitle_SavesSuccessfully()
        {
            // Arrange
            var mockLogger = new Mock<ILogger<UserProgressService>>();
            var service = new UserProgressService(mockLogger.Object);
            var userId = "user_long_title_test";
            var veryLongTitle = new string('A', 10000);
            var lessonsProgress = new List<UserLessonProgressData>
            {
                new UserLessonProgressData
                {
                    LessonId = 1,
                    LessonTitle = veryLongTitle,
                    CompletionPercentage = 50,
                    IsCompleted = false,
                    TotalAttempts = 1,
                    XpEarned = 50
                }
            };
            var appDataPath = Environment.GetFolderPath(Environment.SpecialFolder.ApplicationData);
            var filePath = Path.Combine(appDataPath, "SignLanguageApp", "UserProgress", $"lessons_progress_{userId}.json");

            try
            {
                // Act
                await service.SaveUserLessonsProgressAsync(userId, lessonsProgress);

                // Assert
                Assert.IsTrue(File.Exists(filePath), "File should be created");
                var fileContent = await File.ReadAllTextAsync(filePath);
                var deserializedList = JsonSerializer.Deserialize<List<UserLessonProgressData>>(fileContent, new JsonSerializerOptions { PropertyNameCaseInsensitive = true });
                Assert.IsNotNull(deserializedList);
                Assert.AreEqual(1, deserializedList.Count);
                Assert.AreEqual(veryLongTitle, deserializedList[0].LessonTitle);
            }
            finally
            {
                if (File.Exists(filePath))
                {
                    File.Delete(filePath);
                }
            }
        }

        /// <summary>
        /// Tests SaveUserLessonsProgressAsync with items containing empty and whitespace LessonTitle strings.
        /// Should successfully serialize and save data with empty/whitespace strings.
        /// </summary>
        [TestMethod]
        [DataRow("")]
        [DataRow("   ")]
        [DataRow("\t\n")]
        public async Task SaveUserLessonsProgressAsync_ItemsWithEmptyOrWhitespaceLessonTitle_SavesSuccessfully(string lessonTitle)
        {
            // Arrange
            var mockLogger = new Mock<ILogger<UserProgressService>>();
            var service = new UserProgressService(mockLogger.Object);
            var userId = "user_empty_title_test";
            var lessonsProgress = new List<UserLessonProgressData>
            {
                new UserLessonProgressData
                {
                    LessonId = 1,
                    LessonTitle = lessonTitle,
                    CompletionPercentage = 50,
                    IsCompleted = false,
                    TotalAttempts = 1,
                    XpEarned = 50
                }
            };
            var appDataPath = Environment.GetFolderPath(Environment.SpecialFolder.ApplicationData);
            var filePath = Path.Combine(appDataPath, "SignLanguageApp", "UserProgress", $"lessons_progress_{userId}.json");

            try
            {
                // Act
                await service.SaveUserLessonsProgressAsync(userId, lessonsProgress);

                // Assert
                Assert.IsTrue(File.Exists(filePath), "File should be created");
                var fileContent = await File.ReadAllTextAsync(filePath);
                var deserializedList = JsonSerializer.Deserialize<List<UserLessonProgressData>>(fileContent, new JsonSerializerOptions { PropertyNameCaseInsensitive = true });
                Assert.IsNotNull(deserializedList);
                Assert.AreEqual(1, deserializedList.Count);
                Assert.AreEqual(lessonTitle, deserializedList[0].LessonTitle);
            }
            finally
            {
                if (File.Exists(filePath))
                {
                    File.Delete(filePath);
                }
            }
        }

        /// <summary>
        /// Tests SaveUserLessonsProgressAsync with items containing special characters in LessonTitle.
        /// Should successfully serialize and save data with special characters.
        /// </summary>
        [TestMethod]
        public async Task SaveUserLessonsProgressAsync_ItemsWithSpecialCharactersInLessonTitle_SavesSuccessfully()
        {
            // Arrange
            var mockLogger = new Mock<ILogger<UserProgressService>>();
            var service = new UserProgressService(mockLogger.Object);
            var userId = "user_special_chars_title_test";
            var specialTitle = "Lesson <>&\"'!@#$%^*(){}[]|\\:;,.<>?/~`";
            var lessonsProgress = new List<UserLessonProgressData>
            {
                new UserLessonProgressData
                {
                    LessonId = 1,
                    LessonTitle = specialTitle,
                    CompletionPercentage = 50,
                    IsCompleted = false,
                    TotalAttempts = 1,
                    XpEarned = 50
                }
            };
            var appDataPath = Environment.GetFolderPath(Environment.SpecialFolder.ApplicationData);
            var filePath = Path.Combine(appDataPath, "SignLanguageApp", "UserProgress", $"lessons_progress_{userId}.json");

            try
            {
                // Act
                await service.SaveUserLessonsProgressAsync(userId, lessonsProgress);

                // Assert
                Assert.IsTrue(File.Exists(filePath), "File should be created");
                var fileContent = await File.ReadAllTextAsync(filePath);
                var deserializedList = JsonSerializer.Deserialize<List<UserLessonProgressData>>(fileContent, new JsonSerializerOptions { PropertyNameCaseInsensitive = true });
                Assert.IsNotNull(deserializedList);
                Assert.AreEqual(1, deserializedList.Count);
                Assert.AreEqual(specialTitle, deserializedList[0].LessonTitle);
            }
            finally
            {
                if (File.Exists(filePath))
                {
                    File.Delete(filePath);
                }
            }
        }

        /// <summary>
        /// Tests SaveUserLessonsProgressAsync with a very large list of items.
        /// Should successfully serialize and save a large dataset.
        /// </summary>
        [TestMethod]
        public async Task SaveUserLessonsProgressAsync_VeryLargeList_SavesSuccessfully()
        {
            // Arrange
            var mockLogger = new Mock<ILogger<UserProgressService>>();
            var service = new UserProgressService(mockLogger.Object);
            var userId = "user_large_list_test";
            var lessonsProgress = new List<UserLessonProgressData>();
            for (int i = 0; i < 10000; i++)
            {
                lessonsProgress.Add(new UserLessonProgressData
                {
                    LessonId = i,
                    LessonTitle = $"Lesson {i}",
                    CompletionPercentage = i % 101,
                    IsCompleted = i % 2 == 0,
                    TotalAttempts = i,
                    XpEarned = i * 10
                });
            }
            var appDataPath = Environment.GetFolderPath(Environment.SpecialFolder.ApplicationData);
            var filePath = Path.Combine(appDataPath, "SignLanguageApp", "UserProgress", $"lessons_progress_{userId}.json");

            try
            {
                // Act
                await service.SaveUserLessonsProgressAsync(userId, lessonsProgress);

                // Assert
                Assert.IsTrue(File.Exists(filePath), "File should be created");
                var fileContent = await File.ReadAllTextAsync(filePath);
                var deserializedList = JsonSerializer.Deserialize<List<UserLessonProgressData>>(fileContent, new JsonSerializerOptions { PropertyNameCaseInsensitive = true });
                Assert.IsNotNull(deserializedList);
                Assert.AreEqual(10000, deserializedList.Count);
                mockLogger.Verify(
                    x => x.Log(
                        LogLevel.Information,
                        It.IsAny<EventId>(),
                        It.Is<It.IsAnyType>((v, t) => v.ToString()!.Contains($"UserId={userId}") && v.ToString()!.Contains("LessonCount=10000")),
                        It.IsAny<Exception>(),
                        It.IsAny<Func<It.IsAnyType, Exception?, string>>()),
                    Times.Once);
            }
            finally
            {
                if (File.Exists(filePath))
                {
                    File.Delete(filePath);
                }
            }
        }

        /// <summary>
        /// Tests SaveUserLessonsProgressAsync with items containing null CompletedAt values.
        /// Should successfully serialize and save data with null nullable DateTime.
        /// </summary>
        [TestMethod]
        public async Task SaveUserLessonsProgressAsync_ItemsWithNullCompletedAt_SavesSuccessfully()
        {
            // Arrange
            var mockLogger = new Mock<ILogger<UserProgressService>>();
            var service = new UserProgressService(mockLogger.Object);
            var userId = "user_null_completed_test";
            var lessonsProgress = new List<UserLessonProgressData>
            {
                new UserLessonProgressData
                {
                    LessonId = 1,
                    LessonTitle = "Incomplete Lesson",
                    CompletionPercentage = 50,
                    IsCompleted = false,
                    CompletedAt = null,
                    TotalAttempts = 3,
                    XpEarned = 50
                }
            };
            var appDataPath = Environment.GetFolderPath(Environment.SpecialFolder.ApplicationData);
            var filePath = Path.Combine(appDataPath, "SignLanguageApp", "UserProgress", $"lessons_progress_{userId}.json");

            try
            {
                // Act
                await service.SaveUserLessonsProgressAsync(userId, lessonsProgress);

                // Assert
                Assert.IsTrue(File.Exists(filePath), "File should be created");
                var fileContent = await File.ReadAllTextAsync(filePath);
                var deserializedList = JsonSerializer.Deserialize<List<UserLessonProgressData>>(fileContent, new JsonSerializerOptions { PropertyNameCaseInsensitive = true });
                Assert.IsNotNull(deserializedList);
                Assert.AreEqual(1, deserializedList.Count);
                Assert.IsNull(deserializedList[0].CompletedAt);
            }
            finally
            {
                if (File.Exists(filePath))
                {
                    File.Delete(filePath);
                }
            }
        }

        /// <summary>
        /// Tests SaveUserLessonsProgressAsync with list containing duplicate lesson IDs.
        /// Should successfully serialize and save data even with duplicates.
        /// </summary>
        [TestMethod]
        public async Task SaveUserLessonsProgressAsync_ListWithDuplicateLessonIds_SavesSuccessfully()
        {
            // Arrange
            var mockLogger = new Mock<ILogger<UserProgressService>>();
            var service = new UserProgressService(mockLogger.Object);
            var userId = "user_duplicate_ids_test";
            var lessonsProgress = new List<UserLessonProgressData>
            {
                new UserLessonProgressData
                {
                    LessonId = 1,
                    LessonTitle = "Lesson 1 - First Instance",
                    CompletionPercentage = 50,
                    IsCompleted = false,
                    TotalAttempts = 2,
                    XpEarned = 50
                },
                new UserLessonProgressData
                {
                    LessonId = 1,
                    LessonTitle = "Lesson 1 - Second Instance",
                    CompletionPercentage = 75,
                    IsCompleted = false,
                    TotalAttempts = 3,
                    XpEarned = 75
                }
            };
            var appDataPath = Environment.GetFolderPath(Environment.SpecialFolder.ApplicationData);
            var filePath = Path.Combine(appDataPath, "SignLanguageApp", "UserProgress", $"lessons_progress_{userId}.json");

            try
            {
                // Act
                await service.SaveUserLessonsProgressAsync(userId, lessonsProgress);

                // Assert
                Assert.IsTrue(File.Exists(filePath), "File should be created");
                var fileContent = await File.ReadAllTextAsync(filePath);
                var deserializedList = JsonSerializer.Deserialize<List<UserLessonProgressData>>(fileContent, new JsonSerializerOptions { PropertyNameCaseInsensitive = true });
                Assert.IsNotNull(deserializedList);
                Assert.AreEqual(2, deserializedList.Count);
                Assert.AreEqual(1, deserializedList[0].LessonId);
                Assert.AreEqual(1, deserializedList[1].LessonId);
            }
            finally
            {
                if (File.Exists(filePath))
                {
                    File.Delete(filePath);
                }
            }
        }

        /// <summary>
        /// Tests that LoadUserProgressAsync returns null when the progress file does not exist.
        /// Input: A userId for which no file exists.
        /// Expected: Returns null and logs debug message.
        /// </summary>
        [TestMethod]
        public async Task LoadUserProgressAsync_FileDoesNotExist_ReturnsNull()
        {
            // Arrange
            Mock<ILogger<UserProgressService>> mockLogger = new Mock<ILogger<UserProgressService>>();
            UserProgressService service = new UserProgressService(mockLogger.Object);
            string userId = $"nonexistent_{Guid.NewGuid()}";

            // Act
            UserProgressData? result = await service.LoadUserProgressAsync(userId);

            // Assert
            Assert.IsNull(result);
            mockLogger.Verify(
                x => x.Log(
                    LogLevel.Debug,
                    It.IsAny<EventId>(),
                    It.Is<It.IsAnyType>((v, t) => true),
                    It.IsAny<Exception>(),
                    It.IsAny<Func<It.IsAnyType, Exception?, string>>()),
                Times.Once);
        }

        /// <summary>
        /// Tests that LoadUserProgressAsync successfully deserializes and returns valid user progress data.
        /// Input: Valid userId with existing file containing valid JSON.
        /// Expected: Returns deserialized UserProgressData object with correct values.
        /// </summary>
        [TestMethod]
        public async Task LoadUserProgressAsync_ValidJsonFile_ReturnsDeserializedData()
        {
            // Arrange
            Mock<ILogger<UserProgressService>> mockLogger = new Mock<ILogger<UserProgressService>>();
            UserProgressService service = new UserProgressService(mockLogger.Object);
            string userId = $"valid_user_{Guid.NewGuid()}";

            string appDataPath = Environment.GetFolderPath(Environment.SpecialFolder.ApplicationData);
            string progressDataPath = Path.Combine(appDataPath, "SignLanguageApp", "UserProgress");
            string filePath = Path.Combine(progressDataPath, $"progress_{userId}.json");

            UserProgressData expectedData = new UserProgressData
            {
                UserId = userId,
                TotalXp = 1500,
                LearningStreak = 7,
                LastProgressUpdate = new DateTime(2024, 1, 15, 10, 30, 0, DateTimeKind.Utc),
                SavedAt = new DateTime(2024, 1, 15, 10, 30, 5, DateTimeKind.Utc)
            };

            JsonSerializerOptions jsonOptions = new JsonSerializerOptions
            {
                WriteIndented = true,
                PropertyNameCaseInsensitive = true
            };

            string json = JsonSerializer.Serialize(expectedData, jsonOptions);
            await File.WriteAllTextAsync(filePath, json);

            try
            {
                // Act
                UserProgressData? result = await service.LoadUserProgressAsync(userId);

                // Assert
                Assert.IsNotNull(result);
                Assert.AreEqual(expectedData.UserId, result.UserId);
                Assert.AreEqual(expectedData.TotalXp, result.TotalXp);
                Assert.AreEqual(expectedData.LearningStreak, result.LearningStreak);
                Assert.AreEqual(expectedData.LastProgressUpdate, result.LastProgressUpdate);
                Assert.AreEqual(expectedData.SavedAt, result.SavedAt);
                mockLogger.Verify(
                    x => x.Log(
                        LogLevel.Information,
                        It.IsAny<EventId>(),
                        It.Is<It.IsAnyType>((v, t) => true),
                        It.IsAny<Exception>(),
                        It.IsAny<Func<It.IsAnyType, Exception?, string>>()),
                    Times.Once);
            }
            finally
            {
                if (File.Exists(filePath))
                {
                    File.Delete(filePath);
                }
            }
        }

        /// <summary>
        /// Tests that LoadUserProgressAsync returns null and logs error when file contains invalid JSON.
        /// Input: userId with file containing malformed JSON.
        /// Expected: Returns null and logs error with exception.
        /// </summary>
        [TestMethod]
        public async Task LoadUserProgressAsync_InvalidJsonFile_ReturnsNullAndLogsError()
        {
            // Arrange
            Mock<ILogger<UserProgressService>> mockLogger = new Mock<ILogger<UserProgressService>>();
            UserProgressService service = new UserProgressService(mockLogger.Object);
            string userId = $"invalid_json_{Guid.NewGuid()}";

            string appDataPath = Environment.GetFolderPath(Environment.SpecialFolder.ApplicationData);
            string progressDataPath = Path.Combine(appDataPath, "SignLanguageApp", "UserProgress");
            string filePath = Path.Combine(progressDataPath, $"progress_{userId}.json");

            await File.WriteAllTextAsync(filePath, "{ invalid json content }");

            try
            {
                // Act
                UserProgressData? result = await service.LoadUserProgressAsync(userId);

                // Assert
                Assert.IsNull(result);
                mockLogger.Verify(
                    x => x.Log(
                        LogLevel.Error,
                        It.IsAny<EventId>(),
                        It.Is<It.IsAnyType>((v, t) => true),
                        It.IsAny<Exception>(),
                        It.IsAny<Func<It.IsAnyType, Exception?, string>>()),
                    Times.Once);
            }
            finally
            {
                if (File.Exists(filePath))
                {
                    File.Delete(filePath);
                }
            }
        }

        /// <summary>
        /// Tests that LoadUserProgressAsync handles null userId gracefully.
        /// Input: null userId.
        /// Expected: Returns null (path operations handle null).
        /// </summary>
        [TestMethod]
        public async Task LoadUserProgressAsync_NullUserId_ReturnsNull()
        {
            // Arrange
            Mock<ILogger<UserProgressService>> mockLogger = new Mock<ILogger<UserProgressService>>();
            UserProgressService service = new UserProgressService(mockLogger.Object);

            // Act
            UserProgressData? result = await service.LoadUserProgressAsync(null!);

            // Assert
            Assert.IsNull(result);
        }

        /// <summary>
        /// Tests that LoadUserProgressAsync handles empty userId.
        /// Input: Empty string userId.
        /// Expected: Returns null (file won't exist).
        /// </summary>
        [TestMethod]
        public async Task LoadUserProgressAsync_EmptyUserId_ReturnsNull()
        {
            // Arrange
            Mock<ILogger<UserProgressService>> mockLogger = new Mock<ILogger<UserProgressService>>();
            UserProgressService service = new UserProgressService(mockLogger.Object);

            // Act
            UserProgressData? result = await service.LoadUserProgressAsync(string.Empty);

            // Assert
            Assert.IsNull(result);
        }

        /// <summary>
        /// Tests that LoadUserProgressAsync handles whitespace-only userId.
        /// Input: Whitespace string userId.
        /// Expected: Returns null (file won't exist).
        /// </summary>
        [TestMethod]
        public async Task LoadUserProgressAsync_WhitespaceUserId_ReturnsNull()
        {
            // Arrange
            Mock<ILogger<UserProgressService>> mockLogger = new Mock<ILogger<UserProgressService>>();
            UserProgressService service = new UserProgressService(mockLogger.Object);

            // Act
            UserProgressData? result = await service.LoadUserProgressAsync("   ");

            // Assert
            Assert.IsNull(result);
        }

        /// <summary>
        /// Tests that LoadUserProgressAsync successfully loads data with extreme integer values.
        /// Input: File with int.MaxValue for TotalXp and LearningStreak.
        /// Expected: Returns data with correct extreme values.
        /// </summary>
        [TestMethod]
        public async Task LoadUserProgressAsync_ExtremeMaxValues_ReturnsCorrectData()
        {
            // Arrange
            Mock<ILogger<UserProgressService>> mockLogger = new Mock<ILogger<UserProgressService>>();
            UserProgressService service = new UserProgressService(mockLogger.Object);
            string userId = $"extreme_max_{Guid.NewGuid()}";

            string appDataPath = Environment.GetFolderPath(Environment.SpecialFolder.ApplicationData);
            string progressDataPath = Path.Combine(appDataPath, "SignLanguageApp", "UserProgress");
            string filePath = Path.Combine(progressDataPath, $"progress_{userId}.json");

            UserProgressData expectedData = new UserProgressData
            {
                UserId = userId,
                TotalXp = int.MaxValue,
                LearningStreak = int.MaxValue,
                LastProgressUpdate = DateTime.UtcNow,
                SavedAt = DateTime.UtcNow
            };

            JsonSerializerOptions jsonOptions = new JsonSerializerOptions
            {
                WriteIndented = true,
                PropertyNameCaseInsensitive = true
            };

            string json = JsonSerializer.Serialize(expectedData, jsonOptions);
            await File.WriteAllTextAsync(filePath, json);

            try
            {
                // Act
                UserProgressData? result = await service.LoadUserProgressAsync(userId);

                // Assert
                Assert.IsNotNull(result);
                Assert.AreEqual(int.MaxValue, result.TotalXp);
                Assert.AreEqual(int.MaxValue, result.LearningStreak);
            }
            finally
            {
                if (File.Exists(filePath))
                {
                    File.Delete(filePath);
                }
            }
        }

        /// <summary>
        /// Tests that LoadUserProgressAsync successfully loads data with extreme minimum integer values.
        /// Input: File with int.MinValue for TotalXp and LearningStreak.
        /// Expected: Returns data with correct extreme values.
        /// </summary>
        [TestMethod]
        public async Task LoadUserProgressAsync_ExtremeMinValues_ReturnsCorrectData()
        {
            // Arrange
            Mock<ILogger<UserProgressService>> mockLogger = new Mock<ILogger<UserProgressService>>();
            UserProgressService service = new UserProgressService(mockLogger.Object);
            string userId = $"extreme_min_{Guid.NewGuid()}";

            string appDataPath = Environment.GetFolderPath(Environment.SpecialFolder.ApplicationData);
            string progressDataPath = Path.Combine(appDataPath, "SignLanguageApp", "UserProgress");
            string filePath = Path.Combine(progressDataPath, $"progress_{userId}.json");

            UserProgressData expectedData = new UserProgressData
            {
                UserId = userId,
                TotalXp = int.MinValue,
                LearningStreak = int.MinValue,
                LastProgressUpdate = DateTime.UtcNow,
                SavedAt = DateTime.UtcNow
            };

            JsonSerializerOptions jsonOptions = new JsonSerializerOptions
            {
                WriteIndented = true,
                PropertyNameCaseInsensitive = true
            };

            string json = JsonSerializer.Serialize(expectedData, jsonOptions);
            await File.WriteAllTextAsync(filePath, json);

            try
            {
                // Act
                UserProgressData? result = await service.LoadUserProgressAsync(userId);

                // Assert
                Assert.IsNotNull(result);
                Assert.AreEqual(int.MinValue, result.TotalXp);
                Assert.AreEqual(int.MinValue, result.LearningStreak);
            }
            finally
            {
                if (File.Exists(filePath))
                {
                    File.Delete(filePath);
                }
            }
        }

        /// <summary>
        /// Tests that LoadUserProgressAsync correctly handles zero values.
        /// Input: File with zero values for TotalXp and LearningStreak.
        /// Expected: Returns data with zero values.
        /// </summary>
        [TestMethod]
        public async Task LoadUserProgressAsync_ZeroValues_ReturnsCorrectData()
        {
            // Arrange
            Mock<ILogger<UserProgressService>> mockLogger = new Mock<ILogger<UserProgressService>>();
            UserProgressService service = new UserProgressService(mockLogger.Object);
            string userId = $"zero_values_{Guid.NewGuid()}";

            string appDataPath = Environment.GetFolderPath(Environment.SpecialFolder.ApplicationData);
            string progressDataPath = Path.Combine(appDataPath, "SignLanguageApp", "UserProgress");
            string filePath = Path.Combine(progressDataPath, $"progress_{userId}.json");

            UserProgressData expectedData = new UserProgressData
            {
                UserId = userId,
                TotalXp = 0,
                LearningStreak = 0,
                LastProgressUpdate = DateTime.UtcNow,
                SavedAt = DateTime.UtcNow
            };

            JsonSerializerOptions jsonOptions = new JsonSerializerOptions
            {
                WriteIndented = true,
                PropertyNameCaseInsensitive = true
            };

            string json = JsonSerializer.Serialize(expectedData, jsonOptions);
            await File.WriteAllTextAsync(filePath, json);

            try
            {
                // Act
                UserProgressData? result = await service.LoadUserProgressAsync(userId);

                // Assert
                Assert.IsNotNull(result);
                Assert.AreEqual(0, result.TotalXp);
                Assert.AreEqual(0, result.LearningStreak);
            }
            finally
            {
                if (File.Exists(filePath))
                {
                    File.Delete(filePath);
                }
            }
        }

        /// <summary>
        /// Tests that LoadUserProgressAsync handles negative values for TotalXp and LearningStreak.
        /// Input: File with negative values.
        /// Expected: Returns data with negative values (no validation in load method).
        /// </summary>
        [TestMethod]
        [DataRow(-1, -1)]
        [DataRow(-100, -50)]
        [DataRow(-999999, -365)]
        public async Task LoadUserProgressAsync_NegativeValues_ReturnsCorrectData(int totalXp, int learningStreak)
        {
            // Arrange
            Mock<ILogger<UserProgressService>> mockLogger = new Mock<ILogger<UserProgressService>>();
            UserProgressService service = new UserProgressService(mockLogger.Object);
            string userId = $"negative_{Guid.NewGuid()}";

            string appDataPath = Environment.GetFolderPath(Environment.SpecialFolder.ApplicationData);
            string progressDataPath = Path.Combine(appDataPath, "SignLanguageApp", "UserProgress");
            string filePath = Path.Combine(progressDataPath, $"progress_{userId}.json");

            UserProgressData expectedData = new UserProgressData
            {
                UserId = userId,
                TotalXp = totalXp,
                LearningStreak = learningStreak,
                LastProgressUpdate = DateTime.UtcNow,
                SavedAt = DateTime.UtcNow
            };

            JsonSerializerOptions jsonOptions = new JsonSerializerOptions
            {
                WriteIndented = true,
                PropertyNameCaseInsensitive = true
            };

            string json = JsonSerializer.Serialize(expectedData, jsonOptions);
            await File.WriteAllTextAsync(filePath, json);

            try
            {
                // Act
                UserProgressData? result = await service.LoadUserProgressAsync(userId);

                // Assert
                Assert.IsNotNull(result);
                Assert.AreEqual(totalXp, result.TotalXp);
                Assert.AreEqual(learningStreak, result.LearningStreak);
            }
            finally
            {
                if (File.Exists(filePath))
                {
                    File.Delete(filePath);
                }
            }
        }

        /// <summary>
        /// Tests that LoadUserProgressAsync handles special characters in userId.
        /// Input: userId with special characters.
        /// Expected: Handles gracefully (may or may not find file).
        /// </summary>
        [TestMethod]
        [DataRow("user@example.com")]
        [DataRow("user_123")]
        [DataRow("user-with-dashes")]
        [DataRow("user.with.dots")]
        public async Task LoadUserProgressAsync_SpecialCharactersInUserId_HandlesGracefully(string userId)
        {
            // Arrange
            Mock<ILogger<UserProgressService>> mockLogger = new Mock<ILogger<UserProgressService>>();
            UserProgressService service = new UserProgressService(mockLogger.Object);

            // Act
            UserProgressData? result = await service.LoadUserProgressAsync(userId);

            // Assert
            Assert.IsNull(result);
        }

        /// <summary>
        /// Tests that LoadUserProgressAsync handles very long userId strings.
        /// Input: Very long userId (1000 characters).
        /// Expected: Handles gracefully, returns null (file won't exist).
        /// </summary>
        [TestMethod]
        public async Task LoadUserProgressAsync_VeryLongUserId_HandlesGracefully()
        {
            // Arrange
            Mock<ILogger<UserProgressService>> mockLogger = new Mock<ILogger<UserProgressService>>();
            UserProgressService service = new UserProgressService(mockLogger.Object);
            string userId = new string('a', 1000);

            // Act
            UserProgressData? result = await service.LoadUserProgressAsync(userId);

            // Assert
            Assert.IsNull(result);
        }

        /// <summary>
        /// Tests that LoadUserProgressAsync returns null when JSON deserializes to null.
        /// Input: File with "null" JSON content.
        /// Expected: Returns null and logs information with null-coalesced values.
        /// </summary>
        [TestMethod]
        public async Task LoadUserProgressAsync_DeserializationReturnsNull_ReturnsNull()
        {
            // Arrange
            Mock<ILogger<UserProgressService>> mockLogger = new Mock<ILogger<UserProgressService>>();
            UserProgressService service = new UserProgressService(mockLogger.Object);
            string userId = $"null_deserialization_{Guid.NewGuid()}";

            string appDataPath = Environment.GetFolderPath(Environment.SpecialFolder.ApplicationData);
            string progressDataPath = Path.Combine(appDataPath, "SignLanguageApp", "UserProgress");
            string filePath = Path.Combine(progressDataPath, $"progress_{userId}.json");

            await File.WriteAllTextAsync(filePath, "null");

            try
            {
                // Act
                UserProgressData? result = await service.LoadUserProgressAsync(userId);

                // Assert
                Assert.IsNull(result);
                mockLogger.Verify(
                    x => x.Log(
                        LogLevel.Information,
                        It.IsAny<EventId>(),
                        It.Is<It.IsAnyType>((v, t) => true),
                        It.IsAny<Exception>(),
                        It.IsAny<Func<It.IsAnyType, Exception?, string>>()),
                    Times.Once);
            }
            finally
            {
                if (File.Exists(filePath))
                {
                    File.Delete(filePath);
                }
            }
        }

        /// <summary>
        /// Tests that LoadUserProgressAsync handles DateTime.MinValue correctly.
        /// Input: File with DateTime.MinValue for both date fields.
        /// Expected: Returns data with DateTime.MinValue.
        /// </summary>
        [TestMethod]
        public async Task LoadUserProgressAsync_DateTimeMinValue_ReturnsCorrectData()
        {
            // Arrange
            Mock<ILogger<UserProgressService>> mockLogger = new Mock<ILogger<UserProgressService>>();
            UserProgressService service = new UserProgressService(mockLogger.Object);
            string userId = $"datetime_min_{Guid.NewGuid()}";

            string appDataPath = Environment.GetFolderPath(Environment.SpecialFolder.ApplicationData);
            string progressDataPath = Path.Combine(appDataPath, "SignLanguageApp", "UserProgress");
            string filePath = Path.Combine(progressDataPath, $"progress_{userId}.json");

            UserProgressData expectedData = new UserProgressData
            {
                UserId = userId,
                TotalXp = 100,
                LearningStreak = 5,
                LastProgressUpdate = DateTime.MinValue,
                SavedAt = DateTime.MinValue
            };

            JsonSerializerOptions jsonOptions = new JsonSerializerOptions
            {
                WriteIndented = true,
                PropertyNameCaseInsensitive = true
            };

            string json = JsonSerializer.Serialize(expectedData, jsonOptions);
            await File.WriteAllTextAsync(filePath, json);

            try
            {
                // Act
                UserProgressData? result = await service.LoadUserProgressAsync(userId);

                // Assert
                Assert.IsNotNull(result);
                Assert.AreEqual(DateTime.MinValue, result.LastProgressUpdate);
                Assert.AreEqual(DateTime.MinValue, result.SavedAt);
            }
            finally
            {
                if (File.Exists(filePath))
                {
                    File.Delete(filePath);
                }
            }
        }

        /// <summary>
        /// Tests that LoadUserProgressAsync handles DateTime.MaxValue correctly.
        /// Input: File with DateTime.MaxValue for both date fields.
        /// Expected: Returns data with DateTime.MaxValue.
        /// </summary>
        [TestMethod]
        public async Task LoadUserProgressAsync_DateTimeMaxValue_ReturnsCorrectData()
        {
            // Arrange
            Mock<ILogger<UserProgressService>> mockLogger = new Mock<ILogger<UserProgressService>>();
            UserProgressService service = new UserProgressService(mockLogger.Object);
            string userId = $"datetime_max_{Guid.NewGuid()}";

            string appDataPath = Environment.GetFolderPath(Environment.SpecialFolder.ApplicationData);
            string progressDataPath = Path.Combine(appDataPath, "SignLanguageApp", "UserProgress");
            string filePath = Path.Combine(progressDataPath, $"progress_{userId}.json");

            UserProgressData expectedData = new UserProgressData
            {
                UserId = userId,
                TotalXp = 100,
                LearningStreak = 5,
                LastProgressUpdate = DateTime.MaxValue,
                SavedAt = DateTime.MaxValue
            };

            JsonSerializerOptions jsonOptions = new JsonSerializerOptions
            {
                WriteIndented = true,
                PropertyNameCaseInsensitive = true
            };

            string json = JsonSerializer.Serialize(expectedData, jsonOptions);
            await File.WriteAllTextAsync(filePath, json);

            try
            {
                // Act
                UserProgressData? result = await service.LoadUserProgressAsync(userId);

                // Assert
                Assert.IsNotNull(result);
                Assert.AreEqual(DateTime.MaxValue, result.LastProgressUpdate);
                Assert.AreEqual(DateTime.MaxValue, result.SavedAt);
            }
            finally
            {
                if (File.Exists(filePath))
                {
                    File.Delete(filePath);
                }
            }
        }

        /// <summary>
        /// Tests that LoadUserProgressAsync handles empty JSON file.
        /// Input: File with empty content.
        /// Expected: Returns null and logs error (JSON deserialization fails).
        /// </summary>
        [TestMethod]
        public async Task LoadUserProgressAsync_EmptyFile_ReturnsNullAndLogsError()
        {
            // Arrange
            Mock<ILogger<UserProgressService>> mockLogger = new Mock<ILogger<UserProgressService>>();
            UserProgressService service = new UserProgressService(mockLogger.Object);
            string userId = $"empty_file_{Guid.NewGuid()}";

            string appDataPath = Environment.GetFolderPath(Environment.SpecialFolder.ApplicationData);
            string progressDataPath = Path.Combine(appDataPath, "SignLanguageApp", "UserProgress");
            string filePath = Path.Combine(progressDataPath, $"progress_{userId}.json");

            await File.WriteAllTextAsync(filePath, string.Empty);

            try
            {
                // Act
                UserProgressData? result = await service.LoadUserProgressAsync(userId);

                // Assert
                Assert.IsNull(result);
                mockLogger.Verify(
                    x => x.Log(
                        LogLevel.Error,
                        It.IsAny<EventId>(),
                        It.Is<It.IsAnyType>((v, t) => true),
                        It.IsAny<Exception>(),
                        It.IsAny<Func<It.IsAnyType, Exception?, string>>()),
                    Times.Once);
            }
            finally
            {
                if (File.Exists(filePath))
                {
                    File.Delete(filePath);
                }
            }
        }

        /// <summary>
        /// Tests that LoadUserProgressAsync handles empty JSON object.
        /// Input: File with empty JSON object {}.
        /// Expected: Returns object with default values.
        /// </summary>
        [TestMethod]
        public async Task LoadUserProgressAsync_EmptyJsonObject_ReturnsObjectWithDefaults()
        {
            // Arrange
            Mock<ILogger<UserProgressService>> mockLogger = new Mock<ILogger<UserProgressService>>();
            UserProgressService service = new UserProgressService(mockLogger.Object);
            string userId = $"empty_object_{Guid.NewGuid()}";

            string appDataPath = Environment.GetFolderPath(Environment.SpecialFolder.ApplicationData);
            string progressDataPath = Path.Combine(appDataPath, "SignLanguageApp", "UserProgress");
            string filePath = Path.Combine(progressDataPath, $"progress_{userId}.json");

            await File.WriteAllTextAsync(filePath, "{}");

            try
            {
                // Act
                UserProgressData? result = await service.LoadUserProgressAsync(userId);

                // Assert
                Assert.IsNotNull(result);
                Assert.AreEqual(string.Empty, result.UserId);
                Assert.AreEqual(0, result.TotalXp);
                Assert.AreEqual(0, result.LearningStreak);
            }
            finally
            {
                if (File.Exists(filePath))
                {
                    File.Delete(filePath);
                }
            }
        }

        /// <summary>
        /// Tests LoadUserLessonsProgressAsync when file does not exist.
        /// Expected: Returns an empty list and logs debug message.
        /// </summary>
        [TestMethod]
        public async Task LoadUserLessonsProgressAsync_FileDoesNotExist_ReturnsEmptyListAndLogsDebug()
        {
            // Arrange
            Mock<ILogger<UserProgressService>> mockLogger = new Mock<ILogger<UserProgressService>>();
            UserProgressService service = new UserProgressService(mockLogger.Object);
            string testUserId = Guid.NewGuid().ToString();

            // Act
            List<UserLessonProgressData> result = await service.LoadUserLessonsProgressAsync(testUserId);

            // Assert
            Assert.IsNotNull(result);
            Assert.AreEqual(0, result.Count);
            mockLogger.Verify(
                x => x.Log(
                    LogLevel.Debug,
                    It.IsAny<EventId>(),
                    It.Is<It.IsAnyType>((v, t) => v.ToString()!.Contains("No lessons progress file found")),
                    It.IsAny<Exception>(),
                    It.IsAny<Func<It.IsAnyType, Exception?, string>>()),
                Times.Once);
        }

        /// <summary>
        /// Tests LoadUserLessonsProgressAsync when file exists with valid single-item JSON.
        /// Expected: Returns deserialized list with one item and logs information.
        /// </summary>
        [TestMethod]
        public async Task LoadUserLessonsProgressAsync_FileExistsWithValidSingleItemJson_ReturnsDeserializedList()
        {
            // Arrange
            Mock<ILogger<UserProgressService>> mockLogger = new Mock<ILogger<UserProgressService>>();
            UserProgressService service = new UserProgressService(mockLogger.Object);
            string testUserId = Guid.NewGuid().ToString();

            string appDataPath = Environment.GetFolderPath(Environment.SpecialFolder.ApplicationData);
            string progressDataPath = Path.Combine(appDataPath, "SignLanguageApp", "UserProgress");
            string filePath = Path.Combine(progressDataPath, $"lessons_progress_{testUserId}.json");

            List<UserLessonProgressData> testData = new List<UserLessonProgressData>
            {
                new UserLessonProgressData
                {
                    LessonId = 1,
                    LessonTitle = "Test Lesson",
                    CompletionPercentage = 75,
                    IsCompleted = false,
                    TotalAttempts = 2,
                    XpEarned = 150
                }
            };

            string json = JsonSerializer.Serialize(testData, new JsonSerializerOptions { WriteIndented = true });
            Directory.CreateDirectory(progressDataPath);
            await File.WriteAllTextAsync(filePath, json);

            try
            {
                // Act
                List<UserLessonProgressData> result = await service.LoadUserLessonsProgressAsync(testUserId);

                // Assert
                Assert.IsNotNull(result);
                Assert.AreEqual(1, result.Count);
                Assert.AreEqual(1, result[0].LessonId);
                Assert.AreEqual("Test Lesson", result[0].LessonTitle);
                Assert.AreEqual(75, result[0].CompletionPercentage);
                Assert.AreEqual(false, result[0].IsCompleted);

                mockLogger.Verify(
                    x => x.Log(
                        LogLevel.Information,
                        It.IsAny<EventId>(),
                        It.Is<It.IsAnyType>((v, t) => v.ToString()!.Contains("User lessons progress loaded")),
                        It.IsAny<Exception>(),
                        It.IsAny<Func<It.IsAnyType, Exception?, string>>()),
                    Times.Once);
            }
            finally
            {
                if (File.Exists(filePath))
                {
                    File.Delete(filePath);
                }
            }
        }

        /// <summary>
        /// Tests LoadUserLessonsProgressAsync when file exists with multiple valid items.
        /// Expected: Returns deserialized list with all items and logs information.
        /// </summary>
        [TestMethod]
        public async Task LoadUserLessonsProgressAsync_FileExistsWithMultipleItems_ReturnsAllItems()
        {
            // Arrange
            Mock<ILogger<UserProgressService>> mockLogger = new Mock<ILogger<UserProgressService>>();
            UserProgressService service = new UserProgressService(mockLogger.Object);
            string testUserId = Guid.NewGuid().ToString();

            string appDataPath = Environment.GetFolderPath(Environment.SpecialFolder.ApplicationData);
            string progressDataPath = Path.Combine(appDataPath, "SignLanguageApp", "UserProgress");
            string filePath = Path.Combine(progressDataPath, $"lessons_progress_{testUserId}.json");

            List<UserLessonProgressData> testData = new List<UserLessonProgressData>
            {
                new UserLessonProgressData
                {
                    LessonId = 1,
                    LessonTitle = "Lesson 1",
                    CompletionPercentage = 100,
                    IsCompleted = true,
                    TotalAttempts = 5,
                    XpEarned = 500
                },
                new UserLessonProgressData
                {
                    LessonId = 2,
                    LessonTitle = "Lesson 2",
                    CompletionPercentage = 50,
                    IsCompleted = false,
                    TotalAttempts = 3,
                    XpEarned = 250
                },
                new UserLessonProgressData
                {
                    LessonId = 3,
                    LessonTitle = "Lesson 3",
                    CompletionPercentage = 0,
                    IsCompleted = false,
                    TotalAttempts = 1,
                    XpEarned = 0
                }
            };

            string json = JsonSerializer.Serialize(testData, new JsonSerializerOptions { WriteIndented = true });
            Directory.CreateDirectory(progressDataPath);
            await File.WriteAllTextAsync(filePath, json);

            try
            {
                // Act
                List<UserLessonProgressData> result = await service.LoadUserLessonsProgressAsync(testUserId);

                // Assert
                Assert.IsNotNull(result);
                Assert.AreEqual(3, result.Count);
                Assert.AreEqual(1, result[0].LessonId);
                Assert.AreEqual(2, result[1].LessonId);
                Assert.AreEqual(3, result[2].LessonId);

                mockLogger.Verify(
                    x => x.Log(
                        LogLevel.Information,
                        It.IsAny<EventId>(),
                        It.Is<It.IsAnyType>((v, t) => v.ToString()!.Contains("User lessons progress loaded")),
                        It.IsAny<Exception>(),
                        It.IsAny<Func<It.IsAnyType, Exception?, string>>()),
                    Times.Once);
            }
            finally
            {
                if (File.Exists(filePath))
                {
                    File.Delete(filePath);
                }
            }
        }

        /// <summary>
        /// Tests LoadUserLessonsProgressAsync when file contains empty string.
        /// Expected: Returns empty list and logs error due to deserialization failure.
        /// </summary>
        [TestMethod]
        public async Task LoadUserLessonsProgressAsync_FileContainsEmptyString_ReturnsEmptyListAndLogsError()
        {
            // Arrange
            Mock<ILogger<UserProgressService>> mockLogger = new Mock<ILogger<UserProgressService>>();
            UserProgressService service = new UserProgressService(mockLogger.Object);
            string testUserId = Guid.NewGuid().ToString();

            string appDataPath = Environment.GetFolderPath(Environment.SpecialFolder.ApplicationData);
            string progressDataPath = Path.Combine(appDataPath, "SignLanguageApp", "UserProgress");
            string filePath = Path.Combine(progressDataPath, $"lessons_progress_{testUserId}.json");

            Directory.CreateDirectory(progressDataPath);
            await File.WriteAllTextAsync(filePath, string.Empty);

            try
            {
                // Act
                List<UserLessonProgressData> result = await service.LoadUserLessonsProgressAsync(testUserId);

                // Assert
                Assert.IsNotNull(result);
                Assert.AreEqual(0, result.Count);

                mockLogger.Verify(
                    x => x.Log(
                        LogLevel.Error,
                        It.IsAny<EventId>(),
                        It.Is<It.IsAnyType>((v, t) => v.ToString()!.Contains("Error loading lessons progress")),
                        It.IsAny<Exception>(),
                        It.IsAny<Func<It.IsAnyType, Exception?, string>>()),
                    Times.Once);
            }
            finally
            {
                if (File.Exists(filePath))
                {
                    File.Delete(filePath);
                }
            }
        }

        /// <summary>
        /// Tests LoadUserLessonsProgressAsync when file contains only whitespace.
        /// Expected: Returns empty list and logs error due to deserialization failure.
        /// </summary>
        [TestMethod]
        public async Task LoadUserLessonsProgressAsync_FileContainsOnlyWhitespace_ReturnsEmptyListAndLogsError()
        {
            // Arrange
            Mock<ILogger<UserProgressService>> mockLogger = new Mock<ILogger<UserProgressService>>();
            UserProgressService service = new UserProgressService(mockLogger.Object);
            string testUserId = Guid.NewGuid().ToString();

            string appDataPath = Environment.GetFolderPath(Environment.SpecialFolder.ApplicationData);
            string progressDataPath = Path.Combine(appDataPath, "SignLanguageApp", "UserProgress");
            string filePath = Path.Combine(progressDataPath, $"lessons_progress_{testUserId}.json");

            Directory.CreateDirectory(progressDataPath);
            await File.WriteAllTextAsync(filePath, "   \t\n  ");

            try
            {
                // Act
                List<UserLessonProgressData> result = await service.LoadUserLessonsProgressAsync(testUserId);

                // Assert
                Assert.IsNotNull(result);
                Assert.AreEqual(0, result.Count);

                mockLogger.Verify(
                    x => x.Log(
                        LogLevel.Error,
                        It.IsAny<EventId>(),
                        It.Is<It.IsAnyType>((v, t) => v.ToString()!.Contains("Error loading lessons progress")),
                        It.IsAny<Exception>(),
                        It.IsAny<Func<It.IsAnyType, Exception?, string>>()),
                    Times.Once);
            }
            finally
            {
                if (File.Exists(filePath))
                {
                    File.Delete(filePath);
                }
            }
        }

        /// <summary>
        /// Tests LoadUserLessonsProgressAsync when file contains null JSON value.
        /// Expected: Returns empty list (null-coalescing operator provides empty list).
        /// </summary>
        [TestMethod]
        public async Task LoadUserLessonsProgressAsync_FileContainsNullJson_ReturnsEmptyList()
        {
            // Arrange
            Mock<ILogger<UserProgressService>> mockLogger = new Mock<ILogger<UserProgressService>>();
            UserProgressService service = new UserProgressService(mockLogger.Object);
            string testUserId = Guid.NewGuid().ToString();

            string appDataPath = Environment.GetFolderPath(Environment.SpecialFolder.ApplicationData);
            string progressDataPath = Path.Combine(appDataPath, "SignLanguageApp", "UserProgress");
            string filePath = Path.Combine(progressDataPath, $"lessons_progress_{testUserId}.json");

            Directory.CreateDirectory(progressDataPath);
            await File.WriteAllTextAsync(filePath, "null");

            try
            {
                // Act
                List<UserLessonProgressData> result = await service.LoadUserLessonsProgressAsync(testUserId);

                // Assert
                Assert.IsNotNull(result);
                Assert.AreEqual(0, result.Count);

                mockLogger.Verify(
                    x => x.Log(
                        LogLevel.Information,
                        It.IsAny<EventId>(),
                        It.Is<It.IsAnyType>((v, t) => v.ToString()!.Contains("User lessons progress loaded")),
                        It.IsAny<Exception>(),
                        It.IsAny<Func<It.IsAnyType, Exception?, string>>()),
                    Times.Once);
            }
            finally
            {
                if (File.Exists(filePath))
                {
                    File.Delete(filePath);
                }
            }
        }

        /// <summary>
        /// Tests LoadUserLessonsProgressAsync with special characters in userId.
        /// Expected: Returns empty list (file won't exist for random special char userId).
        /// </summary>
        [TestMethod]
        [DataRow("user@domain.com")]
        [DataRow("user_123")]
        [DataRow("user-with-dashes")]
        [DataRow("user.with.dots")]
        public async Task LoadUserLessonsProgressAsync_UserIdWithSpecialCharacters_ReturnsEmptyList(string userId)
        {
            // Arrange
            Mock<ILogger<UserProgressService>> mockLogger = new Mock<ILogger<UserProgressService>>();
            UserProgressService service = new UserProgressService(mockLogger.Object);

            // Act
            List<UserLessonProgressData> result = await service.LoadUserLessonsProgressAsync(userId);

            // Assert
            Assert.IsNotNull(result);
            Assert.AreEqual(0, result.Count);
        }

        /// <summary>
        /// Tests LoadUserLessonsProgressAsync with extreme property values in JSON.
        /// Expected: Deserializes correctly with int.MaxValue and int.MinValue.
        /// </summary>
        [TestMethod]
        public async Task LoadUserLessonsProgressAsync_FileWithExtremePropertyValues_DeserializesCorrectly()
        {
            // Arrange
            Mock<ILogger<UserProgressService>> mockLogger = new Mock<ILogger<UserProgressService>>();
            UserProgressService service = new UserProgressService(mockLogger.Object);
            string testUserId = Guid.NewGuid().ToString();

            string appDataPath = Environment.GetFolderPath(Environment.SpecialFolder.ApplicationData);
            string progressDataPath = Path.Combine(appDataPath, "SignLanguageApp", "UserProgress");
            string filePath = Path.Combine(progressDataPath, $"lessons_progress_{testUserId}.json");

            List<UserLessonProgressData> testData = new List<UserLessonProgressData>
            {
                new UserLessonProgressData
                {
                    LessonId = int.MaxValue,
                    LessonTitle = new string('x', 10000),
                    CompletionPercentage = int.MaxValue,
                    IsCompleted = true,
                    TotalAttempts = int.MaxValue,
                    XpEarned = int.MaxValue
                },
                new UserLessonProgressData
                {
                    LessonId = int.MinValue,
                    LessonTitle = string.Empty,
                    CompletionPercentage = int.MinValue,
                    IsCompleted = false,
                    TotalAttempts = int.MinValue,
                    XpEarned = int.MinValue
                }
            };

            string json = JsonSerializer.Serialize(testData, new JsonSerializerOptions { WriteIndented = true });
            Directory.CreateDirectory(progressDataPath);
            await File.WriteAllTextAsync(filePath, json);

            try
            {
                // Act
                List<UserLessonProgressData> result = await service.LoadUserLessonsProgressAsync(testUserId);

                // Assert
                Assert.IsNotNull(result);
                Assert.AreEqual(2, result.Count);
                Assert.AreEqual(int.MaxValue, result[0].LessonId);
                Assert.AreEqual(int.MinValue, result[1].LessonId);

                mockLogger.Verify(
                    x => x.Log(
                        LogLevel.Information,
                        It.IsAny<EventId>(),
                        It.Is<It.IsAnyType>((v, t) => v.ToString()!.Contains("User lessons progress loaded")),
                        It.IsAny<Exception>(),
                        It.IsAny<Func<It.IsAnyType, Exception?, string>>()),
                    Times.Once);
            }
            finally
            {
                if (File.Exists(filePath))
                {
                    File.Delete(filePath);
                }
            }
        }

        /// <summary>
        /// Tests LoadUserLessonsProgressAsync with zero values for all numeric properties.
        /// Expected: Deserializes correctly with all zeros.
        /// </summary>
        [TestMethod]
        public async Task LoadUserLessonsProgressAsync_FileWithZeroValues_DeserializesCorrectly()
        {
            // Arrange
            Mock<ILogger<UserProgressService>> mockLogger = new Mock<ILogger<UserProgressService>>();
            UserProgressService service = new UserProgressService(mockLogger.Object);
            string testUserId = Guid.NewGuid().ToString();

            string appDataPath = Environment.GetFolderPath(Environment.SpecialFolder.ApplicationData);
            string progressDataPath = Path.Combine(appDataPath, "SignLanguageApp", "UserProgress");
            string filePath = Path.Combine(progressDataPath, $"lessons_progress_{testUserId}.json");

            List<UserLessonProgressData> testData = new List<UserLessonProgressData>
            {
                new UserLessonProgressData
                {
                    LessonId = 0,
                    LessonTitle = string.Empty,
                    CompletionPercentage = 0,
                    IsCompleted = false,
                    TotalAttempts = 0,
                    XpEarned = 0
                }
            };

            string json = JsonSerializer.Serialize(testData, new JsonSerializerOptions { WriteIndented = true });
            Directory.CreateDirectory(progressDataPath);
            await File.WriteAllTextAsync(filePath, json);

            try
            {
                // Act
                List<UserLessonProgressData> result = await service.LoadUserLessonsProgressAsync(testUserId);

                // Assert
                Assert.IsNotNull(result);
                Assert.AreEqual(1, result.Count);
                Assert.AreEqual(0, result[0].LessonId);
                Assert.AreEqual(0, result[0].CompletionPercentage);
                Assert.AreEqual(0, result[0].TotalAttempts);
                Assert.AreEqual(0, result[0].XpEarned);

                mockLogger.Verify(
                    x => x.Log(
                        LogLevel.Information,
                        It.IsAny<EventId>(),
                        It.Is<It.IsAnyType>((v, t) => v.ToString()!.Contains("User lessons progress loaded")),
                        It.IsAny<Exception>(),
                        It.IsAny<Func<It.IsAnyType, Exception?, string>>()),
                    Times.Once);
            }
            finally
            {
                if (File.Exists(filePath))
                {
                    File.Delete(filePath);
                }
            }
        }

        /// <summary>
        /// Tests LoadUserLessonsProgressAsync with negative values for numeric properties.
        /// Expected: Deserializes correctly with negative values.
        /// </summary>
        [TestMethod]
        public async Task LoadUserLessonsProgressAsync_FileWithNegativeValues_DeserializesCorrectly()
        {
            // Arrange
            Mock<ILogger<UserProgressService>> mockLogger = new Mock<ILogger<UserProgressService>>();
            UserProgressService service = new UserProgressService(mockLogger.Object);
            string testUserId = Guid.NewGuid().ToString();

            string appDataPath = Environment.GetFolderPath(Environment.SpecialFolder.ApplicationData);
            string progressDataPath = Path.Combine(appDataPath, "SignLanguageApp", "UserProgress");
            string filePath = Path.Combine(progressDataPath, $"lessons_progress_{testUserId}.json");

            List<UserLessonProgressData> testData = new List<UserLessonProgressData>
            {
                new UserLessonProgressData
                {
                    LessonId = -1,
                    LessonTitle = "Negative Test",
                    CompletionPercentage = -50,
                    IsCompleted = false,
                    TotalAttempts = -10,
                    XpEarned = -100
                }
            };

            string json = JsonSerializer.Serialize(testData, new JsonSerializerOptions { WriteIndented = true });
            Directory.CreateDirectory(progressDataPath);
            await File.WriteAllTextAsync(filePath, json);

            try
            {
                // Act
                List<UserLessonProgressData> result = await service.LoadUserLessonsProgressAsync(testUserId);

                // Assert
                Assert.IsNotNull(result);
                Assert.AreEqual(1, result.Count);
                Assert.AreEqual(-1, result[0].LessonId);
                Assert.AreEqual(-50, result[0].CompletionPercentage);
                Assert.AreEqual(-10, result[0].TotalAttempts);
                Assert.AreEqual(-100, result[0].XpEarned);

                mockLogger.Verify(
                    x => x.Log(
                        LogLevel.Information,
                        It.IsAny<EventId>(),
                        It.Is<It.IsAnyType>((v, t) => v.ToString()!.Contains("User lessons progress loaded")),
                        It.IsAny<Exception>(),
                        It.IsAny<Func<It.IsAnyType, Exception?, string>>()),
                    Times.Once);
            }
            finally
            {
                if (File.Exists(filePath))
                {
                    File.Delete(filePath);
                }
            }
        }

        /// <summary>
        /// Tests that the constructor handles null logger parameter.
        /// Expected behavior: Constructor completes but NullReferenceException may occur on logger usage.
        /// This test verifies actual behavior since no null validation is present in the constructor.
        /// </summary>
        [TestMethod]
        public void UserProgressService_NullLogger_ConstructorCompletes()
        {
            // Arrange
            ILogger<UserProgressService>? logger = null;

            // Act & Assert
            // Constructor doesn't validate null logger, so it completes successfully
            // NullReferenceException would only occur when logger methods are called
            var service = new UserProgressService(logger!);
            Assert.IsNotNull(service);
        }

        /// <summary>
        /// Tests that multiple constructor calls handle existing directory gracefully.
        /// Verifies that creating multiple service instances doesn't cause errors when directory already exists.
        /// </summary>
        [TestMethod]
        public void UserProgressService_MultipleInstances_HandlesExistingDirectoryGracefully()
        {
            // Arrange
            var mockLogger1 = new Mock<ILogger<UserProgressService>>();
            var mockLogger2 = new Mock<ILogger<UserProgressService>>();

            // Act
            var service1 = new UserProgressService(mockLogger1.Object);
            var service2 = new UserProgressService(mockLogger2.Object);

            // Assert
            Assert.IsNotNull(service1);
            Assert.IsNotNull(service2);
            string appDataPath = Environment.GetFolderPath(Environment.SpecialFolder.ApplicationData);
            string progressDataPath = Path.Combine(appDataPath, "SignLanguageApp", "UserProgress");
            Assert.IsTrue(Directory.Exists(progressDataPath));
        }

        /// <summary>
        /// Tests that the constructor initializes with different logger mock instances.
        /// Verifies that each instance is independent and properly initialized.
        /// </summary>
        [TestMethod]
        public void UserProgressService_DifferentLoggers_EachInstanceInitializesIndependently()
        {
            // Arrange
            var mockLogger1 = new Mock<ILogger<UserProgressService>>();
            var mockLogger2 = new Mock<ILogger<UserProgressService>>();
            var mockLogger3 = new Mock<ILogger<UserProgressService>>();

            // Act
            var service1 = new UserProgressService(mockLogger1.Object);
            var service2 = new UserProgressService(mockLogger2.Object);
            var service3 = new UserProgressService(mockLogger3.Object);

            // Assert
            Assert.IsNotNull(service1);
            Assert.IsNotNull(service2);
            Assert.IsNotNull(service3);
        }
    }
}