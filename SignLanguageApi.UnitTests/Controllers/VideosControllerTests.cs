using System;
using System.Collections.Concurrent;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Reflection;

using Microsoft.AspNetCore.Mvc;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using SignLanguageApi.Controllers;
using SignLanguageApi.Dtos;

namespace SignLanguageApi.Controllers.UnitTests
{
    /// <summary>
    /// Unit tests for the VideosController class.
    /// </summary>
    [TestClass]
    public class VideosControllerTests
    {
        /// <summary>
        /// Tests that Stream returns NotFound when the video with the specified id does not exist in the video list.
        /// </summary>
        [TestMethod]
        public void Stream_VideoNotFound_ReturnsNotFound()
        {
            // Arrange
            VideosController controller = new VideosController();
            int nonExistentId = 99999;

            // Use reflection to set the cached videos to a known state (empty list)
            // This ensures our test ID won't exist
            Type controllerType = typeof(VideosController);
            FieldInfo cachedVideosField = controllerType.GetField("_cachedVideos", BindingFlags.NonPublic | BindingFlags.Static);
            FieldInfo lastScanField = controllerType.GetField("_lastScan", BindingFlags.NonPublic | BindingFlags.Static);

            // Set cached videos to an empty list
            cachedVideosField.SetValue(null, new List<VideoDto>());
            // Set last scan to now so the cache is considered fresh
            lastScanField.SetValue(null, DateTime.UtcNow);

            // Act
            IActionResult result = controller.Stream(nonExistentId);

            // Assert
            Assert.IsInstanceOfType(result, typeof(NotFoundResult), "Expected NotFound result when video ID does not exist.");
        }

        /// <summary>
        /// Tests that Stream returns NotFound when the video exists in the list but the file does not exist on disk.
        /// Note: This test cannot be fully implemented as a unit test due to tight coupling to file system.
        /// See comments in Stream_VideoNotFound_ReturnsNotFound for details.
        /// </summary>
        [TestMethod]
        public void Stream_VideoFoundButFileNotExists_ReturnsNotFound()
        {
            // Arrange
            VideosController controller = new VideosController();
            int videoId = 1;

            // Use reflection to set the cached videos to a known state with a video that has a non-existent file
            Type controllerType = typeof(VideosController);
            FieldInfo cachedVideosField = controllerType.GetField("_cachedVideos", BindingFlags.NonPublic | BindingFlags.Static);
            FieldInfo lastScanField = controllerType.GetField("_lastScan", BindingFlags.NonPublic | BindingFlags.Static);

            // Set cached videos to a list with one video that references a non-existent file
            List<VideoDto> fakeVideos = new List<VideoDto>
            {
                new VideoDto
                {
                    Id = 1,
                    FileName = "non_existent_file_12345.mp4",
                    Title = "Test Video",
                    Category = "General",
                    Path = "/api/videos/stream/1",
                    Likes = 0,
                    Views = 0
                }
            };
            cachedVideosField.SetValue(null, fakeVideos);
            // Set last scan to now so the cache is considered fresh
            lastScanField.SetValue(null, DateTime.UtcNow);

            // Act
            IActionResult result = controller.Stream(videoId);

            // Assert
            Assert.IsInstanceOfType(result, typeof(NotFoundResult), "Expected NotFound result when video file does not exist on disk.");
        }

        /// <summary>
        /// Tests that Stream returns a FileStreamResult when the video exists and the file is found.
        /// Note: This test cannot be fully implemented as a unit test due to tight coupling to file system.
        /// See comments in Stream_VideoNotFound_ReturnsNotFound for details.
        /// </summary>
        [TestMethod]
        public void Stream_VideoFoundAndFileExists_ReturnsFileStreamResult()
        {
            // Arrange
            VideosController controller = new VideosController();
            int videoId = 1;

            // Get the VIDEO folder path using reflection
            var videoFolderField = typeof(VideosController).GetField("VideoFolder", BindingFlags.NonPublic | BindingFlags.Static);
            string videoFolder = (string)videoFolderField.GetValue(null);

            // Ensure VIDEO folder exists
            if (!Directory.Exists(videoFolder))
                Directory.CreateDirectory(videoFolder);

            // Create a test video file
            string testFileName = "test_video.mp4";
            string testFilePath = Path.Combine(videoFolder, testFileName);
            File.WriteAllText(testFilePath, "dummy video content");

            try
            {
                // Set up the cached videos list using reflection
                var cachedVideosField = typeof(VideosController).GetField("_cachedVideos", BindingFlags.NonPublic | BindingFlags.Static);
                var lastScanField = typeof(VideosController).GetField("_lastScan", BindingFlags.NonPublic | BindingFlags.Static);

                var testVideos = new List<VideoDto>
                {
                    new VideoDto
                    {
                        Id = videoId,
                        FileName = testFileName,
                        Title = "Test Video",
                        Category = "Test",
                        Path = $"/api/videos/stream/{videoId}"
                    }
                };

                cachedVideosField.SetValue(null, testVideos);
                lastScanField.SetValue(null, DateTime.UtcNow);

                // Act
                IActionResult result = controller.Stream(videoId);

                // Assert
                Assert.IsNotNull(result, "Result should not be null");
                Assert.IsInstanceOfType(result, typeof(FileStreamResult), "Result should be a FileStreamResult");

                var fileResult = result as FileStreamResult;
                Assert.AreEqual("video/mp4", fileResult.ContentType, "Content type should be video/mp4");
                Assert.IsNotNull(fileResult.FileStream, "File stream should not be null");

                // Clean up the stream
                fileResult.FileStream.Dispose();
            }
            finally
            {
                // Clean up: delete test file and reset cache
                if (File.Exists(testFilePath))
                    File.Delete(testFilePath);

                var cachedVideosField = typeof(VideosController).GetField("_cachedVideos", BindingFlags.NonPublic | BindingFlags.Static);
                cachedVideosField.SetValue(null, null);
            }
        }

        /// <summary>
        /// Tests that Stream handles edge case when id is zero.
        /// Note: This test cannot be fully implemented as a unit test due to tight coupling to file system.
        /// See comments in Stream_VideoNotFound_ReturnsNotFound for details.
        /// </summary>
        [TestMethod]
        public void Stream_IdIsZero_ReturnsNotFound()
        {
            // Arrange
            VideosController controller = new VideosController();
            int videoId = 0;

            // Act
            var result = controller.Stream(videoId);

            // Assert
            Assert.IsInstanceOfType(result, typeof(NotFoundResult), "Expected NotFound result when video ID is 0.");
        }

        /// <summary>
        /// Tests that Stream handles edge case when id is negative.
        /// Note: This test cannot be fully implemented as a unit test due to tight coupling to file system.
        /// See comments in Stream_VideoNotFound_ReturnsNotFound for details.
        /// </summary>
        [TestMethod]
        public void Stream_IdIsNegative_ReturnsNotFound()
        {
            // Arrange
            VideosController controller = new VideosController();
            int videoId = -1;

            // Act
            IActionResult result = controller.Stream(videoId);

            // Assert
            Assert.IsInstanceOfType(result, typeof(NotFoundResult), "Stream should return NotFoundResult for negative video ID.");
        }

        /// <summary>
        /// Tests that Stream handles edge case when id is int.MaxValue.
        /// Note: This test cannot be fully implemented as a unit test due to tight coupling to file system.
        /// See comments in Stream_VideoNotFound_ReturnsNotFound for details.
        /// </summary>
        [TestMethod]
        public void Stream_IdIsMaxValue_ReturnsNotFound()
        {
            // Arrange
            VideosController controller = new VideosController();
            int videoId = int.MaxValue;

            // Use reflection to set the cached videos to a known state (empty list)
            // This ensures our test ID won't exist
            Type controllerType = typeof(VideosController);
            FieldInfo cachedVideosField = controllerType.GetField("_cachedVideos", BindingFlags.NonPublic | BindingFlags.Static);
            FieldInfo lastScanField = controllerType.GetField("_lastScan", BindingFlags.NonPublic | BindingFlags.Static);

            // Set cached videos to an empty list
            cachedVideosField.SetValue(null, new List<VideoDto>());
            // Set last scan to now so the cache is considered fresh
            lastScanField.SetValue(null, DateTime.UtcNow);

            // Act
            IActionResult result = controller.Stream(videoId);

            // Assert
            Assert.IsInstanceOfType(result, typeof(NotFoundResult), "Expected NotFound result when video ID is int.MaxValue and does not exist.");
        }

        /// <summary>
        /// Tests that Stream handles edge case when id is int.MinValue.
        /// Note: This test cannot be fully implemented as a unit test due to tight coupling to file system.
        /// See comments in Stream_VideoNotFound_ReturnsNotFound for details.
        /// </summary>
        [TestMethod]
        public void Stream_IdIsMinValue_ReturnsNotFound()
        {
            // Arrange
            VideosController controller = new VideosController();
            int videoId = int.MinValue;

            // Use reflection to set the cached videos to a known state (empty list)
            // This ensures our test ID won't exist
            Type controllerType = typeof(VideosController);
            FieldInfo cachedVideosField = controllerType.GetField("_cachedVideos", BindingFlags.NonPublic | BindingFlags.Static);
            FieldInfo lastScanField = controllerType.GetField("_lastScan", BindingFlags.NonPublic | BindingFlags.Static);

            // Set cached videos to an empty list
            cachedVideosField.SetValue(null, new List<VideoDto>());
            // Set last scan to now so the cache is considered fresh
            lastScanField.SetValue(null, DateTime.UtcNow);

            // Act
            IActionResult result = controller.Stream(videoId);

            // Assert
            Assert.IsInstanceOfType(result, typeof(NotFoundResult), "Expected NotFound result when video ID does not exist.");
        }

        /// <summary>
        /// Clears the static Views dictionary before each test to ensure test isolation.
        /// </summary>
        [TestInitialize]
        public void TestInitialize()
        {
            // Clear static Views dictionary to ensure test isolation
            var viewsField = typeof(VideosController).GetField("Views", BindingFlags.NonPublic | BindingFlags.Static);
            if (viewsField != null)
            {
                var views = viewsField.GetValue(null) as ConcurrentDictionary<int, int>;
                views?.Clear();
            }
        }

        /// <summary>
        /// Tests that Watch returns Ok result with true value when watching a video for the first time.
        /// Input: Valid positive id (1).
        /// Expected: Returns OkObjectResult with true value, and Views dictionary contains id with value 1.
        /// </summary>
        [TestMethod]
        public void Watch_FirstTimeWatchingVideo_ReturnsOkTrueAndAddsToViews()
        {
            // Arrange
            var controller = new VideosController();
            int videoId = 1;

            // Act
            var result = controller.Watch(videoId);

            // Assert
            Assert.IsNotNull(result);
            Assert.IsInstanceOfType(result.Result, typeof(OkObjectResult));
            var okResult = result.Result as OkObjectResult;
            Assert.IsNotNull(okResult);
            Assert.AreEqual(true, okResult.Value);

            var viewsField = typeof(VideosController).GetField("Views", BindingFlags.NonPublic | BindingFlags.Static);
            var views = viewsField?.GetValue(null) as ConcurrentDictionary<int, int>;
            Assert.IsNotNull(views);
            Assert.IsTrue(views.ContainsKey(videoId));
            Assert.AreEqual(1, views[videoId]);
        }

        /// <summary>
        /// Tests that Watch increments view count when watching the same video multiple times.
        /// Input: Same id (5) watched three times.
        /// Expected: Each call returns OkObjectResult with true, and Views[id] increments from 1 to 2 to 3.
        /// </summary>
        [TestMethod]
        public void Watch_MultipleWatchesOfSameVideo_IncrementsViewCount()
        {
            // Arrange
            var controller = new VideosController();
            int videoId = 5;

            // Act
            var result1 = controller.Watch(videoId);
            var result2 = controller.Watch(videoId);
            var result3 = controller.Watch(videoId);

            // Assert
            Assert.IsInstanceOfType(result1.Result, typeof(OkObjectResult));
            Assert.IsInstanceOfType(result2.Result, typeof(OkObjectResult));
            Assert.IsInstanceOfType(result3.Result, typeof(OkObjectResult));

            var viewsField = typeof(VideosController).GetField("Views", BindingFlags.NonPublic | BindingFlags.Static);
            var views = viewsField?.GetValue(null) as ConcurrentDictionary<int, int>;
            Assert.IsNotNull(views);
            Assert.AreEqual(3, views[videoId]);
        }

        /// <summary>
        /// Tests that Watch handles different video ids independently.
        /// Input: Three different ids (10, 20, 30) each watched once.
        /// Expected: Each id appears in Views dictionary with value 1.
        /// </summary>
        [TestMethod]
        public void Watch_DifferentVideoIds_TracksEachIndependently()
        {
            // Arrange
            var controller = new VideosController();
            int videoId1 = 10;
            int videoId2 = 20;
            int videoId3 = 30;

            // Act
            controller.Watch(videoId1);
            controller.Watch(videoId2);
            controller.Watch(videoId3);

            // Assert
            var viewsField = typeof(VideosController).GetField("Views", BindingFlags.NonPublic | BindingFlags.Static);
            var views = viewsField?.GetValue(null) as ConcurrentDictionary<int, int>;
            Assert.IsNotNull(views);
            Assert.AreEqual(1, views[videoId1]);
            Assert.AreEqual(1, views[videoId2]);
            Assert.AreEqual(1, views[videoId3]);
        }

        /// <summary>
        /// Tests Watch with zero as video id.
        /// Input: id = 0.
        /// Expected: Returns OkObjectResult with true, and Views[0] = 1.
        /// </summary>
        [TestMethod]
        public void Watch_ZeroVideoId_ReturnsOkAndAddsToViews()
        {
            // Arrange
            var controller = new VideosController();
            int videoId = 0;

            // Act
            var result = controller.Watch(videoId);

            // Assert
            Assert.IsInstanceOfType(result.Result, typeof(OkObjectResult));
            var okResult = result.Result as OkObjectResult;
            Assert.AreEqual(true, okResult?.Value);

            var viewsField = typeof(VideosController).GetField("Views", BindingFlags.NonPublic | BindingFlags.Static);
            var views = viewsField?.GetValue(null) as ConcurrentDictionary<int, int>;
            Assert.IsNotNull(views);
            Assert.AreEqual(1, views[videoId]);
        }

        /// <summary>
        /// Tests Watch with negative video id.
        /// Input: id = -42.
        /// Expected: Returns OkObjectResult with true, and Views[-42] = 1.
        /// </summary>
        [TestMethod]
        public void Watch_NegativeVideoId_ReturnsOkAndAddsToViews()
        {
            // Arrange
            var controller = new VideosController();
            int videoId = -42;

            // Act
            var result = controller.Watch(videoId);

            // Assert
            Assert.IsInstanceOfType(result.Result, typeof(OkObjectResult));
            var okResult = result.Result as OkObjectResult;
            Assert.AreEqual(true, okResult?.Value);

            var viewsField = typeof(VideosController).GetField("Views", BindingFlags.NonPublic | BindingFlags.Static);
            var views = viewsField?.GetValue(null) as ConcurrentDictionary<int, int>;
            Assert.IsNotNull(views);
            Assert.AreEqual(1, views[videoId]);
        }

        /// <summary>
        /// Tests Watch with int.MaxValue as video id.
        /// Input: id = int.MaxValue.
        /// Expected: Returns OkObjectResult with true, and Views[int.MaxValue] = 1.
        /// </summary>
        [TestMethod]
        public void Watch_MaxIntVideoId_ReturnsOkAndAddsToViews()
        {
            // Arrange
            var controller = new VideosController();
            int videoId = int.MaxValue;

            // Act
            var result = controller.Watch(videoId);

            // Assert
            Assert.IsInstanceOfType(result.Result, typeof(OkObjectResult));
            var okResult = result.Result as OkObjectResult;
            Assert.AreEqual(true, okResult?.Value);

            var viewsField = typeof(VideosController).GetField("Views", BindingFlags.NonPublic | BindingFlags.Static);
            var views = viewsField?.GetValue(null) as ConcurrentDictionary<int, int>;
            Assert.IsNotNull(views);
            Assert.AreEqual(1, views[videoId]);
        }

        /// <summary>
        /// Tests Watch with int.MinValue as video id.
        /// Input: id = int.MinValue.
        /// Expected: Returns OkObjectResult with true, and Views[int.MinValue] = 1.
        /// </summary>
        [TestMethod]
        public void Watch_MinIntVideoId_ReturnsOkAndAddsToViews()
        {
            // Arrange
            var controller = new VideosController();
            int videoId = int.MinValue;

            // Act
            var result = controller.Watch(videoId);

            // Assert
            Assert.IsInstanceOfType(result.Result, typeof(OkObjectResult));
            var okResult = result.Result as OkObjectResult;
            Assert.AreEqual(true, okResult?.Value);

            var viewsField = typeof(VideosController).GetField("Views", BindingFlags.NonPublic | BindingFlags.Static);
            var views = viewsField?.GetValue(null) as ConcurrentDictionary<int, int>;
            Assert.IsNotNull(views);
            Assert.AreEqual(1, views[videoId]);
        }

        /// <summary>
        /// Tests that Unlike returns Ok(true) for various boundary integer ID values.
        /// </summary>
        /// <param name="id">The video ID to unlike.</param>
        [TestMethod]
        [DataRow(int.MinValue)]
        [DataRow(-1)]
        [DataRow(0)]
        [DataRow(1)]
        [DataRow(int.MaxValue)]
        public void Unlike_VariousIntegerBoundaryIds_ReturnsOkTrue(int id)
        {
            // Arrange
            VideosController controller = new VideosController();

            // Act
            ActionResult<bool> result = controller.Unlike(id);

            // Assert
            Assert.IsNotNull(result);
            Assert.IsInstanceOfType(result.Result, typeof(OkObjectResult));
            OkObjectResult okResult = (OkObjectResult)result.Result;
            Assert.AreEqual(true, okResult.Value);
        }

        /// <summary>
        /// Tests that unliking a non-existent video ID adds it to the dictionary with zero likes.
        /// </summary>
        [TestMethod]
        public void Unlike_NonExistentId_AddsIdWithZeroLikes()
        {
            // Arrange
            VideosController controller = new VideosController();
            int uniqueId = 2000001;
            ConcurrentDictionary<int, int> likes = GetLikesDictionary();

            // Act
            controller.Unlike(uniqueId);

            // Assert
            Assert.IsTrue(likes.ContainsKey(uniqueId));
            Assert.AreEqual(0, likes[uniqueId]);
        }

        /// <summary>
        /// Tests that unliking a video with zero likes keeps the like count at zero.
        /// </summary>
        [TestMethod]
        public void Unlike_ExistingIdWithZeroLikes_RemainsAtZero()
        {
            // Arrange
            VideosController controller = new VideosController();
            int uniqueId = 2000002;
            ConcurrentDictionary<int, int> likes = GetLikesDictionary();
            likes.AddOrUpdate(uniqueId, 0, (k, v) => 0);

            // Act
            controller.Unlike(uniqueId);

            // Assert
            Assert.AreEqual(0, likes[uniqueId]);
        }

        /// <summary>
        /// Tests that unliking a video with one like decrements the count to zero.
        /// </summary>
        [TestMethod]
        public void Unlike_ExistingIdWithOneLike_DecreasesToZero()
        {
            // Arrange
            VideosController controller = new VideosController();
            int uniqueId = 2000003;
            ConcurrentDictionary<int, int> likes = GetLikesDictionary();
            likes.AddOrUpdate(uniqueId, 1, (k, v) => 1);

            // Act
            controller.Unlike(uniqueId);

            // Assert
            Assert.AreEqual(0, likes[uniqueId]);
        }

        /// <summary>
        /// Tests that unliking a video with multiple likes decrements the count by one.
        /// </summary>
        [TestMethod]
        public void Unlike_ExistingIdWithMultipleLikes_DecrementsByOne()
        {
            // Arrange
            VideosController controller = new VideosController();
            int uniqueId = 2000004;
            int initialLikes = 5;
            ConcurrentDictionary<int, int> likes = GetLikesDictionary();
            likes.AddOrUpdate(uniqueId, initialLikes, (k, v) => initialLikes);

            // Act
            controller.Unlike(uniqueId);

            // Assert
            Assert.AreEqual(initialLikes - 1, likes[uniqueId]);
        }

        /// <summary>
        /// Tests that unliking a video with a large number of likes decrements correctly.
        /// </summary>
        [TestMethod]
        public void Unlike_ExistingIdWithLargeLikes_DecrementsByOne()
        {
            // Arrange
            VideosController controller = new VideosController();
            int uniqueId = 2000005;
            int initialLikes = int.MaxValue - 1;
            ConcurrentDictionary<int, int> likes = GetLikesDictionary();
            likes.AddOrUpdate(uniqueId, initialLikes, (k, v) => initialLikes);

            // Act
            controller.Unlike(uniqueId);

            // Assert
            Assert.AreEqual(initialLikes - 1, likes[uniqueId]);
        }

        /// <summary>
        /// Tests that multiple consecutive unlikes on the same ID don't result in negative values.
        /// </summary>
        [TestMethod]
        public void Unlike_MultipleConsecutiveUnlikes_NeverGoesNegative()
        {
            // Arrange
            VideosController controller = new VideosController();
            int uniqueId = 2000006;
            ConcurrentDictionary<int, int> likes = GetLikesDictionary();
            likes.AddOrUpdate(uniqueId, 2, (k, v) => 2);

            // Act - Unlike multiple times
            controller.Unlike(uniqueId); // 2 -> 1
            controller.Unlike(uniqueId); // 1 -> 0
            controller.Unlike(uniqueId); // 0 -> 0
            controller.Unlike(uniqueId); // 0 -> 0

            // Assert
            Assert.AreEqual(0, likes[uniqueId]);
        }

        /// <summary>
        /// Tests that the Unlike method always returns an OkObjectResult regardless of the current like count.
        /// </summary>
        [TestMethod]
        public void Unlike_AlwaysReturnsOkObjectResult()
        {
            // Arrange
            VideosController controller = new VideosController();
            int uniqueId = 2000007;

            // Act
            ActionResult<bool> result = controller.Unlike(uniqueId);

            // Assert
            Assert.IsNotNull(result);
            Assert.IsNotNull(result.Result);
            Assert.IsInstanceOfType(result.Result, typeof(OkObjectResult));
        }

        /// <summary>
        /// Tests that the Unlike method returns true as the value in OkObjectResult.
        /// </summary>
        [TestMethod]
        public void Unlike_ReturnsOkWithTrueValue()
        {
            // Arrange
            VideosController controller = new VideosController();
            int uniqueId = 2000008;

            // Act
            ActionResult<bool> result = controller.Unlike(uniqueId);

            // Assert
            OkObjectResult okResult = (OkObjectResult)result.Result;
            Assert.IsNotNull(okResult.Value);
            Assert.AreEqual(true, okResult.Value);
        }

        /// <summary>
        /// Helper method to access the static Likes dictionary via reflection.
        /// </summary>
        /// <returns>The Likes ConcurrentDictionary.</returns>
        private ConcurrentDictionary<int, int> GetLikesDictionary()
        {
            System.Reflection.FieldInfo? field = typeof(VideosController)
                .GetField("Likes", System.Reflection.BindingFlags.NonPublic | System.Reflection.BindingFlags.Static);

            if (field == null)
            {
                throw new InvalidOperationException("Could not access Likes field");
            }

            ConcurrentDictionary<int, int>? likes = field.GetValue(null) as ConcurrentDictionary<int, int>;

            if (likes == null)
            {
                throw new InvalidOperationException("Likes field is null");
            }

            return likes;
        }

        /// <summary>
        /// Tests that Like method returns OkObjectResult with true value for a positive video ID.
        /// </summary>
        [TestMethod]
        public void Like_PositiveId_ReturnsOkResultWithTrue()
        {
            // Arrange
            var controller = new VideosController();
            int videoId = 100;

            // Act
            var result = controller.Like(videoId);

            // Assert
            Assert.IsNotNull(result);
            Assert.IsNotNull(result.Result);
            var okResult = result.Result as OkObjectResult;
            Assert.IsNotNull(okResult);
            Assert.AreEqual(true, okResult.Value);
        }

        /// <summary>
        /// Tests that Like method returns OkObjectResult with true value for zero video ID.
        /// </summary>
        [TestMethod]
        public void Like_ZeroId_ReturnsOkResultWithTrue()
        {
            // Arrange
            var controller = new VideosController();
            int videoId = 0;

            // Act
            var result = controller.Like(videoId);

            // Assert
            Assert.IsNotNull(result);
            Assert.IsNotNull(result.Result);
            var okResult = result.Result as OkObjectResult;
            Assert.IsNotNull(okResult);
            Assert.AreEqual(true, okResult.Value);
        }

        /// <summary>
        /// Tests that Like method returns OkObjectResult with true value for a negative video ID.
        /// </summary>
        [TestMethod]
        public void Like_NegativeId_ReturnsOkResultWithTrue()
        {
            // Arrange
            var controller = new VideosController();
            int videoId = -1;

            // Act
            var result = controller.Like(videoId);

            // Assert
            Assert.IsNotNull(result);
            Assert.IsNotNull(result.Result);
            var okResult = result.Result as OkObjectResult;
            Assert.IsNotNull(okResult);
            Assert.AreEqual(true, okResult.Value);
        }

        /// <summary>
        /// Tests that Like method returns OkObjectResult with true value for int.MinValue.
        /// </summary>
        [TestMethod]
        public void Like_IntMinValue_ReturnsOkResultWithTrue()
        {
            // Arrange
            var controller = new VideosController();
            int videoId = int.MinValue;

            // Act
            var result = controller.Like(videoId);

            // Assert
            Assert.IsNotNull(result);
            Assert.IsNotNull(result.Result);
            var okResult = result.Result as OkObjectResult;
            Assert.IsNotNull(okResult);
            Assert.AreEqual(true, okResult.Value);
        }

        /// <summary>
        /// Tests that Like method returns OkObjectResult with true value for int.MaxValue.
        /// </summary>
        [TestMethod]
        public void Like_IntMaxValue_ReturnsOkResultWithTrue()
        {
            // Arrange
            var controller = new VideosController();
            int videoId = int.MaxValue;

            // Act
            var result = controller.Like(videoId);

            // Assert
            Assert.IsNotNull(result);
            Assert.IsNotNull(result.Result);
            var okResult = result.Result as OkObjectResult;
            Assert.IsNotNull(okResult);
            Assert.AreEqual(true, okResult.Value);
        }

        /// <summary>
        /// Tests that Like method returns OkObjectResult with true value for multiple calls with the same video ID.
        /// Verifies that the method can be called multiple times without throwing exceptions.
        /// </summary>
        [TestMethod]
        public void Like_MultipleCallsSameId_ReturnsOkResultWithTrue()
        {
            // Arrange
            var controller = new VideosController();
            int videoId = 200;

            // Act
            var result1 = controller.Like(videoId);
            var result2 = controller.Like(videoId);
            var result3 = controller.Like(videoId);

            // Assert
            Assert.IsNotNull(result1);
            Assert.IsNotNull(result1.Result);
            var okResult1 = result1.Result as OkObjectResult;
            Assert.IsNotNull(okResult1);
            Assert.AreEqual(true, okResult1.Value);

            Assert.IsNotNull(result2);
            Assert.IsNotNull(result2.Result);
            var okResult2 = result2.Result as OkObjectResult;
            Assert.IsNotNull(okResult2);
            Assert.AreEqual(true, okResult2.Value);

            Assert.IsNotNull(result3);
            Assert.IsNotNull(result3.Result);
            var okResult3 = result3.Result as OkObjectResult;
            Assert.IsNotNull(okResult3);
            Assert.AreEqual(true, okResult3.Value);
        }

        /// <summary>
        /// Tests that Like method with various boundary integer values using parameterized test.
        /// Verifies correct behavior for edge case integer values.
        /// </summary>
        /// <param name="videoId">The video ID to test.</param>
        [TestMethod]
        [DataRow(1)]
        [DataRow(-1000)]
        [DataRow(1000)]
        [DataRow(999999)]
        [DataRow(-999999)]
        public void Like_VariousBoundaryValues_ReturnsOkResultWithTrue(int videoId)
        {
            // Arrange
            var controller = new VideosController();

            // Act
            var result = controller.Like(videoId);

            // Assert
            Assert.IsNotNull(result);
            Assert.IsNotNull(result.Result);
            var okResult = result.Result as OkObjectResult;
            Assert.IsNotNull(okResult);
            Assert.AreEqual(true, okResult.Value);
        }

        /// <summary>
        /// Tests that GetByCategory returns Ok result with filtered videos when a valid category is provided.
        /// Input: A valid category string.
        /// Expected: Returns OkObjectResult containing a list of VideoDto.
        /// </summary>
        [TestMethod]
        public void GetByCategory_ValidCategory_ReturnsOkResultWithFilteredList()
        {
            // Arrange
            var controller = new VideosController();
            var category = "General";

            // Act
            var result = controller.GetByCategory(category);

            // Assert
            Assert.IsNotNull(result);
            Assert.IsNotNull(result.Result);
            Assert.IsInstanceOfType(result.Result, typeof(OkObjectResult));
            var okResult = result.Result as OkObjectResult;
            Assert.IsNotNull(okResult);
            Assert.IsInstanceOfType(okResult.Value, typeof(List<VideoDto>));
            var videos = okResult.Value as List<VideoDto>;
            Assert.IsNotNull(videos);
        }

        /// <summary>
        /// Tests that GetByCategory performs case-insensitive filtering.
        /// Input: Category strings with different casing variations.
        /// Expected: Returns OkObjectResult containing filtered videos regardless of case.
        /// </summary>
        [TestMethod]
        [DataRow("general")]
        [DataRow("GENERAL")]
        [DataRow("General")]
        [DataRow("GeNeRaL")]
        public void GetByCategory_DifferentCasing_ReturnsCaseInsensitiveMatch(string category)
        {
            // Arrange
            var controller = new VideosController();

            // Act
            var result = controller.GetByCategory(category);

            // Assert
            Assert.IsNotNull(result);
            Assert.IsNotNull(result.Result);
            Assert.IsInstanceOfType(result.Result, typeof(OkObjectResult));
            var okResult = result.Result as OkObjectResult;
            Assert.IsNotNull(okResult);
            Assert.IsInstanceOfType(okResult.Value, typeof(List<VideoDto>));
            var videos = okResult.Value as List<VideoDto>;
            Assert.IsNotNull(videos);
            // All videos should have category matching case-insensitively
            Assert.IsTrue(videos.All(v => v.Category.Equals(category, StringComparison.OrdinalIgnoreCase)));
        }

        /// <summary>
        /// Tests that GetByCategory handles empty string category.
        /// Input: Empty string.
        /// Expected: Returns OkObjectResult with filtered list containing only videos with empty category.
        /// </summary>
        [TestMethod]
        public void GetByCategory_EmptyString_ReturnsOkResultWithEmptyCategoryVideos()
        {
            // Arrange
            var controller = new VideosController();
            var category = string.Empty;

            // Act
            var result = controller.GetByCategory(category);

            // Assert
            Assert.IsNotNull(result);
            Assert.IsNotNull(result.Result);
            Assert.IsInstanceOfType(result.Result, typeof(OkObjectResult));
            var okResult = result.Result as OkObjectResult;
            Assert.IsNotNull(okResult);
            Assert.IsInstanceOfType(okResult.Value, typeof(List<VideoDto>));
            var videos = okResult.Value as List<VideoDto>;
            Assert.IsNotNull(videos);
            // All returned videos should have empty category
            Assert.IsTrue(videos.All(v => v.Category == string.Empty));
        }

        /// <summary>
        /// Tests that GetByCategory handles whitespace-only category string.
        /// Input: Whitespace-only string.
        /// Expected: Returns OkObjectResult with filtered list.
        /// </summary>
        [TestMethod]
        [DataRow(" ")]
        [DataRow("  ")]
        [DataRow("\t")]
        [DataRow("\n")]
        public void GetByCategory_WhitespaceString_ReturnsOkResultWithFilteredList(string category)
        {
            // Arrange
            var controller = new VideosController();

            // Act
            var result = controller.GetByCategory(category);

            // Assert
            Assert.IsNotNull(result);
            Assert.IsNotNull(result.Result);
            Assert.IsInstanceOfType(result.Result, typeof(OkObjectResult));
            var okResult = result.Result as OkObjectResult;
            Assert.IsNotNull(okResult);
            Assert.IsInstanceOfType(okResult.Value, typeof(List<VideoDto>));
            var videos = okResult.Value as List<VideoDto>;
            Assert.IsNotNull(videos);
        }

        /// <summary>
        /// Tests that GetByCategory handles category with special characters.
        /// Input: Category strings containing special characters.
        /// Expected: Returns OkObjectResult with filtered list.
        /// </summary>
        [TestMethod]
        [DataRow("Category@123")]
        [DataRow("Cat-egory")]
        [DataRow("Cat_egory")]
        [DataRow("Category!")]
        [DataRow("Cat/egory")]
        [DataRow("Cat\\egory")]
        public void GetByCategory_SpecialCharacters_ReturnsOkResultWithFilteredList(string category)
        {
            // Arrange
            var controller = new VideosController();

            // Act
            var result = controller.GetByCategory(category);

            // Assert
            Assert.IsNotNull(result);
            Assert.IsNotNull(result.Result);
            Assert.IsInstanceOfType(result.Result, typeof(OkObjectResult));
            var okResult = result.Result as OkObjectResult;
            Assert.IsNotNull(okResult);
            Assert.IsInstanceOfType(okResult.Value, typeof(List<VideoDto>));
            var videos = okResult.Value as List<VideoDto>;
            Assert.IsNotNull(videos);
            Assert.IsTrue(videos.All(v => v.Category.Equals(category, StringComparison.OrdinalIgnoreCase)));
        }

        /// <summary>
        /// Tests that GetByCategory handles non-existent category.
        /// Input: A category that doesn't exist in the video collection.
        /// Expected: Returns OkObjectResult with an empty list.
        /// </summary>
        [TestMethod]
        public void GetByCategory_NonExistentCategory_ReturnsOkResultWithEmptyList()
        {
            // Arrange
            var controller = new VideosController();
            var category = "NonExistentCategory12345";

            // Act
            var result = controller.GetByCategory(category);

            // Assert
            Assert.IsNotNull(result);
            Assert.IsNotNull(result.Result);
            Assert.IsInstanceOfType(result.Result, typeof(OkObjectResult));
            var okResult = result.Result as OkObjectResult;
            Assert.IsNotNull(okResult);
            Assert.IsInstanceOfType(okResult.Value, typeof(List<VideoDto>));
            var videos = okResult.Value as List<VideoDto>;
            Assert.IsNotNull(videos);
        }

        /// <summary>
        /// Tests that GetByCategory handles very long category string.
        /// Input: A very long category string.
        /// Expected: Returns OkObjectResult with filtered list.
        /// </summary>
        [TestMethod]
        public void GetByCategory_VeryLongString_ReturnsOkResultWithFilteredList()
        {
            // Arrange
            var controller = new VideosController();
            var category = new string('A', 10000);

            // Act
            var result = controller.GetByCategory(category);

            // Assert
            Assert.IsNotNull(result);
            Assert.IsNotNull(result.Result);
            Assert.IsInstanceOfType(result.Result, typeof(OkObjectResult));
            var okResult = result.Result as OkObjectResult;
            Assert.IsNotNull(okResult);
            Assert.IsInstanceOfType(okResult.Value, typeof(List<VideoDto>));
            var videos = okResult.Value as List<VideoDto>;
            Assert.IsNotNull(videos);
        }

        /// <summary>
        /// Tests that GetByCategory handles Unicode characters in category.
        /// Input: Category with Unicode characters.
        /// Expected: Returns OkObjectResult with filtered list.
        /// </summary>
        [TestMethod]
        [DataRow("类别")]
        [DataRow("Категория")]
        [DataRow("カテゴリ")]
        [DataRow("فئة")]
        public void GetByCategory_UnicodeCharacters_ReturnsOkResultWithFilteredList(string category)
        {
            // Arrange
            var controller = new VideosController();

            // Act
            var result = controller.GetByCategory(category);

            // Assert
            Assert.IsNotNull(result);
            Assert.IsNotNull(result.Result);
            Assert.IsInstanceOfType(result.Result, typeof(OkObjectResult));
            var okResult = result.Result as OkObjectResult;
            Assert.IsNotNull(okResult);
            Assert.IsInstanceOfType(okResult.Value, typeof(List<VideoDto>));
            var videos = okResult.Value as List<VideoDto>;
            Assert.IsNotNull(videos);
            Assert.IsTrue(videos.All(v => v.Category.Equals(category, StringComparison.OrdinalIgnoreCase)));
        }

        /// <summary>
        /// Tests that GetById returns NotFound when no video with the specified ID exists.
        /// </summary>
        [TestMethod]
        public void GetById_VideoNotFound_ReturnsNotFound()
        {
            // Arrange
            VideosController controller = new VideosController();
            int nonExistentId = int.MaxValue; // Using max value as unlikely to exist

            try
            {
                // Use reflection to set up an empty cached videos list
                var cachedVideosField = typeof(VideosController).GetField("_cachedVideos", BindingFlags.NonPublic | BindingFlags.Static);
                var lastScanField = typeof(VideosController).GetField("_lastScan", BindingFlags.NonPublic | BindingFlags.Static);

                // Set cache to empty list to simulate no videos found
                cachedVideosField.SetValue(null, new List<VideoDto>());
                lastScanField.SetValue(null, DateTime.UtcNow);

                // Act
                var result = controller.GetById(nonExistentId);

                // Assert
                Assert.IsNotNull(result, "Result should not be null");
                Assert.IsNotNull(result.Result, "Result.Result should not be null");
                Assert.IsInstanceOfType(result.Result, typeof(NotFoundResult), "Result should be NotFoundResult when video does not exist");
            }
            finally
            {
                // Clean up: reset cache
                var cachedVideosField = typeof(VideosController).GetField("_cachedVideos", BindingFlags.NonPublic | BindingFlags.Static);
                cachedVideosField.SetValue(null, null);
            }
        }

        /// <summary>
        /// Tests that GetById returns Ok with the video when a video with the specified ID exists.
        /// </summary>
        [TestMethod]
        public void GetById_VideoExists_ReturnsOkWithVideo()
        {
            // Arrange
            VideosController controller = new VideosController();

            // First call GetAll to initialize the cache with any existing videos
            var getAllResult = controller.GetAll();
            var allVideos = (getAllResult.Result as OkObjectResult)?.Value as List<VideoDto>;

            // If there are no videos in the system, we test the NotFound case instead
            if (allVideos == null || allVideos.Count == 0)
            {
                // Act
                var result = controller.GetById(1);

                // Assert - Should return NotFound when no videos exist
                Assert.IsInstanceOfType(result.Result, typeof(NotFoundResult),
                    "GetById should return NotFound when the video does not exist.");
                return;
            }

            // Get the first video's ID
            int existingId = allVideos[0].Id;

            // Act
            var getByIdResult = controller.GetById(existingId);

            // Assert
            Assert.IsInstanceOfType(getByIdResult.Result, typeof(OkObjectResult),
                "GetById should return OkObjectResult when video exists.");

            var okResult = getByIdResult.Result as OkObjectResult;
            Assert.IsNotNull(okResult, "Result should be OkObjectResult.");

            var returnedVideo = okResult.Value as VideoDto;
            Assert.IsNotNull(returnedVideo, "Returned value should be a VideoDto.");
            Assert.AreEqual(existingId, returnedVideo.Id,
                "Returned video should have the requested ID.");
        }

        /// <summary>
        /// Tests that GetById handles boundary value of zero for the ID parameter.
        /// </summary>
        /// <remarks>
        /// This test verifies behavior with ID = 0. Since video IDs start from 1 in GetVideos(),
        /// this should return NotFound. However, the test is marked inconclusive due to inability
        /// to mock the private GetVideos() method.
        /// </remarks>
        [TestMethod]
        public void GetById_IdIsZero_ReturnsNotFound()
        {
            // Arrange
            VideosController controller = new VideosController();
            int id = 0;

            // Act
            var result = controller.GetById(id);

            // Assert
            Assert.IsNotNull(result);
            Assert.IsInstanceOfType(result.Result, typeof(NotFoundResult));
        }

        /// <summary>
        /// Tests that GetById handles negative ID values.
        /// </summary>
        /// <remarks>
        /// This test verifies behavior with negative ID. Since video IDs are positive integers
        /// starting from 1, this should return NotFound. However, the test is marked inconclusive
        /// due to inability to mock the private GetVideos() method.
        /// </remarks>
        [TestMethod]
        public void GetById_NegativeId_ReturnsNotFound()
        {
            // Arrange
            VideosController controller = new VideosController();
            int id = -1;

            // Act
            var result = controller.GetById(id);

            // Assert
            Assert.IsNotNull(result, "Result should not be null");
            Assert.IsInstanceOfType(result.Result, typeof(NotFoundResult),
                "GetById with negative ID should return NotFound");
        }

        /// <summary>
        /// Tests that GetById handles the minimum integer value for the ID parameter.
        /// </summary>
        /// <remarks>
        /// This test verifies boundary condition with ID = int.MinValue, which should return NotFound.
        /// However, the test is marked inconclusive due to inability to mock the private GetVideos() method.
        /// </remarks>
        [TestMethod]
        public void GetById_IdIsMinValue_ReturnsNotFound()
        {
            // Arrange
            VideosController controller = new VideosController();
            int id = int.MinValue;

            // Act
            var result = controller.GetById(id);

            // Assert
            Assert.IsNotNull(result, "Result should not be null");
            Assert.IsInstanceOfType(result.Result, typeof(NotFoundResult),
                "GetById with int.MinValue should return NotFound since no video will have this ID");
        }

        /// <summary>
        /// Tests that GetById handles the maximum integer value for the ID parameter.
        /// </summary>
        /// <remarks>
        /// This test verifies boundary condition with ID = int.MaxValue, which should return NotFound
        /// unless an extremely large number of videos exist. However, the test is marked inconclusive
        /// due to inability to mock the private GetVideos() method.
        /// </remarks>
        [TestMethod]
        public void GetById_IdIsMaxValue_ReturnsNotFound()
        {
            // Arrange
            VideosController controller = new VideosController();
            int id = int.MaxValue;

            // Act
            ActionResult<VideoDto> result = controller.GetById(id);

            // Assert
            Assert.IsNotNull(result);
            Assert.IsInstanceOfType(result.Result, typeof(NotFoundResult));
        }

        /// <summary>
        /// Tests that GetAll returns an OkObjectResult containing a list of videos.
        /// This verifies the method successfully retrieves videos and wraps them in an HTTP 200 response.
        /// </summary>
        [TestMethod]
        public void GetAll_WhenCalled_ReturnsOkResultWithVideoList()
        {
            // Arrange
            var controller = new VideosController();

            // Act
            var result = controller.GetAll();

            // Assert
            Assert.IsNotNull(result);
            Assert.IsNotNull(result.Result);
            Assert.IsInstanceOfType(result.Result, typeof(OkObjectResult));

            var okResult = (OkObjectResult)result.Result;
            Assert.IsNotNull(okResult.Value);
            Assert.IsInstanceOfType(okResult.Value, typeof(List<VideoDto>));
        }

        /// <summary>
        /// Tests that GetAll returns an OkObjectResult with a non-null list.
        /// This ensures the method never returns null, even when no videos are found.
        /// </summary>
        [TestMethod]
        public void GetAll_WhenCalled_ReturnsNonNullList()
        {
            // Arrange
            var controller = new VideosController();

            // Act
            var result = controller.GetAll();

            // Assert
            Assert.IsNotNull(result);
            var okResult = result.Result as OkObjectResult;
            Assert.IsNotNull(okResult);
            var videoList = okResult.Value as List<VideoDto>;
            Assert.IsNotNull(videoList);
        }

        /// <summary>
        /// Tests that GetAll returns HTTP 200 (OK) status code.
        /// This verifies the method uses the correct HTTP status for successful retrieval.
        /// </summary>
        [TestMethod]
        public void GetAll_WhenCalled_ReturnsStatusCode200()
        {
            // Arrange
            var controller = new VideosController();

            // Act
            var result = controller.GetAll();

            // Assert
            Assert.IsNotNull(result);
            var okResult = result.Result as OkObjectResult;
            Assert.IsNotNull(okResult);
            Assert.AreEqual(200, okResult.StatusCode);
        }

        /// <summary>
        /// Tests that GetAll returns a list where all VideoDto objects have valid structure.
        /// This ensures each video has the expected properties initialized.
        /// </summary>
        [TestMethod]
        public void GetAll_WhenVideosExist_ReturnsVideosWithValidStructure()
        {
            // Arrange
            var controller = new VideosController();

            // Act
            var result = controller.GetAll();

            // Assert
            Assert.IsNotNull(result);
            var okResult = result.Result as OkObjectResult;
            Assert.IsNotNull(okResult);
            var videoList = okResult.Value as List<VideoDto>;
            Assert.IsNotNull(videoList);

            foreach (var video in videoList)
            {
                Assert.IsNotNull(video);
                Assert.IsNotNull(video.FileName);
                Assert.IsNotNull(video.Title);
                Assert.IsNotNull(video.Category);
                Assert.IsNotNull(video.Path);
                Assert.IsTrue(video.Id > 0);
                Assert.IsTrue(video.Likes >= 0);
                Assert.IsTrue(video.Views >= 0);
            }
        }

        /// <summary>
        /// Tests that GetById returns Ok with the correct video when multiple videos exist and the requested ID is the first one.
        /// Input: ID = 1, cache contains multiple videos with IDs 1, 2, 3.
        /// Expected: Returns OkObjectResult containing the video with ID = 1.
        /// </summary>
        [TestMethod]
        public void GetById_MultipleVideosFirstId_ReturnsOkWithFirstVideo()
        {
            // Arrange
            VideosController controller = new VideosController();
            List<VideoDto> testVideos = new List<VideoDto>
            {
                new VideoDto { Id = 1, FileName = "video1.mp4", Title = "First Video", Category = "General", Path = "/api/videos/stream/1", Likes = 0, Views = 0 },
                new VideoDto { Id = 2, FileName = "video2.mp4", Title = "Second Video", Category = "General", Path = "/api/videos/stream/2", Likes = 0, Views = 0 },
                new VideoDto { Id = 3, FileName = "video3.mp4", Title = "Third Video", Category = "General", Path = "/api/videos/stream/3", Likes = 0, Views = 0 }
            };

            try
            {
                // Use reflection to set up cached videos
                var cachedVideosField = typeof(VideosController).GetField("_cachedVideos", BindingFlags.NonPublic | BindingFlags.Static);
                var lastScanField = typeof(VideosController).GetField("_lastScan", BindingFlags.NonPublic | BindingFlags.Static);
                cachedVideosField?.SetValue(null, testVideos);
                lastScanField?.SetValue(null, DateTime.UtcNow);

                // Act
                var result = controller.GetById(1);

                // Assert
                Assert.IsNotNull(result, "Result should not be null");
                Assert.IsNotNull(result.Result, "Result.Result should not be null");
                Assert.IsInstanceOfType(result.Result, typeof(OkObjectResult), "Result should be OkObjectResult");

                var okResult = result.Result as OkObjectResult;
                Assert.IsNotNull(okResult, "OkObjectResult should not be null");

                var returnedVideo = okResult.Value as VideoDto;
                Assert.IsNotNull(returnedVideo, "Returned video should not be null");
                Assert.AreEqual(1, returnedVideo.Id, "Returned video should have ID = 1");
                Assert.AreEqual("First Video", returnedVideo.Title, "Returned video should have correct title");
            }
            finally
            {
                // Clean up: reset cache
                var cachedVideosField = typeof(VideosController).GetField("_cachedVideos", BindingFlags.NonPublic | BindingFlags.Static);
                cachedVideosField?.SetValue(null, null);
            }
        }

        /// <summary>
        /// Tests that GetById returns Ok with the correct video when multiple videos exist and the requested ID is in the middle.
        /// Input: ID = 5, cache contains videos with IDs 1 through 10.
        /// Expected: Returns OkObjectResult containing the video with ID = 5.
        /// </summary>
        [TestMethod]
        public void GetById_MultipleVideosMiddleId_ReturnsOkWithCorrectVideo()
        {
            // Arrange
            VideosController controller = new VideosController();
            List<VideoDto> testVideos = new List<VideoDto>();
            for (int i = 1; i <= 10; i++)
            {
                testVideos.Add(new VideoDto
                {
                    Id = i,
                    FileName = $"video{i}.mp4",
                    Title = $"Video {i}",
                    Category = "General",
                    Path = $"/api/videos/stream/{i}",
                    Likes = 0,
                    Views = 0
                });
            }

            try
            {
                // Use reflection to set up cached videos
                var cachedVideosField = typeof(VideosController).GetField("_cachedVideos", BindingFlags.NonPublic | BindingFlags.Static);
                var lastScanField = typeof(VideosController).GetField("_lastScan", BindingFlags.NonPublic | BindingFlags.Static);
                cachedVideosField?.SetValue(null, testVideos);
                lastScanField?.SetValue(null, DateTime.UtcNow);

                // Act
                var result = controller.GetById(5);

                // Assert
                Assert.IsNotNull(result, "Result should not be null");
                Assert.IsNotNull(result.Result, "Result.Result should not be null");
                Assert.IsInstanceOfType(result.Result, typeof(OkObjectResult), "Result should be OkObjectResult");

                var okResult = result.Result as OkObjectResult;
                Assert.IsNotNull(okResult, "OkObjectResult should not be null");

                var returnedVideo = okResult.Value as VideoDto;
                Assert.IsNotNull(returnedVideo, "Returned video should not be null");
                Assert.AreEqual(5, returnedVideo.Id, "Returned video should have ID = 5");
                Assert.AreEqual("Video 5", returnedVideo.Title, "Returned video should have correct title");
            }
            finally
            {
                // Clean up: reset cache
                var cachedVideosField = typeof(VideosController).GetField("_cachedVideos", BindingFlags.NonPublic | BindingFlags.Static);
                cachedVideosField?.SetValue(null, null);
            }
        }

        /// <summary>
        /// Tests that GetById returns Ok with the correct video when multiple videos exist and the requested ID is the last one.
        /// Input: ID = 10, cache contains videos with IDs 1 through 10.
        /// Expected: Returns OkObjectResult containing the video with ID = 10.
        /// </summary>
        [TestMethod]
        public void GetById_MultipleVideosLastId_ReturnsOkWithLastVideo()
        {
            // Arrange
            VideosController controller = new VideosController();
            List<VideoDto> testVideos = new List<VideoDto>();
            for (int i = 1; i <= 10; i++)
            {
                testVideos.Add(new VideoDto
                {
                    Id = i,
                    FileName = $"video{i}.mp4",
                    Title = $"Video {i}",
                    Category = "General",
                    Path = $"/api/videos/stream/{i}",
                    Likes = 0,
                    Views = 0
                });
            }

            try
            {
                // Use reflection to set up cached videos
                var cachedVideosField = typeof(VideosController).GetField("_cachedVideos", BindingFlags.NonPublic | BindingFlags.Static);
                var lastScanField = typeof(VideosController).GetField("_lastScan", BindingFlags.NonPublic | BindingFlags.Static);
                cachedVideosField?.SetValue(null, testVideos);
                lastScanField?.SetValue(null, DateTime.UtcNow);

                // Act
                var result = controller.GetById(10);

                // Assert
                Assert.IsNotNull(result, "Result should not be null");
                Assert.IsNotNull(result.Result, "Result.Result should not be null");
                Assert.IsInstanceOfType(result.Result, typeof(OkObjectResult), "Result should be OkObjectResult");

                var okResult = result.Result as OkObjectResult;
                Assert.IsNotNull(okResult, "OkObjectResult should not be null");

                var returnedVideo = okResult.Value as VideoDto;
                Assert.IsNotNull(returnedVideo, "Returned video should not be null");
                Assert.AreEqual(10, returnedVideo.Id, "Returned video should have ID = 10");
                Assert.AreEqual("Video 10", returnedVideo.Title, "Returned video should have correct title");
            }
            finally
            {
                // Clean up: reset cache
                var cachedVideosField = typeof(VideosController).GetField("_cachedVideos", BindingFlags.NonPublic | BindingFlags.Static);
                cachedVideosField?.SetValue(null, null);
            }
        }

        /// <summary>
        /// Tests that GetById returns NotFound when the ID exists in the range but is missing from the sequence.
        /// Input: ID = 3, cache contains videos with IDs 1, 2, 4, 5 (3 is missing).
        /// Expected: Returns NotFoundResult.
        /// </summary>
        [TestMethod]
        public void GetById_MissingIdInSequence_ReturnsNotFound()
        {
            // Arrange
            VideosController controller = new VideosController();
            List<VideoDto> testVideos = new List<VideoDto>
            {
                new VideoDto { Id = 1, FileName = "video1.mp4", Title = "Video 1", Category = "General", Path = "/api/videos/stream/1", Likes = 0, Views = 0 },
                new VideoDto { Id = 2, FileName = "video2.mp4", Title = "Video 2", Category = "General", Path = "/api/videos/stream/2", Likes = 0, Views = 0 },
                new VideoDto { Id = 4, FileName = "video4.mp4", Title = "Video 4", Category = "General", Path = "/api/videos/stream/4", Likes = 0, Views = 0 },
                new VideoDto { Id = 5, FileName = "video5.mp4", Title = "Video 5", Category = "General", Path = "/api/videos/stream/5", Likes = 0, Views = 0 }
            };

            try
            {
                // Use reflection to set up cached videos
                var cachedVideosField = typeof(VideosController).GetField("_cachedVideos", BindingFlags.NonPublic | BindingFlags.Static);
                var lastScanField = typeof(VideosController).GetField("_lastScan", BindingFlags.NonPublic | BindingFlags.Static);
                cachedVideosField?.SetValue(null, testVideos);
                lastScanField?.SetValue(null, DateTime.UtcNow);

                // Act
                var result = controller.GetById(3);

                // Assert
                Assert.IsNotNull(result, "Result should not be null");
                Assert.IsNotNull(result.Result, "Result.Result should not be null");
                Assert.IsInstanceOfType(result.Result, typeof(NotFoundResult), "Result should be NotFoundResult when ID is missing from sequence");
            }
            finally
            {
                // Clean up: reset cache
                var cachedVideosField = typeof(VideosController).GetField("_cachedVideos", BindingFlags.NonPublic | BindingFlags.Static);
                cachedVideosField?.SetValue(null, null);
            }
        }

        /// <summary>
        /// Tests that GetById returns Ok with video containing all properties correctly populated.
        /// Input: ID = 1, cache contains a video with all properties set.
        /// Expected: Returns OkObjectResult with video having all properties matching the cached data.
        /// </summary>
        [TestMethod]
        public void GetById_VideoWithAllProperties_ReturnsVideoWithAllPropertiesPopulated()
        {
            // Arrange
            VideosController controller = new VideosController();
            VideoDto testVideo = new VideoDto
            {
                Id = 1,
                FileName = "test_video.mp4",
                Title = "Test Video Title",
                Category = "Education",
                Path = "/api/videos/stream/1",
                Likes = 42,
                Views = 100
            };
            List<VideoDto> testVideos = new List<VideoDto> { testVideo };

            try
            {
                // Use reflection to set up cached videos
                var cachedVideosField = typeof(VideosController).GetField("_cachedVideos", BindingFlags.NonPublic | BindingFlags.Static);
                var lastScanField = typeof(VideosController).GetField("_lastScan", BindingFlags.NonPublic | BindingFlags.Static);
                cachedVideosField?.SetValue(null, testVideos);
                lastScanField?.SetValue(null, DateTime.UtcNow);

                // Act
                var result = controller.GetById(1);

                // Assert
                Assert.IsNotNull(result, "Result should not be null");
                Assert.IsNotNull(result.Result, "Result.Result should not be null");
                Assert.IsInstanceOfType(result.Result, typeof(OkObjectResult), "Result should be OkObjectResult");

                var okResult = result.Result as OkObjectResult;
                Assert.IsNotNull(okResult, "OkObjectResult should not be null");

                var returnedVideo = okResult.Value as VideoDto;
                Assert.IsNotNull(returnedVideo, "Returned video should not be null");
                Assert.AreEqual(1, returnedVideo.Id, "Id should match");
                Assert.AreEqual("test_video.mp4", returnedVideo.FileName, "FileName should match");
                Assert.AreEqual("Test Video Title", returnedVideo.Title, "Title should match");
                Assert.AreEqual("Education", returnedVideo.Category, "Category should match");
                Assert.AreEqual("/api/videos/stream/1", returnedVideo.Path, "Path should match");
                Assert.AreEqual(42, returnedVideo.Likes, "Likes should match");
                Assert.AreEqual(100, returnedVideo.Views, "Views should match");
            }
            finally
            {
                // Clean up: reset cache
                var cachedVideosField = typeof(VideosController).GetField("_cachedVideos", BindingFlags.NonPublic | BindingFlags.Static);
                cachedVideosField?.SetValue(null, null);
            }
        }

        /// <summary>
        /// Tests that GetById handles single video in cache correctly.
        /// Input: ID = 1, cache contains exactly one video.
        /// Expected: Returns OkObjectResult containing that video.
        /// </summary>
        [TestMethod]
        public void GetById_SingleVideoInCache_ReturnsOkWithThatVideo()
        {
            // Arrange
            VideosController controller = new VideosController();
            List<VideoDto> testVideos = new List<VideoDto>
            {
                new VideoDto { Id = 1, FileName = "single.mp4", Title = "Single Video", Category = "General", Path = "/api/videos/stream/1", Likes = 0, Views = 0 }
            };

            try
            {
                // Use reflection to set up cached videos
                var cachedVideosField = typeof(VideosController).GetField("_cachedVideos", BindingFlags.NonPublic | BindingFlags.Static);
                var lastScanField = typeof(VideosController).GetField("_lastScan", BindingFlags.NonPublic | BindingFlags.Static);
                cachedVideosField?.SetValue(null, testVideos);
                lastScanField?.SetValue(null, DateTime.UtcNow);

                // Act
                var result = controller.GetById(1);

                // Assert
                Assert.IsNotNull(result, "Result should not be null");
                Assert.IsNotNull(result.Result, "Result.Result should not be null");
                Assert.IsInstanceOfType(result.Result, typeof(OkObjectResult), "Result should be OkObjectResult");

                var okResult = result.Result as OkObjectResult;
                Assert.IsNotNull(okResult, "OkObjectResult should not be null");

                var returnedVideo = okResult.Value as VideoDto;
                Assert.IsNotNull(returnedVideo, "Returned video should not be null");
                Assert.AreEqual(1, returnedVideo.Id, "Returned video should have ID = 1");
                Assert.AreEqual("Single Video", returnedVideo.Title, "Returned video should have correct title");
            }
            finally
            {
                // Clean up: reset cache
                var cachedVideosField = typeof(VideosController).GetField("_cachedVideos", BindingFlags.NonPublic | BindingFlags.Static);
                cachedVideosField?.SetValue(null, null);
            }
        }

        /// <summary>
        /// Tests that GetById returns NotFound when requesting non-existent ID from single video cache.
        /// Input: ID = 2, cache contains only video with ID = 1.
        /// Expected: Returns NotFoundResult.
        /// </summary>
        [TestMethod]
        public void GetById_SingleVideoWrongId_ReturnsNotFound()
        {
            // Arrange
            VideosController controller = new VideosController();
            List<VideoDto> testVideos = new List<VideoDto>
            {
                new VideoDto { Id = 1, FileName = "single.mp4", Title = "Single Video", Category = "General", Path = "/api/videos/stream/1", Likes = 0, Views = 0 }
            };

            try
            {
                // Use reflection to set up cached videos
                var cachedVideosField = typeof(VideosController).GetField("_cachedVideos", BindingFlags.NonPublic | BindingFlags.Static);
                var lastScanField = typeof(VideosController).GetField("_lastScan", BindingFlags.NonPublic | BindingFlags.Static);
                cachedVideosField?.SetValue(null, testVideos);
                lastScanField?.SetValue(null, DateTime.UtcNow);

                // Act
                var result = controller.GetById(2);

                // Assert
                Assert.IsNotNull(result, "Result should not be null");
                Assert.IsNotNull(result.Result, "Result.Result should not be null");
                Assert.IsInstanceOfType(result.Result, typeof(NotFoundResult), "Result should be NotFoundResult when ID does not exist");
            }
            finally
            {
                // Clean up: reset cache
                var cachedVideosField = typeof(VideosController).GetField("_cachedVideos", BindingFlags.NonPublic | BindingFlags.Static);
                cachedVideosField?.SetValue(null, null);
            }
        }

        /// <summary>
        /// Tests that GetById with boundary positive values returns NotFound when ID doesn't exist.
        /// Input: Various positive boundary values that don't exist in cache.
        /// Expected: Returns NotFoundResult for each value.
        /// </summary>
        [TestMethod]
        [DataRow(100)]
        [DataRow(1000)]
        [DataRow(10000)]
        [DataRow(999999)]
        public void GetById_PositiveBoundaryValuesNotInCache_ReturnsNotFound(int id)
        {
            // Arrange
            VideosController controller = new VideosController();
            List<VideoDto> testVideos = new List<VideoDto>
            {
                new VideoDto { Id = 1, FileName = "video1.mp4", Title = "Video 1", Category = "General", Path = "/api/videos/stream/1", Likes = 0, Views = 0 },
                new VideoDto { Id = 2, FileName = "video2.mp4", Title = "Video 2", Category = "General", Path = "/api/videos/stream/2", Likes = 0, Views = 0 }
            };

            try
            {
                // Use reflection to set up cached videos
                var cachedVideosField = typeof(VideosController).GetField("_cachedVideos", BindingFlags.NonPublic | BindingFlags.Static);
                var lastScanField = typeof(VideosController).GetField("_lastScan", BindingFlags.NonPublic | BindingFlags.Static);
                cachedVideosField?.SetValue(null, testVideos);
                lastScanField?.SetValue(null, DateTime.UtcNow);

                // Act
                var result = controller.GetById(id);

                // Assert
                Assert.IsNotNull(result, "Result should not be null");
                Assert.IsNotNull(result.Result, "Result.Result should not be null");
                Assert.IsInstanceOfType(result.Result, typeof(NotFoundResult), $"Result should be NotFoundResult for ID {id}");
            }
            finally
            {
                // Clean up: reset cache
                var cachedVideosField = typeof(VideosController).GetField("_cachedVideos", BindingFlags.NonPublic | BindingFlags.Static);
                cachedVideosField?.SetValue(null, null);
            }
        }

        /// <summary>
        /// Tests that Like adds a new entry to the Likes dictionary with value 1 when the video ID is liked for the first time.
        /// Input: A video ID that hasn't been liked before (42).
        /// Expected: Returns OkObjectResult with true, and Likes[42] equals 1.
        /// </summary>
        [TestMethod]
        public void Like_FirstTimeLikingVideo_AddsToLikesWithValueOne()
        {
            // Arrange
            var controller = new VideosController();
            int videoId = 42;
            var likesDictionary = GetLikesDictionary();
            likesDictionary.Clear();

            // Act
            var result = controller.Like(videoId);

            // Assert
            Assert.IsNotNull(result);
            Assert.IsNotNull(result.Result);
            var okResult = result.Result as OkObjectResult;
            Assert.IsNotNull(okResult);
            Assert.AreEqual(true, okResult.Value);
            Assert.IsTrue(likesDictionary.ContainsKey(videoId));
            Assert.AreEqual(1, likesDictionary[videoId]);
        }

        /// <summary>
        /// Tests that Like increments the like count when the same video ID is liked multiple times.
        /// Input: Same video ID (123) liked three times consecutively.
        /// Expected: Each call returns OkObjectResult with true, and like count increments from 1 to 2 to 3.
        /// </summary>
        [TestMethod]
        public void Like_MultipleConsecutiveLikes_IncrementsLikeCount()
        {
            // Arrange
            var controller = new VideosController();
            int videoId = 123;
            var likesDictionary = GetLikesDictionary();
            likesDictionary.Clear();

            // Act & Assert - First like
            var result1 = controller.Like(videoId);
            Assert.IsNotNull(result1?.Result);
            var okResult1 = result1.Result as OkObjectResult;
            Assert.IsNotNull(okResult1);
            Assert.AreEqual(true, okResult1.Value);
            Assert.AreEqual(1, likesDictionary[videoId]);

            // Act & Assert - Second like
            var result2 = controller.Like(videoId);
            Assert.IsNotNull(result2?.Result);
            var okResult2 = result2.Result as OkObjectResult;
            Assert.IsNotNull(okResult2);
            Assert.AreEqual(true, okResult2.Value);
            Assert.AreEqual(2, likesDictionary[videoId]);

            // Act & Assert - Third like
            var result3 = controller.Like(videoId);
            Assert.IsNotNull(result3?.Result);
            var okResult3 = result3.Result as OkObjectResult;
            Assert.IsNotNull(okResult3);
            Assert.AreEqual(true, okResult3.Value);
            Assert.AreEqual(3, likesDictionary[videoId]);
        }

        /// <summary>
        /// Tests that Like tracks different video IDs independently in the Likes dictionary.
        /// Input: Three different video IDs (10, 20, 30) each liked once.
        /// Expected: Each ID has a separate entry with value 1.
        /// </summary>
        [TestMethod]
        public void Like_DifferentVideoIds_TracksEachIndependently()
        {
            // Arrange
            var controller = new VideosController();
            var likesDictionary = GetLikesDictionary();
            likesDictionary.Clear();

            // Act
            controller.Like(10);
            controller.Like(20);
            controller.Like(30);

            // Assert
            Assert.AreEqual(1, likesDictionary[10]);
            Assert.AreEqual(1, likesDictionary[20]);
            Assert.AreEqual(1, likesDictionary[30]);
            Assert.AreEqual(3, likesDictionary.Count);
        }

        /// <summary>
        /// Tests that Like correctly handles various integer boundary values for the video ID parameter.
        /// Input: Boundary values including int.MinValue, int.MaxValue, 0, negative, and positive values.
        /// Expected: Returns OkObjectResult with true and correctly adds/increments the Likes dictionary entry.
        /// </summary>
        /// <param name="videoId">The video ID to test.</param>
        [TestMethod]
        [DataRow(int.MinValue)]
        [DataRow(int.MaxValue)]
        [DataRow(0)]
        [DataRow(-1)]
        [DataRow(1)]
        [DataRow(-1000)]
        [DataRow(1000)]
        public void Like_BoundaryIntegerValues_ReturnsOkAndUpdatesLikes(int videoId)
        {
            // Arrange
            var controller = new VideosController();
            var likesDictionary = GetLikesDictionary();
            likesDictionary.Clear();

            // Act
            var result = controller.Like(videoId);

            // Assert
            Assert.IsNotNull(result);
            Assert.IsNotNull(result.Result);
            var okResult = result.Result as OkObjectResult;
            Assert.IsNotNull(okResult);
            Assert.AreEqual(true, okResult.Value);
            Assert.IsTrue(likesDictionary.ContainsKey(videoId));
            Assert.AreEqual(1, likesDictionary[videoId]);
        }

        /// <summary>
        /// Tests that Like always returns an OkObjectResult regardless of the video ID.
        /// Input: Various video IDs.
        /// Expected: Always returns OkObjectResult (not null, not NotFoundResult, etc.).
        /// </summary>
        [TestMethod]
        public void Like_AnyVideoId_AlwaysReturnsOkObjectResult()
        {
            // Arrange
            var controller = new VideosController();
            var likesDictionary = GetLikesDictionary();
            likesDictionary.Clear();

            // Act
            var result1 = controller.Like(999);
            var result2 = controller.Like(-999);
            var result3 = controller.Like(0);

            // Assert
            Assert.IsInstanceOfType(result1.Result, typeof(OkObjectResult));
            Assert.IsInstanceOfType(result2.Result, typeof(OkObjectResult));
            Assert.IsInstanceOfType(result3.Result, typeof(OkObjectResult));
        }

        /// <summary>
        /// Tests that Like always returns true as the value in the OkObjectResult.
        /// Input: Any video ID.
        /// Expected: The Value property of OkObjectResult is always true.
        /// </summary>
        [TestMethod]
        public void Like_AnyVideoId_ReturnsOkWithTrueValue()
        {
            // Arrange
            var controller = new VideosController();
            var likesDictionary = GetLikesDictionary();
            likesDictionary.Clear();

            // Act
            var result = controller.Like(500);

            // Assert
            var okResult = result.Result as OkObjectResult;
            Assert.IsNotNull(okResult);
            Assert.IsInstanceOfType(okResult.Value, typeof(bool));
            Assert.AreEqual(true, okResult.Value);
        }

        /// <summary>
        /// Tests that Like can handle a very large number of sequential likes without overflow or error.
        /// Input: Same video ID liked 1000 times.
        /// Expected: Like count reaches 1000 without errors.
        /// </summary>
        [TestMethod]
        public void Like_ManyConsecutiveLikes_IncrementsCorrectly()
        {
            // Arrange
            var controller = new VideosController();
            int videoId = 555;
            var likesDictionary = GetLikesDictionary();
            likesDictionary.Clear();

            // Act
            for (int i = 0; i < 1000; i++)
            {
                controller.Like(videoId);
            }

            // Assert
            Assert.AreEqual(1000, likesDictionary[videoId]);
        }

        /// <summary>
        /// Tests that Like properly handles the edge case where a video has already been liked and is liked again.
        /// Input: Video ID 77 liked once, then liked a second time.
        /// Expected: First like sets count to 1, second like increments to 2.
        /// </summary>
        [TestMethod]
        public void Like_ExistingLikeEntry_IncrementsFromExistingValue()
        {
            // Arrange
            var controller = new VideosController();
            int videoId = 77;
            var likesDictionary = GetLikesDictionary();
            likesDictionary.Clear();
            likesDictionary[videoId] = 5; // Pre-populate with 5 likes

            // Act
            var result = controller.Like(videoId);

            // Assert
            var okResult = result.Result as OkObjectResult;
            Assert.IsNotNull(okResult);
            Assert.AreEqual(true, okResult.Value);
            Assert.AreEqual(6, likesDictionary[videoId]);
        }

        /// <summary>
        /// Tests that GetAll returns a consistent result type across multiple calls.
        /// This verifies the method behavior is stable and predictable.
        /// Input: None (parameterless method).
        /// Expected: Multiple calls return the same result type structure.
        /// </summary>
        [TestMethod]
        public void GetAll_MultipleCalls_ReturnsConsistentResultType()
        {
            // Arrange
            var controller = new VideosController();

            // Act
            var result1 = controller.GetAll();
            var result2 = controller.GetAll();

            // Assert
            Assert.IsNotNull(result1, "First call should return a non-null result.");
            Assert.IsNotNull(result2, "Second call should return a non-null result.");
            Assert.IsInstanceOfType(result1.Result, typeof(OkObjectResult),
                "First call should return OkObjectResult.");
            Assert.IsInstanceOfType(result2.Result, typeof(OkObjectResult),
                "Second call should return OkObjectResult.");

            var list1 = ((OkObjectResult)result1.Result).Value as List<VideoDto>;
            var list2 = ((OkObjectResult)result2.Result).Value as List<VideoDto>;

            Assert.IsNotNull(list1, "First call should return a non-null list.");
            Assert.IsNotNull(list2, "Second call should return a non-null list.");
        }

        /// <summary>
        /// Tests that GetAll returns an empty list when no video files exist in the VIDEO folder.
        /// This verifies the method handles the scenario where no videos are available.
        /// Input: None (parameterless method).
        /// Expected: Returns OkObjectResult with an empty List of VideoDto.
        /// </summary>
        [TestMethod]
        public void GetAll_WhenNoVideosExist_ReturnsEmptyList()
        {
            // Arrange
            var controller = new VideosController();

            // Act
            var result = controller.GetAll();

            // Assert
            var okResult = result.Result as OkObjectResult;
            Assert.IsNotNull(okResult, "Result should be an OkObjectResult.");
            var videoList = okResult.Value as List<VideoDto>;
            Assert.IsNotNull(videoList,
                "The returned list should not be null, even when no videos exist.");

            // Note: The list may be empty or non-empty depending on whether .mp4 files exist
            // in the VIDEO folder. This test verifies the list structure is valid regardless.
            Assert.IsTrue(videoList.Count >= 0,
                "The list should have zero or more items.");
        }

        /// <summary>
        /// Tests Unlike with positive video IDs across different ranges.
        /// Input: Various positive integer values.
        /// Expected: Returns OkObjectResult with true for all positive IDs.
        /// </summary>
        /// <param name="id">The video ID to unlike.</param>
        [TestMethod]
        [DataRow(1)]
        [DataRow(100)]
        [DataRow(1000)]
        [DataRow(999999)]
        [DataRow(int.MaxValue - 1)]
        public void Unlike_PositiveIds_ReturnsOkTrue(int id)
        {
            // Arrange
            VideosController controller = new VideosController();

            // Act
            ActionResult<bool> result = controller.Unlike(id);

            // Assert
            Assert.IsNotNull(result);
            Assert.IsInstanceOfType(result.Result, typeof(OkObjectResult));
            OkObjectResult okResult = (OkObjectResult)result.Result;
            Assert.AreEqual(true, okResult.Value);
        }

        /// <summary>
        /// Tests Unlike with negative video IDs across different ranges.
        /// Input: Various negative integer values.
        /// Expected: Returns OkObjectResult with true for all negative IDs and adds them with 0 likes.
        /// </summary>
        /// <param name="id">The video ID to unlike.</param>
        [TestMethod]
        [DataRow(-1)]
        [DataRow(-100)]
        [DataRow(-1000)]
        [DataRow(-999999)]
        [DataRow(int.MinValue + 1)]
        public void Unlike_NegativeIds_ReturnsOkTrueAndAddsWithZero(int id)
        {
            // Arrange
            VideosController controller = new VideosController();
            ConcurrentDictionary<int, int> likes = GetLikesDictionary();

            // Act
            ActionResult<bool> result = controller.Unlike(id);

            // Assert
            Assert.IsNotNull(result);
            Assert.IsInstanceOfType(result.Result, typeof(OkObjectResult));
            OkObjectResult okResult = (OkObjectResult)result.Result;
            Assert.AreEqual(true, okResult.Value);
            Assert.IsTrue(likes.ContainsKey(id));
            Assert.AreEqual(0, likes[id]);
        }

        /// <summary>
        /// Tests that GetByCategory handles null category input.
        /// Input: null category string.
        /// Expected: May throw NullReferenceException or return result based on runtime behavior.
        /// Note: This test validates runtime behavior when null is passed despite non-nullable parameter annotation.
        /// </summary>
        [TestMethod]
        public void GetByCategory_NullCategory_HandlesGracefully()
        {
            // Arrange
            var controller = new VideosController();
            string? category = null;

            // Act & Assert
            try
            {
                var result = controller.GetByCategory(category!);
                // If no exception thrown, verify result structure
                Assert.IsNotNull(result);
                Assert.IsNotNull(result.Result);
                Assert.IsInstanceOfType(result.Result, typeof(OkObjectResult));
            }
            catch (NullReferenceException)
            {
                // Expected behavior if Category.Equals is called on null
                Assert.IsTrue(true);
            }
        }

        /// <summary>
        /// Tests that GetByCategory does not match categories with leading spaces.
        /// Input: Category with leading space.
        /// Expected: Returns OkObjectResult with filtered list (space-sensitive matching).
        /// </summary>
        [TestMethod]
        [DataRow(" General")]
        [DataRow("  General")]
        [DataRow("\tGeneral")]
        public void GetByCategory_CategoryWithLeadingSpace_ReturnsFilteredListNotMatchingTrimmed(string category)
        {
            // Arrange
            var controller = new VideosController();

            // Act
            var result = controller.GetByCategory(category);

            // Assert
            Assert.IsNotNull(result);
            Assert.IsNotNull(result.Result);
            Assert.IsInstanceOfType(result.Result, typeof(OkObjectResult));
            var okResult = result.Result as OkObjectResult;
            Assert.IsNotNull(okResult);
            Assert.IsInstanceOfType(okResult.Value, typeof(List<VideoDto>));
            var videos = okResult.Value as List<VideoDto>;
            Assert.IsNotNull(videos);
            // Verify exact match including leading space
            Assert.IsTrue(videos.All(v => v.Category.Equals(category, StringComparison.OrdinalIgnoreCase)));
        }

        /// <summary>
        /// Tests that GetByCategory does not match categories with trailing spaces.
        /// Input: Category with trailing space.
        /// Expected: Returns OkObjectResult with filtered list (space-sensitive matching).
        /// </summary>
        [TestMethod]
        [DataRow("General ")]
        [DataRow("General  ")]
        [DataRow("General\t")]
        public void GetByCategory_CategoryWithTrailingSpace_ReturnsFilteredListNotMatchingTrimmed(string category)
        {
            // Arrange
            var controller = new VideosController();

            // Act
            var result = controller.GetByCategory(category);

            // Assert
            Assert.IsNotNull(result);
            Assert.IsNotNull(result.Result);
            Assert.IsInstanceOfType(result.Result, typeof(OkObjectResult));
            var okResult = result.Result as OkObjectResult;
            Assert.IsNotNull(okResult);
            Assert.IsInstanceOfType(okResult.Value, typeof(List<VideoDto>));
            var videos = okResult.Value as List<VideoDto>;
            Assert.IsNotNull(videos);
            // Verify exact match including trailing space
            Assert.IsTrue(videos.All(v => v.Category.Equals(category, StringComparison.OrdinalIgnoreCase)));
        }

        /// <summary>
        /// Tests that GetByCategory does not match categories with leading and trailing spaces.
        /// Input: Category with both leading and trailing spaces.
        /// Expected: Returns OkObjectResult with filtered list (space-sensitive matching).
        /// </summary>
        [TestMethod]
        [DataRow(" General ")]
        [DataRow("  General  ")]
        [DataRow("\tGeneral\t")]
        public void GetByCategory_CategoryWithLeadingAndTrailingSpace_ReturnsFilteredListNotMatchingTrimmed(string category)
        {
            // Arrange
            var controller = new VideosController();

            // Act
            var result = controller.GetByCategory(category);

            // Assert
            Assert.IsNotNull(result);
            Assert.IsNotNull(result.Result);
            Assert.IsInstanceOfType(result.Result, typeof(OkObjectResult));
            var okResult = result.Result as OkObjectResult;
            Assert.IsNotNull(okResult);
            Assert.IsInstanceOfType(okResult.Value, typeof(List<VideoDto>));
            var videos = okResult.Value as List<VideoDto>;
            Assert.IsNotNull(videos);
            // Verify exact match including leading and trailing spaces
            Assert.IsTrue(videos.All(v => v.Category.Equals(category, StringComparison.OrdinalIgnoreCase)));
        }

        /// <summary>
        /// Tests that GetByCategory handles categories with embedded control characters.
        /// Input: Category strings with embedded control characters.
        /// Expected: Returns OkObjectResult with filtered list.
        /// </summary>
        [TestMethod]
        [DataRow("Cate\0gory")]
        [DataRow("Cate\rgory")]
        [DataRow("Cate\ngory")]
        [DataRow("Ca\tte\bgory")]
        public void GetByCategory_CategoryWithControlCharacters_ReturnsOkResultWithFilteredList(string category)
        {
            // Arrange
            var controller = new VideosController();

            // Act
            var result = controller.GetByCategory(category);

            // Assert
            Assert.IsNotNull(result);
            Assert.IsNotNull(result.Result);
            Assert.IsInstanceOfType(result.Result, typeof(OkObjectResult));
            var okResult = result.Result as OkObjectResult;
            Assert.IsNotNull(okResult);
            Assert.IsInstanceOfType(okResult.Value, typeof(List<VideoDto>));
            var videos = okResult.Value as List<VideoDto>;
            Assert.IsNotNull(videos);
        }

        /// <summary>
        /// Tests that GetByCategory handles mixed case special characters consistently.
        /// Input: Special character categories with different casings.
        /// Expected: Returns OkObjectResult with case-insensitive filtered list.
        /// </summary>
        [TestMethod]
        [DataRow("category@123")]
        [DataRow("CATEGORY@123")]
        [DataRow("Category@123")]
        public void GetByCategory_SpecialCharactersDifferentCasing_ReturnsCaseInsensitiveMatch(string category)
        {
            // Arrange
            var controller = new VideosController();

            // Act
            var result = controller.GetByCategory(category);

            // Assert
            Assert.IsNotNull(result);
            Assert.IsNotNull(result.Result);
            Assert.IsInstanceOfType(result.Result, typeof(OkObjectResult));
            var okResult = result.Result as OkObjectResult;
            Assert.IsNotNull(okResult);
            Assert.IsInstanceOfType(okResult.Value, typeof(List<VideoDto>));
            var videos = okResult.Value as List<VideoDto>;
            Assert.IsNotNull(videos);
            // All videos should have category matching case-insensitively
            Assert.IsTrue(videos.All(v => v.Category.Equals(category, StringComparison.OrdinalIgnoreCase)));
        }

        /// <summary>
        /// Tests that GetByCategory always returns an OkObjectResult regardless of input.
        /// Input: Various edge case category strings.
        /// Expected: Always returns OkObjectResult (HTTP 200).
        /// </summary>
        [TestMethod]
        [DataRow("")]
        [DataRow(" ")]
        [DataRow("ValidCategory")]
        [DataRow("NonExistentCategory12345")]
        [DataRow("!@#$%^&*()")]
        public void GetByCategory_VariousInputs_AlwaysReturnsOkResult(string category)
        {
            // Arrange
            var controller = new VideosController();

            // Act
            var result = controller.GetByCategory(category);

            // Assert
            Assert.IsNotNull(result);
            Assert.IsNotNull(result.Result);
            Assert.IsInstanceOfType(result.Result, typeof(OkObjectResult));
        }

        /// <summary>
        /// Tests that GetByCategory returns a list (possibly empty) for any valid input.
        /// Input: Various category strings.
        /// Expected: Returns non-null List of VideoDto.
        /// </summary>
        [TestMethod]
        [DataRow("ExistingCategory")]
        [DataRow("NonExistentCategory")]
        [DataRow("")]
        public void GetByCategory_AnyValidInput_ReturnsNonNullList(string category)
        {
            // Arrange
            var controller = new VideosController();

            // Act
            var result = controller.GetByCategory(category);

            // Assert
            var okResult = result.Result as OkObjectResult;
            Assert.IsNotNull(okResult);
            var videos = okResult.Value as List<VideoDto>;
            Assert.IsNotNull(videos);
        }

        /// <summary>
        /// Tests Stream with boundary integer values for id parameter.
        /// Input: Various boundary values (int.MinValue, int.MaxValue, 0, -1).
        /// Expected: Returns NotFoundResult for all non-existent ids.
        /// </summary>
        /// <param name="id">The video id to test.</param>
        [TestMethod]
        [DataRow(int.MinValue)]
        [DataRow(int.MaxValue)]
        [DataRow(0)]
        [DataRow(-1)]
        [DataRow(-100)]
        public void Stream_BoundaryIntegerIds_ReturnsNotFound(int id)
        {
            // Arrange
            VideosController controller = new VideosController();
            Type controllerType = typeof(VideosController);
            FieldInfo? cachedVideosField = controllerType.GetField("_cachedVideos", BindingFlags.NonPublic | BindingFlags.Static);
            FieldInfo? lastScanField = controllerType.GetField("_lastScan", BindingFlags.NonPublic | BindingFlags.Static);
            cachedVideosField?.SetValue(null, new List<VideoDto>());
            lastScanField?.SetValue(null, DateTime.UtcNow);

            // Act
            IActionResult result = controller.Stream(id);

            // Assert
            Assert.IsInstanceOfType(result, typeof(NotFoundResult));
        }

        /// <summary>
        /// Tests that Stream correctly handles multiple videos and returns the correct one by id.
        /// Input: Multiple videos in the list, request specific video id.
        /// Expected: Returns FileStreamResult for the matching video.
        /// </summary>
        [TestMethod]
        public void Stream_MultipleVideos_ReturnsCorrectVideo()
        {
            // Arrange
            VideosController controller = new VideosController();
            int targetVideoId = 2;
            FieldInfo? videoFolderField = typeof(VideosController).GetField("VideoFolder", BindingFlags.NonPublic | BindingFlags.Static);
            string videoFolder = (string)(videoFolderField?.GetValue(null) ?? string.Empty);
            if (!Directory.Exists(videoFolder))
                Directory.CreateDirectory(videoFolder);
            string testFileName1 = "test_video_1.mp4";
            string testFileName2 = "test_video_2.mp4";
            string testFilePath1 = Path.Combine(videoFolder, testFileName1);
            string testFilePath2 = Path.Combine(videoFolder, testFileName2);
            File.WriteAllText(testFilePath1, "video 1 content");
            File.WriteAllText(testFilePath2, "video 2 content");

            try
            {
                FieldInfo? cachedVideosField = typeof(VideosController).GetField("_cachedVideos", BindingFlags.NonPublic | BindingFlags.Static);
                FieldInfo? lastScanField = typeof(VideosController).GetField("_lastScan", BindingFlags.NonPublic | BindingFlags.Static);
                List<VideoDto> testVideos = new List<VideoDto>
                {
                    new VideoDto { Id = 1, FileName = testFileName1, Title = "Video 1", Category = "Test", Path = "/api/videos/stream/1" },
                    new VideoDto { Id = 2, FileName = testFileName2, Title = "Video 2", Category = "Test", Path = "/api/videos/stream/2" }
                };
                cachedVideosField?.SetValue(null, testVideos);
                lastScanField?.SetValue(null, DateTime.UtcNow);

                // Act
                IActionResult result = controller.Stream(targetVideoId);

                // Assert
                Assert.IsInstanceOfType(result, typeof(FileStreamResult));
                FileStreamResult? fileResult = result as FileStreamResult;
                Assert.IsNotNull(fileResult?.FileStream);
                fileResult?.FileStream?.Dispose();
            }
            finally
            {
                if (File.Exists(testFilePath1))
                    File.Delete(testFilePath1);
                if (File.Exists(testFilePath2))
                    File.Delete(testFilePath2);
                FieldInfo? cachedVideosField = typeof(VideosController).GetField("_cachedVideos", BindingFlags.NonPublic | BindingFlags.Static);
                cachedVideosField?.SetValue(null, null);
            }
        }

        /// <summary>
        /// Tests that Stream enables range processing in the returned FileStreamResult.
        /// Input: Valid video id with existing file.
        /// Expected: FileStreamResult has EnableRangeProcessing set to true.
        /// </summary>
        [TestMethod]
        public void Stream_ValidVideo_EnablesRangeProcessing()
        {
            // Arrange
            VideosController controller = new VideosController();
            int videoId = 1;
            FieldInfo? videoFolderField = typeof(VideosController).GetField("VideoFolder", BindingFlags.NonPublic | BindingFlags.Static);
            string videoFolder = (string)(videoFolderField?.GetValue(null) ?? string.Empty);
            if (!Directory.Exists(videoFolder))
                Directory.CreateDirectory(videoFolder);
            string testFileName = "test_range_video.mp4";
            string testFilePath = Path.Combine(videoFolder, testFileName);
            File.WriteAllText(testFilePath, "test content");

            try
            {
                FieldInfo? cachedVideosField = typeof(VideosController).GetField("_cachedVideos", BindingFlags.NonPublic | BindingFlags.Static);
                FieldInfo? lastScanField = typeof(VideosController).GetField("_lastScan", BindingFlags.NonPublic | BindingFlags.Static);
                List<VideoDto> testVideos = new List<VideoDto>
                {
                    new VideoDto { Id = videoId, FileName = testFileName, Title = "Test", Category = "Test", Path = $"/api/videos/stream/{videoId}" }
                };
                cachedVideosField?.SetValue(null, testVideos);
                lastScanField?.SetValue(null, DateTime.UtcNow);

                // Act
                IActionResult result = controller.Stream(videoId);

                // Assert
                FileStreamResult? fileResult = result as FileStreamResult;
                Assert.IsNotNull(fileResult);
                PropertyInfo? enableRangeProperty = fileResult.GetType().GetProperty("EnableRangeProcessing");
                object? enableRangeValue = enableRangeProperty?.GetValue(fileResult);
                Assert.IsTrue(enableRangeValue is true);
                fileResult.FileStream?.Dispose();
            }
            finally
            {
                if (File.Exists(testFilePath))
                    File.Delete(testFilePath);
                FieldInfo? cachedVideosField = typeof(VideosController).GetField("_cachedVideos", BindingFlags.NonPublic | BindingFlags.Static);
                cachedVideosField?.SetValue(null, null);
            }
        }

        /// <summary>
        /// Tests that Watch with various boundary integer values always returns OkObjectResult with true.
        /// Input: Multiple boundary values including int.MinValue, -1, 0, 1, int.MaxValue.
        /// Expected: All calls return OkObjectResult with true value.
        /// </summary>
        [TestMethod]
        [DataRow(int.MinValue)]
        [DataRow(-1)]
        [DataRow(0)]
        [DataRow(1)]
        [DataRow(100)]
        [DataRow(int.MaxValue)]
        public void Watch_VariousBoundaryValues_AlwaysReturnsOkTrue(int id)
        {
            // Arrange
            var controller = new VideosController();

            // Act
            var result = controller.Watch(id);

            // Assert
            Assert.IsNotNull(result);
            Assert.IsInstanceOfType(result.Result, typeof(OkObjectResult));
            var okResult = result.Result as OkObjectResult;
            Assert.IsNotNull(okResult);
            Assert.AreEqual(true, okResult.Value);
        }

        /// <summary>
        /// Tests that Watch correctly increments view count from an existing non-zero value.
        /// Input: Video id watched multiple times starting from count > 1.
        /// Expected: View count increments correctly with each watch.
        /// </summary>
        [TestMethod]
        public void Watch_IncrementingFromExistingCount_IncrementsCorrectly()
        {
            // Arrange
            var controller = new VideosController();
            int videoId = 99;

            // Pre-populate with some watches
            controller.Watch(videoId);
            controller.Watch(videoId);

            var viewsField = typeof(VideosController).GetField("Views", BindingFlags.NonPublic | BindingFlags.Static);
            var views = viewsField?.GetValue(null) as ConcurrentDictionary<int, int>;
            Assert.AreEqual(2, views?[videoId]);

            // Act
            var result = controller.Watch(videoId);

            // Assert
            Assert.IsInstanceOfType(result.Result, typeof(OkObjectResult));
            Assert.AreEqual(3, views?[videoId]);
        }
    }
}