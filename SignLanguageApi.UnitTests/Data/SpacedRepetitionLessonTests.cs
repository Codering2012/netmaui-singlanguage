using System;

using Microsoft.VisualStudio.TestTools.UnitTesting;
using SignLanguageApi.Data;

namespace SignLanguageApi.Data.UnitTests
{
    /// <summary>
    /// Contains unit tests for the <see cref="SpacedRepetitionLesson"/> class.
    /// </summary>
    [TestClass]
    public class SpacedRepetitionLessonTests
    {
        /// <summary>
        /// Tests that IsReviewDue returns true when DueDate is in the past.
        /// Input: DueDate set to a date clearly in the past (January 1, 2000).
        /// Expected: IsReviewDue should return true because current UTC time is greater than the past date.
        /// </summary>
        [TestMethod]
        public void IsReviewDue_WhenDueDateIsInPast_ReturnsTrue()
        {
            // Arrange
            var lesson = new SpacedRepetitionLesson
            {
                DueDate = new DateTime(2000, 1, 1, 0, 0, 0, DateTimeKind.Utc)
            };

            // Act
            var result = lesson.IsReviewDue;

            // Assert
            Assert.IsTrue(result);
        }

        /// <summary>
        /// Tests that IsReviewDue returns false when DueDate is in the future.
        /// Input: DueDate set to a date clearly in the future (January 1, 2100).
        /// Expected: IsReviewDue should return false because current UTC time is less than the future date.
        /// </summary>
        [TestMethod]
        public void IsReviewDue_WhenDueDateIsInFuture_ReturnsFalse()
        {
            // Arrange
            var lesson = new SpacedRepetitionLesson
            {
                DueDate = new DateTime(2100, 1, 1, 0, 0, 0, DateTimeKind.Utc)
            };

            // Act
            var result = lesson.IsReviewDue;

            // Assert
            Assert.IsFalse(result);
        }

        /// <summary>
        /// Tests that IsReviewDue returns true when DueDate is DateTime.MinValue.
        /// Input: DueDate set to DateTime.MinValue (the earliest possible date).
        /// Expected: IsReviewDue should return true because current UTC time is always greater than DateTime.MinValue.
        /// </summary>
        [TestMethod]
        public void IsReviewDue_WhenDueDateIsMinValue_ReturnsTrue()
        {
            // Arrange
            var lesson = new SpacedRepetitionLesson
            {
                DueDate = DateTime.MinValue
            };

            // Act
            var result = lesson.IsReviewDue;

            // Assert
            Assert.IsTrue(result);
        }

        /// <summary>
        /// Tests that IsReviewDue returns false when DueDate is DateTime.MaxValue.
        /// Input: DueDate set to DateTime.MaxValue (the latest possible date).
        /// Expected: IsReviewDue should return false because current UTC time is always less than DateTime.MaxValue.
        /// </summary>
        [TestMethod]
        public void IsReviewDue_WhenDueDateIsMaxValue_ReturnsFalse()
        {
            // Arrange
            var lesson = new SpacedRepetitionLesson
            {
                DueDate = DateTime.MaxValue
            };

            // Act
            var result = lesson.IsReviewDue;

            // Assert
            Assert.IsFalse(result);
        }

        /// <summary>
        /// Tests that IsReviewDue returns true when DueDate is exactly one second in the past.
        /// Input: DueDate set to one second before the current time.
        /// Expected: IsReviewDue should return true because current UTC time is greater than DueDate.
        /// </summary>
        [TestMethod]
        public void IsReviewDue_WhenDueDateIsOneSecondInPast_ReturnsTrue()
        {
            // Arrange
            var lesson = new SpacedRepetitionLesson
            {
                DueDate = DateTime.UtcNow.AddSeconds(-1)
            };

            // Act
            var result = lesson.IsReviewDue;

            // Assert
            Assert.IsTrue(result);
        }

        /// <summary>
        /// Tests that IsReviewDue returns false when DueDate is exactly one second in the future.
        /// Input: DueDate set to one second after the current time.
        /// Expected: IsReviewDue should return false because current UTC time is less than DueDate.
        /// </summary>
        [TestMethod]
        public void IsReviewDue_WhenDueDateIsOneSecondInFuture_ReturnsFalse()
        {
            // Arrange
            var lesson = new SpacedRepetitionLesson
            {
                DueDate = DateTime.UtcNow.AddSeconds(1)
            };

            // Act
            var result = lesson.IsReviewDue;

            // Assert
            Assert.IsFalse(result);
        }

        /// <summary>
        /// Tests that IsReviewDue returns the expected result based on various DueDate values.
        /// Input: Different DueDate values including past dates, future dates, and boundary values.
        /// Expected: IsReviewDue returns true when DueDate is less than or equal to current UTC time, false otherwise.
        /// </summary>
        /// <param name="year">The year component of the DueDate.</param>
        /// <param name="month">The month component of the DueDate.</param>
        /// <param name="day">The day component of the DueDate.</param>
        /// <param name="expectedResult">The expected result of IsReviewDue property.</param>
        [TestMethod]
        [DataRow(2000, 1, 1, true, DisplayName = "Past date (2000-01-01) should return true")]
        [DataRow(1990, 6, 15, true, DisplayName = "Past date (1990-06-15) should return true")]
        [DataRow(2100, 1, 1, false, DisplayName = "Future date (2100-01-01) should return false")]
        [DataRow(2099, 12, 31, false, DisplayName = "Future date (2099-12-31) should return false")]
        public void IsReviewDue_WithVariousDueDates_ReturnsExpectedResult(int year, int month, int day, bool expectedResult)
        {
            // Arrange
            var lesson = new SpacedRepetitionLesson
            {
                DueDate = new DateTime(year, month, day, 0, 0, 0, DateTimeKind.Utc)
            };

            // Act
            var result = lesson.IsReviewDue;

            // Assert
            Assert.AreEqual(expectedResult, result);
        }

        /// <summary>
        /// Tests that IsReviewDue returns true when DueDate is one millisecond in the past.
        /// Input: DueDate set to one millisecond before the current time.
        /// Expected: IsReviewDue should return true because current UTC time is greater than DueDate.
        /// </summary>
        [TestMethod]
        public void IsReviewDue_WhenDueDateIsOneMillisecondInPast_ReturnsTrue()
        {
            // Arrange
            var lesson = new SpacedRepetitionLesson
            {
                DueDate = DateTime.UtcNow.AddMilliseconds(-1)
            };

            // Act
            var result = lesson.IsReviewDue;

            // Assert
            Assert.IsTrue(result);
        }

        /// <summary>
        /// Tests that IsReviewDue returns true when DueDate is one day in the past.
        /// Input: DueDate set to one day before the current time.
        /// Expected: IsReviewDue should return true because current UTC time is greater than DueDate.
        /// </summary>
        [TestMethod]
        public void IsReviewDue_WhenDueDateIsOneDayInPast_ReturnsTrue()
        {
            // Arrange
            var lesson = new SpacedRepetitionLesson
            {
                DueDate = DateTime.UtcNow.AddDays(-1)
            };

            // Act
            var result = lesson.IsReviewDue;

            // Assert
            Assert.IsTrue(result);
        }

        /// <summary>
        /// Tests that IsReviewDue returns false when DueDate is one day in the future.
        /// Input: DueDate set to one day after the current time.
        /// Expected: IsReviewDue should return false because current UTC time is less than DueDate.
        /// </summary>
        [TestMethod]
        public void IsReviewDue_WhenDueDateIsOneDayInFuture_ReturnsFalse()
        {
            // Arrange
            var lesson = new SpacedRepetitionLesson
            {
                DueDate = DateTime.UtcNow.AddDays(1)
            };

            // Act
            var result = lesson.IsReviewDue;

            // Assert
            Assert.IsFalse(result);
        }
    }
}