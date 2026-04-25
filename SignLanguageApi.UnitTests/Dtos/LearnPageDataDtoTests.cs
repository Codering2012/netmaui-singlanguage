using System;

using Microsoft.VisualStudio.TestTools.UnitTesting;
using SignLanguageApi.Dtos;

namespace SignLanguageApi.Dtos.UnitTests
{
    [TestClass]
    public class LearnPageDataDtoTests
    {
        /// <summary>
        /// Tests that DailyGoalProgress returns 0 when DailyGoalTotal is zero to prevent division by zero.
        /// </summary>
        [TestMethod]
        public void DailyGoalProgress_WhenDailyGoalTotalIsZero_ReturnsZero()
        {
            // Arrange
            var dto = new LearnPageDataDto
            {
                DailyGoalCompleted = 5,
                DailyGoalTotal = 0
            };

            // Act
            var result = dto.DailyGoalProgress;

            // Assert
            Assert.AreEqual(0.0, result, 0.0001);
        }

        /// <summary>
        /// Tests that DailyGoalProgress returns 0 when DailyGoalTotal is negative.
        /// </summary>
        [TestMethod]
        public void DailyGoalProgress_WhenDailyGoalTotalIsNegative_ReturnsZero()
        {
            // Arrange
            var dto = new LearnPageDataDto
            {
                DailyGoalCompleted = 5,
                DailyGoalTotal = -1
            };

            // Act
            var result = dto.DailyGoalProgress;

            // Assert
            Assert.AreEqual(0.0, result, 0.0001);
        }

        /// <summary>
        /// Tests DailyGoalProgress calculation with various valid DailyGoalCompleted and DailyGoalTotal combinations.
        /// Input: DailyGoalCompleted, DailyGoalTotal
        /// Expected: Calculated progress ratio
        /// </summary>
        [TestMethod]
        [DataRow(0, 5, 0.0, DisplayName = "Zero completed out of 5 returns 0.0")]
        [DataRow(1, 5, 0.2, DisplayName = "1 out of 5 returns 0.2")]
        [DataRow(3, 5, 0.6, DisplayName = "3 out of 5 returns 0.6")]
        [DataRow(5, 5, 1.0, DisplayName = "5 out of 5 returns 1.0")]
        [DataRow(10, 5, 2.0, DisplayName = "10 out of 5 returns 2.0 (over 100%)")]
        [DataRow(2, 10, 0.2, DisplayName = "2 out of 10 returns 0.2")]
        [DataRow(1, 1, 1.0, DisplayName = "1 out of 1 returns 1.0")]
        [DataRow(0, 1, 0.0, DisplayName = "0 out of 1 returns 0.0")]
        public void DailyGoalProgress_WithVariousValidInputs_ReturnsCorrectRatio(int completed, int total, double expected)
        {
            // Arrange
            var dto = new LearnPageDataDto
            {
                DailyGoalCompleted = completed,
                DailyGoalTotal = total
            };

            // Act
            var result = dto.DailyGoalProgress;

            // Assert
            Assert.AreEqual(expected, result, 0.0001);
        }

        /// <summary>
        /// Tests that DailyGoalProgress correctly handles negative DailyGoalCompleted values when DailyGoalTotal is positive.
        /// </summary>
        [TestMethod]
        public void DailyGoalProgress_WhenDailyGoalCompletedIsNegativeAndTotalIsPositive_ReturnsNegativeRatio()
        {
            // Arrange
            var dto = new LearnPageDataDto
            {
                DailyGoalCompleted = -5,
                DailyGoalTotal = 10
            };

            // Act
            var result = dto.DailyGoalProgress;

            // Assert
            Assert.AreEqual(-0.5, result, 0.0001);
        }

        /// <summary>
        /// Tests DailyGoalProgress with extreme boundary values (int.MaxValue).
        /// </summary>
        [TestMethod]
        public void DailyGoalProgress_WithMaxIntValues_ReturnsCorrectRatio()
        {
            // Arrange
            var dto = new LearnPageDataDto
            {
                DailyGoalCompleted = int.MaxValue,
                DailyGoalTotal = int.MaxValue
            };

            // Act
            var result = dto.DailyGoalProgress;

            // Assert
            Assert.AreEqual(1.0, result, 0.0001);
        }

        /// <summary>
        /// Tests DailyGoalProgress when DailyGoalTotal is int.MaxValue and DailyGoalCompleted is half of that.
        /// </summary>
        [TestMethod]
        public void DailyGoalProgress_WithMaxIntTotal_ReturnsCorrectRatio()
        {
            // Arrange
            var dto = new LearnPageDataDto
            {
                DailyGoalCompleted = int.MaxValue / 2,
                DailyGoalTotal = int.MaxValue
            };

            // Act
            var result = dto.DailyGoalProgress;

            // Assert
            Assert.AreEqual(0.5, result, 0.0001);
        }

        /// <summary>
        /// Tests DailyGoalProgress with extreme negative boundary values (int.MinValue for DailyGoalTotal).
        /// Expected: Returns 0 since DailyGoalTotal is not greater than 0.
        /// </summary>
        [TestMethod]
        public void DailyGoalProgress_WithMinIntTotal_ReturnsZero()
        {
            // Arrange
            var dto = new LearnPageDataDto
            {
                DailyGoalCompleted = 5,
                DailyGoalTotal = int.MinValue
            };

            // Act
            var result = dto.DailyGoalProgress;

            // Assert
            Assert.AreEqual(0.0, result, 0.0001);
        }

        /// <summary>
        /// Tests DailyGoalProgress with both DailyGoalCompleted and DailyGoalTotal as int.MinValue.
        /// Expected: Returns 0 since DailyGoalTotal is not greater than 0.
        /// </summary>
        [TestMethod]
        public void DailyGoalProgress_WithBothMinIntValues_ReturnsZero()
        {
            // Arrange
            var dto = new LearnPageDataDto
            {
                DailyGoalCompleted = int.MinValue,
                DailyGoalTotal = int.MinValue
            };

            // Act
            var result = dto.DailyGoalProgress;

            // Assert
            Assert.AreEqual(0.0, result, 0.0001);
        }

        /// <summary>
        /// Tests DailyGoalProgress with default values (DailyGoalCompleted = 0, DailyGoalTotal = 5).
        /// </summary>
        [TestMethod]
        public void DailyGoalProgress_WithDefaultValues_ReturnsZero()
        {
            // Arrange
            var dto = new LearnPageDataDto();

            // Act
            var result = dto.DailyGoalProgress;

            // Assert
            Assert.AreEqual(0.0, result, 0.0001);
        }

        /// <summary>
        /// Tests DailyGoalProgress when DailyGoalCompleted is 1 and DailyGoalTotal is int.MaxValue.
        /// Expected: Returns a very small positive value close to zero.
        /// </summary>
        [TestMethod]
        public void DailyGoalProgress_WhenCompletedIsOneAndTotalIsMaxInt_ReturnsVerySmallValue()
        {
            // Arrange
            var dto = new LearnPageDataDto
            {
                DailyGoalCompleted = 1,
                DailyGoalTotal = int.MaxValue
            };

            // Act
            var result = dto.DailyGoalProgress;

            // Assert
            Assert.IsTrue(result > 0);
            Assert.IsTrue(result < 0.0001);
        }

        /// <summary>
        /// Tests DailyGoalProgress when DailyGoalCompleted is int.MinValue and DailyGoalTotal is positive.
        /// Expected: Returns a very negative ratio.
        /// </summary>
        [TestMethod]
        public void DailyGoalProgress_WhenCompletedIsMinIntAndTotalIsPositive_ReturnsVeryNegativeRatio()
        {
            // Arrange
            var dto = new LearnPageDataDto
            {
                DailyGoalCompleted = int.MinValue,
                DailyGoalTotal = 1
            };

            // Act
            var result = dto.DailyGoalProgress;

            // Assert
            Assert.AreEqual((double)int.MinValue, result, 1.0);
        }

        /// <summary>
        /// Tests DailyGoalProgress when DailyGoalCompleted exceeds DailyGoalTotal significantly.
        /// Input: DailyGoalCompleted = 100, DailyGoalTotal = 1
        /// Expected: 100.0
        /// </summary>
        [TestMethod]
        public void DailyGoalProgress_WhenCompletedGreatlyExceedsTotal_ReturnsHighRatio()
        {
            // Arrange
            var dto = new LearnPageDataDto
            {
                DailyGoalCompleted = 100,
                DailyGoalTotal = 1
            };

            // Act
            var result = dto.DailyGoalProgress;

            // Assert
            Assert.AreEqual(100.0, result, 0.0001);
        }

        /// <summary>
        /// Tests DailyGoalProgress with both zero values.
        /// Expected: Returns 0 since DailyGoalTotal is not greater than 0.
        /// </summary>
        [TestMethod]
        public void DailyGoalProgress_WithBothZeroValues_ReturnsZero()
        {
            // Arrange
            var dto = new LearnPageDataDto
            {
                DailyGoalCompleted = 0,
                DailyGoalTotal = 0
            };

            // Act
            var result = dto.DailyGoalProgress;

            // Assert
            Assert.AreEqual(0, result);
        }

        /// <summary>
        /// Tests DailyGoalProgress with DailyGoalTotal equal to 1 and various DailyGoalCompleted values.
        /// Input: DailyGoalCompleted, DailyGoalTotal = 1
        /// Expected: DailyGoalCompleted as double
        /// </summary>
        [TestMethod]
        [DataRow(-100, -100.0, DisplayName = "Completed = -100 with Total = 1 returns -100.0")]
        [DataRow(-1, -1.0, DisplayName = "Completed = -1 with Total = 1 returns -1.0")]
        [DataRow(0, 0.0, DisplayName = "Completed = 0 with Total = 1 returns 0.0")]
        [DataRow(50, 50.0, DisplayName = "Completed = 50 with Total = 1 returns 50.0")]
        public void DailyGoalProgress_WithTotalOfOne_ReturnsCompletedAsDouble(int completed, double expected)
        {
            // Arrange
            var dto = new LearnPageDataDto
            {
                DailyGoalCompleted = completed,
                DailyGoalTotal = 1
            };

            // Act
            var result = dto.DailyGoalProgress;

            // Assert
            Assert.AreEqual(expected, result, 0.0001);
        }
    }
}