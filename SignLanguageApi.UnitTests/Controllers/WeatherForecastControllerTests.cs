using System;
using System.Collections.Generic;
using System.Linq;

using Microsoft.AspNetCore.Mvc;
using Microsoft.Extensions.Logging;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using Moq;
using SignLanguageApi;
using SignLanguageApi.Controllers;

namespace SignLanguageApi.Controllers.UnitTests
{
    /// <summary>
    /// Unit tests for the <see cref="WeatherForecastController"/> class.
    /// </summary>
    [TestClass]
    public class WeatherForecastControllerTests
    {
        /// <summary>
        /// Tests that the constructor successfully creates an instance when provided with a valid logger.
        /// Expected: The constructor completes without throwing and returns a valid instance.
        /// </summary>
        [TestMethod]
        public void Constructor_WithValidLogger_CreatesInstance()
        {
            // Arrange
            var mockLogger = new Mock<ILogger<WeatherForecastController>>();

            // Act
            var controller = new WeatherForecastController(mockLogger.Object);

            // Assert
            Assert.IsNotNull(controller);
            Assert.IsInstanceOfType(controller, typeof(WeatherForecastController));
            Assert.IsInstanceOfType(controller, typeof(ControllerBase));
        }

        /// <summary>
        /// Tests that the constructor behavior when provided with a null logger.
        /// Input: null logger reference.
        /// Expected: The constructor completes without throwing (no explicit null check in implementation).
        /// Note: While the parameter is non-nullable, C# does not enforce this at runtime by default.
        /// </summary>
        [TestMethod]
        public void Constructor_WithNullLogger_CompletesWithoutThrowing()
        {
            // Arrange
            ILogger<WeatherForecastController>? logger = null;

            // Act
            var controller = new WeatherForecastController(logger!);

            // Assert
            Assert.IsNotNull(controller);
        }

        /// <summary>
        /// Tests that the constructor properly initializes the logger dependency.
        /// </summary>
        [TestMethod]
        public void Constructor_WithValidLogger_InitializesController()
        {
            // Arrange
            var mockLogger = new Mock<ILogger<WeatherForecastController>>();

            // Act
            var controller = new WeatherForecastController(mockLogger.Object);

            // Assert
            Assert.IsNotNull(controller);
        }

        /// <summary>
        /// Tests that Get returns a non-null collection of weather forecasts.
        /// </summary>
        [TestMethod]
        public void Get_WhenCalled_ReturnsNonNullResult()
        {
            // Arrange
            var mockLogger = new Mock<ILogger<WeatherForecastController>>();
            var controller = new WeatherForecastController(mockLogger.Object);

            // Act
            var result = controller.Get();

            // Assert
            Assert.IsNotNull(result);
        }

        /// <summary>
        /// Tests that Get returns exactly 5 weather forecast elements.
        /// Input: No parameters.
        /// Expected: Collection with exactly 5 elements.
        /// </summary>
        [TestMethod]
        public void Get_WhenCalled_ReturnsExactlyFiveElements()
        {
            // Arrange
            var mockLogger = new Mock<ILogger<WeatherForecastController>>();
            var controller = new WeatherForecastController(mockLogger.Object);

            // Act
            var result = controller.Get();

            // Assert
            Assert.AreEqual(5, result.Count());
        }

        /// <summary>
        /// Tests that Get returns a collection where all elements are non-null.
        /// Input: No parameters.
        /// Expected: All 5 elements are not null.
        /// </summary>
        [TestMethod]
        public void Get_WhenCalled_ReturnsAllNonNullElements()
        {
            // Arrange
            var mockLogger = new Mock<ILogger<WeatherForecastController>>();
            var controller = new WeatherForecastController(mockLogger.Object);

            // Act
            var result = controller.Get();

            // Assert
            foreach (var forecast in result)
            {
                Assert.IsNotNull(forecast);
            }
        }

        /// <summary>
        /// Tests that Get returns forecasts with Date values set in ascending order.
        /// Input: No parameters.
        /// Expected: Date values increase sequentially by one day each.
        /// </summary>
        [TestMethod]
        public void Get_WhenCalled_ReturnsForecastsWithAscendingDates()
        {
            // Arrange
            var mockLogger = new Mock<ILogger<WeatherForecastController>>();
            var controller = new WeatherForecastController(mockLogger.Object);

            // Act
            var result = controller.Get().ToList();

            // Assert
            for (int i = 1; i < result.Count; i++)
            {
                Assert.IsTrue(result[i].Date > result[i - 1].Date,
                    $"Date at index {i} should be greater than date at index {i - 1}");
            }
        }

        /// <summary>
        /// Tests that Get returns forecasts with Summary values that are all valid entries from the Summaries array.
        /// Input: No parameters.
        /// Expected: All Summary values match one of the predefined summaries.
        /// </summary>
        [TestMethod]
        public void Get_WhenCalled_ReturnsForecastsWithValidSummaries()
        {
            // Arrange
            var mockLogger = new Mock<ILogger<WeatherForecastController>>();
            var controller = new WeatherForecastController(mockLogger.Object);
            var validSummaries = new[]
            {
                "Freezing", "Bracing", "Chilly", "Cool", "Mild",
                "Warm", "Balmy", "Hot", "Sweltering", "Scorching"
            };

            // Act
            var result = controller.Get();

            // Assert
            foreach (var forecast in result)
            {
                Assert.IsTrue(validSummaries.Contains(forecast.Summary),
                    $"Summary '{forecast.Summary}' is not in the valid summaries list");
            }
        }

        /// <summary>
        /// Tests that Get returns forecasts with Date values that are in the future relative to the current date.
        /// Input: No parameters.
        /// Expected: All Date values are greater than today's date.
        /// </summary>
        [TestMethod]
        public void Get_WhenCalled_ReturnsForecastsWithFutureDates()
        {
            // Arrange
            var mockLogger = new Mock<ILogger<WeatherForecastController>>();
            var controller = new WeatherForecastController(mockLogger.Object);
            var today = DateOnly.FromDateTime(DateTime.Now);

            // Act
            var result = controller.Get();

            // Assert
            foreach (var forecast in result)
            {
                Assert.IsTrue(forecast.Date > today,
                    $"Forecast date {forecast.Date} should be greater than today {today}");
            }
        }

        /// <summary>
        /// Tests that Get returns forecasts with TemperatureC values set (initialized).
        /// Input: No parameters.
        /// Expected: All TemperatureC values are initialized (not default 0, but any valid int).
        /// Note: Due to randomness, we verify the value exists and is a valid integer.
        /// </summary>
        [TestMethod]
        public void Get_WhenCalled_ReturnsForecastsWithTemperatureSet()
        {
            // Arrange
            var mockLogger = new Mock<ILogger<WeatherForecastController>>();
            var controller = new WeatherForecastController(mockLogger.Object);

            // Act
            var result = controller.Get();

            // Assert
            foreach (var forecast in result)
            {
                // Temperature should be within the random range of -20 to 54 (inclusive)
                Assert.IsTrue(forecast.TemperatureC >= -20 && forecast.TemperatureC < 55,
                    $"TemperatureC {forecast.TemperatureC} should be between -20 and 54");
            }
        }

        /// <summary>
        /// Tests that Get returns forecasts with all properties properly initialized.
        /// Input: No parameters.
        /// Expected: Each forecast has Date, TemperatureC, and Summary all set to valid values.
        /// </summary>
        [TestMethod]
        public void Get_WhenCalled_ReturnsForecastsWithAllPropertiesInitialized()
        {
            // Arrange
            var mockLogger = new Mock<ILogger<WeatherForecastController>>();
            var controller = new WeatherForecastController(mockLogger.Object);

            // Act
            var result = controller.Get();

            // Assert
            foreach (var forecast in result)
            {
                Assert.IsNotNull(forecast);
                Assert.AreNotEqual(default(DateOnly), forecast.Date, "Date should be initialized");
                Assert.IsNotNull(forecast.Summary, "Summary should not be null");
                Assert.IsFalse(string.IsNullOrWhiteSpace(forecast.Summary), "Summary should not be empty or whitespace");
            }
        }

        /// <summary>
        /// Tests that multiple calls to Get produce different results due to randomness.
        /// Input: No parameters.
        /// Expected: At least one property differs between two consecutive calls (due to Random and DateTime.Now).
        /// </summary>
        [TestMethod]
        public void Get_WhenCalledMultipleTimes_ProducesDifferentResults()
        {
            // Arrange
            var mockLogger = new Mock<ILogger<WeatherForecastController>>();
            var controller = new WeatherForecastController(mockLogger.Object);

            // Act
            var result1 = controller.Get().ToList();
            var result2 = controller.Get().ToList();

            // Assert
            // Due to randomness in temperature and summary, at least one forecast should differ
            bool foundDifference = false;
            for (int i = 0; i < result1.Count && i < result2.Count; i++)
            {
                if (result1[i].TemperatureC != result2[i].TemperatureC ||
                    result1[i].Summary != result2[i].Summary)
                {
                    foundDifference = true;
                    break;
                }
            }

            Assert.IsTrue(foundDifference,
                "Multiple calls should produce different results due to randomness");
        }

        /// <summary>
        /// Tests that Get returns forecasts with TemperatureC values within the expected range.
        /// Input: No parameters.
        /// Expected: All TemperatureC values are between -20 (inclusive) and 55 (exclusive).
        /// </summary>
        [TestMethod]
        public void Get_WhenCalled_ReturnsForecastsWithTemperatureInValidRange()
        {
            // Arrange
            var mockLogger = new Mock<ILogger<WeatherForecastController>>();
            var controller = new WeatherForecastController(mockLogger.Object);

            // Act
            var result = controller.Get();

            // Assert
            foreach (var forecast in result)
            {
                Assert.IsTrue(forecast.TemperatureC >= -20);
                Assert.IsTrue(forecast.TemperatureC < 55);
            }
        }

        /// <summary>
        /// Tests that Get returns forecasts starting from tomorrow (current date + 1 day).
        /// Input: No parameters.
        /// Expected: The first forecast's Date is exactly one day after today.
        /// </summary>
        [TestMethod]
        public void Get_WhenCalled_ReturnsFirstForecastForTomorrow()
        {
            // Arrange
            var mockLogger = new Mock<ILogger<WeatherForecastController>>();
            var controller = new WeatherForecastController(mockLogger.Object);
            var tomorrow = DateOnly.FromDateTime(DateTime.Now.AddDays(1));

            // Act
            var result = controller.Get().ToList();

            // Assert
            Assert.AreEqual(tomorrow, result[0].Date);
        }

        /// <summary>
        /// Tests that multiple calls to Get produce potentially different results due to randomness.
        /// Input: No parameters.
        /// Expected: At least one forecast differs between consecutive calls (due to Random.Shared).
        /// </summary>
        [TestMethod]
        public void Get_WhenCalledMultipleTimes_MayProduceDifferentResults()
        {
            // Arrange
            var mockLogger = new Mock<ILogger<WeatherForecastController>>();
            var controller = new WeatherForecastController(mockLogger.Object);

            // Act
            var result1 = controller.Get().ToList();
            var result2 = controller.Get().ToList();

            // Assert
            // Due to randomness, at least one temperature or summary should differ
            // Note: This test may occasionally fail due to random chance, but is extremely unlikely
            bool anyDifference = false;
            for (int i = 0; i < result1.Count; i++)
            {
                if (result1[i].TemperatureC != result2[i].TemperatureC || result1[i].Summary != result2[i].Summary)
                {
                    anyDifference = true;
                    break;
                }
            }

            Assert.IsTrue(anyDifference, "Expected at least one difference between two consecutive Get() calls due to randomness");
        }

        /// <summary>
        /// Tests that Get returns forecasts with non-null and non-empty Summary values.
        /// Input: No parameters.
        /// Expected: All Summary values are not null and not empty or whitespace.
        /// </summary>
        [TestMethod]
        public void Get_WhenCalled_ReturnsForecastsWithNonEmptySummaries()
        {
            // Arrange
            var mockLogger = new Mock<ILogger<WeatherForecastController>>();
            var controller = new WeatherForecastController(mockLogger.Object);

            // Act
            var result = controller.Get();

            // Assert
            foreach (var forecast in result)
            {
                Assert.IsNotNull(forecast.Summary);
                Assert.IsFalse(string.IsNullOrEmpty(forecast.Summary));
                Assert.IsFalse(string.IsNullOrWhiteSpace(forecast.Summary));
            }
        }

        /// <summary>
        /// Tests that Get returns a result that can be enumerated multiple times.
        /// Input: No parameters.
        /// Expected: The result can be iterated multiple times successfully (array behavior).
        /// </summary>
        [TestMethod]
        public void Get_WhenCalled_ReturnsEnumerableResultThatCanBeEnumeratedMultipleTimes()
        {
            // Arrange
            var mockLogger = new Mock<ILogger<WeatherForecastController>>();
            var controller = new WeatherForecastController(mockLogger.Object);

            // Act
            var result = controller.Get();

            // Assert
            var firstCount = result.Count();
            var secondCount = result.Count();
            Assert.AreEqual(firstCount, secondCount);
            Assert.AreEqual(5, firstCount);
        }

        /// <summary>
        /// Tests that Get returns forecasts with the last date being 5 days from today.
        /// Input: No parameters.
        /// Expected: The fifth forecast's Date is exactly 5 days after today.
        /// </summary>
        [TestMethod]
        public void Get_WhenCalled_ReturnsLastForecastForFiveDaysFromToday()
        {
            // Arrange
            var mockLogger = new Mock<ILogger<WeatherForecastController>>();
            var controller = new WeatherForecastController(mockLogger.Object);
            var fiveDaysFromNow = DateOnly.FromDateTime(DateTime.Now.AddDays(5));

            // Act
            var result = controller.Get().ToList();

            // Assert
            Assert.AreEqual(fiveDaysFromNow, result[4].Date);
        }
    }
}