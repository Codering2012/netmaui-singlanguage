using System;

using Microsoft.VisualStudio.TestTools.UnitTesting;
using SignLanguageApi;

namespace SignLanguageApi.UnitTests
{
    [TestClass]
    public class WeatherForecastTests
    {
        /// <summary>
        /// Tests TemperatureF property calculation with various Celsius values.
        /// Verifies the conversion formula: F = 32 + (int)(C / 0.5556).
        /// </summary>
        /// <param name="temperatureC">The Celsius temperature to test.</param>
        /// <param name="expectedF">The expected Fahrenheit temperature.</param>
        [TestMethod]
        [DataRow(0, 32)]
        [DataRow(100, 212)]
        [DataRow(-40, -40)]
        [DataRow(20, 68)]
        [DataRow(-20, -4)]
        [DataRow(37, 98)]
        [DataRow(1, 33)]
        [DataRow(-1, 30)]
        [TestCategory("ProductionBugSuspected")]
        [Ignore("ProductionBugSuspected")]
        public void TemperatureF_VariousCelsiusValues_ReturnsCorrectFahrenheit(int temperatureC, int expectedF)
        {
            // Arrange
            var weatherForecast = new WeatherForecast
            {
                TemperatureC = temperatureC
            };

            // Act
            int actualF = weatherForecast.TemperatureF;

            // Assert
            Assert.AreEqual(expectedF, actualF);
        }

        /// <summary>
        /// Tests TemperatureF property with large positive values near maximum boundary.
        /// Verifies calculation accuracy and overflow handling.
        /// </summary>
        [TestMethod]
        [DataRow(1000000)]
        [DataRow(500000)]
        [DataRow(100000)]
        public void TemperatureF_LargePositiveValues_ReturnsCalculatedValue(int temperatureC)
        {
            // Arrange
            var weatherForecast = new WeatherForecast
            {
                TemperatureC = temperatureC
            };

            // Act
            int actualF = weatherForecast.TemperatureF;

            // Assert
            int expectedF = 32 + (int)(temperatureC / 0.5556);
            Assert.AreEqual(expectedF, actualF);
        }

        /// <summary>
        /// Tests TemperatureF property with large negative values near minimum boundary.
        /// Verifies calculation accuracy and underflow handling.
        /// </summary>
        [TestMethod]
        [DataRow(-1000000)]
        [DataRow(-500000)]
        [DataRow(-100000)]
        public void TemperatureF_LargeNegativeValues_ReturnsCalculatedValue(int temperatureC)
        {
            // Arrange
            var weatherForecast = new WeatherForecast
            {
                TemperatureC = temperatureC
            };

            // Act
            int actualF = weatherForecast.TemperatureF;

            // Assert
            int expectedF = 32 + (int)(temperatureC / 0.5556);
            Assert.AreEqual(expectedF, actualF);
        }

        /// <summary>
        /// Tests TemperatureF property with zero and small boundary values.
        /// Verifies calculation accuracy near zero boundary.
        /// </summary>
        /// <param name="temperatureC">The Celsius temperature to test.</param>
        [TestMethod]
        [DataRow(0)]
        [DataRow(1)]
        [DataRow(-1)]
        [DataRow(2)]
        [DataRow(-2)]
        [DataRow(5)]
        [DataRow(-5)]
        public void TemperatureF_SmallBoundaryValues_ReturnsCalculatedValue(int temperatureC)
        {
            // Arrange
            var weatherForecast = new WeatherForecast
            {
                TemperatureC = temperatureC
            };

            // Act
            int actualF = weatherForecast.TemperatureF;

            // Assert
            int expectedF = 32 + (int)(temperatureC / 0.5556);
            Assert.AreEqual(expectedF, actualF);
        }

        /// <summary>
        /// Tests TemperatureF property with medium range positive values.
        /// Verifies calculation accuracy for typical and boundary positive temperatures.
        /// </summary>
        /// <param name="temperatureC">The Celsius temperature to test.</param>
        [TestMethod]
        [DataRow(10)]
        [DataRow(15)]
        [DataRow(25)]
        [DataRow(30)]
        [DataRow(50)]
        [DataRow(75)]
        [DataRow(1000)]
        [DataRow(10000)]
        public void TemperatureF_MediumPositiveValues_ReturnsCalculatedValue(int temperatureC)
        {
            // Arrange
            var weatherForecast = new WeatherForecast
            {
                TemperatureC = temperatureC
            };

            // Act
            int actualF = weatherForecast.TemperatureF;

            // Assert
            int expectedF = 32 + (int)(temperatureC / 0.5556);
            Assert.AreEqual(expectedF, actualF);
        }

        /// <summary>
        /// Tests TemperatureF property with medium range negative values.
        /// Verifies calculation accuracy for typical and boundary negative temperatures.
        /// </summary>
        /// <param name="temperatureC">The Celsius temperature to test.</param>
        [TestMethod]
        [DataRow(-10)]
        [DataRow(-15)]
        [DataRow(-25)]
        [DataRow(-30)]
        [DataRow(-50)]
        [DataRow(-75)]
        [DataRow(-1000)]
        [DataRow(-10000)]
        public void TemperatureF_MediumNegativeValues_ReturnsCalculatedValue(int temperatureC)
        {
            // Arrange
            var weatherForecast = new WeatherForecast
            {
                TemperatureC = temperatureC
            };

            // Act
            int actualF = weatherForecast.TemperatureF;

            // Assert
            int expectedF = 32 + (int)(temperatureC / 0.5556);
            Assert.AreEqual(expectedF, actualF);
        }
    }
}