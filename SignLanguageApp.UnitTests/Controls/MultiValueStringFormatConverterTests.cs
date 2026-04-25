using System;
using System.Globalization;

using Microsoft.Maui.Controls;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using SignLanguageApp.Converters;

namespace SignLanguageApp.Converters.UnitTests
{
    /// <summary>
    /// Unit tests for the <see cref="MultiValueStringFormatConverter"/> class.
    /// </summary>
    [TestClass]
    public class MultiValueStringFormatConverterTests
    {
        /// <summary>
        /// Tests that Convert returns an empty string when values parameter is null.
        /// </summary>
        [TestMethod]
        public void Convert_NullValues_ReturnsEmptyString()
        {
            // Arrange
            MultiValueStringFormatConverter converter = new MultiValueStringFormatConverter();
            object?[]? values = null;

            // Act
            object? result = converter.Convert(values, null, null, null);

            // Assert
            Assert.AreEqual(string.Empty, result);
        }

        /// <summary>
        /// Tests that Convert returns an empty string when values array is empty.
        /// </summary>
        [TestMethod]
        public void Convert_EmptyValuesArray_ReturnsEmptyString()
        {
            // Arrange
            MultiValueStringFormatConverter converter = new MultiValueStringFormatConverter();
            object?[] values = Array.Empty<object?>();

            // Act
            object? result = converter.Convert(values, null, null, null);

            // Assert
            Assert.AreEqual(string.Empty, result);
        }

        /// <summary>
        /// Tests that Convert returns an empty string when values array contains only one element.
        /// </summary>
        [TestMethod]
        public void Convert_SingleElementArray_ReturnsEmptyString()
        {
            // Arrange
            MultiValueStringFormatConverter converter = new MultiValueStringFormatConverter();
            object?[] values = new object?[] { "first" };

            // Act
            object? result = converter.Convert(values, null, null, null);

            // Assert
            Assert.AreEqual(string.Empty, result);
        }

        /// <summary>
        /// Tests that Convert returns default format when values has two elements and parameter is null.
        /// </summary>
        [TestMethod]
        public void Convert_TwoElements_NullParameter_ReturnsDefaultFormat()
        {
            // Arrange
            MultiValueStringFormatConverter converter = new MultiValueStringFormatConverter();
            object?[] values = new object?[] { "first", "second" };

            // Act
            object? result = converter.Convert(values, null, null, null);

            // Assert
            Assert.AreEqual("first / second", result);
        }

        /// <summary>
        /// Tests that Convert returns default format when parameter is not a string.
        /// </summary>
        [TestMethod]
        [DataRow(123)]
        [DataRow(true)]
        [DataRow(45.67)]
        public void Convert_TwoElements_NonStringParameter_ReturnsDefaultFormat(object parameter)
        {
            // Arrange
            MultiValueStringFormatConverter converter = new MultiValueStringFormatConverter();
            object?[] values = new object?[] { "first", "second" };

            // Act
            object? result = converter.Convert(values, null, parameter, null);

            // Assert
            Assert.AreEqual("first / second", result);
        }

        /// <summary>
        /// Tests that Convert uses the format string when parameter is a string.
        /// </summary>
        [TestMethod]
        public void Convert_TwoElements_StringParameter_ReturnsFormattedString()
        {
            // Arrange
            MultiValueStringFormatConverter converter = new MultiValueStringFormatConverter();
            object?[] values = new object?[] { "Hello", "World" };
            string formatString = "{0} - {1}";

            // Act
            object? result = converter.Convert(values, null, formatString, null);

            // Assert
            Assert.AreEqual("Hello - World", result);
        }

        /// <summary>
        /// Tests that Convert uses InvariantCulture when culture parameter is null and format string is provided.
        /// </summary>
        [TestMethod]
        public void Convert_WithFormatString_NullCulture_UsesInvariantCulture()
        {
            // Arrange
            MultiValueStringFormatConverter converter = new MultiValueStringFormatConverter();
            object?[] values = new object?[] { 1234.56, 7890.12 };
            string formatString = "{0:N2} and {1:N2}";

            // Act
            object? result = converter.Convert(values, null, formatString, null);

            // Assert
            string expected = string.Format(CultureInfo.InvariantCulture, formatString, 1234.56, 7890.12);
            Assert.AreEqual(expected, result);
        }

        /// <summary>
        /// Tests that Convert uses the provided culture for formatting.
        /// </summary>
        [TestMethod]
        public void Convert_WithFormatString_ProvidedCulture_UsesProvidedCulture()
        {
            // Arrange
            MultiValueStringFormatConverter converter = new MultiValueStringFormatConverter();
            object?[] values = new object?[] { 1234.56, 7890.12 };
            string formatString = "{0:N2} and {1:N2}";
            CultureInfo culture = new CultureInfo("fr-FR");

            // Act
            object? result = converter.Convert(values, null, formatString, culture);

            // Assert
            string expected = string.Format(culture, formatString, 1234.56, 7890.12);
            Assert.AreEqual(expected, result);
        }

        /// <summary>
        /// Tests that Convert uses only the first two elements when values array has more than two elements.
        /// </summary>
        [TestMethod]
        public void Convert_MoreThanTwoElements_UsesFirstTwoOnly()
        {
            // Arrange
            MultiValueStringFormatConverter converter = new MultiValueStringFormatConverter();
            object?[] values = new object?[] { "first", "second", "third", "fourth" };

            // Act
            object? result = converter.Convert(values, null, null, null);

            // Assert
            Assert.AreEqual("first / second", result);
        }

        /// <summary>
        /// Tests that Convert handles null elements in the values array.
        /// </summary>
        [TestMethod]
        public void Convert_NullElementsInArray_HandlesGracefully()
        {
            // Arrange
            MultiValueStringFormatConverter converter = new MultiValueStringFormatConverter();
            object?[] values = new object?[] { null, null };

            // Act
            object? result = converter.Convert(values, null, null, null);

            // Assert
            Assert.AreEqual(" / ", result);
        }

        /// <summary>
        /// Tests that Convert handles null elements with format string.
        /// </summary>
        [TestMethod]
        public void Convert_NullElementsWithFormatString_HandlesGracefully()
        {
            // Arrange
            MultiValueStringFormatConverter converter = new MultiValueStringFormatConverter();
            object?[] values = new object?[] { null, "second" };
            string formatString = "First: {0}, Second: {1}";

            // Act
            object? result = converter.Convert(values, null, formatString, null);

            // Assert
            Assert.AreEqual("First: , Second: second", result);
        }

        /// <summary>
        /// Tests that Convert handles different value types correctly.
        /// </summary>
        [TestMethod]
        [DataRow(123, "text", "123 / text")]
        [DataRow(45.67, true, "45.67 / True")]
        [DataRow("string1", "string2", "string1 / string2")]
        public void Convert_DifferentValueTypes_ConvertsToString(object first, object second, string expected)
        {
            // Arrange
            MultiValueStringFormatConverter converter = new MultiValueStringFormatConverter();
            object?[] values = new object?[] { first, second };

            // Act
            object? result = converter.Convert(values, null, null, null);

            // Assert
            Assert.AreEqual(expected, result);
        }

        /// <summary>
        /// Tests that Convert handles empty string as format parameter.
        /// </summary>
        [TestMethod]
        public void Convert_EmptyStringParameter_ReturnsEmptyString()
        {
            // Arrange
            MultiValueStringFormatConverter converter = new MultiValueStringFormatConverter();
            object?[] values = new object?[] { "first", "second" };
            string emptyFormat = "";

            // Act
            object? result = converter.Convert(values, null, emptyFormat, null);

            // Assert
            Assert.AreEqual("", result);
        }

        /// <summary>
        /// Tests that Convert handles format string with only one placeholder.
        /// </summary>
        [TestMethod]
        public void Convert_FormatStringWithOnePlaceholder_UsesOnlyFirstValue()
        {
            // Arrange
            MultiValueStringFormatConverter converter = new MultiValueStringFormatConverter();
            object?[] values = new object?[] { "first", "second" };
            string formatString = "Value: {0}";

            // Act
            object? result = converter.Convert(values, null, formatString, null);

            // Assert
            Assert.AreEqual("Value: first", result);
        }

        /// <summary>
        /// Tests that Convert handles special characters in values.
        /// </summary>
        [TestMethod]
        public void Convert_ValuesWithSpecialCharacters_ReturnsCorrectFormat()
        {
            // Arrange
            MultiValueStringFormatConverter converter = new MultiValueStringFormatConverter();
            object?[] values = new object?[] { "first\nline", "second\ttab" };

            // Act
            object? result = converter.Convert(values, null, null, null);

            // Assert
            Assert.AreEqual("first\nline / second\ttab", result);
        }

        /// <summary>
        /// Tests that Convert handles very long strings.
        /// </summary>
        [TestMethod]
        public void Convert_VeryLongStrings_ReturnsCorrectFormat()
        {
            // Arrange
            MultiValueStringFormatConverter converter = new MultiValueStringFormatConverter();
            string longString1 = new string('a', 10000);
            string longString2 = new string('b', 10000);
            object?[] values = new object?[] { longString1, longString2 };

            // Act
            object? result = converter.Convert(values, null, null, null);

            // Assert
            Assert.AreEqual($"{longString1} / {longString2}", result);
        }

        /// <summary>
        /// Tests that Convert ignores targetType parameter.
        /// </summary>
        [TestMethod]
        public void Convert_WithTargetType_IgnoresTargetType()
        {
            // Arrange
            MultiValueStringFormatConverter converter = new MultiValueStringFormatConverter();
            object?[] values = new object?[] { "first", "second" };
            Type targetType = typeof(int);

            // Act
            object? result = converter.Convert(values, targetType, null, null);

            // Assert
            Assert.AreEqual("first / second", result);
        }

        /// <summary>
        /// Tests that Convert handles numeric extremes with format string.
        /// </summary>
        [TestMethod]
        [DataRow(int.MinValue, int.MaxValue)]
        [DataRow(0, 0)]
        [DataRow(-1, 1)]
        public void Convert_NumericExtremes_FormatsCorrectly(int first, int second)
        {
            // Arrange
            MultiValueStringFormatConverter converter = new MultiValueStringFormatConverter();
            object?[] values = new object?[] { first, second };
            string formatString = "{0} and {1}";

            // Act
            object? result = converter.Convert(values, null, formatString, null);

            // Assert
            string expected = string.Format(CultureInfo.InvariantCulture, formatString, first, second);
            Assert.AreEqual(expected, result);
        }

        /// <summary>
        /// Tests that Convert handles floating-point special values.
        /// </summary>
        [TestMethod]
        [DataRow(double.NaN, double.PositiveInfinity, "NaN / ∞")]
        [DataRow(double.NegativeInfinity, double.PositiveInfinity, "-∞ / ∞")]
        [DataRow(0.0, double.NaN, "0 / NaN")]
        public void Convert_FloatingPointSpecialValues_FormatsCorrectly(double first, double second, string expected)
        {
            // Arrange
            MultiValueStringFormatConverter converter = new MultiValueStringFormatConverter();
            object?[] values = new object?[] { first, second };

            // Act
            object? result = converter.Convert(values, null, null, null);

            // Assert
            Assert.AreEqual(expected, result);
        }

    }
}