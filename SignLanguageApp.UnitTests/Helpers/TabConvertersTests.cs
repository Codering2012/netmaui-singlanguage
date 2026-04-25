using System;
using System.Globalization;

using Microsoft.Maui.Controls;
using Microsoft.Maui.Graphics;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using SignLanguageApp.Helpers;

namespace SignLanguageApp.Helpers.UnitTests
{
    /// <summary>
    /// Unit tests for the <see cref="TabIndexToBgColorConverter"/> class.
    /// </summary>
    [TestClass]
    public class TabIndexToBgColorConverterTests
    {
        /// <summary>
        /// Tests that ConvertBack always returns 0 regardless of input parameters.
        /// Tests various combinations of nullable and non-nullable parameters including null values,
        /// different value types, target types, parameters, and culture info values.
        /// Expected result: Always returns 0.
        /// </summary>
        [TestMethod]
        [DataRow(null, typeof(int), null, null, DisplayName = "All nullable parameters null")]
        [DataRow(0, typeof(int), null, null, DisplayName = "Value is 0, others null")]
        [DataRow(5, typeof(int), null, null, DisplayName = "Value is 5, others null")]
        [DataRow(-1, typeof(int), null, null, DisplayName = "Value is -1, others null")]
        [DataRow(int.MaxValue, typeof(int), null, null, DisplayName = "Value is int.MaxValue")]
        [DataRow(int.MinValue, typeof(int), null, null, DisplayName = "Value is int.MinValue")]
        [DataRow("test", typeof(string), null, null, DisplayName = "Value is string")]
        [DataRow("", typeof(string), null, null, DisplayName = "Value is empty string")]
        [DataRow(true, typeof(bool), null, null, DisplayName = "Value is true")]
        [DataRow(false, typeof(bool), null, null, DisplayName = "Value is false")]
        [DataRow(null, typeof(string), "param", null, DisplayName = "Parameter is non-null string")]
        [DataRow(null, typeof(object), 123, null, DisplayName = "Parameter is int")]
        public void ConvertBack_VariousInputs_AlwaysReturnsZero(object? value, Type targetType, object? parameter, string? cultureString)
        {
            // Arrange
            var converter = new TabIndexToBgColorConverter();
            CultureInfo culture = cultureString != null ? new CultureInfo(cultureString) : CultureInfo.InvariantCulture;

            // Act
            var result = converter.ConvertBack(value, targetType, parameter, culture);

            // Assert
            Assert.IsNotNull(result);
            Assert.AreEqual(0, result);
            Assert.IsInstanceOfType(result, typeof(int));
        }

        /// <summary>
        /// Tests that ConvertBack returns 0 with different CultureInfo values.
        /// Expected result: Always returns 0 regardless of culture.
        /// </summary>
        [TestMethod]
        [DataRow("en-US", DisplayName = "English (United States) culture")]
        [DataRow("fr-FR", DisplayName = "French (France) culture")]
        [DataRow("de-DE", DisplayName = "German (Germany) culture")]
        [DataRow("ja-JP", DisplayName = "Japanese (Japan) culture")]
        public void ConvertBack_DifferentCultures_AlwaysReturnsZero(string cultureName)
        {
            // Arrange
            var converter = new TabIndexToBgColorConverter();
            var culture = new CultureInfo(cultureName);
            object? value = "test";
            Type targetType = typeof(int);
            object? parameter = null;

            // Act
            var result = converter.ConvertBack(value, targetType, parameter, culture);

            // Assert
            Assert.IsNotNull(result);
            Assert.AreEqual(0, result);
        }

        /// <summary>
        /// Tests that ConvertBack returns 0 with InvariantCulture.
        /// Expected result: Returns 0.
        /// </summary>
        [TestMethod]
        public void ConvertBack_InvariantCulture_ReturnsZero()
        {
            // Arrange
            var converter = new TabIndexToBgColorConverter();
            var culture = CultureInfo.InvariantCulture;

            // Act
            var result = converter.ConvertBack(null, typeof(int), null, culture);

            // Assert
            Assert.IsNotNull(result);
            Assert.AreEqual(0, result);
        }

        /// <summary>
        /// Tests that ConvertBack returns 0 with CurrentCulture.
        /// Expected result: Returns 0.
        /// </summary>
        [TestMethod]
        public void ConvertBack_CurrentCulture_ReturnsZero()
        {
            // Arrange
            var converter = new TabIndexToBgColorConverter();
            var culture = CultureInfo.CurrentCulture;

            // Act
            var result = converter.ConvertBack(null, typeof(int), null, culture);

            // Assert
            Assert.IsNotNull(result);
            Assert.AreEqual(0, result);
        }

        /// <summary>
        /// Tests that ConvertBack returns 0 with various target types.
        /// Expected result: Always returns 0 regardless of target type.
        /// </summary>
        [TestMethod]
        [DataRow(typeof(int), DisplayName = "Target type int")]
        [DataRow(typeof(string), DisplayName = "Target type string")]
        [DataRow(typeof(bool), DisplayName = "Target type bool")]
        [DataRow(typeof(double), DisplayName = "Target type double")]
        [DataRow(typeof(object), DisplayName = "Target type object")]
        [DataRow(typeof(DateTime), DisplayName = "Target type DateTime")]
        public void ConvertBack_VariousTargetTypes_AlwaysReturnsZero(Type targetType)
        {
            // Arrange
            var converter = new TabIndexToBgColorConverter();
            var culture = CultureInfo.InvariantCulture;

            // Act
            var result = converter.ConvertBack(null, targetType, null, culture);

            // Assert
            Assert.IsNotNull(result);
            Assert.AreEqual(0, result);
        }

        /// <summary>
        /// Tests that ConvertBack returns 0 with whitespace string value.
        /// Expected result: Returns 0.
        /// </summary>
        [TestMethod]
        public void ConvertBack_WhitespaceStringValue_ReturnsZero()
        {
            // Arrange
            var converter = new TabIndexToBgColorConverter();
            var culture = CultureInfo.InvariantCulture;

            // Act
            var result = converter.ConvertBack("   ", typeof(string), null, culture);

            // Assert
            Assert.IsNotNull(result);
            Assert.AreEqual(0, result);
        }

        /// <summary>
        /// Tests that ConvertBack returns 0 with very long string value.
        /// Expected result: Returns 0.
        /// </summary>
        [TestMethod]
        public void ConvertBack_VeryLongStringValue_ReturnsZero()
        {
            // Arrange
            var converter = new TabIndexToBgColorConverter();
            var culture = CultureInfo.InvariantCulture;
            var longString = new string('a', 10000);

            // Act
            var result = converter.ConvertBack(longString, typeof(string), null, culture);

            // Assert
            Assert.IsNotNull(result);
            Assert.AreEqual(0, result);
        }

        /// <summary>
        /// Tests that ConvertBack returns 0 with special characters in string value.
        /// Expected result: Returns 0.
        /// </summary>
        [TestMethod]
        public void ConvertBack_StringWithSpecialCharacters_ReturnsZero()
        {
            // Arrange
            var converter = new TabIndexToBgColorConverter();
            var culture = CultureInfo.InvariantCulture;
            var specialString = "!@#$%^&*()_+{}|:\"<>?~`-=[]\\;',./";

            // Act
            var result = converter.ConvertBack(specialString, typeof(string), null, culture);

            // Assert
            Assert.IsNotNull(result);
            Assert.AreEqual(0, result);
        }

        /// <summary>
        /// Tests that ConvertBack returns 0 with double values including edge cases.
        /// Expected result: Always returns 0.
        /// </summary>
        [TestMethod]
        [DataRow(double.NaN, DisplayName = "Value is double.NaN")]
        [DataRow(double.PositiveInfinity, DisplayName = "Value is double.PositiveInfinity")]
        [DataRow(double.NegativeInfinity, DisplayName = "Value is double.NegativeInfinity")]
        [DataRow(double.MaxValue, DisplayName = "Value is double.MaxValue")]
        [DataRow(double.MinValue, DisplayName = "Value is double.MinValue")]
        [DataRow(0.0, DisplayName = "Value is 0.0")]
        public void ConvertBack_DoubleEdgeCases_AlwaysReturnsZero(double value)
        {
            // Arrange
            var converter = new TabIndexToBgColorConverter();
            var culture = CultureInfo.InvariantCulture;

            // Act
            var result = converter.ConvertBack(value, typeof(double), null, culture);

            // Assert
            Assert.IsNotNull(result);
            Assert.AreEqual(0, result);
        }

        /// <summary>
        /// Tests that ConvertBack always returns 0 regardless of input parameters.
        /// </summary>
        /// <param name="value">The value parameter to test.</param>
        /// <param name="targetType">The target type parameter to test.</param>
        /// <param name="parameter">The parameter to test.</param>
        /// <param name="culture">The culture info to test.</param>
        [TestMethod]
        [DataRow(null, typeof(int), null, null, DisplayName = "All nullable parameters are null")]
        [DataRow(5, typeof(string), "param", null, DisplayName = "Value is int, parameter is string, culture is null")]
        [DataRow("test", typeof(bool), 123, null, DisplayName = "Value is string, parameter is int, culture is null")]
        [DataRow(true, typeof(object), false, null, DisplayName = "Value and parameter are booleans, culture is null")]
        [DataRow(null, null, null, null, DisplayName = "All parameters are null")]
        public void ConvertBack_VariousInputs_ReturnsZero(object? value, Type? targetType, object? parameter, CultureInfo? culture)
        {
            // Arrange
            var converter = new TabIndexToBgColorConverter();

            // Act
            var result = converter.ConvertBack(value, targetType!, parameter, culture!);

            // Assert
            Assert.IsNotNull(result);
            Assert.AreEqual(0, result);
        }

        /// <summary>
        /// Tests that ConvertBack returns 0 when value is an integer.
        /// </summary>
        [TestMethod]
        [DataRow(0, DisplayName = "Value is zero")]
        [DataRow(1, DisplayName = "Value is positive integer")]
        [DataRow(-1, DisplayName = "Value is negative integer")]
        [DataRow(int.MaxValue, DisplayName = "Value is int.MaxValue")]
        [DataRow(int.MinValue, DisplayName = "Value is int.MinValue")]
        public void ConvertBack_IntegerValues_ReturnsZero(int value)
        {
            // Arrange
            var converter = new TabIndexToBgColorConverter();
            var targetType = typeof(int);
            var culture = CultureInfo.InvariantCulture;

            // Act
            var result = converter.ConvertBack(value, targetType, null, culture);

            // Assert
            Assert.IsNotNull(result);
            Assert.AreEqual(0, result);
        }

        /// <summary>
        /// Tests that ConvertBack returns 0 when value is a string.
        /// </summary>
        [TestMethod]
        [DataRow(null, DisplayName = "Value is null")]
        [DataRow("", DisplayName = "Value is empty string")]
        [DataRow("   ", DisplayName = "Value is whitespace")]
        [DataRow("test", DisplayName = "Value is normal string")]
        [DataRow("very long string with special characters !@#$%^&*()", DisplayName = "Value is long string with special characters")]
        public void ConvertBack_StringValues_ReturnsZero(string? value)
        {
            // Arrange
            var converter = new TabIndexToBgColorConverter();
            var targetType = typeof(string);
            var culture = CultureInfo.InvariantCulture;

            // Act
            var result = converter.ConvertBack(value, targetType, null, culture);

            // Assert
            Assert.IsNotNull(result);
            Assert.AreEqual(0, result);
        }

        /// <summary>
        /// Tests that ConvertBack returns 0 with various CultureInfo values.
        /// </summary>
        [TestMethod]
        public void ConvertBack_VariousCultures_ReturnsZero()
        {
            // Arrange
            var converter = new TabIndexToBgColorConverter();
            var cultures = new CultureInfo?[]
            {
                null,
                CultureInfo.InvariantCulture,
                CultureInfo.CurrentCulture,
                new CultureInfo("en-US"),
                new CultureInfo("fr-FR"),
                new CultureInfo("ja-JP")
            };

            foreach (var culture in cultures)
            {
                // Act
                var result = converter.ConvertBack(1, typeof(int), null, culture!);

                // Assert
                Assert.IsNotNull(result, $"Result should not be null for culture: {culture?.Name ?? "null"}");
                Assert.AreEqual(0, result, $"Result should be 0 for culture: {culture?.Name ?? "null"}");
            }
        }

        /// <summary>
        /// Tests that ConvertBack returns 0 with various target types.
        /// </summary>
        [TestMethod]
        [DataRow(typeof(int), DisplayName = "Target type is int")]
        [DataRow(typeof(string), DisplayName = "Target type is string")]
        [DataRow(typeof(bool), DisplayName = "Target type is bool")]
        [DataRow(typeof(object), DisplayName = "Target type is object")]
        [DataRow(typeof(double), DisplayName = "Target type is double")]
        public void ConvertBack_VariousTargetTypes_ReturnsZero(Type targetType)
        {
            // Arrange
            var converter = new TabIndexToBgColorConverter();
            var culture = CultureInfo.InvariantCulture;

            // Act
            var result = converter.ConvertBack(1, targetType, null, culture);

            // Assert
            Assert.IsNotNull(result);
            Assert.AreEqual(0, result);
        }

        /// <summary>
        /// Tests that ConvertBack returns 0 when parameter has various values.
        /// </summary>
        [TestMethod]
        [DataRow(null, DisplayName = "Parameter is null")]
        [DataRow(0, DisplayName = "Parameter is zero")]
        [DataRow(1, DisplayName = "Parameter is one")]
        [DataRow(-1, DisplayName = "Parameter is negative")]
        [DataRow("string", DisplayName = "Parameter is string")]
        [DataRow(true, DisplayName = "Parameter is boolean true")]
        [DataRow(false, DisplayName = "Parameter is boolean false")]
        public void ConvertBack_VariousParameters_ReturnsZero(object? parameter)
        {
            // Arrange
            var converter = new TabIndexToBgColorConverter();
            var targetType = typeof(int);
            var culture = CultureInfo.InvariantCulture;

            // Act
            var result = converter.ConvertBack(1, targetType, parameter, culture);

            // Assert
            Assert.IsNotNull(result);
            Assert.AreEqual(0, result);
        }

        /// <summary>
        /// Tests that ConvertBack returns 0 for various input combinations including null values,
        /// different value types, different target types, and different cultures.
        /// </summary>
        /// <param name="value">The value to convert back.</param>
        /// <param name="targetTypeName">The name of the target type (null, int, string, or object).</param>
        /// <param name="parameter">The converter parameter.</param>
        /// <param name="cultureName">The culture name (null, invariant, or en-US).</param>
        [TestMethod]
        [DataRow(null, "int", null, "invariant")]
        [DataRow(null, "string", null, "en-US")]
        [DataRow(null, null, null, null)]
        [DataRow(5, "int", "test", "invariant")]
        [DataRow("test", "string", 123, "en-US")]
        [DataRow(true, "object", false, "fr-FR")]
        [DataRow(0, null, "", "invariant")]
        [DataRow(-1, "int", null, null)]
        [DataRow(int.MaxValue, "int", int.MinValue, "invariant")]
        [DataRow(double.NaN, "double", double.PositiveInfinity, "de-DE")]
        public void ConvertBack_VariousInputs_AlwaysReturnsZero(object? value, string? targetTypeName, object? parameter, string? cultureName)
        {
            // Arrange
            var converter = new TabIndexToBgColorConverter();
            Type? targetType = targetTypeName switch
            {
                "int" => typeof(int),
                "string" => typeof(string),
                "object" => typeof(object),
                "double" => typeof(double),
                null => null,
                _ => typeof(object)
            };
            CultureInfo? culture = cultureName switch
            {
                "invariant" => CultureInfo.InvariantCulture,
                "en-US" => new CultureInfo("en-US"),
                "fr-FR" => new CultureInfo("fr-FR"),
                "de-DE" => new CultureInfo("de-DE"),
                null => null,
                _ => CultureInfo.InvariantCulture
            };

            // Act
            var result = converter.ConvertBack(value, targetType!, parameter, culture!);

            // Assert
            Assert.IsNotNull(result);
            Assert.AreEqual(0, result);
            Assert.IsInstanceOfType(result, typeof(int));
        }

        /// <summary>
        /// Tests that ConvertBack returns 0 when all parameters are null.
        /// This verifies the method handles the edge case of all null inputs gracefully.
        /// </summary>
        [TestMethod]
        public void ConvertBack_AllNullParameters_ReturnsZero()
        {
            // Arrange
            var converter = new TabIndexToBgColorConverter();

            // Act
            var result = converter.ConvertBack(null, null!, null, null!);

            // Assert
            Assert.IsNotNull(result);
            Assert.AreEqual(0, result);
        }

        /// <summary>
        /// Tests that ConvertBack returns 0 for extreme numeric values.
        /// This validates behavior with boundary values for different numeric types.
        /// </summary>
        /// <param name="value">The extreme numeric value to test.</param>
        [TestMethod]
        [DataRow(int.MinValue)]
        [DataRow(int.MaxValue)]
        [DataRow(0)]
        [DataRow(-1)]
        [DataRow(1)]
        [DataRow(long.MinValue)]
        [DataRow(long.MaxValue)]
        public void ConvertBack_ExtremeNumericValues_ReturnsZero(object value)
        {
            // Arrange
            var converter = new TabIndexToBgColorConverter();
            var targetType = typeof(int);
            var culture = CultureInfo.InvariantCulture;

            // Act
            var result = converter.ConvertBack(value, targetType, null, culture);

            // Assert
            Assert.IsNotNull(result);
            Assert.AreEqual(0, result);
        }

        /// <summary>
        /// Tests that ConvertBack returns 0 for extreme floating point values including NaN and infinities.
        /// This ensures the method handles special floating point values correctly.
        /// </summary>
        /// <param name="value">The floating point value to test.</param>
        [TestMethod]
        [DataRow(double.NaN)]
        [DataRow(double.PositiveInfinity)]
        [DataRow(double.NegativeInfinity)]
        [DataRow(double.MinValue)]
        [DataRow(double.MaxValue)]
        [DataRow(0.0)]
        [DataRow(-0.0)]
        public void ConvertBack_FloatingPointValues_ReturnsZero(double value)
        {
            // Arrange
            var converter = new TabIndexToBgColorConverter();
            var targetType = typeof(double);
            var culture = CultureInfo.InvariantCulture;

            // Act
            var result = converter.ConvertBack(value, targetType, null, culture);

            // Assert
            Assert.IsNotNull(result);
            Assert.AreEqual(0, result);
        }

        /// <summary>
        /// Tests that ConvertBack returns 0 for various parameter values.
        /// This ensures the converter parameter doesn't affect the return value.
        /// </summary>
        /// <param name="parameter">The parameter value to test.</param>
        [TestMethod]
        [DataRow(null)]
        [DataRow("")]
        [DataRow("0")]
        [DataRow("1")]
        [DataRow("test")]
        [DataRow(123)]
        [DataRow(true)]
        [DataRow(false)]
        public void ConvertBack_VariousParameterValues_ReturnsZero(object? parameter)
        {
            // Arrange
            var converter = new TabIndexToBgColorConverter();
            var value = 5;
            var targetType = typeof(int);
            var culture = CultureInfo.InvariantCulture;

            // Act
            var result = converter.ConvertBack(value, targetType, parameter, culture);

            // Assert
            Assert.IsNotNull(result);
            Assert.AreEqual(0, result);
        }

        /// <summary>
        /// Tests that ConvertBack returns 0 for different culture settings.
        /// This validates that culture information doesn't affect the return value.
        /// </summary>
        /// <param name="cultureName">The culture name to test.</param>
        [TestMethod]
        [DataRow("en-US")]
        [DataRow("fr-FR")]
        [DataRow("de-DE")]
        [DataRow("ja-JP")]
        [DataRow("ar-SA")]
        [DataRow("")]
        public void ConvertBack_VariousCultures_ReturnsZero(string cultureName)
        {
            // Arrange
            var converter = new TabIndexToBgColorConverter();
            var value = 5;
            var targetType = typeof(int);
            var culture = string.IsNullOrEmpty(cultureName) ? CultureInfo.InvariantCulture : new CultureInfo(cultureName);

            // Act
            var result = converter.ConvertBack(value, targetType, null, culture);

            // Assert
            Assert.IsNotNull(result);
            Assert.AreEqual(0, result);
        }

        /// <summary>
        /// Tests that ConvertBack doesn't throw exceptions for any combination of inputs.
        /// This is a safety test to ensure the method is robust against unexpected inputs.
        /// </summary>
        [TestMethod]
        public void ConvertBack_UnexpectedInputCombinations_DoesNotThrowException()
        {
            // Arrange
            var converter = new TabIndexToBgColorConverter();
            var testCases = new[]
            {
                (value: (object?)new object(), targetType: typeof(int), parameter: (object?)new object(), culture: CultureInfo.InvariantCulture),
                (value: (object?)DBNull.Value, targetType: typeof(string), parameter: (object?)DBNull.Value, culture: CultureInfo.CurrentCulture),
                (value: (object?)DateTime.Now, targetType: typeof(DateTime), parameter: (object?)"param", culture: CultureInfo.InvariantCulture),
            };

            foreach (var (value, targetType, parameter, culture) in testCases)
            {
                // Act
                var result = converter.ConvertBack(value, targetType, parameter, culture);

                // Assert
                Assert.IsNotNull(result);
                Assert.AreEqual(0, result);
            }
        }

        /// <summary>
        /// Tests that Convert returns Transparent when value is null.
        /// </summary>
        [TestMethod]
        public void Convert_ValueIsNull_ReturnsTransparent()
        {
            // Arrange
            var converter = new TabIndexToBgColorConverter();
            object? value = null;
            object? parameter = "0";
            Type targetType = typeof(Color);
            CultureInfo culture = CultureInfo.InvariantCulture;

            // Act
            var result = converter.Convert(value, targetType, parameter, culture);

            // Assert
            Assert.AreEqual(Colors.Transparent, result);
        }

        /// <summary>
        /// Tests that Convert returns Transparent when value is not an integer.
        /// </summary>
        /// <param name="value">The non-integer value to test.</param>
        [TestMethod]
        [DataRow("5")]
        [DataRow(5.5)]
        [DataRow(true)]
        public void Convert_ValueIsNotInt_ReturnsTransparent(object value)
        {
            // Arrange
            var converter = new TabIndexToBgColorConverter();
            object? parameter = "0";
            Type targetType = typeof(Color);
            CultureInfo culture = CultureInfo.InvariantCulture;

            // Act
            var result = converter.Convert(value, targetType, parameter, culture);

            // Assert
            Assert.AreEqual(Colors.Transparent, result);
        }

        /// <summary>
        /// Tests that Convert returns Transparent when parameter is null.
        /// </summary>
        [TestMethod]
        public void Convert_ParameterIsNull_ReturnsTransparent()
        {
            // Arrange
            var converter = new TabIndexToBgColorConverter();
            object? value = 0;
            object? parameter = null;
            Type targetType = typeof(Color);
            CultureInfo culture = CultureInfo.InvariantCulture;

            // Act
            var result = converter.Convert(value, targetType, parameter, culture);

            // Assert
            Assert.AreEqual(Colors.Transparent, result);
        }

        /// <summary>
        /// Tests that Convert returns Transparent when parameter is not a string.
        /// </summary>
        /// <param name="parameter">The non-string parameter to test.</param>
        [TestMethod]
        [DataRow(5)]
        [DataRow(5.5)]
        [DataRow(true)]
        public void Convert_ParameterIsNotString_ReturnsTransparent(object parameter)
        {
            // Arrange
            var converter = new TabIndexToBgColorConverter();
            object? value = 0;
            Type targetType = typeof(Color);
            CultureInfo culture = CultureInfo.InvariantCulture;

            // Act
            var result = converter.Convert(value, targetType, parameter, culture);

            // Assert
            Assert.AreEqual(Colors.Transparent, result);
        }

        /// <summary>
        /// Tests that Convert returns Transparent when parameter string cannot be parsed to int.
        /// </summary>
        /// <param name="parameterStr">The unparsable string parameter.</param>
        [TestMethod]
        [DataRow("")]
        [DataRow("   ")]
        [DataRow("abc")]
        [DataRow("12.5")]
        [DataRow("12a")]
        [DataRow("a12")]
        [DataRow("!@#")]
        [DataRow("2147483648")]
        [DataRow("-2147483649")]
        public void Convert_ParameterStringNotParsable_ReturnsTransparent(string parameterStr)
        {
            // Arrange
            var converter = new TabIndexToBgColorConverter();
            object? value = 0;
            object? parameter = parameterStr;
            Type targetType = typeof(Color);
            CultureInfo culture = CultureInfo.InvariantCulture;

            // Act
            var result = converter.Convert(value, targetType, parameter, culture);

            // Assert
            Assert.AreEqual(Colors.Transparent, result);
        }

        /// <summary>
        /// Tests that Convert returns purple color when value and parameter match.
        /// </summary>
        /// <param name="tabIndex">The matching tab index value.</param>
        [TestMethod]
        [DataRow(0)]
        [DataRow(1)]
        [DataRow(5)]
        [DataRow(-1)]
        [DataRow(-10)]
        [DataRow(2147483647)]
        [DataRow(-2147483648)]
        public void Convert_ValueMatchesParameter_ReturnsPurpleColor(int tabIndex)
        {
            // Arrange
            var converter = new TabIndexToBgColorConverter();
            object? value = tabIndex;
            object? parameter = tabIndex.ToString();
            Type targetType = typeof(Color);
            CultureInfo culture = CultureInfo.InvariantCulture;
            var expectedColor = Color.FromArgb("#7C3AED");

            // Act
            var result = converter.Convert(value, targetType, parameter, culture);

            // Assert
            Assert.IsNotNull(result);
            var resultColor = result as Color;
            Assert.IsNotNull(resultColor);
            Assert.AreEqual(expectedColor.ToArgbHex(), resultColor.ToArgbHex());
        }

        /// <summary>
        /// Tests that Convert returns Transparent when value does not match parameter.
        /// </summary>
        /// <param name="currentIndex">The current index value.</param>
        /// <param name="parameterStr">The parameter string representing target index.</param>
        [TestMethod]
        [DataRow(0, "1")]
        [DataRow(1, "0")]
        [DataRow(5, "6")]
        [DataRow(-1, "1")]
        [DataRow(10, "-10")]
        [DataRow(2147483647, "0")]
        [DataRow(-2147483648, "0")]
        public void Convert_ValueDoesNotMatchParameter_ReturnsTransparent(int currentIndex, string parameterStr)
        {
            // Arrange
            var converter = new TabIndexToBgColorConverter();
            object? value = currentIndex;
            object? parameter = parameterStr;
            Type targetType = typeof(Color);
            CultureInfo culture = CultureInfo.InvariantCulture;

            // Act
            var result = converter.Convert(value, targetType, parameter, culture);

            // Assert
            Assert.AreEqual(Colors.Transparent, result);
        }

        /// <summary>
        /// Tests that Convert handles both value and parameter being null gracefully.
        /// </summary>
        [TestMethod]
        public void Convert_BothValueAndParameterNull_ReturnsTransparent()
        {
            // Arrange
            var converter = new TabIndexToBgColorConverter();
            object? value = null;
            object? parameter = null;
            Type targetType = typeof(Color);
            CultureInfo culture = CultureInfo.InvariantCulture;

            // Act
            var result = converter.Convert(value, targetType, parameter, culture);

            // Assert
            Assert.AreEqual(Colors.Transparent, result);
        }

        /// <summary>
        /// Tests that Convert works correctly with different CultureInfo values.
        /// The implementation doesn't use culture, but this verifies it doesn't cause issues.
        /// </summary>
        [TestMethod]
        public void Convert_WithDifferentCultures_WorksCorrectly()
        {
            // Arrange
            var converter = new TabIndexToBgColorConverter();
            object? value = 5;
            object? parameter = "5";
            Type targetType = typeof(Color);
            var expectedColor = Color.FromArgb("#7C3AED");

            // Act
            var resultInvariant = converter.Convert(value, targetType, parameter, CultureInfo.InvariantCulture);
            var resultEnUs = converter.Convert(value, targetType, parameter, new CultureInfo("en-US"));
            var resultFrFr = converter.Convert(value, targetType, parameter, new CultureInfo("fr-FR"));

            // Assert
            Assert.IsNotNull(resultInvariant);
            var colorInvariant = resultInvariant as Color;
            Assert.IsNotNull(colorInvariant);
            Assert.AreEqual(expectedColor.ToArgbHex(), colorInvariant.ToArgbHex());

            Assert.IsNotNull(resultEnUs);
            var colorEnUs = resultEnUs as Color;
            Assert.IsNotNull(colorEnUs);
            Assert.AreEqual(expectedColor.ToArgbHex(), colorEnUs.ToArgbHex());

            Assert.IsNotNull(resultFrFr);
            var colorFrFr = resultFrFr as Color;
            Assert.IsNotNull(colorFrFr);
            Assert.AreEqual(expectedColor.ToArgbHex(), colorFrFr.ToArgbHex());
        }
    }

    /// <summary>
    /// Unit tests for the <see cref="TabIndexToVisibilityConverter"/> class.
    /// </summary>
    [TestClass]
    public class TabIndexToVisibilityConverterTests
    {
        /// <summary>
        /// Tests that Convert returns false when value parameter is null.
        /// </summary>
        [TestMethod]
        public void Convert_ValueIsNull_ReturnsFalse()
        {
            // Arrange
            var converter = new TabIndexToVisibilityConverter();
            object? value = null;
            object? parameter = "0";

            // Act
            var result = converter.Convert(value, typeof(bool), parameter, CultureInfo.InvariantCulture);

            // Assert
            Assert.AreEqual(false, result);
        }

        /// <summary>
        /// Tests that Convert returns false when parameter is null.
        /// </summary>
        [TestMethod]
        public void Convert_ParameterIsNull_ReturnsFalse()
        {
            // Arrange
            var converter = new TabIndexToVisibilityConverter();
            object? value = 0;
            object? parameter = null;

            // Act
            var result = converter.Convert(value, typeof(bool), parameter, CultureInfo.InvariantCulture);

            // Assert
            Assert.AreEqual(false, result);
        }

        /// <summary>
        /// Tests that Convert returns false when both value and parameter are null.
        /// </summary>
        [TestMethod]
        public void Convert_BothValueAndParameterAreNull_ReturnsFalse()
        {
            // Arrange
            var converter = new TabIndexToVisibilityConverter();
            object? value = null;
            object? parameter = null;

            // Act
            var result = converter.Convert(value, typeof(bool), parameter, CultureInfo.InvariantCulture);

            // Assert
            Assert.AreEqual(false, result);
        }

        /// <summary>
        /// Tests that Convert returns false when value is not an integer type.
        /// </summary>
        /// <param name="value">The non-integer value to test.</param>
        [TestMethod]
        [DataRow("5")]
        [DataRow(5.5)]
        [DataRow(true)]
        public void Convert_ValueIsNotInt_ReturnsFalse(object value)
        {
            // Arrange
            var converter = new TabIndexToVisibilityConverter();
            object? parameter = "0";

            // Act
            var result = converter.Convert(value, typeof(bool), parameter, CultureInfo.InvariantCulture);

            // Assert
            Assert.AreEqual(false, result);
        }

        /// <summary>
        /// Tests that Convert returns false when parameter is not a string type.
        /// </summary>
        /// <param name="parameter">The non-string parameter to test.</param>
        [TestMethod]
        [DataRow(5)]
        [DataRow(5.5)]
        [DataRow(true)]
        public void Convert_ParameterIsNotString_ReturnsFalse(object parameter)
        {
            // Arrange
            var converter = new TabIndexToVisibilityConverter();
            object? value = 0;

            // Act
            var result = converter.Convert(value, typeof(bool), parameter, CultureInfo.InvariantCulture);

            // Assert
            Assert.AreEqual(false, result);
        }

        /// <summary>
        /// Tests that Convert returns false when parameter string cannot be parsed as an integer.
        /// </summary>
        /// <param name="parameterString">The unparseable parameter string.</param>
        [TestMethod]
        [DataRow("")]
        [DataRow(" ")]
        [DataRow("abc")]
        [DataRow("12.5")]
        [DataRow("12a")]
        [DataRow("a12")]
        [DataRow("!@#")]
        [DataRow("2147483648")]
        [DataRow("-2147483649")]
        [DataRow("999999999999999999999")]
        public void Convert_ParameterStringIsNotValidInt_ReturnsFalse(string parameterString)
        {
            // Arrange
            var converter = new TabIndexToVisibilityConverter();
            object? value = 0;

            // Act
            var result = converter.Convert(value, typeof(bool), parameterString, CultureInfo.InvariantCulture);

            // Assert
            Assert.AreEqual(false, result);
        }

        /// <summary>
        /// Tests that Convert returns true when value matches the parsed parameter.
        /// </summary>
        /// <param name="intValue">The integer value to test.</param>
        /// <param name="parameterString">The parameter string that should match.</param>
        [TestMethod]
        [DataRow(0, "0")]
        [DataRow(1, "1")]
        [DataRow(-1, "-1")]
        [DataRow(42, "42")]
        [DataRow(100, "100")]
        [DataRow(2147483647, "2147483647")]
        [DataRow(-2147483648, "-2147483648")]
        public void Convert_ValueMatchesParameter_ReturnsTrue(int intValue, string parameterString)
        {
            // Arrange
            var converter = new TabIndexToVisibilityConverter();

            // Act
            var result = converter.Convert(intValue, typeof(bool), parameterString, CultureInfo.InvariantCulture);

            // Assert
            Assert.AreEqual(true, result);
        }

        /// <summary>
        /// Tests that Convert returns false when value does not match the parsed parameter.
        /// </summary>
        /// <param name="intValue">The integer value to test.</param>
        /// <param name="parameterString">The parameter string that should not match.</param>
        [TestMethod]
        [DataRow(0, "1")]
        [DataRow(1, "0")]
        [DataRow(5, "10")]
        [DataRow(-1, "1")]
        [DataRow(2147483647, "2147483646")]
        [DataRow(-2147483648, "-2147483647")]
        public void Convert_ValueDoesNotMatchParameter_ReturnsFalse(int intValue, string parameterString)
        {
            // Arrange
            var converter = new TabIndexToVisibilityConverter();

            // Act
            var result = converter.Convert(intValue, typeof(bool), parameterString, CultureInfo.InvariantCulture);

            // Assert
            Assert.AreEqual(false, result);
        }

        /// <summary>
        /// Tests that Convert handles boundary integer values correctly when they match.
        /// </summary>
        [TestMethod]
        public void Convert_BoundaryValues_ReturnsExpectedResults()
        {
            // Arrange
            var converter = new TabIndexToVisibilityConverter();

            // Act & Assert - int.MaxValue matches
            var resultMaxMatch = converter.Convert(int.MaxValue, typeof(bool), int.MaxValue.ToString(), CultureInfo.InvariantCulture);
            Assert.AreEqual(true, resultMaxMatch);

            // Act & Assert - int.MinValue matches
            var resultMinMatch = converter.Convert(int.MinValue, typeof(bool), int.MinValue.ToString(), CultureInfo.InvariantCulture);
            Assert.AreEqual(true, resultMinMatch);

            // Act & Assert - Zero matches
            var resultZeroMatch = converter.Convert(0, typeof(bool), "0", CultureInfo.InvariantCulture);
            Assert.AreEqual(true, resultZeroMatch);
        }

        /// <summary>
        /// Tests that Convert works correctly regardless of targetType parameter value.
        /// </summary>
        [TestMethod]
        public void Convert_DifferentTargetTypes_DoesNotAffectResult()
        {
            // Arrange
            var converter = new TabIndexToVisibilityConverter();

            // Act
            var resultWithBoolType = converter.Convert(5, typeof(bool), "5", CultureInfo.InvariantCulture);
            var resultWithStringType = converter.Convert(5, typeof(string), "5", CultureInfo.InvariantCulture);
            var resultWithObjectType = converter.Convert(5, typeof(object), "5", CultureInfo.InvariantCulture);

            // Assert
            Assert.AreEqual(true, resultWithBoolType);
            Assert.AreEqual(true, resultWithStringType);
            Assert.AreEqual(true, resultWithObjectType);
        }

        /// <summary>
        /// Tests that Convert works correctly regardless of culture parameter value.
        /// </summary>
        [TestMethod]
        public void Convert_DifferentCultures_DoesNotAffectResult()
        {
            // Arrange
            var converter = new TabIndexToVisibilityConverter();

            // Act
            var resultInvariant = converter.Convert(5, typeof(bool), "5", CultureInfo.InvariantCulture);
            var resultEnUs = converter.Convert(5, typeof(bool), "5", new CultureInfo("en-US"));
            var resultDeDe = converter.Convert(5, typeof(bool), "5", new CultureInfo("de-DE"));

            // Assert
            Assert.AreEqual(true, resultInvariant);
            Assert.AreEqual(true, resultEnUs);
            Assert.AreEqual(true, resultDeDe);
        }
    }

    /// <summary>
    /// Tests for the StringToFirstCharConverter class.
    /// </summary>
    [TestClass]
    public class StringToFirstCharConverterTests
    {
        /// <summary>
        /// Tests that Convert returns "?" when value is null.
        /// </summary>
        [TestMethod]
        public void Convert_NullValue_ReturnsQuestionMark()
        {
            // Arrange
            var converter = new StringToFirstCharConverter();
            object? value = null;
            Type targetType = typeof(string);
            object? parameter = null;
            CultureInfo culture = CultureInfo.InvariantCulture;

            // Act
            var result = converter.Convert(value, targetType, parameter, culture);

            // Assert
            Assert.AreEqual("?", result);
        }

        /// <summary>
        /// Tests that Convert returns "?" when value is an empty string.
        /// </summary>
        [TestMethod]
        public void Convert_EmptyString_ReturnsQuestionMark()
        {
            // Arrange
            var converter = new StringToFirstCharConverter();
            object? value = string.Empty;
            Type targetType = typeof(string);
            object? parameter = null;
            CultureInfo culture = CultureInfo.InvariantCulture;

            // Act
            var result = converter.Convert(value, targetType, parameter, culture);

            // Assert
            Assert.AreEqual("?", result);
        }

        /// <summary>
        /// Tests that Convert returns "?" when value is not a string type.
        /// </summary>
        /// <param name="value">The non-string value to test.</param>
        /// <param name="description">Description of the test case.</param>
        [TestMethod]
        [DataRow(123, "integer value")]
        [DataRow(45.67, "double value")]
        [DataRow(true, "boolean value")]
        public void Convert_NonStringValue_ReturnsQuestionMark(object value, string description)
        {
            // Arrange
            var converter = new StringToFirstCharConverter();
            Type targetType = typeof(string);
            object? parameter = null;
            CultureInfo culture = CultureInfo.InvariantCulture;

            // Act
            var result = converter.Convert(value, targetType, parameter, culture);

            // Assert
            Assert.AreEqual("?", result, $"Failed for {description}");
        }

        /// <summary>
        /// Tests that Convert returns the first character uppercase for various valid string inputs.
        /// </summary>
        /// <param name="input">The input string.</param>
        /// <param name="expected">The expected output.</param>
        [TestMethod]
        [DataRow("a", "A")]
        [DataRow("A", "A")]
        [DataRow("hello", "H")]
        [DataRow("HELLO", "H")]
        [DataRow("Hello", "H")]
        [DataRow("world123", "W")]
        [DataRow("1abc", "1")]
        [DataRow("!test", "!")]
        [DataRow("@special", "@")]
        [DataRow("#hashtag", "#")]
        [DataRow(" space", " ")]
        [DataRow("\ttab", "\t")]
        public void Convert_ValidStringInput_ReturnsFirstCharacterUppercase(string input, string expected)
        {
            // Arrange
            var converter = new StringToFirstCharConverter();
            object? value = input;
            Type targetType = typeof(string);
            object? parameter = null;
            CultureInfo culture = CultureInfo.InvariantCulture;

            // Act
            var result = converter.Convert(value, targetType, parameter, culture);

            // Assert
            Assert.AreEqual(expected, result);
        }

        /// <summary>
        /// Tests that Convert handles single character strings correctly.
        /// </summary>
        /// <param name="input">The single character input.</param>
        /// <param name="expected">The expected uppercase character.</param>
        [TestMethod]
        [DataRow("x", "X")]
        [DataRow("Z", "Z")]
        [DataRow("9", "9")]
        [DataRow("$", "$")]
        public void Convert_SingleCharacterString_ReturnsUppercaseCharacter(string input, string expected)
        {
            // Arrange
            var converter = new StringToFirstCharConverter();
            object? value = input;
            Type targetType = typeof(string);
            object? parameter = null;
            CultureInfo culture = CultureInfo.InvariantCulture;

            // Act
            var result = converter.Convert(value, targetType, parameter, culture);

            // Assert
            Assert.AreEqual(expected, result);
        }

        /// <summary>
        /// Tests that Convert handles Unicode and special culture-specific characters.
        /// </summary>
        [TestMethod]
        public void Convert_UnicodeCharacters_ReturnsUppercaseFirstCharacter()
        {
            // Arrange
            var converter = new StringToFirstCharConverter();
            object? value = "äpfel";
            Type targetType = typeof(string);
            object? parameter = null;
            CultureInfo culture = CultureInfo.InvariantCulture;

            // Act
            var result = converter.Convert(value, targetType, parameter, culture);

            // Assert
            Assert.AreEqual("Ä", result);
        }

        /// <summary>
        /// Tests that Convert handles very long strings by returning only the first character uppercase.
        /// </summary>
        [TestMethod]
        public void Convert_VeryLongString_ReturnsFirstCharacterUppercase()
        {
            // Arrange
            var converter = new StringToFirstCharConverter();
            object? value = new string('a', 10000) + "bcdefghijklmnopqrstuvwxyz";
            Type targetType = typeof(string);
            object? parameter = null;
            CultureInfo culture = CultureInfo.InvariantCulture;

            // Act
            var result = converter.Convert(value, targetType, parameter, culture);

            // Assert
            Assert.AreEqual("A", result);
        }

        /// <summary>
        /// Tests that Convert handles string with whitespace only (not empty but has content).
        /// </summary>
        [TestMethod]
        public void Convert_WhitespaceString_ReturnsFirstCharacterAsIs()
        {
            // Arrange
            var converter = new StringToFirstCharConverter();
            object? value = "   ";
            Type targetType = typeof(string);
            object? parameter = null;
            CultureInfo culture = CultureInfo.InvariantCulture;

            // Act
            var result = converter.Convert(value, targetType, parameter, culture);

            // Assert
            Assert.AreEqual(" ", result);
        }

        /// <summary>
        /// Tests that Convert returns "?" when value is an empty object (not string).
        /// </summary>
        [TestMethod]
        public void Convert_EmptyObject_ReturnsQuestionMark()
        {
            // Arrange
            var converter = new StringToFirstCharConverter();
            object? value = new object();
            Type targetType = typeof(string);
            object? parameter = null;
            CultureInfo culture = CultureInfo.InvariantCulture;

            // Act
            var result = converter.Convert(value, targetType, parameter, culture);

            // Assert
            Assert.AreEqual("?", result);
        }

        /// <summary>
        /// Tests that Convert handles strings with newline characters.
        /// </summary>
        [TestMethod]
        public void Convert_StringWithNewline_ReturnsFirstCharacterUppercase()
        {
            // Arrange
            var converter = new StringToFirstCharConverter();
            object? value = "\ntest";
            Type targetType = typeof(string);
            object? parameter = null;
            CultureInfo culture = CultureInfo.InvariantCulture;

            // Act
            var result = converter.Convert(value, targetType, parameter, culture);

            // Assert
            Assert.AreEqual("\n", result);
        }

        /// <summary>
        /// Tests that Convert handles strings starting with lowercase accented characters.
        /// </summary>
        [TestMethod]
        public void Convert_AccentedLowercaseCharacter_ReturnsUppercaseAccentedCharacter()
        {
            // Arrange
            var converter = new StringToFirstCharConverter();
            object? value = "étoile";
            Type targetType = typeof(string);
            object? parameter = null;
            CultureInfo culture = CultureInfo.InvariantCulture;

            // Act
            var result = converter.Convert(value, targetType, parameter, culture);

            // Assert
            Assert.AreEqual("É", result);
        }

        /// <summary>
        /// Tests that ConvertBack always returns an empty string regardless of input parameters.
        /// </summary>
        /// <param name="value">The value parameter to test.</param>
        /// <param name="parameter">The parameter to test.</param>
        /// <param name="cultureName">The culture name to test (null for null culture).</param>
        [TestMethod]
        [DataRow(null, null, null, DisplayName = "All nullable parameters are null")]
        [DataRow("test", null, null, DisplayName = "Value is string, others null")]
        [DataRow(123, null, null, DisplayName = "Value is int, others null")]
        [DataRow(null, "param", null, DisplayName = "Parameter has value, others null")]
        [DataRow(null, null, "en-US", DisplayName = "Culture is en-US, others null")]
        [DataRow("test", "param", "en-US", DisplayName = "All parameters have values")]
        [DataRow("", "", "fr-FR", DisplayName = "Empty strings with fr-FR culture")]
        [DataRow("   ", "param", "de-DE", DisplayName = "Whitespace value with de-DE culture")]
        [DataRow(int.MaxValue, int.MinValue, "ja-JP", DisplayName = "Numeric boundary values")]
        [DataRow(true, false, "ar-SA", DisplayName = "Boolean values")]
        public void ConvertBack_WithVariousInputs_ReturnsEmptyString(object? value, object? parameter, string? cultureName)
        {
            // Arrange
            var converter = new StringToFirstCharConverter();
            var targetType = typeof(string);
            var culture = cultureName != null ? new CultureInfo(cultureName) : null;

            // Act
            var result = converter.ConvertBack(value, targetType, parameter, culture);

            // Assert
            Assert.IsNotNull(result);
            Assert.AreEqual(string.Empty, result);
            Assert.IsInstanceOfType(result, typeof(string));
        }

        /// <summary>
        /// Tests that ConvertBack returns empty string when targetType is different from string type.
        /// </summary>
        [TestMethod]
        public void ConvertBack_WithNonStringTargetType_ReturnsEmptyString()
        {
            // Arrange
            var converter = new StringToFirstCharConverter();
            var value = "test";
            var targetType = typeof(int);
            var parameter = "param";
            var culture = CultureInfo.InvariantCulture;

            // Act
            var result = converter.ConvertBack(value, targetType, parameter, culture);

            // Assert
            Assert.IsNotNull(result);
            Assert.AreEqual(string.Empty, result);
        }

        /// <summary>
        /// Tests that ConvertBack returns empty string when value is a complex object.
        /// </summary>
        [TestMethod]
        public void ConvertBack_WithComplexObjectValue_ReturnsEmptyString()
        {
            // Arrange
            var converter = new StringToFirstCharConverter();
            var value = new { Name = "Test", Value = 42 };
            var targetType = typeof(object);
            var parameter = new object();
            var culture = new CultureInfo("en-US");

            // Act
            var result = converter.ConvertBack(value, targetType, parameter, culture);

            // Assert
            Assert.IsNotNull(result);
            Assert.AreEqual(string.Empty, result);
        }

        /// <summary>
        /// Tests that ConvertBack returns the same empty string instance consistently.
        /// </summary>
        [TestMethod]
        public void ConvertBack_CalledMultipleTimes_ReturnsSameEmptyStringValue()
        {
            // Arrange
            var converter = new StringToFirstCharConverter();
            var targetType = typeof(string);
            var culture = CultureInfo.CurrentCulture;

            // Act
            var result1 = converter.ConvertBack("value1", targetType, null, culture);
            var result2 = converter.ConvertBack("value2", targetType, "param", culture);
            var result3 = converter.ConvertBack(null, targetType, null, null);

            // Assert
            Assert.AreEqual(string.Empty, result1);
            Assert.AreEqual(string.Empty, result2);
            Assert.AreEqual(string.Empty, result3);
            Assert.AreEqual(result1, result2);
            Assert.AreEqual(result2, result3);
        }
    }
}

namespace SignLanguageApp.Helpers.UnitTests
{
    /// <summary>
    /// Unit tests for the <see cref="IntToNotIntConverter"/> class.
    /// </summary>
    [TestClass]
    public class IntToNotIntConverterTests
    {
        /// <summary>
        /// Tests that Convert returns true when value is a non-zero integer.
        /// </summary>
        /// <param name="value">The integer value to test.</param>
        [TestMethod]
        [DataRow(1)]
        [DataRow(-1)]
        [DataRow(5)]
        [DataRow(-5)]
        [DataRow(100)]
        [DataRow(-100)]
        [DataRow(int.MaxValue)]
        [DataRow(int.MinValue)]
        public void Convert_NonZeroIntValue_ReturnsTrue(int value)
        {
            // Arrange
            var converter = new IntToNotIntConverter();

            // Act
            var result = converter.Convert(value, typeof(bool), null, CultureInfo.InvariantCulture);

            // Assert
            Assert.IsNotNull(result);
            Assert.IsInstanceOfType(result, typeof(bool));
            Assert.AreEqual(true, result);
        }

        /// <summary>
        /// Tests that Convert returns false when value is zero.
        /// </summary>
        [TestMethod]
        public void Convert_ZeroIntValue_ReturnsFalse()
        {
            // Arrange
            var converter = new IntToNotIntConverter();

            // Act
            var result = converter.Convert(0, typeof(bool), null, CultureInfo.InvariantCulture);

            // Assert
            Assert.IsNotNull(result);
            Assert.IsInstanceOfType(result, typeof(bool));
            Assert.AreEqual(false, result);
        }

        /// <summary>
        /// Tests that Convert returns false when value is null.
        /// </summary>
        [TestMethod]
        public void Convert_NullValue_ReturnsFalse()
        {
            // Arrange
            var converter = new IntToNotIntConverter();

            // Act
            var result = converter.Convert(null, typeof(bool), null, CultureInfo.InvariantCulture);

            // Assert
            Assert.IsNotNull(result);
            Assert.IsInstanceOfType(result, typeof(bool));
            Assert.AreEqual(false, result);
        }

        /// <summary>
        /// Tests that Convert returns false when value is not an integer type.
        /// </summary>
        /// <param name="value">The non-integer value to test.</param>
        [TestMethod]
        [DataRow("string")]
        [DataRow("0")]
        [DataRow("")]
        [DataRow(5.5)]
        [DataRow(0.0)]
        [DataRow(true)]
        [DataRow(false)]
        public void Convert_NonIntValue_ReturnsFalse(object value)
        {
            // Arrange
            var converter = new IntToNotIntConverter();

            // Act
            var result = converter.Convert(value, typeof(bool), null, CultureInfo.InvariantCulture);

            // Assert
            Assert.IsNotNull(result);
            Assert.IsInstanceOfType(result, typeof(bool));
            Assert.AreEqual(false, result);
        }

        /// <summary>
        /// Tests that Convert returns false when value is a long type (not an int).
        /// </summary>
        [TestMethod]
        public void Convert_LongValue_ReturnsFalse()
        {
            // Arrange
            var converter = new IntToNotIntConverter();
            long longValue = 5L;

            // Act
            var result = converter.Convert(longValue, typeof(bool), null, CultureInfo.InvariantCulture);

            // Assert
            Assert.IsNotNull(result);
            Assert.IsInstanceOfType(result, typeof(bool));
            Assert.AreEqual(false, result);
        }

        /// <summary>
        /// Tests that Convert returns false when value is a short type (not an int).
        /// </summary>
        [TestMethod]
        public void Convert_ShortValue_ReturnsFalse()
        {
            // Arrange
            var converter = new IntToNotIntConverter();
            short shortValue = 5;

            // Act
            var result = converter.Convert(shortValue, typeof(bool), null, CultureInfo.InvariantCulture);

            // Assert
            Assert.IsNotNull(result);
            Assert.IsInstanceOfType(result, typeof(bool));
            Assert.AreEqual(false, result);
        }

        /// <summary>
        /// Tests that Convert works correctly when targetType parameter varies.
        /// Verifies that the targetType parameter does not affect the result.
        /// </summary>
        [TestMethod]
        public void Convert_DifferentTargetType_ReturnsExpectedResult()
        {
            // Arrange
            var converter = new IntToNotIntConverter();

            // Act
            var result = converter.Convert(5, typeof(string), null, CultureInfo.InvariantCulture);

            // Assert
            Assert.IsNotNull(result);
            Assert.IsInstanceOfType(result, typeof(bool));
            Assert.AreEqual(true, result);
        }

        /// <summary>
        /// Tests that Convert works correctly when parameter argument varies.
        /// Verifies that the parameter argument does not affect the result.
        /// </summary>
        [TestMethod]
        public void Convert_DifferentParameter_ReturnsExpectedResult()
        {
            // Arrange
            var converter = new IntToNotIntConverter();
            var parameterValue = new object();

            // Act
            var result = converter.Convert(5, typeof(bool), parameterValue, CultureInfo.InvariantCulture);

            // Assert
            Assert.IsNotNull(result);
            Assert.IsInstanceOfType(result, typeof(bool));
            Assert.AreEqual(true, result);
        }

        /// <summary>
        /// Tests that Convert works correctly when culture parameter varies.
        /// Verifies that the culture parameter does not affect the result.
        /// </summary>
        [TestMethod]
        public void Convert_DifferentCulture_ReturnsExpectedResult()
        {
            // Arrange
            var converter = new IntToNotIntConverter();

            // Act
            var result = converter.Convert(5, typeof(bool), null, new CultureInfo("fr-FR"));

            // Assert
            Assert.IsNotNull(result);
            Assert.IsInstanceOfType(result, typeof(bool));
            Assert.AreEqual(true, result);
        }
    }
}