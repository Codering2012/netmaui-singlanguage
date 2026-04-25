using System;
using System.Globalization;

using Microsoft.Maui.Controls;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using SignLanguageApp.Converters;

namespace SignLanguageApp.Converters.UnitTests;


/// <summary>
/// Unit tests for the CategoryToEmojiConverter class.
/// </summary>
[TestClass]
public class CategoryToEmojiConverterTests
{
    /// <summary>
    /// Tests that Convert method returns the correct emoji for all known category strings in lowercase.
    /// </summary>
    /// <param name="category">The category name to convert.</param>
    /// <param name="expectedEmoji">The expected emoji result.</param>
    [TestMethod]
    [DataRow("basics", "🤟")]
    [DataRow("numbers", "🔢")]
    [DataRow("phrases", "💬")]
    [DataRow("advanced", "🎯")]
    [DataRow("alphabet", "🔤")]
    [DataRow("emotions", "😊")]
    [DataRow("daily", "📅")]
    [DataRow("greetings", "👋")]
    [DataRow("colors", "🎨")]
    [DataRow("animals", "🦁")]
    [DataRow("family", "👨‍👩‍👧")]
    [DataRow("food", "🍔")]
    public void Convert_KnownCategoryLowercase_ReturnsCorrectEmoji(string category, string expectedEmoji)
    {
        // Arrange
        var converter = new CategoryToEmojiConverter();

        // Act
        var result = converter.Convert(category, typeof(string), null, null);

        // Assert
        Assert.AreEqual(expectedEmoji, result);
    }

    /// <summary>
    /// Tests that Convert method is case-insensitive and returns correct emoji for uppercase and mixed case category strings.
    /// </summary>
    /// <param name="category">The category name in various cases.</param>
    /// <param name="expectedEmoji">The expected emoji result.</param>
    [TestMethod]
    [DataRow("BASICS", "🤟")]
    [DataRow("Basics", "🤟")]
    [DataRow("BaSiCs", "🤟")]
    [DataRow("NUMBERS", "🔢")]
    [DataRow("Numbers", "🔢")]
    [DataRow("GREETINGS", "👋")]
    [DataRow("Greetings", "👋")]
    [DataRow("FOOD", "🍔")]
    [DataRow("Food", "🍔")]
    public void Convert_KnownCategoryVariousCases_ReturnsCorrectEmoji(string category, string expectedEmoji)
    {
        // Arrange
        var converter = new CategoryToEmojiConverter();

        // Act
        var result = converter.Convert(category, typeof(string), null, null);

        // Assert
        Assert.AreEqual(expectedEmoji, result);
    }

    /// <summary>
    /// Tests that Convert method returns default emoji for unknown category strings.
    /// </summary>
    /// <param name="category">The unknown category name.</param>
    [TestMethod]
    [DataRow("unknown")]
    [DataRow("invalid")]
    [DataRow("test")]
    [DataRow("random")]
    [DataRow("xyz")]
    [DataRow("category")]
    public void Convert_UnknownCategoryString_ReturnsDefaultEmoji(string category)
    {
        // Arrange
        var converter = new CategoryToEmojiConverter();
        var expectedEmoji = "📹";

        // Act
        var result = converter.Convert(category, typeof(string), null, null);

        // Assert
        Assert.AreEqual(expectedEmoji, result);
    }

    /// <summary>
    /// Tests that Convert method returns default emoji when value is null.
    /// </summary>
    [TestMethod]
    public void Convert_NullValue_ReturnsDefaultEmoji()
    {
        // Arrange
        var converter = new CategoryToEmojiConverter();
        var expectedEmoji = "📹";

        // Act
        var result = converter.Convert(null, typeof(string), null, null);

        // Assert
        Assert.AreEqual(expectedEmoji, result);
    }

    /// <summary>
    /// Tests that Convert method returns default emoji for empty string.
    /// </summary>
    [TestMethod]
    public void Convert_EmptyString_ReturnsDefaultEmoji()
    {
        // Arrange
        var converter = new CategoryToEmojiConverter();
        var expectedEmoji = "📹";

        // Act
        var result = converter.Convert(string.Empty, typeof(string), null, null);

        // Assert
        Assert.AreEqual(expectedEmoji, result);
    }

    /// <summary>
    /// Tests that Convert method returns default emoji for whitespace-only strings.
    /// </summary>
    /// <param name="value">The whitespace string to test.</param>
    [TestMethod]
    [DataRow("   ")]
    [DataRow("\t")]
    [DataRow("\n")]
    [DataRow("\r\n")]
    [DataRow(" \t \n ")]
    public void Convert_WhitespaceString_ReturnsDefaultEmoji(string value)
    {
        // Arrange
        var converter = new CategoryToEmojiConverter();
        var expectedEmoji = "📹";

        // Act
        var result = converter.Convert(value, typeof(string), null, null);

        // Assert
        Assert.AreEqual(expectedEmoji, result);
    }

    /// <summary>
    /// Tests that Convert method returns default emoji for non-string value types.
    /// </summary>
    /// <param name="value">The non-string value to test.</param>
    [TestMethod]
    [DataRow(123)]
    [DataRow(0)]
    [DataRow(-456)]
    [DataRow(int.MaxValue)]
    [DataRow(int.MinValue)]
    public void Convert_IntegerValue_ReturnsDefaultEmoji(int value)
    {
        // Arrange
        var converter = new CategoryToEmojiConverter();
        var expectedEmoji = "📹";

        // Act
        var result = converter.Convert(value, typeof(string), null, null);

        // Assert
        Assert.AreEqual(expectedEmoji, result);
    }

    /// <summary>
    /// Tests that Convert method returns default emoji for double values including special values.
    /// </summary>
    /// <param name="value">The double value to test.</param>
    [TestMethod]
    [DataRow(123.45)]
    [DataRow(0.0)]
    [DataRow(-456.78)]
    [DataRow(double.MaxValue)]
    [DataRow(double.MinValue)]
    [DataRow(double.NaN)]
    [DataRow(double.PositiveInfinity)]
    [DataRow(double.NegativeInfinity)]
    public void Convert_DoubleValue_ReturnsDefaultEmoji(double value)
    {
        // Arrange
        var converter = new CategoryToEmojiConverter();
        var expectedEmoji = "📹";

        // Act
        var result = converter.Convert(value, typeof(string), null, null);

        // Assert
        Assert.AreEqual(expectedEmoji, result);
    }

    /// <summary>
    /// Tests that Convert method returns default emoji for boolean values.
    /// </summary>
    /// <param name="value">The boolean value to test.</param>
    [TestMethod]
    [DataRow(true)]
    [DataRow(false)]
    public void Convert_BooleanValue_ReturnsDefaultEmoji(bool value)
    {
        // Arrange
        var converter = new CategoryToEmojiConverter();
        var expectedEmoji = "📹";

        // Act
        var result = converter.Convert(value, typeof(string), null, null);

        // Assert
        Assert.AreEqual(expectedEmoji, result);
    }

    /// <summary>
    /// Tests that Convert method returns default emoji for arbitrary object values.
    /// </summary>
    [TestMethod]
    public void Convert_ObjectValue_ReturnsDefaultEmoji()
    {
        // Arrange
        var converter = new CategoryToEmojiConverter();
        var expectedEmoji = "📹";
        var value = new object();

        // Act
        var result = converter.Convert(value, typeof(string), null, null);

        // Assert
        Assert.AreEqual(expectedEmoji, result);
    }

    /// <summary>
    /// Tests that Convert method returns default emoji for strings with special characters.
    /// </summary>
    /// <param name="value">The string with special characters to test.</param>
    [TestMethod]
    [DataRow("@#$%^&*()")]
    [DataRow("123numbers")]
    [DataRow("!basics!")]
    [DataRow("category\0with\0null")]
    [DataRow("very_long_string_that_does_not_match_any_known_category_name")]
    public void Convert_StringWithSpecialCharacters_ReturnsDefaultEmoji(string value)
    {
        // Arrange
        var converter = new CategoryToEmojiConverter();
        var expectedEmoji = "📹";

        // Act
        var result = converter.Convert(value, typeof(string), null, null);

        // Assert
        Assert.AreEqual(expectedEmoji, result);
    }

    /// <summary>
    /// Tests that Convert method behavior is not affected by the targetType parameter.
    /// </summary>
    [TestMethod]
    public void Convert_DifferentTargetTypes_ReturnsSameResult()
    {
        // Arrange
        var converter = new CategoryToEmojiConverter();
        var category = "basics";
        var expectedEmoji = "🤟";

        // Act
        var result1 = converter.Convert(category, typeof(string), null, null);
        var result2 = converter.Convert(category, typeof(object), null, null);
        var result3 = converter.Convert(category, typeof(int), null, null);

        // Assert
        Assert.AreEqual(expectedEmoji, result1);
        Assert.AreEqual(expectedEmoji, result2);
        Assert.AreEqual(expectedEmoji, result3);
    }

    /// <summary>
    /// Tests that Convert method behavior is not affected by the parameter argument.
    /// </summary>
    [TestMethod]
    public void Convert_DifferentParameters_ReturnsSameResult()
    {
        // Arrange
        var converter = new CategoryToEmojiConverter();
        var category = "numbers";
        var expectedEmoji = "🔢";

        // Act
        var result1 = converter.Convert(category, typeof(string), null, null);
        var result2 = converter.Convert(category, typeof(string), "parameter", null);
        var result3 = converter.Convert(category, typeof(string), 123, null);

        // Assert
        Assert.AreEqual(expectedEmoji, result1);
        Assert.AreEqual(expectedEmoji, result2);
        Assert.AreEqual(expectedEmoji, result3);
    }

    /// <summary>
    /// Tests that Convert method behavior is not affected by the culture parameter.
    /// </summary>
    [TestMethod]
    public void Convert_DifferentCultures_ReturnsSameResult()
    {
        // Arrange
        var converter = new CategoryToEmojiConverter();
        var category = "phrases";
        var expectedEmoji = "💬";

        // Act
        var result1 = converter.Convert(category, typeof(string), null, null);
        var result2 = converter.Convert(category, typeof(string), null, CultureInfo.InvariantCulture);
        var result3 = converter.Convert(category, typeof(string), null, new CultureInfo("en-US"));
        var result4 = converter.Convert(category, typeof(string), null, new CultureInfo("fr-FR"));

        // Assert
        Assert.AreEqual(expectedEmoji, result1);
        Assert.AreEqual(expectedEmoji, result2);
        Assert.AreEqual(expectedEmoji, result3);
        Assert.AreEqual(expectedEmoji, result4);
    }

}