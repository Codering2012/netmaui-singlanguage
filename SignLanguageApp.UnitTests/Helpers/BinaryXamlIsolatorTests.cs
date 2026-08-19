using SignLanguageApp.Helpers;
using Microsoft.VisualStudio.TestTools.UnitTesting;

namespace SignLanguageApp.UnitTests.Helpers;

[TestClass]
public class BinaryXamlIsolatorTests
{
    [TestMethod]
    public void SplitIntoElements_ValidXaml_ReturnsElements()
    {
        // Arrange
        var xaml = @"<ContentPage xmlns=""http://schemas.microsoft.com/dotnet/2021/maui"">
    <VerticalStackLayout>
        <Label Text=""Hello"" />
        <Button Text=""Click me"" />
    </VerticalStackLayout>
</ContentPage>";

        // Act
        var elements = BinaryXamlIsolator.SplitIntoElements(xaml);

        // Assert
        Assert.AreEqual(1, elements.Count); // The root has one child: VerticalStackLayout
    }

    [TestMethod]
    public void ReconstructXaml_ValidParts_ReturnsFullXaml()
    {
        // Arrange
        var original = @"<ContentPage xmlns=""http://schemas.microsoft.com/dotnet/2021/maui""></ContentPage>";
        var elements = new List<string> { @"<Label Text=""Test"" />" };

        // Act
        var result = BinaryXamlIsolator.ReconstructXaml(original, elements);

        // Assert
        Assert.IsTrue(result.Contains(@"<ContentPage"));
        Assert.IsTrue(result.Contains(@"<Label Text=""Test"" />"));
        Assert.IsTrue(result.Contains(@"</ContentPage>"));
    }
}
