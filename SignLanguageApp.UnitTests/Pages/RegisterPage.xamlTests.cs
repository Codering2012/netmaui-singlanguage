using Microsoft.Maui.Controls;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using Moq;
using SignLanguageApp.Pages;
using SignLanguageApp.Services;
using SignLanguageApp.ViewModels;


namespace SignLanguageApp.Pages.UnitTests;

/// <summary>
/// Unit tests for <see cref="RegisterPage"/> class.
/// </summary>
[TestClass]
public partial class RegisterPageTests
{
    /// <summary>
    /// Tests that the constructor correctly initializes the page and sets the BindingContext
    /// when provided with a valid RegisterViewModel instance.
    /// </summary>
    [TestMethod]
    public void Constructor_ValidViewModel_SetsBindingContextToViewModel()
    {
        // Arrange
        var mockApiService = new Mock<IApiService>();
        var viewModel = new RegisterViewModel(mockApiService.Object);

        // Act
        var page = new RegisterPage(viewModel);

        // Assert
        Assert.IsNotNull(page);
        Assert.AreSame(viewModel, page.BindingContext);
    }

    /// <summary>
    /// Tests that the constructor handles a null viewModel parameter.
    /// Even though the parameter is marked as non-nullable, C# allows null to be passed at runtime.
    /// This test verifies that the BindingContext is set to null without throwing an exception
    /// during construction.
    /// </summary>
    [TestMethod]
    public void Constructor_NullViewModel_SetsBindingContextToNull()
    {
        // Arrange
        RegisterViewModel? viewModel = null;

        // Act
        var page = new RegisterPage(viewModel!);

        // Assert
        Assert.IsNotNull(page);
        Assert.IsNull(page.BindingContext);
    }
}