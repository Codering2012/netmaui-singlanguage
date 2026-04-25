using Microsoft.Maui.Controls;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using Moq;
using SignLanguageApp.Pages;
using SignLanguageApp.ViewModels;


namespace SignLanguageApp.Pages.UnitTests
{
    /// <summary>
    /// Unit tests for the HomePage class.
    /// </summary>
    [TestClass]
    public partial class HomePageTests
    {
        /// <summary>
        /// Tests that the constructor successfully initializes the HomePage
        /// with a valid HomeViewModel and sets the BindingContext correctly.
        /// </summary>
        [TestMethod]
        public void Constructor_WithValidViewModel_SetsBindingContextToViewModel()
        {
            // Arrange
            var mockViewModel = new Mock<HomeViewModel>();

            // Act
            var homePage = new HomePage(mockViewModel.Object);

            // Assert
            Assert.IsNotNull(homePage);
            Assert.AreEqual(mockViewModel.Object, homePage.BindingContext);
        }

        /// <summary>
        /// Tests that the constructor accepts a null viewModel parameter
        /// and sets the BindingContext to null without throwing an exception.
        /// Note: Although the parameter is marked as non-nullable, null can be passed at runtime.
        /// </summary>
        [TestMethod]
        public void Constructor_WithNullViewModel_SetsBindingContextToNull()
        {
            // Arrange
            HomeViewModel? nullViewModel = null;

            // Act
            var homePage = new HomePage(nullViewModel!);

            // Assert
            Assert.IsNotNull(homePage);
            Assert.IsNull(homePage.BindingContext);
        }
    }
}