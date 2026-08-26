using System;
using System.Threading;
using System.Threading.Tasks;

using Microsoft.Maui.Controls;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using Moq;
using SignLanguageApp.Pages;
using SignLanguageApp.Services;
using SignLanguageApp.ViewModels;

namespace SignLanguageApp.Pages.UnitTests
{
    /// <summary>
    /// Unit tests for the <see cref="LearnPage"/> class.
    /// </summary>
    [TestClass]
    public partial class LearnPageTests
    {
        /// <summary>
        /// Tests that the constructor properly initializes the page with a valid view model.
        /// Verifies that the BindingContext is set to the provided view model.
        /// </summary>
        [TestMethod]
        public void Constructor_ValidViewModel_SetsBindingContextCorrectly()
        {
            // Arrange
            var mockApiService = new Mock<IApiService>();
            var payloadSecurityService = new LessonPayloadSecurityService();
            var mockMedia = new Mock<IMediaDownloadAndCacheService>();
            var mockDb = new Mock<IDatabaseService>();
            var viewModel = new LearnViewModel(mockApiService.Object, payloadSecurityService, mockMedia.Object, mockDb.Object);

            // Act
            var page = new LearnPage(viewModel);

            // Assert
            Assert.IsNotNull(page);
            Assert.AreSame(viewModel, page.BindingContext);
        }

    }
}