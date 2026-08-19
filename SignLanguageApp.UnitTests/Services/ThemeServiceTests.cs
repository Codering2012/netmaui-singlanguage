using Microsoft.VisualStudio.TestTools.UnitTesting;
using SignLanguageApp.Services;

namespace SignLanguageApp.UnitTests.Services
{
    [TestClass]
    public class ThemeServiceTests
    {
        private ThemeService _themeService = null!;

        [TestInitialize]
        public void Setup()
        {
            _themeService = new ThemeService();
        }

        [TestMethod]
        public void InitializeTheme_DoesNotThrow_WhenApplicationCurrentIsNull()
        {
            // Act & Assert (Should execute without exceptions in headless mode)
            _themeService.InitializeTheme();
            Assert.IsNotNull(_themeService);
        }

        [TestMethod]
        public void SetTheme_DoesNotThrow_WhenApplicationCurrentIsNull()
        {
            // Act & Assert
            _themeService.SetTheme(0);
            _themeService.SetTheme(1);
            _themeService.SetTheme(2);
            Assert.IsNotNull(_themeService);
        }
    }
}
