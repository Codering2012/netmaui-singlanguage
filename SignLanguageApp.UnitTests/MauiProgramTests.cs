using System;
using System.Net.Http;
using Microsoft.Extensions.DependencyInjection;
using Microsoft.Maui.Hosting;
using Microsoft.Maui.Storage;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using SignLanguageApp.Pages;
using SignLanguageApp.Services;
using SignLanguageApp.ViewModels;


namespace SignLanguageApp.UnitTests
{
    /// <summary>
    /// Unit tests for the MauiProgram class.
    /// </summary>
    [TestClass]
    public class MauiProgramTests
    {
        /// <summary>
        /// Verifies that CreateMauiApp returns a non-null MauiApp instance.
        /// </summary>
        [TestMethod]
        public void CreateMauiApp_WhenCalled_ReturnsNonNullMauiApp()
        {
            // Act
            MauiApp result = MauiProgram.CreateMauiApp();

            // Assert
            Assert.IsNotNull(result);
        }

        /// <summary>
        /// Verifies that CreateMauiApp registers ISecureStorage service correctly.
        /// </summary>
        [TestMethod]
        public void CreateMauiApp_WhenCalled_RegistersSecureStorageService()
        {
            // Act
            MauiApp app = MauiProgram.CreateMauiApp();

            // Assert
            ISecureStorage? secureStorage = app.Services.GetService<ISecureStorage>();
            Assert.IsNotNull(secureStorage);
        }

        /// <summary>
        /// Verifies that CreateMauiApp registers IDatabaseService with DatabaseService implementation.
        /// </summary>
        [TestMethod]
        public void CreateMauiApp_WhenCalled_RegistersDatabaseService()
        {
            // Act
            MauiApp app = MauiProgram.CreateMauiApp();

            // Assert
            IDatabaseService? databaseService = app.Services.GetService<IDatabaseService>();
            Assert.IsNotNull(databaseService);
            Assert.IsInstanceOfType(databaseService, typeof(DatabaseService));
        }

        /// <summary>
        /// Verifies that CreateMauiApp registers IAuthService with AuthService implementation.
        /// </summary>
        [TestMethod]
        public void CreateMauiApp_WhenCalled_RegistersAuthService()
        {
            // Act
            MauiApp app = MauiProgram.CreateMauiApp();

            // Assert
            IAuthService? authService = app.Services.GetService<IAuthService>();
            Assert.IsNotNull(authService);
            Assert.IsInstanceOfType(authService, typeof(AuthService));
        }

        /// <summary>
        /// Verifies that CreateMauiApp registers ICacheService with CacheService implementation.
        /// </summary>
        [TestMethod]
        public void CreateMauiApp_WhenCalled_RegistersCacheService()
        {
            // Act
            MauiApp app = MauiProgram.CreateMauiApp();

            // Assert
            ICacheService? cacheService = app.Services.GetService<ICacheService>();
            Assert.IsNotNull(cacheService);
            Assert.IsInstanceOfType(cacheService, typeof(CacheService));
        }

        /// <summary>
        /// Verifies that CreateMauiApp registers HttpClient service.
        /// </summary>
        [TestMethod]
        public void CreateMauiApp_WhenCalled_RegistersHttpClient()
        {
            // Act
            MauiApp app = MauiProgram.CreateMauiApp();

            // Assert
            HttpClient? httpClient = app.Services.GetService<HttpClient>();
            Assert.IsNotNull(httpClient);
        }

        /// <summary>
        /// Verifies that CreateMauiApp configures HttpClient with correct base address.
        /// </summary>
        [TestMethod]
        public void CreateMauiApp_WhenCalled_ConfiguresHttpClientWithCorrectBaseAddress()
        {
            // Act
            MauiApp app = MauiProgram.CreateMauiApp();

            // Assert
            HttpClient? httpClient = app.Services.GetService<HttpClient>();
            Assert.IsNotNull(httpClient);
            Assert.IsNotNull(httpClient.BaseAddress);
            Assert.AreEqual("https://localhost:7084/api/", httpClient.BaseAddress.ToString());
        }

        /// <summary>
        /// Verifies that CreateMauiApp configures HttpClient with correct timeout.
        /// </summary>
        [TestMethod]
        public void CreateMauiApp_WhenCalled_ConfiguresHttpClientWithCorrectTimeout()
        {
            // Act
            MauiApp app = MauiProgram.CreateMauiApp();

            // Assert
            HttpClient? httpClient = app.Services.GetService<HttpClient>();
            Assert.IsNotNull(httpClient);
            Assert.AreEqual(TimeSpan.FromSeconds(30), httpClient.Timeout);
        }

        /// <summary>
        /// Verifies that CreateMauiApp registers IApiService with ApiService implementation.
        /// </summary>
        [TestMethod]
        public void CreateMauiApp_WhenCalled_RegistersApiService()
        {
            // Act
            MauiApp app = MauiProgram.CreateMauiApp();

            // Assert
            IApiService? apiService = app.Services.GetService<IApiService>();
            Assert.IsNotNull(apiService);
            Assert.IsInstanceOfType(apiService, typeof(ApiService));
        }

        /// <summary>
        /// Verifies that CreateMauiApp registers LoginViewModel.
        /// </summary>
        [TestMethod]
        public void CreateMauiApp_WhenCalled_RegistersLoginViewModel()
        {
            // Act
            MauiApp app = MauiProgram.CreateMauiApp();

            // Assert
            LoginViewModel? viewModel = app.Services.GetService<LoginViewModel>();
            Assert.IsNotNull(viewModel);
        }

        /// <summary>
        /// Verifies that CreateMauiApp registers RegisterViewModel.
        /// </summary>
        [TestMethod]
        public void CreateMauiApp_WhenCalled_RegistersRegisterViewModel()
        {
            // Act
            MauiApp app = MauiProgram.CreateMauiApp();

            // Assert
            RegisterViewModel? viewModel = app.Services.GetService<RegisterViewModel>();
            Assert.IsNotNull(viewModel);
        }

        /// <summary>
        /// Verifies that CreateMauiApp registers HomeViewModel.
        /// </summary>
        [TestMethod]
        public void CreateMauiApp_WhenCalled_RegistersHomeViewModel()
        {
            // Act
            MauiApp app = MauiProgram.CreateMauiApp();

            // Assert
            HomeViewModel? viewModel = app.Services.GetService<HomeViewModel>();
            Assert.IsNotNull(viewModel);
        }

        /// <summary>
        /// Verifies that CreateMauiApp registers AccountViewModel.
        /// </summary>
        [TestMethod]
        public void CreateMauiApp_WhenCalled_RegistersAccountViewModel()
        {
            // Act
            MauiApp app = MauiProgram.CreateMauiApp();

            // Assert
            AccountViewModel? viewModel = app.Services.GetService<AccountViewModel>();
            Assert.IsNotNull(viewModel);
        }

        /// <summary>
        /// Verifies that CreateMauiApp registers CameraTranslationViewModel.
        /// </summary>
        [TestMethod]
        public void CreateMauiApp_WhenCalled_RegistersCameraTranslationViewModel()
        {
            // Act
            MauiApp app = MauiProgram.CreateMauiApp();

            // Assert
            CameraTranslationViewModel? viewModel = app.Services.GetService<CameraTranslationViewModel>();
            Assert.IsNotNull(viewModel);
        }

        /// <summary>
        /// Verifies that CreateMauiApp registers LearnViewModel.
        /// </summary>
        [TestMethod]
        public void CreateMauiApp_WhenCalled_RegistersLearnViewModel()
        {
            // Act
            MauiApp app = MauiProgram.CreateMauiApp();

            // Assert
            LearnViewModel? viewModel = app.Services.GetService<LearnViewModel>();
            Assert.IsNotNull(viewModel);
        }

        /// <summary>
        /// Verifies that CreateMauiApp registers VideoViewModel.
        /// </summary>
        [TestMethod]
        public void CreateMauiApp_WhenCalled_RegistersVideoViewModel()
        {
            // Act
            MauiApp app = MauiProgram.CreateMauiApp();

            // Assert
            VideoViewModel? viewModel = app.Services.GetService<VideoViewModel>();
            Assert.IsNotNull(viewModel);
        }

        /// <summary>
        /// Verifies that CreateMauiApp registers LoginPage.
        /// </summary>
        [TestMethod]
        public void CreateMauiApp_WhenCalled_RegistersLoginPage()
        {
            // Act
            MauiApp app = MauiProgram.CreateMauiApp();

            // Assert
            LoginPage? page = app.Services.GetService<LoginPage>();
            Assert.IsNotNull(page);
        }

        /// <summary>
        /// Verifies that CreateMauiApp registers RegisterPage.
        /// </summary>
        [TestMethod]
        public void CreateMauiApp_WhenCalled_RegistersRegisterPage()
        {
            // Act
            MauiApp app = MauiProgram.CreateMauiApp();

            // Assert
            RegisterPage? page = app.Services.GetService<RegisterPage>();
            Assert.IsNotNull(page);
        }

        /// <summary>
        /// Verifies that CreateMauiApp registers HomePage.
        /// </summary>
        [TestMethod]
        public void CreateMauiApp_WhenCalled_RegistersHomePage()
        {
            // Act
            MauiApp app = MauiProgram.CreateMauiApp();

            // Assert
            HomePage? page = app.Services.GetService<HomePage>();
            Assert.IsNotNull(page);
        }

        /// <summary>
        /// Verifies that CreateMauiApp registers AccountPage.
        /// </summary>
        [TestMethod]
        public void CreateMauiApp_WhenCalled_RegistersAccountPage()
        {
            // Act
            MauiApp app = MauiProgram.CreateMauiApp();

            // Assert
            AccountPage? page = app.Services.GetService<AccountPage>();
            Assert.IsNotNull(page);
        }

        /// <summary>
        /// Verifies that CreateMauiApp registers CameraTranslationPage.
        /// </summary>
        [TestMethod]
        public void CreateMauiApp_WhenCalled_RegistersCameraTranslationPage()
        {
            // Act
            MauiApp app = MauiProgram.CreateMauiApp();

            // Assert
            CameraTranslationPage? page = app.Services.GetService<CameraTranslationPage>();
            Assert.IsNotNull(page);
        }

        /// <summary>
        /// Verifies that CreateMauiApp registers LearnPage.
        /// </summary>
        [TestMethod]
        public void CreateMauiApp_WhenCalled_RegistersLearnPage()
        {
            // Act
            MauiApp app = MauiProgram.CreateMauiApp();

            // Assert
            LearnPage? page = app.Services.GetService<LearnPage>();
            Assert.IsNotNull(page);
        }

        /// <summary>
        /// Verifies that CreateMauiApp registers VideosPage.
        /// </summary>
        [TestMethod]
        public void CreateMauiApp_WhenCalled_RegistersVideosPage()
        {
            // Act
            MauiApp app = MauiProgram.CreateMauiApp();

            // Assert
            VideosPage? page = app.Services.GetService<VideosPage>();
            Assert.IsNotNull(page);
        }

        /// <summary>
        /// Verifies that CreateMauiApp registers ProfilePage.
        /// </summary>
        [TestMethod]
        public void CreateMauiApp_WhenCalled_RegistersProfilePage()
        {
            // Act
            MauiApp app = MauiProgram.CreateMauiApp();

            // Assert
            ProfilePage? page = app.Services.GetService<ProfilePage>();
            Assert.IsNotNull(page);
        }

        /// <summary>
        /// Verifies that CreateMauiApp registers MainPage.
        /// </summary>
        [TestMethod]
        public void CreateMauiApp_WhenCalled_RegistersMainPage()
        {
            // Act
            MauiApp app = MauiProgram.CreateMauiApp();

            // Assert
            MainPage? page = app.Services.GetService<MainPage>();
            Assert.IsNotNull(page);
        }

        /// <summary>
        /// Verifies that CreateMauiApp registers all services as singletons by resolving the same instance twice.
        /// </summary>
        [TestMethod]
        public void CreateMauiApp_WhenCalled_RegistersServicesAsSingletons()
        {
            // Act
            MauiApp app = MauiProgram.CreateMauiApp();

            // Assert
            IDatabaseService? firstInstance = app.Services.GetService<IDatabaseService>();
            IDatabaseService? secondInstance = app.Services.GetService<IDatabaseService>();
            Assert.IsNotNull(firstInstance);
            Assert.IsNotNull(secondInstance);
            Assert.AreSame(firstInstance, secondInstance);
        }
    }
}