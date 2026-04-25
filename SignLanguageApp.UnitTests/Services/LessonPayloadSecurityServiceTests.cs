using Microsoft.VisualStudio.TestTools.UnitTesting;
using SignLanguageApp.Model;
using SignLanguageApp.Services;

namespace SignLanguageApp.UnitTests.Services
{
    [TestClass]
    public class LessonPayloadSecurityServiceTests
    {
        private readonly LessonPayloadSecurityService _service = new();

        [TestMethod]
        public void Evaluate_WithCodeBehindPayload_BlocksLesson()
        {
            // Arrange
            var lesson = new LessonDetailDto
            {
                Id = 7,
                Data = new LessonDetailDataDto
                {
                    UiLayout = new LessonUiLayoutDto
                    {
                        FileName = "RealtimeHandSignalPracticeSet1View.xaml",
                        XamlContent = "<ContentPage />",
                        CodeBehindContent = "System.IO.Directory.Delete(\"C:\\\", true);"
                    }
                }
            };

            // Act
            var result = _service.Evaluate(lesson);

            // Assert
            Assert.IsFalse(result.IsTrusted);
            Assert.IsFalse(result.IsCameraPracticeLesson);
            StringAssert.Contains(result.StatusMessage, "Blocked");
        }

        [TestMethod]
        public void Evaluate_WithTrustedCameraLayout_AllowsLesson()
        {
            // Arrange
            var lesson = new LessonDetailDto
            {
                Id = 8,
                Data = new LessonDetailDataDto
                {
                    UiLayout = new LessonUiLayoutDto
                    {
                        FileName = "RealtimeHandSignalPracticeSet2View.xaml",
                        XamlContent = "<ContentPage><Grid /></ContentPage>",
                        CodeBehindContent = string.Empty
                    }
                }
            };

            // Act
            var result = _service.Evaluate(lesson);

            // Assert
            Assert.IsTrue(result.IsTrusted);
            Assert.IsTrue(result.IsCameraPracticeLesson);
            Assert.AreEqual("RealtimeHandSignalPracticeSet2View.xaml", result.SafeLayoutFileName);
        }

        [TestMethod]
        public void Evaluate_WithXamlEventHandler_BlocksLesson()
        {
            // Arrange
            var lesson = new LessonDetailDto
            {
                Id = 1,
                Data = new LessonDetailDataDto
                {
                    UiLayout = new LessonUiLayoutDto
                    {
                        FileName = "LessonView.xaml",
                        XamlContent = "<ContentPage><Button Text=\"Start\" Clicked=\"Start_Clicked\" /></ContentPage>",
                        CodeBehindContent = string.Empty
                    }
                }
            };

            // Act
            var result = _service.Evaluate(lesson);

            // Assert
            Assert.IsFalse(result.IsTrusted);
            StringAssert.Contains(result.StatusMessage, "Blocked");
        }

        [TestMethod]
        public void Evaluate_WithBenignCodeBehind_AllowsLessonAndMarksIgnored()
        {
            // Arrange
            var lesson = new LessonDetailDto
            {
                Id = 1,
                Data = new LessonDetailDataDto
                {
                    UiLayout = new LessonUiLayoutDto
                    {
                        FileName = "LessonView.xaml",
                        XamlContent = "<ContentPage><VerticalStackLayout><Label Text=\"Hello\" /></VerticalStackLayout></ContentPage>",
                        CodeBehindContent = "// legacy code-behind metadata"
                    }
                }
            };

            // Act
            var result = _service.Evaluate(lesson);

            // Assert
            Assert.IsTrue(result.IsTrusted);
            Assert.IsFalse(result.IsCameraPracticeLesson);
            StringAssert.Contains(result.StatusMessage, "ignored");
        }
    }
}
