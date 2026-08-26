using Microsoft.Maui.Controls;
using Microsoft.Maui;
using System;
using System.Threading.Tasks;

namespace SignLanguageApp
{
    public partial class AppShell : Shell
    {
        public AppShell()
        {
            SignLanguageApp.Helpers.FileLogger.Log("[APPSHELL] Constructor started.");
            try
            {
                SignLanguageApp.Helpers.FileLogger.Log("[APPSHELL] Calling InitializeComponent...");
                InitializeComponent();
                SignLanguageApp.Helpers.FileLogger.Log("[APPSHELL] InitializeComponent finished.");
                Routing.RegisterRoute("home_page", typeof(Pages.HomePage));
                Routing.RegisterRoute("account_page", typeof(Pages.AccountPage));
                Routing.RegisterRoute("learn", typeof(Pages.LearnPage));
                Routing.RegisterRoute("camera", typeof(Pages.CameraTranslationPage));
                Routing.RegisterRoute("camera-translation", typeof(Pages.CameraTranslationPage));
                Routing.RegisterRoute("translation", typeof(Pages.CameraTranslationPage));

                // Phase 4 New Routes
                Routing.RegisterRoute("dictionary", typeof(Pages.DictionaryPage));
                Routing.RegisterRoute("community", typeof(Pages.CommunityHubPage));
                Routing.RegisterRoute("time-attack", typeof(Pages.TimeAttackPage));
                Routing.RegisterRoute("mistake-replay", typeof(Pages.MistakeReplayPage));
                Routing.RegisterRoute("achievements", typeof(Pages.AchievementsPage));
                Routing.RegisterRoute("interactive-lesson", typeof(Pages.InteractiveLessonPage));
                Routing.RegisterRoute("leaderboard", typeof(Pages.LeaderboardPage));
                Routing.RegisterRoute("statistics", typeof(Pages.StatisticsPage));
                Routing.RegisterRoute("difficulty-calibration", typeof(Pages.DifficultyCalibrationPage));
                Routing.RegisterRoute("edit-profile", typeof(Pages.EditProfilePage));
                Routing.RegisterRoute("feedback", typeof(Pages.FeedbackPage));
                Routing.RegisterRoute("progress-reports", typeof(Pages.ProgressReportsPage));
                Routing.RegisterRoute("sentence-builder", typeof(Pages.SentenceBuilderPage));
            }
            catch (Exception ex)
            {
                SignLanguageApp.Helpers.GlobalExceptionHandler.HandleException(ex);
                throw; // CRITICAL: Rethrow to prevent MAUI from mounting a broken shell and crashing natively!
            }
        }

        protected override async void OnNavigating(ShellNavigatingEventArgs args)
        {
            base.OnNavigating(args);
        }

        protected override void OnNavigated(ShellNavigatedEventArgs args)
        {
            base.OnNavigated(args);
        }
    }
}
