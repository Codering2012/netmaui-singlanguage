using System;
using Microsoft.Maui.Controls;
using Microsoft.Maui.ApplicationModel;
using SignLanguageApp.Pages;

namespace SignLanguageApp.Helpers
{
    public static class GlobalExceptionHandler
    {
        public static void HandleException(Exception ex)
        {
            if (Application.Current != null)
            {
                MainThread.BeginInvokeOnMainThread(() =>
                {
                    try
                    {
                        var window = Application.Current.Windows.FirstOrDefault();
#pragma warning disable CS0618
                        Application.Current.MainPage = new ErrorDebugPage(ex);
#pragma warning restore CS0618
                    }
                    catch (Exception debugPageEx)
                    {
                        // Absolute fallback: Native Alert
                        try
                        {
                            Application.Current?.MainPage?.DisplayAlert(
                                "CRITICAL CRASH",
                                $"Original Error:\n{ex?.Message}\n\nDebug Page Error:\n{debugPageEx.Message}",
                                "OK");
                        }
                        catch
                        {
                            // Ignore all errors at this point
                        }
                    }
                });
            }
        }
    }
}
