using System;
using Microsoft.Maui.Controls;
using Microsoft.Maui.Graphics;
using Microsoft.Maui.ApplicationModel.DataTransfer;

namespace SignLanguageApp.Pages
{
    public class ErrorDebugPage : ContentPage
    {
        public ErrorDebugPage(Exception exception)
        {
            BackgroundColor = Color.FromArgb("#8B0000"); // Dark red

            var stackLayout = new VerticalStackLayout
            {
                Padding = 20,
                Spacing = 15
            };

            var titleLabel = new Label
            {
                Text = "FATAL ERROR DETECTED",
                TextColor = Colors.White,
                FontSize = 24,
                FontAttributes = FontAttributes.Bold,
                HorizontalOptions = LayoutOptions.Center
            };
            stackLayout.Children.Add(titleLabel);

            var typeText = exception?.GetType()?.FullName ?? "Unknown Exception Type";
            var typeLabel = new Label
            {
                Text = $"Type: {typeText}",
                TextColor = Colors.Yellow,
                FontSize = 16,
                FontAttributes = FontAttributes.Bold
            };
            stackLayout.Children.Add(typeLabel);

            var messageText = exception?.Message ?? "No exception message provided.";
            var messageLabel = new Label
            {
                Text = $"Message:\n{messageText}",
                TextColor = Colors.White,
                FontSize = 14
            };
            stackLayout.Children.Add(messageLabel);

            var stackTraceText = exception?.StackTrace ?? "No stack trace available.";
            var stackTraceLabel = new Label
            {
                Text = $"Stack Trace:\n{stackTraceText}",
                TextColor = Color.FromArgb("#CCCCCC"),
                FontSize = 12
            };
            
            var scrollView = new ScrollView 
            { 
                Content = stackTraceLabel,
                MaximumHeightRequest = 400
            };
            stackLayout.Children.Add(scrollView);

            var innerEx = exception?.InnerException;
            if (innerEx != null)
            {
                var innerLabel = new Label
                {
                    Text = $"Inner Exception:\n{innerEx.Message}\n{innerEx.StackTrace}",
                    TextColor = Colors.Orange,
                    FontSize = 12
                };
                stackLayout.Children.Add(innerLabel);
            }

            var copyButton = new Button
            {
                Text = "Copy Error to Clipboard",
                BackgroundColor = Colors.White,
                TextColor = Colors.Black,
                Margin = new Thickness(0, 20, 0, 0)
            };
            copyButton.Clicked += async (s, e) =>
            {
                var textToCopy = $"Type: {typeText}\nMessage: {messageText}\nStack: {stackTraceText}";
                if (innerEx != null)
                    textToCopy += $"\nInner: {innerEx.Message}\n{innerEx.StackTrace}";
                
                try
                {
                    await Clipboard.Default.SetTextAsync(textToCopy);
                    await DisplayAlert("Copied", "Error details copied to clipboard. Please paste this to Antigravity.", "OK");
                }
                catch { } // Ignore if clipboard fails
            };
            stackLayout.Children.Add(copyButton);

            Content = stackLayout;
        }
    }
}
