using Microsoft.Maui.Controls;
using System;
using System.Threading;
using System.Threading.Tasks;

namespace SignLanguageApp.Controls
{
    public class TypewriterLabel : Label
    {
        public static readonly BindableProperty TypedTextProperty =
            BindableProperty.Create(nameof(TypedText), typeof(string), typeof(TypewriterLabel), string.Empty, propertyChanged: OnTypedTextChanged);

        public string TypedText
        {
            get => (string)GetValue(TypedTextProperty);
            set => SetValue(TypedTextProperty, value);
        }

        private CancellationTokenSource? _cancellationTokenSource;

        private static async void OnTypedTextChanged(BindableObject bindable, object oldValue, object newValue)
        {
            var label = (TypewriterLabel)bindable;
            var text = newValue as string;
            
            if (string.IsNullOrEmpty(text))
            {
                label.Text = string.Empty;
                return;
            }

            label._cancellationTokenSource?.Cancel();
            label._cancellationTokenSource = new CancellationTokenSource();
            var token = label._cancellationTokenSource.Token;

            label.Text = "";
            try
            {
                for (int i = 0; i < text.Length; i++)
                {
                    token.ThrowIfCancellationRequested();
                    
                    // Add character and a blinking block cursor
                    label.Text = text.Substring(0, i + 1) + "_";
                    
                    await Task.Delay(25, token); // Typing speed
                }
                
                // Final text without cursor
                label.Text = text;
            }
            catch (OperationCanceledException)
            {
                // Task was cancelled, do nothing
            }
        }
    }
}
