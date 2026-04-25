using Microsoft.Maui.Controls;
using Microsoft.Maui.Controls.Xaml;
using Microsoft.Maui.Graphics;
using SignLanguageApp.ViewModels;

namespace SignLanguageApp.Pages;

public class DynamicLessonPage : ContentPage
{
    public DynamicLessonPage(string xamlContent, DynamicLessonViewModel viewModel)
    {
        BindingContext = viewModel;

        try
        {
            this.LoadFromXaml(xamlContent);
        }
        catch (Exception ex)
        {
            Content = new Grid
            {
                Children =
                {
                    new Label
                    {
                        Text = $"Error rendering lesson: {ex.Message}",
                        TextColor = Colors.Red,
                        HorizontalTextAlignment = TextAlignment.Center,
                        VerticalTextAlignment = TextAlignment.Center,
                        HorizontalOptions = LayoutOptions.Center,
                        VerticalOptions = LayoutOptions.Center
                    }
                }
            };
        }
    }
}
