using Microsoft.Maui.Controls;

namespace SignLanguageApp.Helpers;

public class CategoryColorConverter : IValueConverter
{
    public object? Convert(object? value, Type targetType, object? parameter, System.Globalization.CultureInfo? culture)
    {
        if (value is string category)
        {
            return category switch
            {
                "All" => Color.FromArgb("#6B5B95"),
                "Basics" => Color.FromArgb("#3B82F6"),
                "Numbers" => Color.FromArgb("#10B981"),
                "Phrases" => Color.FromArgb("#F59E0B"),
                "Advanced" => Color.FromArgb("#8B5CF6"),
                "Alphabet" => Color.FromArgb("#EC4899"),
                _ => Color.FromArgb("#374151")
            };
        }
        return Color.FromArgb("#374151");
    }

    public object? ConvertBack(object? value, Type targetType, object? parameter, System.Globalization.CultureInfo? culture)
    {
        throw new NotImplementedException();
    }
}
