using Microsoft.Maui.Controls;

namespace SignLanguageApp.Helpers;

public class CategoryToEmojiConverter : IValueConverter
{
    public object? Convert(object? value, Type targetType, object? parameter, System.Globalization.CultureInfo? culture)
    {
        if (value is string category)
        {
            return category.ToLowerInvariant() switch
            {
                "basics" => "BSC",
                "numbers" => "123",
                "phrases" => "TXT",
                "advanced" => "ADV",
                "alphabet" => "ABC",
                "emotions" => "EMO",
                "daily" => "DAY",
                "greetings" => "HI",
                "colors" => "CLR",
                "animals" => "ANI",
                "family" => "FAM",
                "food" => "FOOD",
                _ => "VID"
            };
        }

        return "VID";
    }

    public object? ConvertBack(object? value, Type targetType, object? parameter, System.Globalization.CultureInfo? culture)
    {
        throw new NotImplementedException();
    }
}
