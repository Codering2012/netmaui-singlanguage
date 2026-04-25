using Microsoft.Maui.Controls;

namespace SignLanguageApp.Helpers;

public class StringIsNotNullOrEmptyConverter : IValueConverter
{
    public object? Convert(object? value, Type targetType, object? parameter, System.Globalization.CultureInfo? culture)
    {
        if (value is string str)
        {
            return !string.IsNullOrWhiteSpace(str);
        }
        return false;
    }

    public object? ConvertBack(object? value, Type targetType, object? parameter, System.Globalization.CultureInfo? culture)
    {
        throw new NotImplementedException();
    }
}
