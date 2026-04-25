using System.Globalization;
using Microsoft.Maui.Controls;

namespace SignLanguageApp.Helpers
{
    public class TabIndexToBgColorConverter : IValueConverter
    {
        public object? Convert(object? value, Type targetType, object? parameter, CultureInfo culture)
        {
            if (value is not int currentIndex || parameter is not string paramStr)
                return Colors.Transparent;

            if (!int.TryParse(paramStr, out int tabIndex))
                return Colors.Transparent;

            return currentIndex == tabIndex ? Color.FromArgb("#7C3AED") : Colors.Transparent;
        }

        public object? ConvertBack(object? value, Type targetType, object? parameter, CultureInfo culture)
        {
            return 0;
        }
    }

    public class TabIndexToVisibilityConverter : IValueConverter
    {
        public object? Convert(object? value, Type targetType, object? parameter, CultureInfo culture)
        {
            if (value is not int currentIndex || parameter is not string paramStr)
                return false;

            if (!int.TryParse(paramStr, out int tabIndex))
                return false;

            return currentIndex == tabIndex;
        }

        public object? ConvertBack(object? value, Type targetType, object? parameter, CultureInfo culture)
        {
            return 0;
        }
    }

    public class IntToNotIntConverter : IValueConverter
    {
        public object? Convert(object? value, Type targetType, object? parameter, CultureInfo culture)
        {
            return value is int intValue && intValue != 0;
        }

        public object? ConvertBack(object? value, Type targetType, object? parameter, CultureInfo culture)
        {
            return 0;
        }
    }

    public class StringToFirstCharConverter : IValueConverter
    {
        public object? Convert(object? value, Type targetType, object? parameter, CultureInfo culture)
        {
            if (value is not string str || string.IsNullOrEmpty(str))
                return "?";

            return str[0].ToString().ToUpper();
        }

        public object? ConvertBack(object? value, Type targetType, object? parameter, CultureInfo culture)
        {
            return string.Empty;
        }
    }
}
