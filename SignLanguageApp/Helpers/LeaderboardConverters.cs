using System.Globalization;
using Microsoft.Maui.Graphics;

namespace SignLanguageApp.Helpers
{
    public class RankToColorConverter : IValueConverter
    {
        public object Convert(object? value, Type targetType, object? parameter, CultureInfo culture)
        {
            if (value is int rank)
            {
                return rank switch
                {
                    1 => Color.FromArgb("#FFD700"), // Gold
                    2 => Color.FromArgb("#C0C0C0"), // Silver
                    3 => Color.FromArgb("#CD7F32"), // Bronze
                    _ => Color.FromArgb("#9E9E9E")  // Grey
                };
            }
            return Color.FromArgb("#9E9E9E");
        }

        public object ConvertBack(object? value, Type targetType, object? parameter, CultureInfo culture) => throw new NotImplementedException();
    }

    public class BooleanToOpacityConverter : IValueConverter
    {
        public object Convert(object? value, Type targetType, object? parameter, CultureInfo culture)
        {
            if (value is bool isUnlocked)
            {
                return isUnlocked ? 1.0 : 0.4;
            }
            return 0.4;
        }

        public object ConvertBack(object? value, Type targetType, object? parameter, CultureInfo culture) => throw new NotImplementedException();
    }
    
    public class BooleanToColorConverter : IValueConverter
    {
        public object Convert(object? value, Type targetType, object? parameter, CultureInfo culture)
        {
            if (value is bool boolValue)
            {
                if (parameter is string colorKey && Application.Current?.Resources != null && Application.Current.Resources.TryGetValue(colorKey, out var color))
                {
                    return boolValue ? color : Colors.Transparent;
                }
                return boolValue ? Color.FromArgb("#38BDF8") : Colors.Transparent;
            }
            return Colors.Transparent;
        }

        public object ConvertBack(object? value, Type targetType, object? parameter, CultureInfo culture) => throw new NotImplementedException();
    }

    public class NotNullToBooleanConverter : IValueConverter
    {
        public object Convert(object? value, Type targetType, object? parameter, CultureInfo culture)
        {
            return value != null;
        }

        public object ConvertBack(object? value, Type targetType, object? parameter, CultureInfo culture) => throw new NotImplementedException();
    }
}
