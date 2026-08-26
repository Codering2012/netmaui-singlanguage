using System.Globalization;

namespace SignLanguageApp.Helpers
{
    public class XpToHeightConverter : IValueConverter
    {
        public object? Convert(object? value, Type targetType, object? parameter, CultureInfo culture)
        {
            if (value is int xp)
            {
                // Simple scaling for the bar chart
                return Math.Max(5, Math.Min(120, xp * 1.5));
            }
            return 5;
        }

        public object? ConvertBack(object? value, Type targetType, object? parameter, CultureInfo culture) => throw new NotImplementedException();
    }

    public class DateToDayConverter : IValueConverter
    {
        public object? Convert(object? value, Type targetType, object? parameter, CultureInfo culture)
        {
            if (value is string dateStr && DateTime.TryParse(dateStr, out var date))
            {
                return date.ToString("ddd").ToUpperInvariant();
            }
            return string.Empty;
        }

        public object? ConvertBack(object? value, Type targetType, object? parameter, CultureInfo culture) => throw new NotImplementedException();
    }

    public class RatingColorConverter : IValueConverter
    {
        public object? Convert(object? value, Type targetType, object? parameter, CultureInfo culture)
        {
            if (value is int rating && parameter is string targetRatingStr && int.TryParse(targetRatingStr, out var targetRating))
            {
                // Returns a SkyBlue color if selected, SlateGrey otherwise to match dark theme
                return rating == targetRating ? Color.FromArgb("#38BDF8") : Color.FromArgb("#334155");
            }
            return Color.FromArgb("#334155");
        }

        public object? ConvertBack(object? value, Type targetType, object? parameter, CultureInfo culture) => throw new NotImplementedException();
    }
}
