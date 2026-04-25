using Microsoft.Maui.Controls;
using System.Globalization;

namespace SignLanguageApp.Converters
{
    public class MultiValueStringFormatConverter : IMultiValueConverter
    {
        public object? Convert(object?[]? values, Type? targetType, object? parameter, CultureInfo? culture)
        {
            if (values == null || values.Length < 2)
                return string.Empty;

            var first = values[0];
            var second = values[1];

            if (parameter is string format)
                return string.Format(culture ?? CultureInfo.InvariantCulture, format, first, second);

            return $"{first} / {second}";
        }

        public object?[]? ConvertBack(object? value, Type?[]? targetTypes, object? parameter, CultureInfo? culture)
        {
            throw new NotImplementedException();
        }
    }
}