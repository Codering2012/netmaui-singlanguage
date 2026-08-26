using System.Globalization;
using Microsoft.Maui.Controls;
using Microsoft.Maui.Graphics;

namespace SignLanguageApp.Helpers;

public class EnumToBoolConverter : IValueConverter
{
    public object Convert(object? value, Type targetType, object? parameter, CultureInfo culture)
    {
        if (value == null || parameter == null) return false;
        return value.ToString() == parameter.ToString();
    }

    public object ConvertBack(object? value, Type targetType, object? parameter, CultureInfo culture)
    {
        throw new NotImplementedException();
    }
}

public class OptionToStrokeConverter : IValueConverter
{
    public object Convert(object? value, Type targetType, object? parameter, CultureInfo culture)
    {
        if (value == null || parameter == null) return Colors.Transparent;
        var isMatch = value.ToString() == parameter.ToString();
        if (!isMatch) return Colors.Transparent;
        
        return Application.Current?.Resources.TryGetValue("PrimaryColor", out var color) == true ? (Color)color : Colors.Blue;
    }

    public object ConvertBack(object? value, Type targetType, object? parameter, CultureInfo culture)
    {
        throw new NotImplementedException();
    }
}

public class BoolToFeedbackColorConverter : IValueConverter
{
    public object Convert(object? value, Type targetType, object? parameter, CultureInfo culture)
    {
        if (value is bool isCorrect)
        {
            string key = isCorrect ? "SuccessColor" : "ErrorColor";
            if (Application.Current?.Resources.TryGetValue(key, out var color) == true)
                return (Color)color;
            
            return isCorrect ? Colors.Green : Colors.Red;
        }
        return Colors.Gray;
    }

    public object ConvertBack(object? value, Type targetType, object? parameter, CultureInfo culture)
    {
        throw new NotImplementedException();
    }
}

public class BoolToFeedbackTextConverter : IValueConverter
{
    public object Convert(object? value, Type targetType, object? parameter, CultureInfo culture)
    {
        if (value is bool isCorrect)
        {
            return isCorrect ? "Correct! Well done." : "Not quite. Try again!";
        }
        return string.Empty;
    }

    public object ConvertBack(object? value, Type targetType, object? parameter, CultureInfo culture)
    {
        throw new NotImplementedException();
    }
}

public class ContinueEnableConverter : IValueConverter
{
    public object Convert(object? value, Type targetType, object? parameter, CultureInfo culture)
    {
        // In Duolingo, you can always continue if it's a flashcard, 
        // but need to answer if it's a quiz.
        // For simplicity here, we use the HasAnswered binding from the view
        return value ?? true;
    }

    public object ConvertBack(object? value, Type targetType, object? parameter, CultureInfo culture)
    {
        throw new NotImplementedException();
    }
}
