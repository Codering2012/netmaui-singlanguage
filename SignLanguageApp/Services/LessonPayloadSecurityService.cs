using System.Diagnostics;
using System.IO;
using SignLanguageApp.Model;

namespace SignLanguageApp.Services;

public interface ILessonPayloadSecurityService
{
    LessonLayoutSecurityResult Evaluate(LessonDetailDto? lessonDetail);
}

public sealed class LessonLayoutSecurityResult
{
    public bool IsTrusted { get; init; }
    public bool IsCameraPracticeLesson { get; init; }
    public string SafeLayoutFileName { get; init; } = string.Empty;
    public string StatusMessage { get; init; } = string.Empty;
}

public sealed class LessonPayloadSecurityService : ILessonPayloadSecurityService
{
    private const int MaxXamlLength = 20000;

    private static readonly Dictionary<int, string> CameraLessonLayoutMap = new()
    {
        { 7, "RealtimeHandSignalPracticeSet1View.xaml" },
        { 8, "RealtimeHandSignalPracticeSet2View.xaml" },
        { 9, "RealtimeHandSignalPracticeSet3View.xaml" }
    };

    private static readonly HashSet<string> AllowedLayoutFiles =
    [
        "LessonView.xaml",
        "RealtimeHandSignalPracticeSet1View.xaml",
        "RealtimeHandSignalPracticeSet2View.xaml",
        "RealtimeHandSignalPracticeSet3View.xaml",
        "FamilySignsView.xaml",
        "DaysAndTimeView.xaml",
        "ShoppingDialoguesView.xaml",
        "SchoolWorkPhrasesView.xaml",
        "NarrativeClassifiersView.xaml"
    ];

    private static readonly string[] DangerousTokens =
    [
        "System.IO.",
        "Directory.Delete",
        "File.Delete",
        "Process.Start",
        "Environment.Exit",
        "DllImport",
        "Reflection",
        "AppDomain",
        "x:Code",
        "Clicked=",
        "Tapped=",
        "TextChanged=",
        "SelectedIndexChanged=",
        "CheckedChanged=",
        "ValueChanged=",
        "NavigatedTo=",
        "NavigatedFrom=",
        "Loaded="
    ];

    public LessonLayoutSecurityResult Evaluate(LessonDetailDto? lessonDetail)
    {
        var uiLayout = lessonDetail?.Data?.UiLayout;
        if (uiLayout == null)
        {
            return Rejected("No dynamic layout payload found.");
        }

        var rawFileName = uiLayout.FileName?.Trim() ?? string.Empty;
        if (string.IsNullOrWhiteSpace(rawFileName))
        {
            return Rejected("Layout file name was empty.");
        }

        var safeFileName = Path.GetFileName(rawFileName);
        if (!string.Equals(rawFileName, safeFileName, StringComparison.Ordinal))
        {
            return Rejected("Layout file name contained a path and was blocked.");
        }

        if (!IsAllowedLayoutFileName(safeFileName))
        {
            return Rejected($"Layout '{safeFileName}' is not in the trusted allowlist.");
        }

        var hasCodeBehindPayload = !string.IsNullOrWhiteSpace(uiLayout.CodeBehindContent);
        if (hasCodeBehindPayload && ContainsDangerousToken(uiLayout.CodeBehindContent))
        {
            return Rejected("Code-behind payload contained blocked tokens.");
        }

        var xamlContent = uiLayout.XamlContent ?? string.Empty;
        if (xamlContent.Length > MaxXamlLength)
        {
            return Rejected("Layout payload exceeded the maximum allowed size.");
        }

        if (ContainsDangerousToken(xamlContent))
        {
            return Rejected("Layout payload contained blocked tokens.");
        }

        var isCameraLesson = CameraLessonLayoutMap.TryGetValue(lessonDetail!.Id, out var expectedLayout);
        if (isCameraLesson && !string.Equals(expectedLayout, safeFileName, StringComparison.Ordinal))
        {
            return Rejected("Camera lesson layout file does not match expected mapping.");
        }

        return new LessonLayoutSecurityResult
        {
            IsTrusted = true,
            IsCameraPracticeLesson = isCameraLesson,
            SafeLayoutFileName = safeFileName,
            StatusMessage = hasCodeBehindPayload
                ? "Lesson payload validated. Code-behind content was ignored by the client."
                : "Lesson payload validated. Metadata-only rendering mode is active."
        };
    }

    private static bool ContainsDangerousToken(string content)
    {
        foreach (var token in DangerousTokens)
        {
            if (content.Contains(token, StringComparison.OrdinalIgnoreCase))
            {
                Debug.WriteLine($"Blocked lesson payload token detected: {token}");
                return true;
            }
        }

        return false;
    }

    private static bool IsAllowedLayoutFileName(string safeFileName)
    {
        if (AllowedLayoutFiles.Contains(safeFileName))
        {
            return true;
        }

        if (!safeFileName.EndsWith("View.xaml", StringComparison.Ordinal))
        {
            return false;
        }

        var fileStem = Path.GetFileNameWithoutExtension(safeFileName);
        foreach (var ch in fileStem)
        {
            if (!(char.IsLetterOrDigit(ch) || ch == '_'))
            {
                return false;
            }
        }

        return true;
    }

    private static LessonLayoutSecurityResult Rejected(string reason)
    {
        return new LessonLayoutSecurityResult
        {
            IsTrusted = false,
            IsCameraPracticeLesson = false,
            SafeLayoutFileName = string.Empty,
            StatusMessage = $"Blocked: {reason}"
        };
    }
}
