using CommunityToolkit.Mvvm.ComponentModel;
using System.Collections.ObjectModel;

namespace SignLanguageApp.Model;

public enum LessonStepType
{
    Flashcard,
    MultipleChoice,
    CameraPractice,
    MatchingCard,
    SentenceSequence,
    Completion
}

public class MatchingPairItem
{
    public int Id { get; set; }
    public string SignTitle { get; set; } = string.Empty;
    public string TextTranslation { get; set; } = string.Empty;
    public string? ImageUrl { get; set; }
    public string? VideoUrl { get; set; }
}

public class LessonStep
{
    public LessonStepType Type { get; set; }
    public string Title { get; set; } = string.Empty;
    public string Description { get; set; } = string.Empty;
    public string? ImageUrl { get; set; }
    public string? VideoUrl { get; set; }
    public string? VideoDemonstrationUrl { get; set; }
    public string? HandPostureTips { get; set; }
    public string? SignerCreditName { get; set; }
    
    // For MultipleChoice
    public List<string>? Options { get; set; }
    public string? CorrectOption { get; set; }
    
    // For CameraPractice
    public string? TargetGesture { get; set; }

    // For MatchingCard
    public List<MatchingPairItem>? MatchingPairs { get; set; }

    // For SentenceSequence
    public List<string>? SequenceTokens { get; set; }
    public string? TargetSentence { get; set; }
}

public class InteractiveLesson
{
    public int Id { get; set; }
    public string Title { get; set; } = string.Empty;
    public List<LessonStep> Steps { get; set; } = new();
}

