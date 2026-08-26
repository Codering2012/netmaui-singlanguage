using SignLanguageApp.Model;
using SignLanguageApp.ViewModels;

namespace SignLanguageApp.Services;

public sealed class InferenceOutcome
{
    public HandGestureResult? Gesture { get; init; }
    public string Message { get; init; } = "No hand detected";
    public bool ClearPredictionBuffer { get; init; } = true;
}

public interface IGesturePredictionService
{
    Task<InferenceOutcome> PerformGestureInferenceAsync(byte[] frameBytes, string? targetSign = null, CancellationToken ct = default);
}
