using System.Diagnostics;
using SignLanguageApp.Model;
using SignLanguageApp.ViewModels;

namespace SignLanguageApp.Services;

public class GesturePredictionService : IGesturePredictionService
{
    private readonly IApiService _apiService;

    public GesturePredictionService(IApiService apiService)
    {
        _apiService = apiService;
    }

    public async Task<InferenceOutcome> PerformGestureInferenceAsync(byte[] frameBytes, string? targetSign = null, CancellationToken ct = default)
    {
        try
        {
            if (frameBytes == null || frameBytes.Length == 0)
            {
                return new InferenceOutcome { Message = "No hand detected", ClearPredictionBuffer = true };
            }

            if (_apiService == null)
            {
                return new InferenceOutcome { Message = "API Connection Error", ClearPredictionBuffer = true };
            }

            using var timeoutCts = CancellationTokenSource.CreateLinkedTokenSource(ct);
            timeoutCts.CancelAfter(TimeSpan.FromSeconds(5));

            var prediction = await _apiService.PredictGestureFromImageAsync(frameBytes, targetSign, timeoutCts.Token);

            if (prediction == null)
            {
                return new InferenceOutcome { Message = "API Connection Error", ClearPredictionBuffer = true };
            }

            var status = prediction.Status?.Trim() ?? string.Empty;

            if (!string.Equals(status, "success", StringComparison.OrdinalIgnoreCase) || prediction.Data == null)
            {
                var message = string.IsNullOrWhiteSpace(prediction.Message) ? "Low Confidence..." : prediction.Message;
                if (IsNoHandText(message)) message = "No hand detected";
                return new InferenceOutcome { Message = message, ClearPredictionBuffer = true };
            }

            var letter = prediction.Data.Letter?.Trim() ?? string.Empty;
            var isNoGestureLabel = string.IsNullOrWhiteSpace(letter) ||
                                   string.Equals(letter, "none", StringComparison.OrdinalIgnoreCase) ||
                                   string.Equals(letter, "nothing", StringComparison.OrdinalIgnoreCase) ||
                                   string.Equals(letter, "no_hand", StringComparison.OrdinalIgnoreCase) ||
                                   string.Equals(letter, "no hand", StringComparison.OrdinalIgnoreCase) ||
                                   string.Equals(letter, "background", StringComparison.OrdinalIgnoreCase);

            if (isNoGestureLabel)
            {
                return new InferenceOutcome { Message = "Low Confidence...", ClearPredictionBuffer = true };
            }

            if (prediction.Data.Coordinates != null)
            {
                TryGetJpegDimensions(frameBytes, out var fWidth, out var fHeight);
                var usesNormalizedCoordinates = prediction.Data.Coordinates.Count > 0 &&
                                                prediction.Data.Coordinates.All(coord => coord.X is >= 0 and <= 1 && coord.Y is >= 0 and <= 1);

                foreach (var coord in prediction.Data.Coordinates)
                {
                    if (usesNormalizedCoordinates)
                    {
                        coord.X = 1.0 - coord.X;
                    }
                    else if (fWidth > 0)
                    {
                        coord.X = fWidth - coord.X;
                    }
                }
            }

            TryGetJpegDimensions(frameBytes, out var frameWidthResult, out var frameHeightResult);

            var result = new HandGestureResult
            {
                GestureLabel = letter,
                ConfidenceScore = prediction.Data.Confidence,
                ProcessingTimeMs = prediction.Data.ProcessingTimeMs,
                DetectedAt = DateTime.Now,
                Coordinates = prediction.Data.Coordinates,
                SourceFrameWidth = frameWidthResult,
                SourceFrameHeight = frameHeightResult
            };

            return new InferenceOutcome
            {
                Gesture = result,
                Message = "Waiting for sign...",
                ClearPredictionBuffer = false
            };
        }
        catch (OperationCanceledException)
        {
            return new InferenceOutcome { Message = "API Connection Error", ClearPredictionBuffer = true };
        }
        catch (Exception ex)
        {
            Debug.WriteLine($"Gesture inference error: {ex.Message}");
            return new InferenceOutcome { Message = "API Connection Error", ClearPredictionBuffer = true };
        }
    }

    private static bool TryGetJpegDimensions(byte[] data, out int width, out int height)
    {
        width = 0;
        height = 0;

        try
        {
            if (data.Length < 4 || data[0] != 0xFF || data[1] != 0xD8) return false;

            var index = 2;
            while (index + 8 < data.Length)
            {
                if (data[index] != 0xFF)
                {
                    index++;
                    continue;
                }

                var marker = data[index + 1];
                index += 2;

                if (marker == 0xD9 || marker == 0xDA) break;
                if (index + 1 >= data.Length) return false;

                var segmentLength = (data[index] << 8) | data[index + 1];
                if (segmentLength < 2 || index + segmentLength > data.Length) return false;

                var isStartOfFrame = marker is >= 0xC0 and <= 0xCF && marker is not (0xC4 or 0xC8 or 0xCC);
                if (isStartOfFrame)
                {
                    if (index + 7 >= data.Length) return false;

                    height = (data[index + 3] << 8) | data[index + 4];
                    width = (data[index + 5] << 8) | data[index + 6];
                    return width > 0 && height > 0;
                }

                index += segmentLength;
            }
        }
        catch (Exception ex)
        {
            Debug.WriteLine($"Failed to parse JPEG size: {ex.Message}");
        }

        return false;
    }

    private static bool IsNoHandText(string text)
    {
        return text.Contains("no hand", StringComparison.OrdinalIgnoreCase) ||
               text.Contains("no_hand", StringComparison.OrdinalIgnoreCase) ||
               text.Contains("not detect", StringComparison.OrdinalIgnoreCase) ||
               text.Contains("missing hand", StringComparison.OrdinalIgnoreCase);
    }
}
