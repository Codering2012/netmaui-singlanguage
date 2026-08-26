using System.Collections.ObjectModel;
using System.Diagnostics;
using System.IO;
using CommunityToolkit.Mvvm.ComponentModel;
using CommunityToolkit.Mvvm.Input;
using Microsoft.Maui.Controls;
using Microsoft.Maui.Graphics;
using SignLanguageApp.Controls;
using SignLanguageApp.Model;
using SignLanguageApp.Services;

namespace SignLanguageApp.ViewModels;

public partial class HandGestureResult
{
    public string GestureLabel { get; set; } = string.Empty;
    public float ConfidenceScore { get; set; }
    public double ProcessingTimeMs { get; set; }
    public DateTime DetectedAt { get; set; }
    public List<CoordinateDto>? Coordinates { get; set; }
    public int SourceFrameWidth { get; set; }
    public int SourceFrameHeight { get; set; }
    public List<CoordinateDto>? IndexTrail { get; set; }
    public List<CoordinateDto>? PinkyTrail { get; set; }
    public string TrackingLetter { get; set; } = string.Empty;
}

public partial class GestureFrame
{
    public string Label { get; set; } = string.Empty;
    public float Confidence { get; set; }
    public long Timestamp { get; set; }
}

#pragma warning disable MVVMTK0045 // Field using [ObservableProperty] not AOT compatible for WinRT
public partial class CameraTranslationViewModel : ObservableObject
{
    private const int FrameSendIntervalMs = 100;
    private const int MaxDetectionHistoryEntries = 200;
    private static readonly TimeSpan FrameCaptureTimeout = TimeSpan.FromSeconds(5);
    private const float LetterCommitConfidenceThreshold = 0.80f;
    private static readonly TimeSpan WordInactivityTimeout = TimeSpan.FromSeconds(3);

    // LOWERED CONFIDENCE/BUFFER: Changed from 5 to 3. 
    // The UI will now display the translation much faster.
    private const int PredictionBufferSize = 3;

    private readonly Queue<GestureFrame> _predictionBuffer = new();
    private readonly object _predictionSync = new();
    private readonly System.Text.StringBuilder _wordBuffer = new();
    private CancellationTokenSource? _processingCts;
    private readonly IApiService? _apiService;
    private CommunityToolkit.Maui.Views.CameraView? _cameraView;
    private readonly object _captureSync = new();
    private readonly SemaphoreSlim _inferenceLock = new(1, 1);
    private TaskCompletionSource<byte[]?>? _pendingCapture;
    private DateTime _lastInputAtUtc = DateTime.MinValue;
    private string _lastCommittedLetter = string.Empty;

    public ObservableCollection<HandGestureResult> DetectionHistory { get; } = [];

    [ObservableProperty]
    public partial bool IsCameraAvailable { get; set; }

    [ObservableProperty]
    public partial bool IsProcessingFrames { get; set; }

    [ObservableProperty]
    public partial string CurrentGestureLabel { get; set; } = string.Empty;

    [ObservableProperty]
    public partial float CurrentConfidenceScore { get; set; }

    [ObservableProperty]
    public partial double ProcessingTimeMs { get; set; }

    [ObservableProperty]
    public partial int FramesProcessed { get; set; }

    [ObservableProperty]
    public partial string FrameRate { get; set; } = "0 fps";

    [ObservableProperty]
    public partial bool IsRecordingSession { get; set; }

    [ObservableProperty]
    public partial string TranslatedText { get; set; } = string.Empty;

    [ObservableProperty]
    public partial string ConfidenceText { get; set; } = string.Empty;

    [ObservableProperty]
    public partial bool IsProcessing { get; set; }

    [ObservableProperty]
    public partial bool UseNpuAcceleration { get; set; }

    [ObservableProperty]
    public partial bool IsFrameFrozen { get; set; }

    [ObservableProperty]
    public partial IDrawable? Drawable { get; set; }

    public CameraTranslationViewModel() : this(null) { }

    public CameraTranslationViewModel(IApiService? apiService = null)
    {
        _apiService = apiService;
    }

    public void SetCameraView(CommunityToolkit.Maui.Views.CameraView cameraView)
    {
        if (_cameraView != null)
        {
            _cameraView.MediaCaptured -= OnMediaCaptured;
            _cameraView.MediaCaptureFailed -= OnMediaCaptureFailed;
        }

        _cameraView = cameraView;
        _cameraView.MediaCaptured += OnMediaCaptured;
        _cameraView.MediaCaptureFailed += OnMediaCaptureFailed;
    }

    [RelayCommand]
    public async Task StartCameraCapture()
    {
        if (IsProcessingFrames) return;
        _processingCts?.Dispose();
        ResetWordBuffer();
        IsProcessingFrames = true;
        IsProcessing = true;
        TranslatedText = "Waiting for sign...";
        ConfidenceText = string.Empty;
        _processingCts = new CancellationTokenSource();
        _ = Task.Run(() => ProcessFramesAsync(_processingCts.Token));
    }

    [RelayCommand]
    public Task StopCameraCapture()
    {
        IsProcessingFrames = false;
        IsProcessing = false;
        ResetWordBuffer();
        _processingCts?.Cancel();
        _processingCts?.Dispose();
        _processingCts = null;
        return Task.CompletedTask;
    }

    [RelayCommand]
    public async Task Stop()
    {
        await StopCameraCapture();
    }

    [RelayCommand]
    public Task FreezeFrame()
    {
        IsFrameFrozen = !IsFrameFrozen;
        return Task.CompletedTask;
    }

    [RelayCommand]
    public Task ToggleRecording()
    {
        IsRecordingSession = !IsRecordingSession;
        return Task.CompletedTask;
    }

    [RelayCommand]
    public Task ClearHistory()
    {
        DetectionHistory.Clear();
        FramesProcessed = 0;
        return Task.CompletedTask;
    }

    [RelayCommand]
    public async Task ExportDetectionResults()
    {
        var csv = new System.Text.StringBuilder();
        csv.AppendLine("Gesture,Confidence,Timestamp,ProcessingTimeMs");
        foreach (var r in DetectionHistory)
            csv.AppendLine($"{r.GestureLabel},{r.ConfidenceScore:F4},{r.DetectedAt:O},{r.ProcessingTimeMs:F2}");
        await Shell.Current.DisplayAlertAsync("Export", $"Exported {DetectionHistory.Count} gestures.", "OK");
    }

    private async Task ProcessFramesAsync(CancellationToken ct)
    {
        var fps = 0;
        var lastCount = 0;
        var sw = Stopwatch.StartNew();

        try
        {
            while (!ct.IsCancellationRequested && IsProcessingFrames)
            {
                var cycleStartedAt = Stopwatch.GetTimestamp();

                if (IsFrameFrozen)
                {
                    await Task.Delay(100, ct);
                    continue;
                }

                if (!await _inferenceLock.WaitAsync(0, ct))
                {
                    await Task.Delay(50, ct);
                    continue;
                }

                try
                {
                    var inference = await PerformGestureInferenceAsync(ct);
                    if (inference.Gesture != null && !ct.IsCancellationRequested)
                    {
                        var gesture = inference.Gesture;
                        var smoothedGesture = ApplyTemporalSmoothing(gesture);
                        await MainThread.InvokeOnMainThreadAsync(() =>
                        {
                            ProcessingTimeMs = gesture.ProcessingTimeMs;

                            if (smoothedGesture != null)
                            {
                                CurrentGestureLabel = smoothedGesture.GestureLabel;
                                CurrentConfidenceScore = smoothedGesture.ConfidenceScore;
                                FramesProcessed++;

                                RegisterInputActivity();
                                TryAppendLetterToWord(smoothedGesture);
                                TranslatedText = BuildTranslationText($"Sign: {smoothedGesture.GestureLabel}", smoothedGesture.GestureLabel);
                                ConfidenceText = $"{smoothedGesture.ConfidenceScore * 100:F1}%";
                            }
                            else
                            {
                                CurrentGestureLabel = string.Empty;
                                CurrentConfidenceScore = 0;
                                RegisterInputActivity();
                                TranslatedText = BuildTranslationText("Waiting for sign...", gesture.GestureLabel);
                                ConfidenceText = string.Empty;
                            }

                            if (gesture.Coordinates != null && gesture.Coordinates.Count > 0)
                            {
                                Drawable = new SkeletalDrawable(
                                    gesture.Coordinates,
                                    gesture.SourceFrameWidth,
                                    gesture.SourceFrameHeight);
                            }
                            else if (Drawable != null)
                            {
                                Drawable = null;
                            }

                            if (IsRecordingSession && smoothedGesture != null)
                            {
                                DetectionHistory.Add(smoothedGesture);
                                if (DetectionHistory.Count > MaxDetectionHistoryEntries)
                                {
                                    DetectionHistory.RemoveAt(0);
                                }
                            }
                        });
                    }
                    else if (!ct.IsCancellationRequested)
                    {
                        if (inference.ClearPredictionBuffer)
                        {
                            ClearPredictionBuffer();
                        }

                        await MainThread.InvokeOnMainThreadAsync(() =>
                        {
                            CurrentGestureLabel = string.Empty;
                            CurrentConfidenceScore = 0;
                            TranslatedText = BuildTranslationText(inference.Message);
                            ConfidenceText = string.Empty;
                            if (Drawable != null)
                            {
                                Drawable = null;
                            }
                        });
                    }
                }
                catch (Exception ex)
                {
                    Debug.WriteLine($"Frame processing error: {ex.Message}");
                }
                finally
                {
                    _inferenceLock.Release();
                }

                if (sw.ElapsedMilliseconds >= 1000)
                {
                    fps = FramesProcessed - lastCount;
                    lastCount = FramesProcessed;
                    await MainThread.InvokeOnMainThreadAsync(() => FrameRate = $"{fps} fps");
                    sw.Restart();
                }

                var cycleElapsedMs = (Stopwatch.GetTimestamp() - cycleStartedAt) * 1000.0 / Stopwatch.Frequency;
                var remainingDelayMs = FrameSendIntervalMs - (int)cycleElapsedMs;
                if (remainingDelayMs > 0)
                {
                    await Task.Delay(remainingDelayMs, ct);
                }
            }
        }
        catch (OperationCanceledException) { }
        finally
        {
            IsProcessingFrames = false;
            sw.Stop();
        }
    }

    private async Task<InferenceOutcome> PerformGestureInferenceAsync(CancellationToken ct)
    {
        try
        {
            var frameBytes = await CaptureFrameAsJpegAsync(ct);
            if (frameBytes == null || frameBytes.Length == 0)
            {
                return new InferenceOutcome
                {
                    Message = "No hand detected",
                    ClearPredictionBuffer = true
                };
            }

            if (_apiService == null)
            {
                return new InferenceOutcome
                {
                    Message = "API Connection Error",
                    ClearPredictionBuffer = true
                };
            }

            using var timeoutCts = CancellationTokenSource.CreateLinkedTokenSource(ct);
            timeoutCts.CancelAfter(TimeSpan.FromSeconds(5));

            var prediction = await _apiService.PredictGestureFromImageAsync(frameBytes, cancellationToken: timeoutCts.Token);

            if (prediction == null)
            {
                return new InferenceOutcome
                {
                    Message = "API Connection Error",
                    ClearPredictionBuffer = true
                };
            }

            var status = prediction.Status?.Trim() ?? string.Empty;

            if (!string.Equals(status, "success", StringComparison.OrdinalIgnoreCase) || prediction.Data == null)
            {
                var message = string.IsNullOrWhiteSpace(prediction.Message)
                    ? "Low Confidence..."
                    : prediction.Message;

                if (IsNoHandText(message))
                {
                    message = "No hand detected";
                }

                return new InferenceOutcome
                {
                    Message = message,
                    ClearPredictionBuffer = true
                };
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
                return new InferenceOutcome
                {
                    Message = "Low Confidence...",
                    ClearPredictionBuffer = true
                };
            }

            // CRITICAL FIX: MIRRORED COORDINATES
            // The front-facing camera acts as a mirror, but the API coordinates 
            // are unmirrored. We invert the X coordinate (1.0 - X) so the red dots 
            // perfectly align with the user's hand on the screen.
            if (prediction.Data.Coordinates != null)
            {
                var frameWidth = 0;
                var frameHeight = 0;
                TryGetJpegDimensions(frameBytes, out frameWidth, out frameHeight);

                var usesNormalizedCoordinates = prediction.Data.Coordinates.Count > 0 &&
                                                prediction.Data.Coordinates.All(coord =>
                                                    coord.X is >= 0 and <= 1 && coord.Y is >= 0 and <= 1);

                foreach (var coord in prediction.Data.Coordinates)
                {
                    if (usesNormalizedCoordinates)
                    {
                        coord.X = 1.0 - coord.X;
                    }
                    else if (frameWidth > 0)
                    {
                        coord.X = frameWidth - coord.X;
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
            return new InferenceOutcome
            {
                Message = "API Connection Error",
                ClearPredictionBuffer = true
            };
        }
        catch (Exception ex)
        {
            Debug.WriteLine($"Gesture inference error: {ex.Message}");
            return new InferenceOutcome
            {
                Message = "API Connection Error",
                ClearPredictionBuffer = true
            };
        }
    }

    private static bool TryGetJpegDimensions(byte[] data, out int width, out int height)
    {
        width = 0;
        height = 0;

        try
        {
            if (data.Length < 4 || data[0] != 0xFF || data[1] != 0xD8)
            {
                return false;
            }

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

                if (marker == 0xD9 || marker == 0xDA)
                {
                    break;
                }

                if (index + 1 >= data.Length)
                {
                    return false;
                }

                var segmentLength = (data[index] << 8) | data[index + 1];
                if (segmentLength < 2 || index + segmentLength > data.Length)
                {
                    return false;
                }

                var isStartOfFrame = marker is >= 0xC0 and <= 0xCF && marker is not (0xC4 or 0xC8 or 0xCC);
                if (isStartOfFrame)
                {
                    if (index + 7 >= data.Length)
                    {
                        return false;
                    }

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

    private async Task<byte[]?> CaptureFrameAsJpegAsync(CancellationToken ct)
    {
        try
        {
            if (_cameraView == null)
            {
                Debug.WriteLine("CameraView not initialized");
                return null;
            }

            var captureTcs = new TaskCompletionSource<byte[]?>(TaskCreationOptions.RunContinuationsAsynchronously);
            lock (_captureSync)
            {
                _pendingCapture?.TrySetCanceled();
                _pendingCapture = captureTcs;
            }

            Stream? mediaStream;
            if (MainThread.IsMainThread)
            {
                mediaStream = await _cameraView.CaptureImage(ct);
            }
            else
            {
                mediaStream = await MainThread.InvokeOnMainThreadAsync(() => _cameraView.CaptureImage(ct));
            }

            await using (mediaStream)
            {
                if (mediaStream == null)
                {
                    Debug.WriteLine("Camera capture returned no image stream");
                    return null;
                }

                if (mediaStream.CanSeek)
                {
                    mediaStream.Position = 0;
                }

                using var memory = new MemoryStream();
                await mediaStream.CopyToAsync(memory, ct);
                var bytes = memory.ToArray();
                if (bytes.Length > 0)
                {
                    return bytes;
                }
            }

            return await captureTcs.Task.WaitAsync(FrameCaptureTimeout, ct);
        }
        catch (OperationCanceledException)
        {
            Debug.WriteLine("Frame capture cancelled");
            return null;
        }
        catch (Exception ex)
        {
            Debug.WriteLine($"Error capturing frame: {ex.Message}");
            return null;
        }
        finally
        {
            lock (_captureSync)
            {
                _pendingCapture = null;
            }
        }
    }

    private async void OnMediaCaptured(object? sender, CommunityToolkit.Maui.Core.MediaCapturedEventArgs e)
    {
        try
        {
            TaskCompletionSource<byte[]?>? pendingCapture;
            lock (_captureSync)
            {
                pendingCapture = _pendingCapture;
            }

            if (pendingCapture == null)
            {
                return;
            }

            if (e.Media == null)
            {
                pendingCapture.TrySetResult(null);
                return;
            }

            using var memory = new MemoryStream();
            if (e.Media.CanSeek)
            {
                e.Media.Position = 0;
            }

            await e.Media.CopyToAsync(memory);
            pendingCapture.TrySetResult(memory.ToArray());
        }
        catch (Exception ex)
        {
            Debug.WriteLine($"Error handling captured media: {ex.Message}");
            lock (_captureSync)
            {
                _pendingCapture?.TrySetResult(null);
            }
        }
    }

    private void OnMediaCaptureFailed(object? sender, CommunityToolkit.Maui.Core.MediaCaptureFailedEventArgs e)
    {
        Debug.WriteLine($"Media capture failed: {e.FailureReason}");
        lock (_captureSync)
        {
            _pendingCapture?.TrySetResult(null);
        }
    }

    public void OnDisappearing()
    {
        _processingCts?.Cancel();
        _processingCts?.Dispose();
        IsProcessing = false;
    }

    private static bool IsNoHandText(string? text)
    {
        if (string.IsNullOrWhiteSpace(text))
        {
            return false;
        }

        var value = text.Trim();
        return value.Contains("no hand", StringComparison.OrdinalIgnoreCase)
               || value.Contains("not detected", StringComparison.OrdinalIgnoreCase)
               || value.Contains("nothing", StringComparison.OrdinalIgnoreCase)
               || value.Contains("none", StringComparison.OrdinalIgnoreCase)
               || value.Contains("background", StringComparison.OrdinalIgnoreCase);
    }

    private HandGestureResult? ApplyTemporalSmoothing(HandGestureResult current)
    {
        lock (_predictionSync)
        {
            _predictionBuffer.Enqueue(new GestureFrame
            {
                Label = current.GestureLabel,
                Confidence = current.ConfidenceScore,
                Timestamp = DateTimeOffset.UtcNow.ToUnixTimeMilliseconds()
            });

            while (_predictionBuffer.Count > PredictionBufferSize)
            {
                _predictionBuffer.Dequeue();
            }

            if (_predictionBuffer.Count == 0)
            {
                return null;
            }

            var grouped = new Dictionary<string, (int Count, float ConfidenceSum)>();
            foreach (var frame in _predictionBuffer)
            {
                if (grouped.TryGetValue(frame.Label, out var stats))
                {
                    grouped[frame.Label] = (stats.Count + 1, stats.ConfidenceSum + frame.Confidence);
                }
                else
                {
                    grouped[frame.Label] = (1, frame.Confidence);
                }
            }

            string? bestLabel = null;
            var bestCount = 0;
            var bestAverage = 0f;

            foreach (var pair in grouped)
            {
                var count = pair.Value.Count;
                var average = pair.Value.ConfidenceSum / count;
                if (count > bestCount || (count == bestCount && average > bestAverage))
                {
                    bestLabel = pair.Key;
                    bestCount = count;
                    bestAverage = average;
                }
            }

            if (bestLabel == null || bestCount <= PredictionBufferSize / 2)
            {
                return null;
            }

            return new HandGestureResult
            {
                GestureLabel = bestLabel,
                ConfidenceScore = bestAverage,
                ProcessingTimeMs = current.ProcessingTimeMs,
                DetectedAt = current.DetectedAt,
                Coordinates = current.Coordinates,
                SourceFrameWidth = current.SourceFrameWidth,
                SourceFrameHeight = current.SourceFrameHeight
            };
        }
    }

    private void ClearPredictionBuffer()
    {
        lock (_predictionSync)
        {
            _predictionBuffer.Clear();
        }
    }

    private void RegisterInputActivity()
    {
        _lastInputAtUtc = DateTime.UtcNow;
    }

    private void ResetWordBuffer()
    {
        _wordBuffer.Clear();
        _lastCommittedLetter = string.Empty;
        _lastInputAtUtc = DateTime.MinValue;
    }

    private bool TryAppendLetterToWord(HandGestureResult gesture)
    {
        var label = gesture.GestureLabel?.Trim() ?? string.Empty;
        if (label.Length != 1 || !char.IsLetter(label[0]))
        {
            return false;
        }

        if (gesture.ConfidenceScore < LetterCommitConfidenceThreshold)
        {
            return false;
        }

        var normalizedLetter = label.ToLowerInvariant();
        if (string.Equals(_lastCommittedLetter, normalizedLetter, StringComparison.Ordinal))
        {
            return false;
        }

        _wordBuffer.Append(normalizedLetter);
        _lastCommittedLetter = normalizedLetter;
        return true;
    }

    private string BuildTranslationText(string fallbackText, string? currentGestureLabel = null)
    {
        var now = DateTime.UtcNow;
        if (_wordBuffer.Length > 0 &&
            _lastInputAtUtc != DateTime.MinValue &&
            now - _lastInputAtUtc >= WordInactivityTimeout)
        {
            ResetWordBuffer();
        }

        if (_wordBuffer.Length > 0)
        {
            return $"Word: {_wordBuffer}";
        }

        if (!string.IsNullOrWhiteSpace(currentGestureLabel))
        {
            return $"Sign: {currentGestureLabel}";
        }

        return fallbackText;
    }

    private sealed class InferenceOutcome
    {
        public HandGestureResult? Gesture { get; init; }

        public string Message { get; init; } = "No hand detected";

        public bool ClearPredictionBuffer { get; init; } = true;
    }
}
#pragma warning restore MVVMTK0045