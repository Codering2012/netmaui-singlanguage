using System.Collections.ObjectModel;
using System.Linq;
using System.Diagnostics;
using CommunityToolkit.Mvvm.ComponentModel;
using CommunityToolkit.Mvvm.Input;
using SignLanguageApp.Model;
using SignLanguageApp.Services;
using SignLanguageApp.Controls;
using Microsoft.Maui.Graphics;
using System.Text.Json;
using SignLanguageApp.Helpers;

namespace SignLanguageApp.ViewModels;

public partial class InteractiveLessonViewModel : ObservableObject, IQueryAttributable
{
    private readonly IApiService _apiService;
    private readonly IGesturePredictionService _gestureService;
    private readonly IEnvironmentDetectionService _envService;
    private readonly IStudyService _studyService;
    private readonly IFrameBufferService _frameBuffer;
    private readonly IMediaDownloadAndCacheService _mediaCache;
    private CancellationTokenSource? _inferenceCts;

    private DateTime _stepStartTimeUtc = DateTime.MinValue;
    private int _consecutiveIncorrectCount = 0;
    private static readonly TimeSpan StepGracePeriod = TimeSpan.FromSeconds(2.5);

    private VideoProcessingCameraView? _cameraView;
    private readonly object _frameLock = new();
    private byte[]? _latestFrameBytes;
    private CancellationTokenSource? _processingCts;
    private bool _isProcessingLoopRunning;
    private bool _isFinishing;
    private bool _isAdvancing;

    public void SetCameraView(VideoProcessingCameraView cameraView)
    {
        if (_cameraView != null)
        {
            _cameraView.FrameReady -= OnFrameReady;
        }

        _cameraView = cameraView;
        if (_cameraView != null)
        {
            _cameraView.FrameReady += OnFrameReady;
        }
    }

    private void OnFrameReady(object? sender, byte[] frameBytes)
    {
        if (frameBytes != null && frameBytes.Length > 0)
        {
            lock (_frameLock)
            {
                _latestFrameBytes = frameBytes;
            }
        }
    }

    public void PushFallbackFrame(byte[] frameBytes)
    {
        if (frameBytes != null && frameBytes.Length > 0)
        {
            lock (_frameLock)
            {
                _latestFrameBytes = frameBytes;
            }
        }
    }

    public void Cleanup()
    {
        if (_cameraView != null)
        {
            _cameraView.FrameReady -= OnFrameReady;
            _cameraView = null;
        }
        
        var procCts = _processingCts;
        _processingCts = null;
        if (procCts != null)
        {
            try { procCts.Cancel(); } catch { }
            try { procCts.Dispose(); } catch { }
        }

        var infCts = _inferenceCts;
        _inferenceCts = null;
        if (infCts != null)
        {
            try { infCts.Cancel(); } catch { }
            try { infCts.Dispose(); } catch { }
        }
        
        _isProcessingLoopRunning = false;
        Drawable = null;
    }

    partial void OnIsCameraActiveChanged(bool value)
    {
        if (value)
        {
            StartFrameProcessingLoop();
        }
        else
        {
            StopFrameProcessingLoop();
        }
    }

    private void StartFrameProcessingLoop()
    {
        if (_isProcessingLoopRunning) return;
        _processingCts?.Dispose();
        _processingCts = new CancellationTokenSource();
        
        _inferenceCts?.Dispose();
        _inferenceCts = new CancellationTokenSource();

        _isProcessingLoopRunning = true;
        _ = ProcessFramesAsync(_processingCts.Token);
    }

    private void StopFrameProcessingLoop()
    {
        _isProcessingLoopRunning = false;
        try { _processingCts?.Cancel(); } catch { }
        Drawable = null;
    }

    private async Task ProcessFramesAsync(CancellationToken ct)
    {
        try
        {
            while (!ct.IsCancellationRequested && _isProcessingLoopRunning && IsCameraActive)
            {
                if (IsProcessing)
                {
                    await Task.Delay(50, ct);
                    continue;
                }

                byte[]? frameBytes = null;
                lock (_frameLock)
                {
                    frameBytes = _latestFrameBytes;
                    _latestFrameBytes = null;
                }

                if (frameBytes == null)
                {
                    await Task.Delay(20, ct); // Wait for the next frame
                    continue;
                }

                try
                {
                    await ProcessCameraFrameAsync(frameBytes);
                }
                catch (Exception ex)
                {
                    Debug.WriteLine($"Error in frame processing: {ex.Message}");
                }

                // Throttle frame processing to max ~30 FPS for desktop and LAN stability
                await Task.Delay(33, ct);
            }
        }
        catch (OperationCanceledException) { }
        catch (Exception ex)
        {
            Debug.WriteLine($"ProcessFramesAsync error: {ex.Message}");
        }
        finally
        {
            _isProcessingLoopRunning = false;
        }
    }

    [ObservableProperty]
    public partial int LessonId { get; set; }
    
    [ObservableProperty]
    public partial string LessonTitle { get; set; }

    [ObservableProperty]
    [NotifyPropertyChangedFor(nameof(IsEmpty))]
    public partial InteractiveLessonDto? Lesson { get; set; }

    [ObservableProperty]
    public partial int CurrentStepIndex { get; set; }

    [ObservableProperty]
    public partial LessonStepDto? CurrentStep { get; set; }

    [ObservableProperty]
    public partial double Progress { get; set; }

    [ObservableProperty]
    public partial bool IsCameraActive { get; set; }

    [ObservableProperty]
    public partial bool ShowCorrectFeedback { get; set; }

    [ObservableProperty]
    public partial bool ShowIncorrectFeedback { get; set; }

    [ObservableProperty]
    public partial string FeedbackMessage { get; set; }

    [ObservableProperty]
    public partial bool IsProcessing { get; set; }

    [ObservableProperty]
    [NotifyPropertyChangedFor(nameof(IsEmpty))]
    public partial bool IsBusy { get; set; }

    [ObservableProperty]
    public partial string EnvironmentWarning { get; set; }

    [ObservableProperty]
    public partial bool ShowEnvironmentWarning { get; set; }
    
    [ObservableProperty]
    public partial ObservableCollection<CoordinateDto> CurrentTrail { get; set; }

    [ObservableProperty]
    public partial IDrawable? Drawable { get; set; }

    [ObservableProperty]
    [NotifyPropertyChangedFor(nameof(IsEmpty))]
    public partial bool HasError { get; set; }

    [ObservableProperty]
    public partial string ErrorMessage { get; set; }

    // Gamification & New Learning Modes Properties
    [ObservableProperty]
    public partial ObservableCollection<MatchingCardTile> MatchingCards { get; set; }

    [ObservableProperty]
    public partial ObservableCollection<string> AvailableSequenceTokens { get; set; }

    [ObservableProperty]
    public partial ObservableCollection<string> UserSequenceTokens { get; set; }

    [ObservableProperty]
    public partial int ComboStreak { get; set; }

    [ObservableProperty]
    public partial double ComboMultiplier { get; set; }

    [ObservableProperty]
    public partial bool IsVideoTutorialVisible { get; set; }

    [ObservableProperty]
    public partial string PlaybackSpeedText { get; set; }

    private MatchingCardTile? _firstSelectedCard;

    /// <summary>True when the lesson loaded but has no steps (blank-screen guard).</summary>
    public bool IsEmpty => !IsBusy && (Lesson == null || Lesson.Steps == null || !Lesson.Steps.Any());

    public InteractiveLessonViewModel(IApiService apiService, IGesturePredictionService gestureService, IEnvironmentDetectionService envService, IStudyService studyService, IFrameBufferService frameBuffer, IMediaDownloadAndCacheService mediaCache)
    {
        CurrentTrail = [];
        MatchingCards = [];
        AvailableSequenceTokens = [];
        UserSequenceTokens = [];
        ComboStreak = 0;
        ComboMultiplier = 1.0;
        IsVideoTutorialVisible = false;
        PlaybackSpeedText = "1.0x";
        _apiService = apiService;
        _gestureService = gestureService;
        _envService = envService;
        _studyService = studyService;
        _frameBuffer = frameBuffer;
        _mediaCache = mediaCache;
        
        LessonTitle = "Lesson";
        FeedbackMessage = string.Empty;
        EnvironmentWarning = string.Empty;
        ErrorMessage = string.Empty;
    }

    public void ApplyQueryAttributes(IDictionary<string, object> query)
    {
        _isFinishing = false;
        _isAdvancing = false;
        if ((query.TryGetValue("lessonId", out var lessonIdObj) || query.TryGetValue("LessonId", out lessonIdObj)) && lessonIdObj != null)
        {
            if (int.TryParse(lessonIdObj.ToString(), out var lessonId))
            {
                LessonId = lessonId;
                _ = InitializeAsync();
            }
        }
    }

    public async Task InitializeAsync()
    {
        if (IsBusy || LessonId == 0) return;
        IsBusy = true;
        HasError = false;
        ErrorMessage = string.Empty;

        try
        {
            var response = await _apiService.GetInteractiveLessonAsync(LessonId);
            if (response?.Data != null && response.Data.Steps?.Any() == true)
            {
                Lesson = response.Data;
                LessonTitle = Lesson.Title;
                CurrentStepIndex = 0;
                UpdateCurrentStep();
            }
            else if (response?.Data != null)
            {
                // Lesson loaded but has no steps
                Lesson = response.Data;
                LessonTitle = response.Data.Title;
                HasError = true;
                ErrorMessage = "This lesson has no steps yet. Please try a different lesson.";
            }
            else
            {
                var localFile = Path.Combine(FileSystem.AppDataDirectory, "lessons", $"lesson_{LessonId}.json");
                if (File.Exists(localFile))
                {
                    var localJson = await File.ReadAllTextAsync(localFile);
                    var localLesson = System.Text.Json.JsonSerializer.Deserialize(localJson, AppJsonContext.Default.InteractiveLessonDto);
                    if (localLesson?.Steps?.Any() == true)
                    {
                        Lesson = localLesson;
                        LessonTitle = Lesson.Title;
                        CurrentStepIndex = 0;
                        UpdateCurrentStep();
                        return;
                    }
                }

                HasError = true;
                ErrorMessage = "Could not load lesson from API server. Content is loaded strictly from the Sign Language API.";
            }
        }
        catch (Exception ex)
        {
            Debug.WriteLine($"Error loading interactive lesson: {ex.Message}");
            Lesson = new InteractiveLessonDto
            {
                Id = LessonId,
                Title = $"ASL Practice Lesson {LessonId}",
                Steps = new List<LessonStepDto>
                {
                    new LessonStepDto
                    {
                        Type = LessonStepType.Flashcard,
                        Title = "Alphabet A",
                        Description = "Make a fist with your thumb resting against the side of your index finger."
                    },
                    new LessonStepDto
                    {
                        Type = LessonStepType.CameraPractice,
                        Title = "Practice Sign 'A'",
                        Description = "Show the letter 'A' gesture to your camera.",
                        TargetGesture = "A"
                    }
                }
            };
            LessonTitle = Lesson.Title;
            CurrentStepIndex = 0;
            UpdateCurrentStep();
        }
        finally
        {
            IsBusy = false;
        }
    }

    private void UpdateCurrentStep()
    {
        if (Lesson?.Steps == null || CurrentStepIndex < 0 || CurrentStepIndex >= Lesson.Steps.Count)
            return;

        var nextStep = Lesson.Steps[CurrentStepIndex];
        
        SignLanguageApp.Helpers.MainThreadHelper.SafeInvokeOnMainThread(() =>
        {
            CurrentStep = nextStep;
            if (CurrentStep == null) return;
            
            if (!string.IsNullOrEmpty(CurrentStep.ImageUrl))
            {
                var absoluteUrl = _apiService.EnsureAbsoluteUrl(CurrentStep.ImageUrl);
                CurrentStep.ImageUrl = absoluteUrl;
                
                // Fire and forget caching for UI to pick up when ready
                _ = Task.Run(async () =>
                {
                    var localPath = await _mediaCache.GetCachedMediaAsync(absoluteUrl);
                    if (!string.IsNullOrEmpty(localPath))
                    {
                        MainThread.BeginInvokeOnMainThread(() =>
                        {
                            CurrentStep.ImageUrl = localPath;
                            OnPropertyChanged(nameof(CurrentStep));
                        });
                    }
                });
            }
            
            Progress = (double)(CurrentStepIndex + 1) / Lesson.Steps.Count;
            IsCameraActive = CurrentStep.Type == LessonStepType.CameraPractice;

            // Setup MatchingCard mode tiles
            MatchingCards.Clear();
            _firstSelectedCard = null;
            if (CurrentStep.Type == LessonStepType.MatchingCard && CurrentStep.MatchingPairs != null)
            {
                var tiles = new List<MatchingCardTile>();
                foreach (var pair in CurrentStep.MatchingPairs)
                {
                    tiles.Add(new MatchingCardTile { PairId = pair.Id, DisplayText = pair.SignTitle, ImageUrl = _apiService.EnsureAbsoluteUrl(pair.ImageUrl), IsSign = true });
                    tiles.Add(new MatchingCardTile { PairId = pair.Id, DisplayText = pair.TextTranslation, IsSign = false });
                }
                var rnd = new Random();
                foreach (var tile in tiles.OrderBy(_ => rnd.Next()))
                {
                    MatchingCards.Add(tile);
                }
            }

            // Setup SentenceSequence mode tokens
            AvailableSequenceTokens.Clear();
            UserSequenceTokens.Clear();
            if (CurrentStep.Type == LessonStepType.SentenceSequence && CurrentStep.SequenceTokens != null)
            {
                var rnd = new Random();
                foreach (var token in CurrentStep.SequenceTokens.OrderBy(_ => rnd.Next()))
                {
                    AvailableSequenceTokens.Add(token);
                }
            }

            // Reset feedback state & camera step timers
            _stepStartTimeUtc = DateTime.UtcNow;
            _consecutiveIncorrectCount = 0;
            ShowCorrectFeedback = false;
            ShowIncorrectFeedback = false;
            FeedbackMessage = string.Empty;
        });
    }

    [RelayCommand]
    private async Task CheckAnswer(string selectedOption)
    {
        if (CurrentStep == null || IsProcessing) return;

        IsProcessing = true;
        try
        {
            bool isCorrect = string.Equals(selectedOption, CurrentStep.CorrectOption, StringComparison.OrdinalIgnoreCase);

            if (isCorrect)
            {
                SignLanguageApp.Helpers.MainThreadHelper.SafeInvokeOnMainThread(() =>
                {
                    ShowCorrectFeedback = true;
                    ShowIncorrectFeedback = false;
                    FeedbackMessage = "Correct! Well done.";
                });
                
                await Task.Delay(2000);
                await NextStep();
            }
            else
            {
                SignLanguageApp.Helpers.MainThreadHelper.SafeInvokeOnMainThread(() =>
                {
                    ShowCorrectFeedback = false;
                    ShowIncorrectFeedback = true;
                    FeedbackMessage = "Incorrect. Try again!";
                });
                
                await Task.Delay(2000);
                SignLanguageApp.Helpers.MainThreadHelper.SafeInvokeOnMainThread(() => ShowIncorrectFeedback = false);
            }
        }
        finally
        {
            IsProcessing = false;
        }
    }

    [RelayCommand]
    private async Task NextStep()
    {
        if (_isFinishing || _isAdvancing) return;
        _isAdvancing = true;

        try
        {
            if (Lesson == null || CurrentStepIndex >= Lesson.Steps.Count - 1)
            {
                // Finish lesson
                await FinishLessonAsync();
                return;
            }

            CurrentStepIndex++;
            UpdateCurrentStep();
        }
        finally
        {
            _isAdvancing = false;
        }
    }

    [RelayCommand]
    private void PreviousStep()
    {
        if (CurrentStepIndex <= 0) return;

        CurrentStepIndex--;
        UpdateCurrentStep();
    }

    private async Task FinishLessonAsync()
    {
        if (_isFinishing) return;
        _isFinishing = true;

        try { _inferenceCts?.Cancel(); } catch { } // Stop background processing
        try { _processingCts?.Cancel(); } catch { }
        IsCameraActive = false; // Stop camera preview/processing loop immediately

        try
        {
            await _apiService.MarkLessonCompleteAsync(LessonId);
        }
        catch (Exception ex)
        {
            Debug.WriteLine($"Error marking lesson complete: {ex.Message}");
        }

        await SignLanguageApp.Helpers.MainThreadHelper.SafeInvokeOnMainThreadAsync(async () =>
        {
            if (Shell.Current != null)
            {
                await Helpers.NavigationHelper.SafeNavigateAsync("..");
            }
        });
    }

    public async Task ProcessCameraFrameAsync(byte[] frameBytes)
    {
        if (!IsCameraActive || IsProcessing || CurrentStep == null || string.IsNullOrEmpty(CurrentStep.TargetGesture))
            return;

        IsProcessing = true;

        var infCts = _inferenceCts;
        bool needNew = true;
        if (infCts != null)
        {
            try 
            { 
                needNew = infCts.IsCancellationRequested; 
            }
            catch (ObjectDisposedException) 
            { 
                needNew = true; 
            }
        }

        // Ensure we have a active, non-cancelled token
        if (needNew)
        {
            if (infCts != null)
            {
                try { infCts.Dispose(); } catch { }
            }
            _inferenceCts = new CancellationTokenSource();
        }

        var ctsToken = _inferenceCts?.Token ?? CancellationToken.None;

        try
        {
            var targetGesture = CurrentStep?.TargetGesture ?? string.Empty;
            var result = await _gestureService.PerformGestureInferenceAsync(frameBytes, targetGesture, ctsToken);
            
            if (result?.Gesture != null)
            {
                SignLanguageApp.Helpers.MainThreadHelper.SafeInvokeOnMainThread(() =>
                {
                    CurrentTrail.Clear();
                    if (result.Gesture.IndexTrail != null)
                    {
                        foreach (var coord in result.Gesture.IndexTrail)
                            CurrentTrail.Add(coord);
                    }
                    if (result.Gesture.PinkyTrail != null)
                    {
                        foreach (var coord in result.Gesture.PinkyTrail)
                            CurrentTrail.Add(coord);
                    }

                    if (result.Gesture.Coordinates != null && result.Gesture.Coordinates.Count > 0)
                    {
                        var drawable = new SkeletalDrawable(
                            result.Gesture.Coordinates,
                            result.Gesture.SourceFrameWidth,
                            result.Gesture.SourceFrameHeight);
                        
                        drawable.IndexTrail = result.Gesture.IndexTrail ?? new List<CoordinateDto>();
                        drawable.PinkyTrail = result.Gesture.PinkyTrail ?? new List<CoordinateDto>();
                        drawable.TrackingLetter = result.Gesture.TrackingLetter;
                        drawable.IsMirrored = true;
                        Drawable = drawable;
                        Debug.WriteLine($">>> GESTURE DRAWN: {result.Gesture.GestureLabel} Trails: I={result.Gesture.IndexTrail?.Count} P={result.Gesture.PinkyTrail?.Count}");
                    }
                    else
                    {
                        Drawable = null;
                        Debug.WriteLine(">>> NO GESTURE DETECTED - DRAWABLE NULL");
                    }
                });

                // The backend handles the validation logic and returns IsCorrect
                // We can also double check here if needed.
                bool isCorrect = string.Equals(result.Gesture.GestureLabel, targetGesture, StringComparison.OrdinalIgnoreCase);

                if (isCorrect)
                {
                    _consecutiveIncorrectCount = 0;
                    SignLanguageApp.Helpers.MainThreadHelper.SafeInvokeOnMainThread(() =>
                    {
                        ShowCorrectFeedback = true;
                        ShowIncorrectFeedback = false;
                        FeedbackMessage = "Correct! Well done.";
                    });
                    
                    // SRS Update
                    if (!string.IsNullOrEmpty(CurrentStep?.TargetGesture))
                    {
                        await _studyService.UpdateSignPerformanceAsync(CurrentStep.TargetGesture, true, result.Gesture.ConfidenceScore);
                    }

                    // Auto-advance after a short delay if it's correct
                    await Task.Delay(2000);
                    if (ShowCorrectFeedback) // Check if still on the same step
                    {
                        await NextStep();
                    }
                }
                else
                {
                    // Ignore transient wrong signs during the initial 2.5-second camera warmup/positioning grace period
                    var timeInStep = DateTime.UtcNow - _stepStartTimeUtc;
                    if (timeInStep < StepGracePeriod)
                    {
                        _consecutiveIncorrectCount = 0;
                    }
                    else if (result.Gesture.ConfidenceScore > 0.65f)
                    {
                        // Require at least 8 consecutive high-confidence incorrect frames (~1s sustained wrong gesture)
                        _consecutiveIncorrectCount++;
                        if (_consecutiveIncorrectCount >= 8)
                        {
                            _consecutiveIncorrectCount = 0;

                            SignLanguageApp.Helpers.MainThreadHelper.SafeInvokeOnMainThread(() =>
                            {
                                ShowCorrectFeedback = false;
                                ShowIncorrectFeedback = true;
                                FeedbackMessage = $"Incorrect. You signed '{result.Gesture.GestureLabel}'. Try again!";
                            });

                            // SRS Update for incorrect attempt
                            if (!string.IsNullOrEmpty(CurrentStep?.TargetGesture))
                            {
                                await _studyService.UpdateSignPerformanceAsync(CurrentStep.TargetGesture, false, result.Gesture.ConfidenceScore);
                            }

                            // Mistake Replay Navigation
                            var lastFrame = _frameBuffer.GetLastNSecondsOfFrames(3);
                            if (lastFrame != null)
                            {
                                var navigationParameter = new Dictionary<string, object>
                                {
                                    { "targetGesture", CurrentStep?.TargetGesture ?? string.Empty },
                                    { "userFrame", lastFrame }
                                };
                                await SignLanguageApp.Helpers.MainThreadHelper.SafeInvokeOnMainThreadAsync(async () =>
                                {
                                    if (Shell.Current != null)
                                    {
                                        await Helpers.NavigationHelper.SafeNavigateAsync("MistakeReplayPage", navigationParameter);
                                    }
                                });
                            }
                        }
                    }
                    else
                    {
                        _consecutiveIncorrectCount = 0;
                    }
                }
            }

            // Environment Detection & Frame Buffering
            _frameBuffer.AddFrame(frameBytes);
            var envStatus = _envService.CheckEnvironment(frameBytes);
            SignLanguageApp.Helpers.MainThreadHelper.SafeInvokeOnMainThread(() => 
            {
                EnvironmentWarning = envStatus.WarningMessage;
                ShowEnvironmentWarning = !string.IsNullOrEmpty(EnvironmentWarning);
            });
        }
        catch (Exception ex)
        {
            Debug.WriteLine($"Error processing camera frame: {ex.Message}");
        }
        finally
        {
            IsProcessing = false;
        }
    }

    [RelayCommand]
    private async Task CloseLesson()
    {
        _inferenceCts?.Cancel(); // Stop background processing
        _processingCts?.Cancel();
        await Helpers.NavigationHelper.SafeNavigateAsync("..");
    }

    [RelayCommand]
    private async Task SelectMatchingCard(MatchingCardTile? card)
    {
        if (card == null || card.IsMatched || card.IsSelected || IsProcessing) return;

        card.IsSelected = true;

        if (_firstSelectedCard == null)
        {
            _firstSelectedCard = card;
            return;
        }

        // Second card selected
        IsProcessing = true;
        var first = _firstSelectedCard;
        _firstSelectedCard = null;

        if (first.PairId == card.PairId && first.IsSign != card.IsSign)
        {
            // MATCH SUCCESS
            first.IsMatched = true;
            card.IsMatched = true;
            ComboStreak++;
            ComboMultiplier = Math.Min(3.0, 1.0 + (ComboStreak * 0.5));
            FeedbackMessage = $"Match! Combo x{ComboMultiplier:F1} 🔥";
            ShowCorrectFeedback = true;

            await Task.Delay(800);
            ShowCorrectFeedback = false;

            if (MatchingCards.All(c => c.IsMatched))
            {
                FeedbackMessage = "All Pairs Matched! Great job!";
                ShowCorrectFeedback = true;
                await Task.Delay(1500);
                await NextStep();
            }
        }
        else
        {
            // MATCH FAIL
            ComboStreak = 0;
            ComboMultiplier = 1.0;
            FeedbackMessage = "Try again!";
            ShowIncorrectFeedback = true;

            await Task.Delay(1000);
            first.IsSelected = false;
            card.IsSelected = false;
            ShowIncorrectFeedback = false;
        }

        IsProcessing = false;
    }

    [RelayCommand]
    private void AddSequenceToken(string? token)
    {
        if (string.IsNullOrWhiteSpace(token)) return;
        AvailableSequenceTokens.Remove(token);
        UserSequenceTokens.Add(token);
    }

    [RelayCommand]
    private void RemoveSequenceToken(string? token)
    {
        if (string.IsNullOrWhiteSpace(token)) return;
        UserSequenceTokens.Remove(token);
        AvailableSequenceTokens.Add(token);
    }

    [RelayCommand]
    private async Task CheckSequence()
    {
        if (CurrentStep == null || IsProcessing) return;
        IsProcessing = true;

        try
        {
            var userBuiltSentence = string.Join(" ", UserSequenceTokens);
            bool isCorrect = string.Equals(userBuiltSentence.Trim(), CurrentStep.TargetSentence?.Trim(), StringComparison.OrdinalIgnoreCase);

            if (isCorrect)
            {
                ComboStreak++;
                ComboMultiplier = Math.Min(3.0, 1.0 + (ComboStreak * 0.5));
                ShowCorrectFeedback = true;
                FeedbackMessage = $"Perfect Sentence! Combo x{ComboMultiplier:F1} 🔥";
                await Task.Delay(1800);
                await NextStep();
            }
            else
            {
                ComboStreak = 0;
                ComboMultiplier = 1.0;
                ShowIncorrectFeedback = true;
                FeedbackMessage = "Incorrect order. Try rearranging the words!";
                await Task.Delay(1800);
                ShowIncorrectFeedback = false;
            }
        }
        finally
        {
            IsProcessing = false;
        }
    }

    [RelayCommand]
    private void ToggleVideoTutorial()
    {
        IsVideoTutorialVisible = !IsVideoTutorialVisible;
    }

    [RelayCommand]
    private void TogglePlaybackSpeed()
    {
        PlaybackSpeedText = PlaybackSpeedText switch
        {
            "1.0x" => "0.75x",
            "0.75x" => "0.5x",
            "0.5x" => "1.0x",
            _ => "1.0x"
        };
    }

    [RelayCommand]
    private async Task OpenSignerCredits()
    {
        await Helpers.NavigationHelper.SafeNavigateAsync("credits");
    }
}

public partial class MatchingCardTile : ObservableObject
{
    public int PairId { get; set; }
    public string DisplayText { get; set; } = string.Empty;
    public string? ImageUrl { get; set; }
    public bool IsSign { get; set; }

    [ObservableProperty]
    public partial bool IsSelected { get; set; }

    [ObservableProperty]
    public partial bool IsMatched { get; set; }
}





