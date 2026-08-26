using CommunityToolkit.Maui.Views;
using CommunityToolkit.Maui.Core;
using SignLanguageApp.ViewModels;
using SignLanguageApp.Helpers;
using System.Diagnostics;
using System.IO;
using System.Threading;
using System.Linq;

using Microsoft.Extensions.DependencyInjection;

namespace SignLanguageApp.Pages;

public partial class InteractiveLessonPage : ContentPage
{
    private readonly InteractiveLessonViewModel _viewModel;
#if !ANDROID
    private bool _isDisposed;
    private bool _isPreviewRunning;
    private bool _isFallbackLoopRunning;
#endif


    public InteractiveLessonPage(InteractiveLessonViewModel viewModel)
    {
        InitializeComponent();
        BindingContext = _viewModel = viewModel;
        _viewModel.SetCameraView(CameraViewAndroid);

        _viewModel.PropertyChanged += OnViewModelPropertyChanged;
    }

    private void OnViewModelPropertyChanged(object? sender, System.ComponentModel.PropertyChangedEventArgs e)
    {
        var correctBorder = this.FindByName<Border>("CorrectFeedbackBorder");
        var incorrectBorder = this.FindByName<Border>("IncorrectFeedbackBorder");
        var optionsLayout = this.FindByName<Layout>("OptionsLayout");

        if (e.PropertyName == nameof(InteractiveLessonViewModel.ShowCorrectFeedback) && _viewModel.ShowCorrectFeedback)
        {
            if (correctBorder != null)
            {
                correctBorder.TranslationX = 0;
                correctBorder.TranslationY = 0;
                correctBorder.IsVisible = true;
                var successColor = Application.Current?.Resources.TryGetValue("SuccessColor", out var sc) == true ? (Color)sc : Colors.Green;
                
                _ = Task.WhenAll(
                    correctBorder.AnimateColorTransition(Colors.Black, successColor, c => correctBorder.BackgroundColor = c, 450, Easing.CubicOut),
                    correctBorder.FadeToAsync(1, 450, Easing.CubicOut)
                );
            }
            
            var confetti = this.FindByName<Controls.ConfettiView>("ConfettiOverlay");
            if (confetti != null)
            {
                confetti.IsVisible = true;
                confetti.StartConfetti();
            }
        }
        else if (e.PropertyName == nameof(InteractiveLessonViewModel.ShowIncorrectFeedback) && _viewModel.ShowIncorrectFeedback)
        {
            if (incorrectBorder != null)
            {
                incorrectBorder.TranslationX = 0;
                incorrectBorder.TranslationY = 0;
                incorrectBorder.IsVisible = true;
                var errorColor = Application.Current?.Resources.TryGetValue("ErrorColor", out var ec) == true ? (Color)ec : Colors.Red;

                _ = Task.WhenAll(
                    incorrectBorder.AnimateColorTransition(Colors.Black, errorColor, c => incorrectBorder.BackgroundColor = c, 450, Easing.CubicOut),
                    incorrectBorder.FadeToAsync(1, 450, Easing.CubicOut)
                );
            }
        }
        else if (e.PropertyName == nameof(InteractiveLessonViewModel.CurrentStep))
        {
            if (_viewModel.CurrentStep?.Options != null && _viewModel.CurrentStep.Options.Any())
            {
                MainThread.BeginInvokeOnMainThread(async () =>
                {
                    try
                    {
                        await Task.Delay(100); // Wait for BindableLayout to populate children
                        if (optionsLayout != null && optionsLayout.Children.Count > 0)
                        {
                            var children = optionsLayout.Children.OfType<VisualElement>().ToList();
                            for (int i = 0; i < children.Count; i++)
                            {
                                var child = children[i];
                                child.TranslationX = 0;
                                child.TranslationY = 20;
                                child.Opacity = 0;
                                
                                _ = Task.WhenAll(
                                    child.FadeToAsync(1, 400, Easing.CubicOut),
                                    child.TranslateToAsync(0, 0, 400, Easing.CubicOut)
                                );
                                await Task.Delay(50);
                            }
                        }
                    }
                    catch (Exception ex)
                    {
                        Debug.WriteLine($"Error in animation loop: {ex.Message}");
                    }
                });
            }
        }
    }

    protected override async void OnAppearing()
    {
        base.OnAppearing();
        try
        {
#if !ANDROID
            _isDisposed = false;
            _isFallbackLoopRunning = false;
#endif
            await _viewModel.InitializeAsync();
            
            await StartPreviewAsync();
        }
        catch (Exception ex)
        {
            Debug.WriteLine($"Error starting preview in InteractiveLesson: {ex.Message}");
            await DisplayAlertAsync("Page Error", $"An error occurred loading the interactive lesson:\n{ex.Message}", "OK");
        }
    }

    private async Task StartPreviewAsync()
    {
#if !ANDROID
        _isPreviewRunning = false;
        if (CameraViewDefault != null)
        {
            int retryCount = 0;
            while (CameraViewDefault.Handler == null && retryCount < 10)
            {
                await Task.Delay(200);
                retryCount++;
            }

            if (CameraViewDefault.Handler != null)
            {
                try
                {
                    var cameras = await CameraViewDefault.GetAvailableCameras(CancellationToken.None);
                    var camera = cameras.FirstOrDefault(c => c.Position == CameraPosition.Front) 
                                 ?? cameras.FirstOrDefault();

                    if (camera != null)
                    {
                        CameraViewDefault.SelectedCamera = camera;
                        await CameraViewDefault.StartCameraPreview(CancellationToken.None);
                        _isPreviewRunning = true;
                    }
                }
                catch (Exception ex)
                {
                    Debug.WriteLine($"Failed to start fallback camera preview: {ex.Message}");
                    _isPreviewRunning = false;
                }
            }
        }

        if (_isPreviewRunning && !_isFallbackLoopRunning)
        {
            _isFallbackLoopRunning = true;
            StartFallbackCameraLoop();
        }
#endif
    }

#if !ANDROID
    private async void StartFallbackCameraLoop()
    {
        while (!_isDisposed)
        {
            if (_viewModel.IsCameraActive && _isPreviewRunning && CameraViewDefault != null)
            {
                try
                {
                    var stream = await MainThread.InvokeOnMainThreadAsync(() => CameraViewDefault.CaptureImage(CancellationToken.None));
                    if (stream != null)
                    {
                        using var ms = new MemoryStream();
                        await stream.CopyToAsync(ms);
                        var frameBytes = ms.ToArray();
                        _viewModel.PushFallbackFrame(frameBytes);
                    }
                }
                catch (Exception ex)
                {
                    Debug.WriteLine($"Fallback camera loop error: {ex.Message}");
                }
                await Task.Delay(33); // Real-time capture fallback for desktop
            }
            else
            {
                await Task.Delay(500);
            }
        }
    }
#endif

    protected override void OnDisappearing()
    {
        base.OnDisappearing();
        _viewModel.PropertyChanged -= OnViewModelPropertyChanged;
#if !ANDROID
        _isDisposed = true;
        _isPreviewRunning = false;
#endif
        _viewModel.Cleanup();
#if !ANDROID
        try
        {
            if (CameraViewDefault != null)
            {
                CameraViewDefault.StopCameraPreview();
            }
        }
        catch (Exception ex)
        {
            Debug.WriteLine($"Error stopping camera in InteractiveLesson: {ex.Message}");
        }
#endif
    }
}
