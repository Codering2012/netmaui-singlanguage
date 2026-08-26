using SignLanguageApp.ViewModels;

namespace SignLanguageApp.Pages;

public partial class DifficultyCalibrationPage : ContentPage
{
    private readonly DifficultyCalibrationViewModel _viewModel;
    private bool _isCapturing;

    public DifficultyCalibrationPage(DifficultyCalibrationViewModel viewModel)
    {
        InitializeComponent();
        _viewModel = viewModel;
        BindingContext = _viewModel;
    }

    protected override async void OnAppearing()
    {
        base.OnAppearing();
            this.Opacity = 0;
            this.TranslationY = 20;
            MainThread.BeginInvokeOnMainThread(async () => {
            await Task.Delay(100);
            await Task.WhenAll(
                this.FadeToAsync(1, 400, Easing.CubicOut),
                this.TranslateToAsync(0, 0, 400, Easing.CubicOut)
            );
        });
        await StartCameraAsync();
    }

    private async Task StartCameraAsync()
    {
        try
        {
            var status = await Permissions.RequestAsync<Permissions.Camera>();
            if (status == PermissionStatus.Granted)
            {
                await CameraView.StartCameraPreview(CancellationToken.None);
                _isCapturing = true;
                _ = CaptureLoop();
            }
        }
        catch (Exception ex)
        {
            System.Diagnostics.Debug.WriteLine($"Camera start failed: {ex.Message}");
        }
    }

    private async Task CaptureLoop()
    {
        while (_isCapturing)
        {
            try
            {
                using var stream = await CameraView.CaptureImage(CancellationToken.None);
                if (stream != null)
                {
                    using var memoryStream = new MemoryStream();
                    await stream.CopyToAsync(memoryStream);
                    await _viewModel.ProcessFrameAsync(memoryStream.ToArray());
                }
            }
            catch (Exception ex)
            {
                System.Diagnostics.Debug.WriteLine($"Capture loop error: {ex.Message}");
            }
            await Task.Delay(200);
        }
    }

    protected override void OnDisappearing()
    {
        base.OnDisappearing();
        _isCapturing = false;
        CameraView.StopCameraPreview();
    }
}
