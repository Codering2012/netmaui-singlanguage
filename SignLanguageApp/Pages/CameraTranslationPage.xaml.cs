using Microsoft.Maui.Controls;
using Microsoft.Maui.Media;
using SignLanguageApp.ViewModels;
using System.Diagnostics;
using System.ComponentModel;
using System.Linq;
using System.Threading;

namespace SignLanguageApp.Pages
{
    public partial class CameraTranslationPage : ContentPage
    {
        private const int OverlayInvalidateIntervalMs = 33;
        private readonly CameraTranslationViewModel _viewModel;
        private long _lastOverlayInvalidateTick;
        private bool _isPreviewRunning;

        public CameraTranslationPage(CameraTranslationViewModel viewModel)
        {
            InitializeComponent();
            _viewModel = viewModel;
            BindingContext = _viewModel;
            _viewModel.PropertyChanged += OnViewModelPropertyChanged;

            // Pass camera view reference to view model
            _viewModel.SetCameraView(CameraView);
        }

        private void OnViewModelPropertyChanged(object? sender, PropertyChangedEventArgs e)
        {
            if (e.PropertyName == nameof(CameraTranslationViewModel.Drawable))
            {
                var now = Environment.TickCount64;
                var last = Interlocked.Read(ref _lastOverlayInvalidateTick);
                if (now - last < OverlayInvalidateIntervalMs)
                {
                    return;
                }

                Interlocked.Exchange(ref _lastOverlayInvalidateTick, now);
                MainThread.BeginInvokeOnMainThread(() => SkeletalOverlay.Invalidate());
            }
        }

        protected override async void OnAppearing()
        {
            base.OnAppearing();
            try
            {
                // Avoid auto-starting camera on page load to prevent startup crashes
                // on devices/environments where camera handlers are not immediately ready.
                await EnsureCameraPermissionAsync(showDeniedAlert: false);
            }
            catch (OperationCanceledException)
            {
                Debug.WriteLine("Camera translation initialization timed out.");
            }
            catch (Exception ex)
            {
                Debug.WriteLine($"Camera appearance error: {ex.Message}");
                Debug.WriteLine($"Stack trace: {ex.StackTrace}");
            }
        }

        private async Task<bool> EnsureCameraPermissionAsync(bool showDeniedAlert)
        {
            var cameraStatus = await Permissions.CheckStatusAsync<Permissions.Camera>();
            if (cameraStatus != PermissionStatus.Granted)
            {
                cameraStatus = await Permissions.RequestAsync<Permissions.Camera>();
            }

            if (cameraStatus == PermissionStatus.Granted)
            {
                return true;
            }

            if (showDeniedAlert)
            {
                await DisplayAlertAsync("Permission Denied", "Camera permission is required to use this feature.", "OK");
            }

            return false;
        }

        private async Task StartPreviewAsync()
        {
            if (_isPreviewRunning)
            {
                return;
            }

            var startCameraPreviewCts = new CancellationTokenSource(TimeSpan.FromSeconds(10));

            // Android devices are more reliable when preview starts directly
            // without preselecting a camera from GetAvailableCameras.
            if (Microsoft.Maui.Devices.DeviceInfo.Platform == Microsoft.Maui.Devices.DevicePlatform.Android)
            {
                try
                {
                    await CameraView.StartCameraPreview(startCameraPreviewCts.Token);
                    _isPreviewRunning = true;
                    return;
                }
                catch (Exception ex)
                {
                    Debug.WriteLine($"Android direct camera start failed, retrying with explicit selection: {ex.Message}");
                }
            }

            var availableCameras = await CameraView.GetAvailableCameras(startCameraPreviewCts.Token);
            var selectedCamera = availableCameras.FirstOrDefault();

            if (selectedCamera == null)
            {
                throw new InvalidOperationException("No camera available on device.");
            }

            CameraView.SelectedCamera = selectedCamera;
            await CameraView.StartCameraPreview(startCameraPreviewCts.Token);
            _isPreviewRunning = true;
        }

        private async void OnStartClicked(object? sender, EventArgs e)
        {
            StartButton.IsEnabled = false;

            try
            {
                // Wait a moment for handlers to settle on initial navigation.
                if (CameraView?.Handler == null)
                {
                    await Task.Delay(400);
                }

                var hasPermission = await EnsureCameraPermissionAsync(showDeniedAlert: true);
                if (!hasPermission)
                {
                    return;
                }

                await StartPreviewAsync();
                await _viewModel.StartCameraCaptureCommand.ExecuteAsync(null);
            }
            catch (OperationCanceledException)
            {
                await DisplayAlertAsync("Camera Error", "Camera initialization timed out.", "OK");
            }
            catch (Exception ex)
            {
                Debug.WriteLine($"Start camera error: {ex.Message}");
                if (!ex.Message.Contains("No camera available on device", StringComparison.OrdinalIgnoreCase))
                {
                    await DisplayAlertAsync("Camera Error", $"Failed to start camera: {ex.Message}", "OK");
                }
            }
            finally
            {
                StartButton.IsEnabled = true;
            }
        }

        private async void OnStopClicked(object? sender, EventArgs e)
        {
            try
            {
                await _viewModel.StopCommand.ExecuteAsync(null);
            }
            catch (Exception ex)
            {
                Debug.WriteLine($"Stop command error: {ex.Message}");
            }

            try
            {
                CameraView.StopCameraPreview();
            }
            catch (Exception ex)
            {
                Debug.WriteLine($"Camera stop preview error: {ex.Message}");
            }
            finally
            {
                _isPreviewRunning = false;
            }
        }

        protected override void OnDisappearing()
        {
            base.OnDisappearing();
            try
            {
                CameraView.StopCameraPreview();
            }
            catch (Exception ex)
            {
                Debug.WriteLine($"Camera stop error: {ex.Message}");
            }
            if (_viewModel.StopCameraCaptureCommand.CanExecute(null))
                _viewModel.StopCameraCaptureCommand.Execute(null);
            _viewModel.OnDisappearing();
            _isPreviewRunning = false;
        }

        protected override void OnHandlerChanging(HandlerChangingEventArgs args)
        {
            if (args.NewHandler == null)
            {
                _viewModel.PropertyChanged -= OnViewModelPropertyChanged;
            }

            base.OnHandlerChanging(args);
        }
    }
}
