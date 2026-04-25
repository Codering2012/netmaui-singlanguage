using Microsoft.Maui.Controls;
using Microsoft.Maui.Controls.Shapes;

namespace SignLanguageApp.Controls;

public partial class AnimatedBackgroundView : ContentView
{
    private bool _isAnimating = false;
    private CancellationTokenSource? _animationCts;
    private readonly Random _random = new();
    private const uint ANIMATION_DURATION = 6000;

    private Grid? _blob1Container;
    private Grid? _blob2Container;
    private Grid? _blob3Container;
    private Grid? _blob4Container;
    private Grid? _blob5Container;

    private double _waveTime = 0;
    private const double WAVE_SPEED = 0.015;
    private const int ANIMATION_FRAME_DELAY = 50;

    public AnimatedBackgroundView()
    {
        InitializeComponent();
    }

    protected override void OnHandlerChanged()
    {
        base.OnHandlerChanged();

        if (Handler != null)
        {
            InitializeBlobs();
            StartAnimation();
        }
    }

    private void InitializeBlobs()
    {
        _blob1Container = this.FindByName<Grid>("Blob1Container");
        _blob2Container = this.FindByName<Grid>("Blob2Container");
        _blob3Container = this.FindByName<Grid>("Blob3Container");
        _blob4Container = this.FindByName<Grid>("Blob4Container");
        _blob5Container = this.FindByName<Grid>("Blob5Container");
    }

    private void StartAnimation()
    {
        if (_animationCts != null)
            return; // Already running

        _animationCts = new CancellationTokenSource();
        _isAnimating = true;

        // Start animation loop without blocking
        _ = AnimateWavyBackgroundOptimized(_animationCts.Token);
    }

    private async Task AnimateWavyBackgroundOptimized(CancellationToken cancellationToken)
    {
        try
        {
            while (!cancellationToken.IsCancellationRequested && _isAnimating)
            {
                _waveTime += WAVE_SPEED;

                // Calculate positions using sine waves
                double blob1X = Math.Sin(_waveTime) * 25;
                double blob1Y = Math.Cos(_waveTime * 0.8) * 20;

                double blob2X = Math.Sin(_waveTime + Math.PI / 2) * 30;
                double blob2Y = Math.Cos(_waveTime * 0.9 + Math.PI / 3) * 25;

                double blob3X = Math.Sin(_waveTime * 0.7) * 35;
                double blob3Y = Math.Cos(_waveTime * 1.1) * 30;

                double blob4X = Math.Sin(_waveTime + Math.PI) * 28;
                double blob4Y = Math.Cos(_waveTime * 0.85 + Math.PI / 4) * 22;

                double blob5X = Math.Sin(_waveTime * 0.9 + Math.PI / 6) * 32;
                double blob5Y = Math.Cos(_waveTime * 1.0 + Math.PI / 2) * 28;

                // Apply translations without creating task list
                await ApplyAnimationsAsync(
                    (blob1X, blob1Y),
                    (blob2X, blob2Y),
                    (blob3X, blob3Y),
                    (blob4X, blob4Y),
                    (blob5X, blob5Y),
                    cancellationToken);

                await Task.Delay(ANIMATION_FRAME_DELAY, cancellationToken);
            }
        }
        catch (OperationCanceledException)
        {
            // Animation was cancelled - expected behavior
        }
    }

    private async Task ApplyAnimationsAsync(
        (double x, double y) blob1,
        (double x, double y) blob2,
        (double x, double y) blob3,
        (double x, double y) blob4,
        (double x, double y) blob5,
        CancellationToken cancellationToken)
    {
        var tasks = new List<Task>(5);

        if (_blob1Container != null)
            tasks.Add(_blob1Container.TranslateToAsync(blob1.x, blob1.y, ANIMATION_FRAME_DELAY, Easing.Linear));

        if (_blob2Container != null)
            tasks.Add(_blob2Container.TranslateToAsync(blob2.x, blob2.y, ANIMATION_FRAME_DELAY, Easing.Linear));

        if (_blob3Container != null)
            tasks.Add(_blob3Container.TranslateToAsync(blob3.x, blob3.y, ANIMATION_FRAME_DELAY, Easing.Linear));

        if (_blob4Container != null)
            tasks.Add(_blob4Container.TranslateToAsync(blob4.x, blob4.y, ANIMATION_FRAME_DELAY, Easing.Linear));

        if (_blob5Container != null)
            tasks.Add(_blob5Container.TranslateToAsync(blob5.x, blob5.y, ANIMATION_FRAME_DELAY, Easing.Linear));

        if (tasks.Count > 0)
        {
            try
            {
                await Task.WhenAll(tasks);
            }
            catch (OperationCanceledException)
            {
                // Animation cancelled - clear pending tasks
            }
        }
    }

    public void StopAnimation()
    {
        _isAnimating = false;
        _animationCts?.Cancel();
        _animationCts?.Dispose();
        _animationCts = null;
    }

    protected override void OnHandlerChanging(HandlerChangingEventArgs args)
    {
        base.OnHandlerChanging(args);
        StopAnimation();
    }
}

