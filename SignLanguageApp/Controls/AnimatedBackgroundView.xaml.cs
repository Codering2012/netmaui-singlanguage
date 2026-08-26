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

    private class BlobState
    {
        public double X;
        public double Y;
        public double Vx;
        public double Vy;
        public double MinX = -60;
        public double MaxX = 60;
        public double MinY = -60;
        public double MaxY = 60;
    }

    private readonly BlobState[] _blobs = new BlobState[5];

    public AnimatedBackgroundView()
    {
        InitializeComponent();
        
        for (int i = 0; i < 5; i++)
        {
            _blobs[i] = new BlobState
            {
                X = _random.Next(-30, 30),
                Y = _random.Next(-30, 30),
                Vx = (_random.NextDouble() * 1.5 + 0.5) * (_random.Next(2) == 0 ? 1 : -1),
                Vy = (_random.NextDouble() * 1.5 + 0.5) * (_random.Next(2) == 0 ? 1 : -1)
            };
        }
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
                foreach (var b in _blobs)
                {
                    b.X += b.Vx;
                    b.Y += b.Vy;

                    if (b.X <= b.MinX || b.X >= b.MaxX)
                    {
                        b.Vx = -b.Vx;
                        b.X = Math.Clamp(b.X, b.MinX, b.MaxX);
                    }
                    if (b.Y <= b.MinY || b.Y >= b.MaxY)
                    {
                        b.Vy = -b.Vy;
                        b.Y = Math.Clamp(b.Y, b.MinY, b.MaxY);
                    }
                }

                // Apply translations without creating task list
                await ApplyAnimationsAsync(
                    (_blobs[0].X, _blobs[0].Y),
                    (_blobs[1].X, _blobs[1].Y),
                    (_blobs[2].X, _blobs[2].Y),
                    (_blobs[3].X, _blobs[3].Y),
                    (_blobs[4].X, _blobs[4].Y),
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

