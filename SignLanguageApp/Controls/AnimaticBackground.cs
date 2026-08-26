using Microsoft.Maui.Graphics;
using Microsoft.Maui.Devices.Sensors;
using System.Diagnostics;

namespace SignLanguageApp.Controls;

public class AnimaticBackground : GraphicsView
{
    private AnimaticDrawable _drawable;
    private float _accelX = 0;
    private float _accelY = 0;

    public AnimaticBackground()
    {
        _drawable = new AnimaticDrawable();
        this.Drawable = _drawable;
        
        // Start the animation loop
        StartAnimationLoop();
        StartAccelerometer();
    }

    private void StartAccelerometer()
    {
        if (Accelerometer.Default.IsSupported)
        {
            if (!Accelerometer.Default.IsMonitoring)
            {
                try {
                    Accelerometer.Default.Start(SensorSpeed.UI);
                } catch (Exception ex) {
                    Debug.WriteLine($"Accelerometer error: {ex.Message}");
                }
            }
            Accelerometer.Default.ReadingChanged += (s, e) =>
            {
                // Smooth out the readings (low-pass filter)
                _accelX = _accelX * 0.9f + (float)e.Reading.Acceleration.X * 0.1f;
                _accelY = _accelY * 0.9f + (float)e.Reading.Acceleration.Y * 0.1f;
            };
        }
    }

    private async void StartAnimationLoop()
    {
        var dispatcher = this.Dispatcher ?? Application.Current?.Dispatcher;
        if (dispatcher == null) return;
        
        while (true)
        {
            _drawable.Update(0.016f, _accelX, _accelY); // Assume ~60fps
            this.Invalidate();
            await Task.Delay(16); // 16ms
        }
    }
}

public class AnimaticDrawable : IDrawable
{
    private float _time = 0;
    private float _offsetX = 0;
    private float _offsetY = 0;
    
    private class CircleState
    {
        public float X;
        public float Y;
        public float Vx;
        public float Vy;
    }
    
    private CircleState[]? _circles;

    public void Update(float deltaTime, float accelX, float accelY)
    {
        _time += deltaTime;
        // Map acceleration to screen offset (parallax)
        _offsetX = accelX * 100f; // max shift 100px
        _offsetY = accelY * 100f;
    }

    public void Draw(ICanvas canvas, RectF dirtyRect)
    {
        if (_circles == null)
        {
            _circles = new CircleState[]
            {
                new CircleState { X = dirtyRect.Width * 0.5f, Y = dirtyRect.Height * 0.3f, Vx = 1.0f, Vy = 1.5f },
                new CircleState { X = dirtyRect.Width * 0.2f, Y = dirtyRect.Height * 0.7f, Vx = -0.75f, Vy = 1.25f },
                new CircleState { X = dirtyRect.Width * 0.8f, Y = dirtyRect.Height * 0.8f, Vx = 1.25f, Vy = -1.0f }
            };
        }

        foreach (var c in _circles)
        {
            c.X += c.Vx;
            c.Y += c.Vy;

            if (c.X < 0 || c.X > dirtyRect.Width)
            {
                c.Vx = -c.Vx;
                c.X = Math.Clamp(c.X, 0, dirtyRect.Width);
            }
            if (c.Y < 0 || c.Y > dirtyRect.Height)
            {
                c.Vy = -c.Vy;
                c.Y = Math.Clamp(c.Y, 0, dirtyRect.Height);
            }
        }

        // Deep background color based on theme
        bool isDark = Application.Current?.RequestedTheme == AppTheme.Dark;
        canvas.FillColor = isDark ? Color.FromArgb("#050814") : Color.FromArgb("#FFFFFF");
        canvas.FillRectangle(dirtyRect);

        // Draw overlapping moving radial gradients with Parallax Offset
        float cx1 = _circles[0].X - _offsetX * 1.5f;
        float cy1 = _circles[0].Y + _offsetY * 1.5f;
        DrawGlow(canvas, new PointF(cx1, cy1), dirtyRect.Width * 0.8f, isDark ? Color.FromArgb("#2A00F0FF") : Color.FromArgb("#3300F0FF"));

        float cx2 = _circles[1].X - _offsetX * 0.8f;
        float cy2 = _circles[1].Y + _offsetY * 0.8f;
        DrawGlow(canvas, new PointF(cx2, cy2), dirtyRect.Width * 1.0f, isDark ? Color.FromArgb("#2A6366F1") : Color.FromArgb("#336366F1"));

        float cx3 = _circles[2].X - _offsetX * 2.0f;
        float cy3 = _circles[2].Y + _offsetY * 2.0f;
        DrawGlow(canvas, new PointF(cx3, cy3), dirtyRect.Width * 0.9f, isDark ? Color.FromArgb("#2AD946EF") : Color.FromArgb("#33D946EF"));
    }

    private void DrawGlow(ICanvas canvas, PointF center, float radius, Color color)
    {
        canvas.SetShadow(new SizeF(0, 0), radius * 0.5f, color);
        canvas.FillColor = color;
        canvas.FillCircle(center, radius * 0.5f);
    }
}
