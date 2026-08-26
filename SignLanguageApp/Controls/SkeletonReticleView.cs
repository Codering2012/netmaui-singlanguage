using Microsoft.Maui.Controls;
using Microsoft.Maui.Graphics;
using System;
using System.Threading.Tasks;

namespace SignLanguageApp.Controls
{
    public class SkeletonReticleView : GraphicsView
    {
        private SkeletonReticleDrawable _drawable;
        private bool _isRunning = false;

        public static readonly BindableProperty IsScanningProperty =
            BindableProperty.Create(nameof(IsScanning), typeof(bool), typeof(SkeletonReticleView), false, propertyChanged: OnIsScanningChanged);

        public bool IsScanning
        {
            get => (bool)GetValue(IsScanningProperty);
            set => SetValue(IsScanningProperty, value);
        }

        public SkeletonReticleView()
        {
            InputTransparent = true;
            _drawable = new SkeletonReticleDrawable();
            this.Drawable = _drawable;
        }

        private static void OnIsScanningChanged(BindableObject bindable, object oldValue, object newValue)
        {
            var view = (SkeletonReticleView)bindable;
            if ((bool)newValue)
            {
                view.StartScanning();
            }
            else
            {
                view.StopScanning();
            }
        }

        private void StartScanning()
        {
            if (_isRunning) return;
            _isRunning = true;
            StartAnimationLoop();
        }

        private void StopScanning()
        {
            _isRunning = false;
            this.Invalidate();
        }

        private async void StartAnimationLoop()
        {
            var dispatcher = this.Dispatcher ?? Application.Current?.Dispatcher;
            if (dispatcher == null) return;
            
            while (_isRunning)
            {
                _drawable.Update(0.016f); 
                this.Invalidate();
                await Task.Delay(16);
            }
        }
    }

    public class SkeletonReticleDrawable : IDrawable
    {
        private float _time = 0;

        public void Update(float deltaTime)
        {
            _time += deltaTime;
        }

        public void Draw(ICanvas canvas, RectF dirtyRect)
        {
            float cx = dirtyRect.Width / 2;
            float cy = dirtyRect.Height / 2;

            canvas.StrokeColor = Colors.Cyan.WithAlpha(0.6f);
            canvas.StrokeSize = 2;
            
            // Draw scanning line
            float scanY = (float)Math.Sin(_time * 2) * cy + cy;
            canvas.DrawLine(cx - 100, scanY, cx + 100, scanY);

            // Draw face reticle (top)
            canvas.DrawRectangle(cx - 40, cy - 80, 80, 80);
            DrawCorners(canvas, cx - 40, cy - 80, 80, 80, 10);

            // Draw hand reticles (left, right)
            float handOffset = (float)Math.Sin(_time * 5) * 5;
            
            canvas.DrawRectangle(cx - 90 - handOffset, cy + 20, 40, 40);
            DrawCorners(canvas, cx - 90 - handOffset, cy + 20, 40, 40, 5);

            canvas.DrawRectangle(cx + 50 + handOffset, cy + 20, 40, 40);
            DrawCorners(canvas, cx + 50 + handOffset, cy + 20, 40, 40, 5);
        }

        private void DrawCorners(ICanvas canvas, float x, float y, float w, float h, float len)
        {
            canvas.StrokeColor = Colors.White;
            canvas.StrokeSize = 3;
            
            // Top Left
            canvas.DrawLine(x, y, x + len, y);
            canvas.DrawLine(x, y, x, y + len);
            
            // Top Right
            canvas.DrawLine(x + w, y, x + w - len, y);
            canvas.DrawLine(x + w, y, x + w, y + len);
            
            // Bottom Left
            canvas.DrawLine(x, y + h, x + len, y + h);
            canvas.DrawLine(x, y + h, x, y + h - len);
            
            // Bottom Right
            canvas.DrawLine(x + w, y + h, x + w - len, y + h);
            canvas.DrawLine(x + w, y + h, x + w, y + h - len);
        }
    }
}
