using Microsoft.Maui.Controls;
using Microsoft.Maui.Graphics;
using System;

namespace SignLanguageApp.Helpers
{
    public class TiltBehavior : Behavior<View>
    {
        private View? _view;

        protected override void OnAttachedTo(View bindable)
        {
            base.OnAttachedTo(bindable);
            _view = bindable;

            var pointerRecognizer = new PointerGestureRecognizer();
            pointerRecognizer.PointerEntered += OnPointerEntered;
            pointerRecognizer.PointerMoved += OnPointerMoved;
            pointerRecognizer.PointerExited += OnPointerExited;

            _view.GestureRecognizers.Add(pointerRecognizer);
        }

        protected override void OnDetachingFrom(View bindable)
        {
            base.OnDetachingFrom(bindable);
            if (_view != null)
            {
                // Note: Need to find and remove the pointer recognizer in production, 
                // but for this behavior we just let the page GC handle it.
            }
            _view = null;
        }

        private async void OnPointerEntered(object? sender, PointerEventArgs e)
        {
            if (_view == null) return;
            await _view.ScaleToAsync(1.03, 150, Easing.CubicOut);
        }

        private void OnPointerMoved(object? sender, PointerEventArgs e)
        {
            if (_view == null) return;
            
            var position = e.GetPosition(_view);
            if (position.HasValue)
            {
                double width = _view.Width;
                double height = _view.Height;
                
                if (width <= 0 || height <= 0) return;

                // Normalize position to -1.0 to 1.0
                double normX = (position.Value.X / width) * 2 - 1;
                double normY = (position.Value.Y / height) * 2 - 1;

                // Max tilt angle (degrees)
                double maxTilt = 8.0;

                _view.RotationY = normX * maxTilt;
                _view.RotationX = -normY * maxTilt; // Negative to tilt towards pointer
            }
        }

        private async void OnPointerExited(object? sender, PointerEventArgs e)
        {
            if (_view == null) return;
            
            await Task.WhenAll(
                _view.RotateXToAsync(0, 250, Easing.SpringOut),
                _view.RotateYToAsync(0, 250, Easing.SpringOut),
                _view.ScaleToAsync(1.0, 250, Easing.SpringOut)
            );
        }
    }
}
