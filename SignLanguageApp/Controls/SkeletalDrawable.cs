using Microsoft.Maui.Graphics;
using System.Collections.Generic;
using SignLanguageApp.Model; // To access CoordinateDto

namespace SignLanguageApp.Controls
{
    public class SkeletalDrawable : IDrawable
    {
        public bool IsCorrectForm { get; set; } = true;
        
        // Missing properties
        public List<CoordinateDto> IndexTrail { get; set; } = new();
        public List<CoordinateDto> PinkyTrail { get; set; } = new();
        public string TrackingLetter { get; set; } = string.Empty;
        public bool IsMirrored { get; set; } = false;

        private List<CoordinateDto> _coordinates;
        private int _frameWidth;
        private int _frameHeight;

        public SkeletalDrawable()
        {
            _coordinates = new List<CoordinateDto>();
        }

        public SkeletalDrawable(List<CoordinateDto> coordinates, int frameWidth, int frameHeight)
        {
            _coordinates = coordinates ?? new List<CoordinateDto>();
            _frameWidth = frameWidth;
            _frameHeight = frameHeight;
        }

        public void Draw(ICanvas canvas, RectF dirtyRect)
        {
            // Draw neon skeleton
            canvas.StrokeColor = IsCorrectForm ? Colors.Cyan : Colors.Red;
            canvas.StrokeSize = 4;
            // Add a glow effect in production, for now just a bright line
            
            // Draw a dummy hand skeleton
            var centerX = dirtyRect.Center.X;
            var centerY = dirtyRect.Center.Y;
            
            canvas.DrawLine(centerX, centerY + 50, centerX - 30, centerY - 20);
            canvas.DrawLine(centerX, centerY + 50, centerX - 10, centerY - 40);
            canvas.DrawLine(centerX, centerY + 50, centerX + 10, centerY - 40);
            canvas.DrawLine(centerX, centerY + 50, centerX + 30, centerY - 20);
            
            // Draw joints
            canvas.FillColor = IsCorrectForm ? Colors.White : Colors.Yellow;
            canvas.FillCircle(centerX - 30, centerY - 20, 6);
            canvas.FillCircle(centerX - 10, centerY - 40, 6);
            canvas.FillCircle(centerX + 10, centerY - 40, 6);
            canvas.FillCircle(centerX + 30, centerY - 20, 6);
            canvas.FillCircle(centerX, centerY + 50, 8); // wrist
        }
    }
}
