using Microsoft.Maui.Graphics;
using SignLanguageApp.Model;

namespace SignLanguageApp.Controls
{
    /// <summary>
    /// Draws hand skeletal structure with 21 landmarks and connecting lines
    /// Based on MediaPipe hand tracking model
    /// </summary>
    public class SkeletalDrawable : IDrawable
    {
        private readonly List<CoordinateDto> _landmarks;
        private readonly int _sourceFrameWidth;
        private readonly int _sourceFrameHeight;

        // Hand landmark connections (MediaPipe model - 21 landmarks)
        private static readonly int[][] HAND_CONNECTIONS = new[]
        {
            // Thumb
            new[] { 0, 1 }, new[] { 1, 2 }, new[] { 2, 3 }, new[] { 3, 4 },
            // Index finger
            new[] { 0, 5 }, new[] { 5, 6 }, new[] { 6, 7 }, new[] { 7, 8 },
            // Middle finger
            new[] { 0, 9 }, new[] { 9, 10 }, new[] { 10, 11 }, new[] { 11, 12 },
            // Ring finger
            new[] { 0, 13 }, new[] { 13, 14 }, new[] { 14, 15 }, new[] { 15, 16 },
            // Pinky finger
            new[] { 0, 17 }, new[] { 17, 18 }, new[] { 18, 19 }, new[] { 19, 20 },
            // Palm connections
            new[] { 5, 9 }, new[] { 9, 13 }, new[] { 13, 17 }
        };

        public SkeletalDrawable(List<CoordinateDto> landmarks, int sourceFrameWidth = 0, int sourceFrameHeight = 0)
        {
            _landmarks = landmarks ?? [];
            _sourceFrameWidth = sourceFrameWidth;
            _sourceFrameHeight = sourceFrameHeight;
        }

        public void Draw(ICanvas canvas, RectF dirtyRect)
        {
            if (_landmarks == null || _landmarks.Count == 0)
                return;

            canvas.SaveState();

            try
            {
                // Draw connections (lines between joints)
                DrawConnections(canvas, dirtyRect);

                // Draw landmarks (circles at joints)
                DrawLandmarks(canvas, dirtyRect);
            }
            finally
            {
                canvas.RestoreState();
            }
        }

        private void DrawConnections(ICanvas canvas, RectF dirtyRect)
        {
            var drawRect = GetPreviewRect(dirtyRect);
            canvas.StrokeColor = Color.FromRgb(0, 255, 255); // Cyan
            canvas.StrokeSize = 2f;

            foreach (var connection in HAND_CONNECTIONS)
            {
                int startIdx = connection[0];
                int endIdx = connection[1];

                if (startIdx >= _landmarks.Count || endIdx >= _landmarks.Count)
                    continue;

                var start = _landmarks[startIdx];
                var end = _landmarks[endIdx];

                var startPoint = ToCanvasPoint(start, drawRect);
                var endPoint = ToCanvasPoint(end, drawRect);

                canvas.DrawLine(startPoint.X, startPoint.Y, endPoint.X, endPoint.Y);
            }
        }

        private void DrawLandmarks(ICanvas canvas, RectF dirtyRect)
        {
            const float radius = 5f;
            var drawRect = GetPreviewRect(dirtyRect);

            for (int i = 0; i < _landmarks.Count; i++)
            {
                var landmark = _landmarks[i];

                var point = ToCanvasPoint(landmark, drawRect);

                // Color based on landmark type (finger index)
                canvas.FillColor = GetLandmarkColor(i);
                canvas.FillCircle(point.X, point.Y, radius);

                // Draw white outline
                canvas.StrokeColor = Colors.White;
                canvas.StrokeSize = 1f;
                canvas.DrawCircle(point.X, point.Y, radius);
            }
        }

        private PointF ToCanvasPoint(CoordinateDto landmark, RectF drawRect)
        {
            if (_landmarks.Count == 0)
            {
                return new PointF(drawRect.Left, drawRect.Top);
            }

            var normalized = IsNormalizedCoordinates();
            if (normalized)
            {
                return new PointF(
                    (float)(drawRect.Left + (landmark.X * drawRect.Width)),
                    (float)(drawRect.Top + (landmark.Y * drawRect.Height)));
            }

            if (_sourceFrameWidth > 0 && _sourceFrameHeight > 0)
            {
                return new PointF(
                    (float)(drawRect.Left + ((landmark.X / _sourceFrameWidth) * drawRect.Width)),
                    (float)(drawRect.Top + ((landmark.Y / _sourceFrameHeight) * drawRect.Height)));
            }

            var maxX = _landmarks.Max(l => l.X);
            var maxY = _landmarks.Max(l => l.Y);
            var xScale = maxX > 0 ? (float)(landmark.X / maxX) : 0f;
            var yScale = maxY > 0 ? (float)(landmark.Y / maxY) : 0f;

            return new PointF(
                (float)(drawRect.Left + (xScale * drawRect.Width)),
                (float)(drawRect.Top + (yScale * drawRect.Height)));
        }

        private bool IsNormalizedCoordinates()
        {
            return _landmarks.All(l => l.X is >= 0 and <= 1 && l.Y is >= 0 and <= 1);
        }

        /// <summary>
        /// Returns different colors for different hand parts
        /// </summary>
        private Color GetLandmarkColor(int index)
        {
            return index switch
            {
                0 => Color.FromRgb(255, 255, 0),      // Wrist - Yellow
                1 or 2 or 3 or 4 => Color.FromRgb(255, 0, 0),        // Thumb - Red
                5 or 6 or 7 or 8 => Color.FromRgb(0, 255, 0),        // Index - Green
                9 or 10 or 11 or 12 => Color.FromRgb(0, 0, 255),     // Middle - Blue
                13 or 14 or 15 or 16 => Color.FromRgb(128, 0, 128),  // Ring - Purple
                17 or 18 or 19 or 20 => Color.FromRgb(255, 165, 0),  // Pinky - Orange
                _ => Color.FromRgb(128, 128, 128)     // Default - Gray
            };
        }

        private RectF GetPreviewRect(RectF dirtyRect)
        {
            if (_sourceFrameWidth <= 0 || _sourceFrameHeight <= 0)
            {
                return dirtyRect;
            }

            var sourceAspect = _sourceFrameWidth / (float)_sourceFrameHeight;
            if (sourceAspect <= 0)
            {
                return dirtyRect;
            }

            var targetAspect = dirtyRect.Width / dirtyRect.Height;
            if (targetAspect <= 0)
            {
                return dirtyRect;
            }

            if (sourceAspect > targetAspect)
            {
                var height = dirtyRect.Width / sourceAspect;
                var top = dirtyRect.Top + ((dirtyRect.Height - height) / 2f);
                return new RectF(dirtyRect.Left, top, dirtyRect.Width, height);
            }

            var width = dirtyRect.Height * sourceAspect;
            var left = dirtyRect.Left + ((dirtyRect.Width - width) / 2f);
            return new RectF(left, dirtyRect.Top, width, dirtyRect.Height);
        }
    }
}
