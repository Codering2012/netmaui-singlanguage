using Microsoft.Maui.Controls;
using Microsoft.Maui.Graphics;
using System;
using System.Collections.Generic;
using System.Threading.Tasks;

namespace SignLanguageApp.Controls;

public class ConfettiView : GraphicsView
{
    private ConfettiDrawable _drawable;
    private bool _isRunning = false;

    public ConfettiView()
    {
        InputTransparent = true;
        _drawable = new ConfettiDrawable();
        this.Drawable = _drawable;
    }

    public void StartConfetti()
    {
        if (_isRunning) return;
        _isRunning = true;
        _drawable.InitializeParticles((float)Width, (float)Height);
        StartAnimationLoop();
    }

    private async void StartAnimationLoop()
    {
        var dispatcher = this.Dispatcher ?? Application.Current?.Dispatcher;
        if (dispatcher == null) return;
        
        while (_isRunning)
        {
            _isRunning = _drawable.Update(0.016f); 
            this.Invalidate();
            await Task.Delay(16);
        }
    }
}

public class ConfettiParticle
{
    public float X;
    public float Y;
    public float VelocityX;
    public float VelocityY;
    public float Size;
    public Color Color = Colors.Transparent;
    public float Rotation;
    public float RotationSpeed;
}

public class ConfettiDrawable : IDrawable
{
    private List<ConfettiParticle> _particles = new List<ConfettiParticle>();
    private Random _rand = new Random();
    private Color[] _colors = new[] { Colors.Red, Colors.Yellow, Colors.Blue, Colors.Green, Colors.Purple, Colors.Cyan };

    public void InitializeParticles(float width, float height)
    {
        _particles.Clear();
        for (int i = 0; i < 150; i++)
        {
            _particles.Add(new ConfettiParticle
            {
                X = (float)_rand.NextDouble() * width,
                Y = -50 - (float)_rand.NextDouble() * 200,
                VelocityX = (float)(_rand.NextDouble() - 0.5) * 200f,
                VelocityY = 300f + (float)_rand.NextDouble() * 300f,
                Size = 10f + (float)_rand.NextDouble() * 15f,
                Color = _colors[_rand.Next(_colors.Length)],
                Rotation = (float)_rand.NextDouble() * 360f,
                RotationSpeed = (float)(_rand.NextDouble() - 0.5) * 500f
            });
        }
    }

    public bool Update(float deltaTime)
    {
        bool anyActive = false;
        foreach (var p in _particles)
        {
            p.X += p.VelocityX * deltaTime;
            p.Y += p.VelocityY * deltaTime;
            p.Rotation += p.RotationSpeed * deltaTime;
            
            if (p.Y < 3000) // Fall infinitely until out of bounds (assume max height 3000)
            {
                anyActive = true;
            }
        }
        return anyActive;
    }

    public void Draw(ICanvas canvas, RectF dirtyRect)
    {
        foreach (var p in _particles)
        {
            canvas.SaveState();
            canvas.Translate(p.X, p.Y);
            canvas.Rotate(p.Rotation);
            canvas.FillColor = p.Color;
            canvas.FillRectangle(-p.Size/2, -p.Size/2, p.Size, p.Size);
            canvas.RestoreState();
        }
    }
}
