using System.Drawing;
using Microsoft.Maui.Graphics;
using PointF = Microsoft.Maui.Graphics.PointF;

namespace SignLanguageApp.ViewModels;

public partial class DynamicGestureTracker
{
    public float PalmWidth { get; set; } = 100f;
    private readonly Queue<PointF> _idxBuffer = new(3);
    private readonly Queue<PointF> _pkyBuffer = new(3);

    public int State { get; private set; } = 0;
    public string? TrackingLetter { get; private set; }
    public PointF? AnchorPos { get; private set; }
    public PointF? StartPos { get; private set; }
    public List<PointF> Trail { get; private set; } = new();
    
    public float J_Lowest { get; private set; }
    public PointF Z_TopCorner { get; private set; }
    public PointF Z_BotCorner { get; private set; }
    
    public int FramesSinceProgress { get; private set; } = 0;

    private PointF GetMedian(PointF pt, Queue<PointF> buffer)
    {
        if (buffer.Count >= 3) buffer.Dequeue();
        buffer.Enqueue(pt);
        if (buffer.Count < 3) return pt;

        var xs = buffer.Select(p => p.X).OrderBy(x => x).ToList();
        var ys = buffer.Select(p => p.Y).OrderBy(y => y).ToList();
        return new PointF(xs[1], ys[1]);
    }

    public string? Update(PointF idxRaw, PointF pkyRaw, string? staticChar, float staticConf)
    {
        var idxSmooth = GetMedian(idxRaw, _idxBuffer);
        var pkySmooth = GetMedian(pkyRaw, _pkyBuffer);

        FramesSinceProgress++;
        if (FramesSinceProgress > 60)
        {
            Reset();
        }

        if (State == 0)
        {
            if (staticChar == "I" && staticConf > 0.7f)
            {
                if (TrackingLetter != "J")
                {
                    TrackingLetter = "J";
                    AnchorPos = pkySmooth;
                }
                else
                {
                    var dist = MathF.Sqrt(MathF.Pow(pkySmooth.X - AnchorPos!.Value.X, 2) + MathF.Pow(pkySmooth.Y - AnchorPos!.Value.Y, 2));
                    if (dist > PalmWidth * 0.2f)
                    {
                        State = 1;
                        StartPos = AnchorPos;
                        Trail = new List<PointF> { AnchorPos.Value, pkySmooth };
                        FramesSinceProgress = 0;
                    }
                }
            }
            else if ((staticChar == "X" || staticChar == "D") && staticConf > 0.7f)
            {
                if (TrackingLetter != "Z")
                {
                    TrackingLetter = "Z";
                    AnchorPos = idxSmooth;
                }
                else
                {
                    var dist = MathF.Sqrt(MathF.Pow(idxSmooth.X - AnchorPos!.Value.X, 2) + MathF.Pow(idxSmooth.Y - AnchorPos!.Value.Y, 2));
                    if (dist > PalmWidth * 0.2f)
                    {
                        State = 1;
                        StartPos = AnchorPos;
                        Trail = new List<PointF> { AnchorPos.Value, idxSmooth };
                        FramesSinceProgress = 0;
                    }
                }
            }
            else
            {
                Reset();
            }
            return null;
        }

        if (TrackingLetter == "J")
        {
            Trail.Add(pkySmooth);
            float currX = pkySmooth.X, currY = pkySmooth.Y;
            float startX = StartPos!.Value.X, startY = StartPos!.Value.Y;

            if (State == 1)
            {
                if (currY > startY + (PalmWidth * 0.4f))
                {
                    State = 2;
                    J_Lowest = currY;
                    FramesSinceProgress = 0;
                }
            }
            else if (State == 2)
            {
                if (currY > J_Lowest)
                {
                    J_Lowest = currY;
                    FramesSinceProgress = 0;
                }

                if (currY < J_Lowest - (PalmWidth * 0.1f) &&
                    MathF.Abs(currX - startX) > (PalmWidth * 0.1f))
                {
                    Reset();
                    return "J";
                }
            }
        }
        else if (TrackingLetter == "Z")
        {
            Trail.Add(idxSmooth);
            float currX = idxSmooth.X, currY = idxSmooth.Y;
            float startX = StartPos!.Value.X, startY = StartPos!.Value.Y;

            if (State == 1)
            {
                if (MathF.Abs(currX - startX) > (PalmWidth * 0.4f))
                {
                    State = 2;
                    Z_TopCorner = idxSmooth;
                    FramesSinceProgress = 0;
                }
            }
            else if (State == 2)
            {
                if (currY > startY + (PalmWidth * 0.4f))
                {
                    State = 3;
                    Z_BotCorner = idxSmooth;
                    FramesSinceProgress = 0;
                }
            }
            else if (State == 3)
            {
                if (MathF.Abs(currX - Z_BotCorner.X) > (PalmWidth * 0.4f))
                {
                    Reset();
                    return "Z";
                }
            }
        }

        return null;
    }

    public void Reset()
    {
        State = 0;
        TrackingLetter = null;
        AnchorPos = null;
        StartPos = null;
        Trail.Clear();
        FramesSinceProgress = 0;
    }
}
