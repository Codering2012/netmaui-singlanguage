using System.Collections.Concurrent;

namespace SignLanguageApp.Services;

public interface IFrameBufferService
{
    void AddFrame(byte[] frame);
    byte[]? GetLastNSecondsOfFrames(int seconds);
    void Clear();
}

public class FrameBufferService : IFrameBufferService
{
    private const int MaxFrames = 150; // ~5 seconds at 30fps
    private readonly ConcurrentQueue<(byte[] Frame, DateTime Timestamp)> _buffer = new();

    public void AddFrame(byte[] frame)
    {
        _buffer.Enqueue((frame, DateTime.UtcNow));
        
        while (_buffer.Count > MaxFrames)
        {
            _buffer.TryDequeue(out _);
        }
    }

    public byte[]? GetLastNSecondsOfFrames(int seconds)
    {
        // For simplicity in this demo, we'll return the most recent frame or a sequence
        // In a real implementation, this would return a compiled video or a list of frames.
        // For the "Mistake Replay", we'll return the last captured frame to show as a static comparison
        // or a list of frames for a custom player.
        return _buffer.ToArray().OrderByDescending(f => f.Timestamp).FirstOrDefault().Frame;
    }

    public void Clear()
    {
        while (_buffer.TryDequeue(out _)) { }
    }
}
