namespace SignLanguageApp.Controls;

public class VideoProcessingCameraView : ContentView
{
    public event EventHandler<byte[]>? FrameReady;
    
    public void StartCamera()
    {
    }

    public void StopCamera()
    {
    }

    protected virtual void OnFrameReady(byte[] frame)
    {
        FrameReady?.Invoke(this, frame);
    }
}
