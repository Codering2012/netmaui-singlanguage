using System.Diagnostics;

namespace SignLanguageApp.Services;

public class EnvironmentDetectionService : IEnvironmentDetectionService
{
    private const double LowBrightnessThreshold = 40.0;
    private const double HighBrightnessThreshold = 240.0;
    private const double LowContrastThreshold = 20.0;

    public EnvironmentStatus CheckEnvironment(byte[] frameBytes)
    {
        if (frameBytes == null || frameBytes.Length == 0)
            return new EnvironmentStatus { IsLightingAdequate = false, WarningMessage = "No camera data" };

        try
        {
            // Simple heuristic: Sample every Nth byte of the JPEG data
            // Note: This is a very rough approximation as JPEG is compressed.
            // In a production app, we would decode a 16x16 thumbnail for accurate analysis.
            
            long total = 0;
            int count = 0;
            byte min = 255;
            byte max = 0;

            // Sampling every 100th byte as a proxy for raw pixel data (rough)
            for (int i = 0; i < frameBytes.Length; i += 100)
            {
                byte val = frameBytes[i];
                total += val;
                if (val < min) min = val;
                if (val > max) max = val;
                count++;
            }

            double averageBrightness = count > 0 ? (double)total / count : 128;
            double contrast = max - min;

            var status = new EnvironmentStatus
            {
                BrightnessLevel = averageBrightness,
                IsLightingAdequate = averageBrightness > LowBrightnessThreshold && averageBrightness < HighBrightnessThreshold,
                IsContrastAdequate = contrast > LowContrastThreshold
            };

            if (!status.IsLightingAdequate)
            {
                status.WarningMessage = averageBrightness <= LowBrightnessThreshold 
                    ? "Environment too dark. Please turn on more lights." 
                    : "Environment too bright. Avoid direct glare.";
            }
            else if (!status.IsContrastAdequate)
            {
                status.WarningMessage = "Low contrast detected. Ensure your hands are clearly visible against the background.";
            }

            return status;
        }
        catch (Exception ex)
        {
            Debug.WriteLine($"Environment detection failed: {ex.Message}");
            return new EnvironmentStatus();
        }
    }
}
