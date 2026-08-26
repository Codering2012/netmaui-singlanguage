namespace SignLanguageApp.Services;

public interface IEnvironmentDetectionService
{
    EnvironmentStatus CheckEnvironment(byte[] frameBytes);
}

public class EnvironmentStatus
{
    public bool IsLightingAdequate { get; set; } = true;
    public bool IsContrastAdequate { get; set; } = true;
    public string WarningMessage { get; set; } = string.Empty;
    public double BrightnessLevel { get; set; }
}
