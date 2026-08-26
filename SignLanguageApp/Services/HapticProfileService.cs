using System.Diagnostics;
using Microsoft.Maui.Controls;
using Microsoft.Maui.Devices;

namespace SignLanguageApp.Services;

public interface IHapticProfileService
{
    bool IsHapticsEnabled { get; set; }
    void PerformMicroTick();
    void PerformStandardClick();
    void PerformSuccessPattern();
    void VibrateSuccess();
    void VibrateFailure();
    void VibrateAttention();
}

public class HapticProfileService : IHapticProfileService
{
    public bool IsHapticsEnabled
    {
        get
        {
            try { return Preferences.Get("HapticsEnabled", false); } catch { return false; }
        }
        set
        {
            try { Preferences.Set("HapticsEnabled", value); } catch { }
        }
    }

    public void PerformMicroTick()
    {
        if (!IsHapticsEnabled) return;
        try
        {
            if (HapticFeedback.Default.IsSupported)
            {
                HapticFeedback.Default.Perform(HapticFeedbackType.Click);
            }
        }
        catch (Exception ex)
        {
            Debug.WriteLine($"[HapticProfileService] MicroTick error: {ex.Message}");
        }
    }

    public void PerformStandardClick()
    {
        if (!IsHapticsEnabled) return;
        try
        {
            if (HapticFeedback.Default.IsSupported)
            {
                HapticFeedback.Default.Perform(HapticFeedbackType.Click);
            }
        }
        catch (Exception ex)
        {
            Debug.WriteLine($"[HapticProfileService] StandardClick error: {ex.Message}");
        }
    }

    public void PerformSuccessPattern()
    {
        if (!IsHapticsEnabled) return;
        try
        {
            if (HapticFeedback.Default.IsSupported)
            {
                HapticFeedback.Default.Perform(HapticFeedbackType.LongPress);
            }
        }
        catch (Exception ex)
        {
            Debug.WriteLine($"[HapticProfileService] SuccessPattern error: {ex.Message}");
        }
    }

    public void VibrateSuccess()
    {
        PerformSuccessPattern();
    }

    public void VibrateFailure()
    {
        if (!IsHapticsEnabled) return;
        try
        {
            if (HapticFeedback.Default.IsSupported)
            {
                HapticFeedback.Default.Perform(HapticFeedbackType.LongPress);
            }
        }
        catch (Exception ex)
        {
            Debug.WriteLine($"[HapticProfileService] Haptic error: {ex.Message}");
        }
    }

    public void VibrateAttention()
    {
        if (!IsHapticsEnabled) return;
        try
        {
            if (HapticFeedback.Default.IsSupported)
            {
                HapticFeedback.Default.Perform(HapticFeedbackType.Click);
            }
        }
        catch (Exception ex)
        {
            Debug.WriteLine($"[HapticProfileService] Haptic error: {ex.Message}");
        }
    }
}
