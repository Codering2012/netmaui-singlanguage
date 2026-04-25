using SignLanguageApi.Dtos;

namespace SignLanguageApi.Services
{
    public interface IGestureRecognitionService
    {
        Task<GesturePredictionResponseDto> PredictGestureAsync(byte[] imageData, CancellationToken cancellationToken = default);
        Task<GesturePredictionResponseDto> PredictFromLandmarksAsync(float[] rawLandmarks, CancellationToken cancellationToken = default);
        bool ValidateImageData(byte[] imageData);
    }
}