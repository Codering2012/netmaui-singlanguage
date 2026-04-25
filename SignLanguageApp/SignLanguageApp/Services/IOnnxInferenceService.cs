namespace SignLanguageApp.Services
{
    public interface IOnnxInferenceService
    {
        Task InitializeAsync(string modelPath);
        Task<float[]> PredictAsync(byte[] imageData);
        void Dispose();
    }
}
