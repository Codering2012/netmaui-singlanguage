namespace SignLanguageApi.Dtos
{
    public class LandmarkRequestDto
    {
        public float[] RawLandmarks { get; set; } = Array.Empty<float>();
    }
}