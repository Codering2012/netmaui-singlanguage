namespace SignLanguageApi.Dtos
{
    public class Landmark3DDto
    {
        public double X { get; set; }
        public double Y { get; set; }
        public double Z { get; set; }
    }

    public class GestureLandmarksRequestDto
    {
        public List<Landmark3DDto> Landmarks { get; set; } = [];
    }

    /// <summary>
    /// Represents a single hand landmark coordinate (0-1 normalized values)
    /// </summary>
    public class CoordinateDto
    {
        public double X { get; set; }
        public double Y { get; set; }
    }

    /// <summary>
    /// Contains the gesture prediction data with 21 hand landmarks
    /// </summary>
    public class GesturePredictionDataDto
    {
        public int Count { get; set; }
        public int Sequence { get; set; }
        public List<CoordinateDto> Coordinates { get; set; } = [];
        public string? Letter { get; set; }
        public float Confidence { get; set; }
        public double ProcessingTimeMs { get; set; }
    }

    /// <summary>
    /// Response from the gesture prediction API endpoint
    /// </summary>
    public class GesturePredictionResponseDto
    {
        public string Status { get; set; } = string.Empty;
        public string Message { get; set; } = string.Empty;
        public GesturePredictionDataDto? Data { get; set; }
    }
}
