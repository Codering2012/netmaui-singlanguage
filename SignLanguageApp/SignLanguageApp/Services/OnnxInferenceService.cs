using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;

namespace SignLanguageApp.Services
{
    public class OnnxInferenceService : IOnnxInferenceService
    {
        private InferenceSession? _session;
        private bool _isInitialized;

        public async Task InitializeAsync(string modelPath)
        {
            try
            {
                // Load model from application resources
                var assembly = typeof(OnnxInferenceService).Assembly;
                var resourcePath = $"{assembly.GetName().Name}.Resources.Raw.{modelPath}";

                using (var stream = assembly.GetManifestResourceStream(resourcePath))
                {
                    if (stream == null)
                    {
                        throw new FileNotFoundException($"Model file '{modelPath}' not found in resources.");
                    }

                    using (var memoryStream = new MemoryStream())
                    {
                        await stream.CopyToAsync(memoryStream);
                        var modelBytes = memoryStream.ToArray();

                        var sessionOptions = new SessionOptions();
                        _session = new InferenceSession(modelBytes, sessionOptions);
                        _isInitialized = true;
                    }
                }
            }
            catch (Exception ex)
            {
                throw new InvalidOperationException($"Failed to initialize ONNX model: {ex.Message}", ex);
            }
        }

        public async Task<float[]> PredictAsync(byte[] imageData)
        {
            if (!_isInitialized || _session == null)
            {
                throw new InvalidOperationException("ONNX service is not initialized. Call InitializeAsync first.");
            }

            try
            {
                // Convert image data to tensor (assuming grayscale image or specific preprocessing needed)
                var inputData = await ProcessImageDataAsync(imageData);

                // Get input node names from the model
                var inputNames = _session.InputNames.ToList();
                if (inputNames.Count == 0)
                {
                    throw new InvalidOperationException("Model has no input nodes.");
                }

                // Create tensor with preprocessed data
                var inputTensor = new DenseTensor<float>(inputData, new int[] { 1, 224, 224, 3 });
                var inputs = new List<NamedOnnxValue>
                {
                    NamedOnnxValue.CreateFromTensor(inputNames[0], inputTensor)
                };

                // Run inference
                using (var results = _session.Run(inputs))
                {
                    var output = results.FirstOrDefault()?.AsEnumerable<float>().ToArray() ?? Array.Empty<float>();
                    return output;
                }
            }
            catch (Exception ex)
            {
                throw new InvalidOperationException($"Inference failed: {ex.Message}", ex);
            }
        }

        private Task<float[]> ProcessImageDataAsync(byte[] imageData)
        {
            // TODO: Implement proper image preprocessing
            // This should convert the image data to the format expected by the model
            // For hand pose estimation, typically involves:
            // 1. Resize to model input size (usually 224x224 or similar)
            // 2. Normalize pixel values
            // 3. Convert to appropriate tensor format

            // Placeholder implementation
            var floatData = new float[224 * 224 * 3];
            for (int i = 0; i < imageData.Length && i < floatData.Length; i++)
            {
                floatData[i] = imageData[i] / 255.0f;
            }

            return Task.FromResult(floatData);
        }

        public void Dispose()
        {
            _session?.Dispose();
            _isInitialized = false;
        }
    }
}
