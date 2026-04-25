using System.Diagnostics;
using System.Text.Json;
using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;
using SignLanguageApi.Dtos;

namespace SignLanguageApi.Services
{
    public class GestureRecognitionService : IGestureRecognitionService
    {
        private readonly ILogger<GestureRecognitionService> _logger;
        private const float ConfidenceThreshold = 0.45f;

        private static readonly string[] AslLetters = {
            "A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K", "L", "M",
            "N", "O", "P", "Q", "R", "S", "T", "U", "V", "W", "X", "Y", "Z", "del", "space"
        };

        // Thread-safe Lazy loading for ONNX and the Python MediaPipe Worker
        private static readonly Lazy<ModelRuntime?> SharedModelRuntime =
            new(LoadSharedModelRuntime, LazyThreadSafetyMode.ExecutionAndPublication);

        private static readonly Lazy<MediaPipeWorkerClient?> SharedMediaPipeWorker =
            new(LoadSharedMediaPipeWorker, LazyThreadSafetyMode.ExecutionAndPublication);

        private sealed class ModelRuntime
        {
            public InferenceSession Session { get; init; } = default!;
            public string InputName { get; init; } = string.Empty;
        }

        public GestureRecognitionService(ILogger<GestureRecognitionService> logger)
        {
            _logger = logger;
        }

        public bool ValidateImageData(byte[] imageData)
        {
            if (imageData == null || imageData.Length == 0) return false;
            if (imageData.Length < 100 || imageData.Length > 5 * 1024 * 1024) return false;
            if (imageData[0] != 0xFF || imageData[1] != 0xD8) return false; // JPEG check
            return true;
        }

        /// <summary>
        /// Processes raw images from the MAUI App using the background Python MediaPipe worker.
        /// </summary>
        public async Task<GesturePredictionResponseDto> PredictGestureAsync(byte[] imageData, CancellationToken cancellationToken = default)
        {
            if (!ValidateImageData(imageData))
            {
                return new GesturePredictionResponseDto { Status = "error", Message = "Invalid JPEG image." };
            }

            var sw = Stopwatch.StartNew();
            var worker = SharedMediaPipeWorker.Value;

            if (worker == null)
            {
                return new GesturePredictionResponseDto { Status = "error", Message = "MediaPipe worker offline." };
            }

            // 1. Send image to background Python script to extract hand landmarks
            var responseJson = await worker.TryExtractAsync(imageData, cancellationToken);
            if (string.IsNullOrWhiteSpace(responseJson))
            {
                return new GesturePredictionResponseDto { Status = "low_confidence", Message = "No hand detected" };
            }

            using var doc = JsonDocument.Parse(responseJson);
            if (!doc.RootElement.TryGetProperty("success", out var successEl) || !successEl.GetBoolean() ||
                !doc.RootElement.TryGetProperty("landmarks", out var lmEl))
            {
                return new GesturePredictionResponseDto { Status = "low_confidence", Message = "No hand detected" };
            }

            // 2. Pass extracted landmarks into the ONNX pipeline
            var rawLandmarks = lmEl.EnumerateArray().Select(v => v.GetSingle()).ToArray();
            var result = await PredictFromLandmarksAsync(rawLandmarks, cancellationToken);

            if (result.Data != null)
            {
                result.Data.ProcessingTimeMs = sw.ElapsedMilliseconds; // Override with total time
            }

            return result;
        }

        /// <summary>
        /// Directly processes pre-extracted landmarks from the Python Client.
        /// </summary>
        public async Task<GesturePredictionResponseDto> PredictFromLandmarksAsync(float[] rawLandmarks, CancellationToken cancellationToken = default)
        {
            if (rawLandmarks == null || rawLandmarks.Length != 63)
            {
                return new GesturePredictionResponseDto { Status = "error", Message = "Invalid landmarks payload." };
            }

            var sw = Stopwatch.StartNew();

            // 1. Normalize points to the wrist
            var normalizedCoordinates = NormalizeHandCoordinates(rawLandmarks);

            // 2. Predict with ONNX
            var (predictedLetter, confidence) = await InvokeModelAsync(normalizedCoordinates, cancellationToken);
            sw.Stop();

            // 3. Return Raw Result (Stateless - Client handles temporal smoothing)
            if (confidence < ConfidenceThreshold)
            {
                _logger.LogInformation("Prediction low confidence. Letter={Letter}, Confidence={Confidence:F4}, ProcessingTimeMs={ProcessingTimeMs}",
                    predictedLetter, confidence, sw.ElapsedMilliseconds);
                return new GesturePredictionResponseDto { Status = "low_confidence", Message = "Low Confidence..." };
            }

            _logger.LogInformation("Prediction success. Letter={Letter}, Confidence={Confidence:F4}, ProcessingTimeMs={ProcessingTimeMs}",
                predictedLetter, confidence, sw.ElapsedMilliseconds);

            return new GesturePredictionResponseDto
            {
                Status = "success",
                Message = $"Sign: {predictedLetter}",
                Data = new GesturePredictionDataDto
                {
                    Count = 21,
                    Coordinates = ConvertTo2DCoordinates(rawLandmarks),
                    Letter = predictedLetter,
                    Confidence = confidence,
                    ProcessingTimeMs = sw.ElapsedMilliseconds
                }
            };
        }

        // --- MATH & ONNX LOGIC ---
        private float[] NormalizeHandCoordinates(float[] handLandmarks)
        {
            var points = new float[21, 3];
            for (int i = 0; i < 21; i++)
            {
                points[i, 0] = handLandmarks[i * 3];
                points[i, 1] = handLandmarks[i * 3 + 1];
                points[i, 2] = handLandmarks[i * 3 + 2];
            }

            float wristX = points[0, 0], wristY = points[0, 1], wristZ = points[0, 2];
            float maxDistance = 0f;

            for (int i = 0; i < 21; i++)
            {
                points[i, 0] -= wristX;
                points[i, 1] -= wristY;
                points[i, 2] -= wristZ;

                float distance = MathF.Sqrt(points[i, 0] * points[i, 0] + points[i, 1] * points[i, 1] + points[i, 2] * points[i, 2]);
                if (distance > maxDistance) maxDistance = distance;
            }

            maxDistance = MathF.Max(maxDistance, 1e-5f);
            var normalized = new float[63];

            for (int i = 0; i < 21; i++)
            {
                normalized[i * 3] = points[i, 0] / maxDistance;
                normalized[i * 3 + 1] = points[i, 1] / maxDistance;
                normalized[i * 3 + 2] = points[i, 2] / maxDistance;
            }

            return normalized;
        }

        private async Task<(string letter, float confidence)> InvokeModelAsync(float[] normalizedCoordinates, CancellationToken ct)
        {
            var runtime = SharedModelRuntime.Value ?? throw new InvalidOperationException("ONNX model unavailable.");
            var inputTensor = new DenseTensor<float>(normalizedCoordinates, new[] { 1, 63 });

            using var output = runtime.Session.Run(new[] { NamedOnnxValue.CreateFromTensor(runtime.InputName, inputTensor) });
            var logits = output.First().AsEnumerable<float>().ToArray();
            var probabilities = Softmax(logits);

            var bestIndex = Array.IndexOf(probabilities, probabilities.Max());
            return await Task.FromResult((AslLetters[Math.Min(bestIndex, AslLetters.Length - 1)], probabilities[bestIndex]));
        }

        private List<CoordinateDto> ConvertTo2DCoordinates(float[] handLandmarks)
        {
            var coordinates = new List<CoordinateDto>(21);
            for (int i = 0; i < 21; i++)
                coordinates.Add(new CoordinateDto { X = handLandmarks[i * 3], Y = handLandmarks[i * 3 + 1] });
            return coordinates;
        }

        private static float[] Softmax(float[] logits)
        {
            var max = logits.Max();
            var exps = logits.Select(x => MathF.Exp(x - max)).ToArray();
            var sum = exps.Sum();
            return sum <= 0 ? exps : exps.Select(x => x / sum).ToArray();
        }

        // --- BACKGROUND WORKER & STARTUP ---
        private static ModelRuntime? LoadSharedModelRuntime()
        {
            var candidates = new[] { Path.Combine(Directory.GetCurrentDirectory(), "asl_model.onnx"), Path.Combine(AppContext.BaseDirectory, "asl_model.onnx") };
            var modelPath = candidates.FirstOrDefault(File.Exists);
            if (modelPath == null) return null;

            var session = new InferenceSession(modelPath, new Microsoft.ML.OnnxRuntime.SessionOptions
            {
                InterOpNumThreads = Environment.ProcessorCount,
                IntraOpNumThreads = Environment.ProcessorCount,
                ExecutionMode = ExecutionMode.ORT_PARALLEL
            });
            return new ModelRuntime { Session = session, InputName = session.InputMetadata.Keys.First() };
        }

        private static MediaPipeWorkerClient? LoadSharedMediaPipeWorker()
        {
            var candidates = new[] { Path.Combine(Directory.GetCurrentDirectory(), "scripts", "mediapipe_extract_landmarks.py"), Path.Combine(AppContext.BaseDirectory, "scripts", "mediapipe_extract_landmarks.py") };
            var scriptPath = candidates.FirstOrDefault(File.Exists);
            if (scriptPath == null) return null;

            var pythonExe = Environment.GetEnvironmentVariable("MEDIAPIPE_PYTHON_EXE") ?? "python";
            return MediaPipeWorkerClient.Start(pythonExe, scriptPath);
        }

        private sealed class MediaPipeWorkerClient : IDisposable
        {
            private readonly Process _process;
            private readonly StreamWriter _stdin;
            private readonly StreamReader _stdout;
            private readonly SemaphoreSlim _requestLock = new(1, 1);

            private MediaPipeWorkerClient(Process process)
            {
                _process = process;
                _stdin = process.StandardInput;
                _stdout = process.StandardOutput;
            }

            public static MediaPipeWorkerClient? Start(string pythonExe, string scriptPath)
            {
                try
                {
                    var process = Process.Start(new ProcessStartInfo
                    {
                        FileName = pythonExe,
                        Arguments = $"\"{scriptPath}\" --worker",
                        UseShellExecute = false,
                        RedirectStandardInput = true,
                        RedirectStandardOutput = true,
                        CreateNoWindow = true
                    });
                    return process != null ? new MediaPipeWorkerClient(process) : null;
                }
                catch { return null; }
            }

            public async Task<string?> TryExtractAsync(byte[] imageData, CancellationToken ct)
            {
                if (_process.HasExited) return null;
                var payload = JsonSerializer.Serialize(new { imageBase64 = Convert.ToBase64String(imageData) });

                await _requestLock.WaitAsync(ct);
                try
                {
                    await _stdin.WriteLineAsync(payload.AsMemory(), ct);
                    await _stdin.FlushAsync(ct);
                    return await _stdout.ReadLineAsync(ct);
                }
                catch { return null; }
                finally { _requestLock.Release(); }
            }

            public void Dispose()
            {
                try { if (!_process.HasExited) _process.Kill(true); } catch { }
                _process.Dispose(); _requestLock.Dispose();
            }
        }
    }
}