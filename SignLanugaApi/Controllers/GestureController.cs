using Microsoft.AspNetCore.Authorization;
using Microsoft.AspNetCore.Mvc;
using SignLanguageApi.Dtos;
using SignLanguageApi.Services;

namespace SignLanguageApi.Controllers
{
    [ApiController]
    [Route("api/[controller]")]
    public class GestureController : ControllerBase
    {
        private readonly IGestureRecognitionService _gestureService;
        private readonly ILogger<GestureController> _logger;

        public GestureController(
            IGestureRecognitionService gestureService,
            ILogger<GestureController> logger)
        {
            _gestureService = gestureService;
            _logger = logger;
        }

        /// <summary>
        /// ENDPOINT 1: For the fast Python Client (Raw array of 63 floats)
        /// </summary>
        [HttpPost("predict-landmarks")]
        [Consumes("application/json")]
        [Produces("application/json")]
        public async Task<IActionResult> PredictLandmarks(
            [FromBody] LandmarkRequestDto request,
            CancellationToken cancellationToken = default)
        {
            try
            {
                if (request?.RawLandmarks == null || request.RawLandmarks.Length != 63)
                {
                    return BadRequest(new GesturePredictionResponseDto
                    {
                        Status = "error",
                        Message = "Invalid payload. Expected 'rawLandmarks' array of exactly 63 floats.",
                        Data = null
                    });
                }

                var result = await _gestureService.PredictFromLandmarksAsync(request.RawLandmarks, cancellationToken);
                return Ok(result);
            }
            catch (Exception ex)
            {
                _logger.LogError(ex, "Error in landmark prediction endpoint");
                return StatusCode(500, new GesturePredictionResponseDto { Status = "error", Message = "Internal Server Error" });
            }
        }

        /// <summary>
        /// ENDPOINT 2: For the .NET MAUI App (Full JPEG Images)
        /// </summary>
        [HttpPost("predict")]
        [Consumes("multipart/form-data")]
        [Produces("application/json")]
        public async Task<ActionResult<GesturePredictionResponseDto>> Predict(
            IFormFile image,
            CancellationToken cancellationToken = default)
        {
            try
            {
                if (image == null || image.Length == 0 || image.Length > 5 * 1024 * 1024)
                {
                    return BadRequest(new GesturePredictionResponseDto
                    {
                        Status = "error",
                        Message = "Invalid image file. Must be between 1 byte and 5MB.",
                        Data = null
                    });
                }

                using var memoryStream = new MemoryStream();
                await image.CopyToAsync(memoryStream, cancellationToken);
                var imageData = memoryStream.ToArray();

                var result = await _gestureService.PredictGestureAsync(imageData, cancellationToken);
                return Ok(result);
            }
            catch (Exception ex)
            {
                _logger.LogError(ex, "Error in image prediction endpoint");
                return StatusCode(500, new GesturePredictionResponseDto { Status = "error", Message = "Internal Server Error" });
            }
        }

        [HttpGet("health")]
        [AllowAnonymous]
        public ActionResult<object> Health() => Ok(new { status = "healthy", timestamp = DateTime.UtcNow });
    }
}