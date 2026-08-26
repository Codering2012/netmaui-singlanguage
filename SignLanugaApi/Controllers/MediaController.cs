using Microsoft.AspNetCore.Authorization;
using Microsoft.AspNetCore.Http;
using Microsoft.AspNetCore.Mvc;
using Microsoft.AspNetCore.SignalR;
using System;
using System.IO;
using System.Threading.Tasks;

namespace SignLanguageApi.Controllers
{
    [ApiController]
    [Route("api/[controller]")]
    [Authorize]
    public class MediaController : ControllerBase
    {
        private readonly Microsoft.AspNetCore.SignalR.IHubContext<Hubs.BroadcastHub> _hubContext;
        private static readonly string MediaFolder = SignLanguageApi.Helpers.PathResolver.ResolveFolder("MEDIA");
        private static readonly string VideoFolder = SignLanguageApi.Helpers.PathResolver.ResolveFolder("VIDEO");

        public MediaController(Microsoft.AspNetCore.SignalR.IHubContext<Hubs.BroadcastHub> hubContext)
        {
            _hubContext = hubContext;
            if (!Directory.Exists(MediaFolder)) Directory.CreateDirectory(MediaFolder);
            if (!Directory.Exists(VideoFolder)) Directory.CreateDirectory(VideoFolder);
        }

        [HttpPost("upload")]
        [RequestSizeLimit(52428800)] // 50MB
        public async Task<IActionResult> Upload(IFormFile file, [FromQuery] string type = "image")
        {
            if (file == null || file.Length == 0) return BadRequest("No file uploaded.");

            var extension = Path.GetExtension(file.FileName).ToLower();
            string targetFolder = type == "video" ? VideoFolder : MediaFolder;
            
            // Basic signature check for images
            if (type == "image" && extension != ".jpg" && extension != ".jpeg" && extension != ".png")
                return BadRequest("Only JPG and PNG images are allowed.");
            
            if (type == "video" && extension != ".mp4")
                return BadRequest("Only MP4 videos are allowed.");

            var fileName = $"{Guid.NewGuid()}{extension}";
            var filePath = Path.Combine(targetFolder, fileName);

            using (var stream = new FileStream(filePath, FileMode.Create))
            {
                await file.CopyToAsync(stream);
            }

            var url = $"/api/media/{(type == "video" ? "video" : "image")}/{fileName}";
            
            // Broadcast the new media to all connected clients
            await _hubContext.Clients.All.SendAsync("NewMediaAdded", type, fileName, url);

            return Ok(new { url, fileName });
        }

        [HttpGet("image/{fileName}")]
        [AllowAnonymous]
        public IActionResult GetImage(string fileName)
        {
            var safeFileName = Path.GetFileName(fileName);
            var filePath = Path.Combine(MediaFolder, safeFileName);
            if (!System.IO.File.Exists(filePath)) return NotFound();

            var extension = Path.GetExtension(safeFileName).ToLower();
            var contentType = extension == ".png" ? "image/png" : "image/jpeg";
            return PhysicalFile(filePath, contentType);
        }

        [HttpGet]
        public IActionResult GetAllMedia()
        {
            if (!Directory.Exists(MediaFolder)) return Ok(new List<Dtos.MediaDto>());

            var files = Directory.GetFiles(MediaFolder);
            var mediaList = files.Select(f => new Dtos.MediaDto
            {
                FileName = Path.GetFileName(f),
                Url = $"/api/media/image/{Path.GetFileName(f)}",
                Type = "image",
                UploadedAt = System.IO.File.GetCreationTimeUtc(f)
            }).OrderByDescending(m => m.UploadedAt).ToList();

            return Ok(mediaList);
        }
    }
}
