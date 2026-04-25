using Microsoft.AspNetCore.Authorization;
using Microsoft.AspNetCore.Mvc;
using SignLanguageApi.Dtos;
using System.Collections.Concurrent;

namespace SignLanguageApi.Controllers
{
    [ApiController]
    [Route("api/[controller]")]
    [Authorize]
    public class VideosController : ControllerBase
    {
        private static readonly string VideoFolder = Path.Combine(Directory.GetCurrentDirectory(), "VIDEO");
        private static readonly ConcurrentDictionary<int, int> Likes = new();
        private static readonly ConcurrentDictionary<int, int> Views = new();
        private static readonly object ScanLock = new();
        private static List<VideoDto>? _cachedVideos;
        private static DateTime _lastScan = DateTime.MinValue;

        // Scan the VIDEO folder for .mp4 files and cache results for 10 seconds
        private List<VideoDto> GetVideos()
        {
            lock (ScanLock)
            {
                if (_cachedVideos != null && (DateTime.UtcNow - _lastScan).TotalSeconds < 10)
                    return _cachedVideos;

                if (!Directory.Exists(VideoFolder))
                    Directory.CreateDirectory(VideoFolder);

                var files = Directory.GetFiles(VideoFolder, "*.mp4");
                var videos = new List<VideoDto>();
                int id = 1;
                foreach (var file in files)
                {
                    var fileName = Path.GetFileName(file);
                    var title = Path.GetFileNameWithoutExtension(file);
                    var category = "General";
                    // Optionally, parse category from file name (e.g., cat_title.mp4)
                    var parts = title.Split('_', 2);
                    if (parts.Length == 2)
                    {
                        category = parts[0];
                        title = parts[1];
                    }
                    videos.Add(new VideoDto
                    {
                        Id = id,
                        FileName = fileName,
                        Title = title,
                        Category = category,
                        Path = $"/api/videos/stream/{id}",
                        Likes = Likes.GetValueOrDefault(id, 0),
                        Views = Views.GetValueOrDefault(id, 0)
                    });
                    id++;
                }
                _cachedVideos = videos;
                _lastScan = DateTime.UtcNow;
                return videos;
            }
        }

        [HttpGet]
        public ActionResult<List<VideoDto>> GetAll()
        {
            var videos = GetVideos();
            return Ok(videos);
        }

        [HttpGet("{id}")]
        public ActionResult<VideoDto> GetById(int id)
        {
            var videos = GetVideos();
            var video = videos.FirstOrDefault(v => v.Id == id);
            if (video == null) return NotFound();
            return Ok(video);
        }

        [HttpGet("category/{cat}")]
        public ActionResult<List<VideoDto>> GetByCategory(string cat)
        {
            var videos = GetVideos();
            var filtered = videos.Where(v => v.Category.Equals(cat, StringComparison.OrdinalIgnoreCase)).ToList();
            return Ok(filtered);
        }

        [HttpPost("{id}/like")]
        public ActionResult<bool> Like(int id)
        {
            Likes.AddOrUpdate(id, 1, (k, v) => v + 1);
            return Ok(true);
        }

        [HttpPost("{id}/unlike")]
        public ActionResult<bool> Unlike(int id)
        {
            Likes.AddOrUpdate(id, 0, (k, v) => v > 0 ? v - 1 : 0);
            return Ok(true);
        }

        [HttpPost("{id}/watch")]
        public ActionResult<bool> Watch(int id)
        {
            Views.AddOrUpdate(id, 1, (k, v) => v + 1);
            return Ok(true);
        }

        // Optional: Stream video file by id
        [HttpGet("stream/{id}")]
        [AllowAnonymous]
        public IActionResult Stream(int id)
        {
            var videos = GetVideos();
            var video = videos.FirstOrDefault(v => v.Id == id);
            if (video == null) return NotFound();
            var filePath = Path.Combine(VideoFolder, video.FileName);
            if (!System.IO.File.Exists(filePath)) return NotFound();
            var stream = System.IO.File.OpenRead(filePath);
            return File(stream, "video/mp4", enableRangeProcessing: true);
        }
    }
}
