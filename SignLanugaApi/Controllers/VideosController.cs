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
        private static readonly string VideoFolder = SignLanguageApi.Helpers.PathResolver.ResolveFolder("VIDEO");
        private static readonly string AslLettersFolder = SignLanguageApi.Helpers.PathResolver.ResolveFolder("ASL_LETTERS");
        private static readonly ConcurrentDictionary<int, int> Likes = new();
        private static readonly ConcurrentDictionary<int, int> Views = new();
        private static readonly object ScanLock = new();
        private static List<VideoDto>? _cachedVideos;
        private static DateTime _lastScan = DateTime.MinValue;

        private List<VideoDto> GetVideos()
        {
            lock (ScanLock)
            {
                if (_cachedVideos != null && (DateTime.UtcNow - _lastScan).TotalSeconds < 10)
                    return _cachedVideos;

                if (!Directory.Exists(VideoFolder))
                    Directory.CreateDirectory(VideoFolder);

                if (!Directory.Exists(AslLettersFolder))
                    Directory.CreateDirectory(AslLettersFolder);

                var videos = new List<VideoDto>();
                int id = 1;

                // 1. Scan General VIDEO folder
                var videoFiles = Directory.GetFiles(VideoFolder, "*.mp4");
                foreach (var file in videoFiles)
                {
                    var fileName = Path.GetFileName(file);
                    var title = Path.GetFileNameWithoutExtension(file);
                    var category = "General";
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

                // 2. Scan ASL_LETTERS folder (A.mp4 - Z.mp4)
                var letterFiles = Directory.GetFiles(AslLettersFolder, "*.mp4")
                                          .OrderBy(f => Path.GetFileNameWithoutExtension(f))
                                          .ToList();
                foreach (var file in letterFiles)
                {
                    var fileName = Path.GetFileName(file);
                    var letterName = Path.GetFileNameWithoutExtension(file);
                    videos.Add(new VideoDto
                    {
                        Id = id,
                        FileName = fileName,
                        Title = $"ASL Letter: {letterName.ToUpper()}",
                        Category = "Alphabet",
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

        [HttpGet("{id:int}")]
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

        [HttpPost("{id:int}/like")]
        public ActionResult<bool> Like(int id)
        {
            Likes.AddOrUpdate(id, 1, (k, v) => v + 1);
            return Ok(true);
        }

        [HttpPost("{id:int}/unlike")]
        public ActionResult<bool> Unlike(int id)
        {
            Likes.AddOrUpdate(id, 0, (k, v) => v > 0 ? v - 1 : 0);
            return Ok(true);
        }

        [HttpPost("{id:int}/watch")]
        public ActionResult<bool> Watch(int id)
        {
            Views.AddOrUpdate(id, 1, (k, v) => v + 1);
            return Ok(true);
        }

        [HttpGet("stream/{id:int}")]
        [AllowAnonymous]
        public IActionResult Stream(int id)
        {
            var videos = GetVideos();
            var video = videos.FirstOrDefault(v => v.Id == id);
            if (video == null) return NotFound();

            string filePath = Path.Combine(VideoFolder, video.FileName);
            if (!System.IO.File.Exists(filePath))
            {
                filePath = Path.Combine(AslLettersFolder, video.FileName);
                if (!System.IO.File.Exists(filePath))
                {
                    filePath = video.FileName;
                    if (!System.IO.File.Exists(filePath))
                    {
                        return NotFound();
                    }
                }
            }

            var stream = System.IO.File.OpenRead(filePath);
            return File(stream, "video/mp4", enableRangeProcessing: true);
        }

        [HttpGet("letter/{letter}")]
        [AllowAnonymous]
        public IActionResult StreamLetter(string letter)
        {
            if (string.IsNullOrWhiteSpace(letter)) return BadRequest();
            var cleanLetter = letter.Trim().ToUpper();
            var filePath = Path.Combine(AslLettersFolder, $"{cleanLetter}.mp4");
            if (!System.IO.File.Exists(filePath))
            {
                filePath = Path.Combine(VideoFolder, $"Letters_Alphabet {cleanLetter}.mp4");
                if (!System.IO.File.Exists(filePath)) return NotFound();
            }
            var stream = System.IO.File.OpenRead(filePath);
            return File(stream, "video/mp4", enableRangeProcessing: true);
        }
    }
}
