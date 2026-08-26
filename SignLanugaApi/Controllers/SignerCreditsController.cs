using Microsoft.AspNetCore.Authorization;
using Microsoft.AspNetCore.Mvc;
using SignLanguageApi.Dtos;
using System.Collections.Generic;
using System.Threading.Tasks;

namespace SignLanguageApi.Controllers
{
    [ApiController]
    [Route("api/signer-credits")]
    [Authorize]
    public class SignerCreditsController : ControllerBase
    {
        [HttpGet]
        public ActionResult<List<SignerCreditDto>> GetSignerCredits()
        {
            var credits = new List<SignerCreditDto>
            {
                new SignerCreditDto
                {
                    SignerName = "Dr. Bill Vicars (Lifeprint)",
                    AvatarUrl = "https://images.unsplash.com/photo-1544717305-2782549b5136?w=200",
                    SocialLinks = "https://www.lifeprint.com/",
                    LicenseType = "Educational / Public Use",
                    ContributedVideosCount = 142,
                    Bio = "Founder of Lifeprint.com, providing free high-quality ASL instruction to millions.",
                    SourceUrl = "https://www.lifeprint.com/"
                },
                new SignerCreditDto
                {
                    SignerName = "SignSchool",
                    AvatarUrl = "https://images.unsplash.com/photo-1531427186611-ecfd6d936c79?w=200",
                    SocialLinks = "https://www.signschool.com/",
                    LicenseType = "Creative Commons",
                    ContributedVideosCount = 85,
                    Bio = "An online platform offering free, interactive ASL learning tools and community-driven video resources.",
                    SourceUrl = "https://www.signschool.com/"
                },
                new SignerCreditDto
                {
                    SignerName = "Wikimedia Commons ASL",
                    AvatarUrl = "https://images.unsplash.com/photo-1491336477066-31156b5e4f35?w=200",
                    SocialLinks = "https://commons.wikimedia.org/",
                    LicenseType = "Public Domain / CC-BY-SA",
                    ContributedVideosCount = 47,
                    Bio = "Open-source contributor videos documenting American Sign Language for free knowledge dissemination.",
                    SourceUrl = "https://commons.wikimedia.org/wiki/Category:Videos_in_American_Sign_Language"
                },
                new SignerCreditDto
                {
                    SignerName = "ASL Connect",
                    AvatarUrl = "https://images.unsplash.com/photo-1573496359142-b8d87734a5a2?w=200",
                    SocialLinks = "https://gallaudet.edu/asl-connect/",
                    LicenseType = "Educational",
                    ContributedVideosCount = 110,
                    Bio = "High-quality ASL learning resources and modules from Gallaudet University.",
                    SourceUrl = "https://gallaudet.edu/"
                },
                new SignerCreditDto
                {
                    SignerName = "Handspeak",
                    AvatarUrl = "https://images.unsplash.com/photo-1542596594-649edbc13630?w=200",
                    SocialLinks = "https://www.handspeak.com/",
                    LicenseType = "Educational",
                    ContributedVideosCount = 230,
                    Bio = "A comprehensive online ASL dictionary, lessons, and cultural resources.",
                    SourceUrl = "https://www.handspeak.com/"
                }
            };

            return Ok(credits);
        }
    }
}
