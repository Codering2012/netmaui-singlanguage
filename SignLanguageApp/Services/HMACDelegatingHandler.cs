using System.Net.Http;
using System.Security.Cryptography;
using System.Text;
using System.Threading;
using System.Threading.Tasks;

namespace SignLanguageApp.Services;

public class HMACDelegatingHandler : DelegatingHandler
{
    private const string SharedSecret = "GlobalInterface-V2-Super-Secret-Key";

    protected override async Task<HttpResponseMessage> SendAsync(HttpRequestMessage request, CancellationToken cancellationToken)
    {
        // Add Unix Timestamp
        string timestamp = DateTimeOffset.UtcNow.ToUnixTimeSeconds().ToString();
        request.Headers.Add("X-Timestamp", timestamp);

        // Calculate HMAC
        string contentHash = string.Empty;
        if (request.Content != null)
        {
            byte[] contentBytes = await request.Content.ReadAsByteArrayAsync(cancellationToken);
            using var sha256 = SHA256.Create();
            byte[] hashBytes = sha256.ComputeHash(contentBytes);
            contentHash = Convert.ToBase64String(hashBytes);
        }

        // Signature format: HTTPMethod + RequestURI + Timestamp + ContentHash
        string signatureRaw = $"{request.Method.Method}{request.RequestUri?.AbsolutePath}{timestamp}{contentHash}";
        
        using (var hmac = new HMACSHA256(Encoding.UTF8.GetBytes(SharedSecret)))
        {
            byte[] signatureBytes = hmac.ComputeHash(Encoding.UTF8.GetBytes(signatureRaw));
            string signature = Convert.ToBase64String(signatureBytes);
            request.Headers.Add("X-HMAC-Signature", signature);
        }

        return await base.SendAsync(request, cancellationToken);
    }
}
