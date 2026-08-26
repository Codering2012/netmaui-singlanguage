using System.Net;
using System.Net.Sockets;
using System.Text;

namespace SignLanguageApi.Services
{
    public class DiscoveryService : BackgroundService
    {
        private const int DiscoveryPort = 50001;
        private const string DiscoveryMessagePrefix = "SIGN_LANGUAGE_API|";
        private readonly ILogger<DiscoveryService> _logger;
        private readonly IConfiguration _configuration;

        public DiscoveryService(ILogger<DiscoveryService> logger, IConfiguration configuration)
        {
            _logger = logger;
            _configuration = configuration;
        }

        protected override async Task ExecuteAsync(CancellationToken stoppingToken)
        {
            _logger.LogInformation("Discovery Service started.");

            while (!stoppingToken.IsCancellationRequested)
            {
                try
                {
                    using var client = new UdpClient();
                    client.EnableBroadcast = true;
                    var endpoint = new IPEndPoint(IPAddress.Broadcast, DiscoveryPort);
                    var localEndpoint = new IPEndPoint(IPAddress.Loopback, DiscoveryPort);

                    // 1. Broadcast localhost (for same-machine clients)
                    string localhostHttps = $"{DiscoveryMessagePrefix}https://127.0.0.1:8443/api/";
                    string localhostHttp = $"{DiscoveryMessagePrefix}http://127.0.0.1:8080/api/";
                    
                    byte[] lhHttpsData = Encoding.UTF8.GetBytes(localhostHttps);
                    await client.SendAsync(lhHttpsData, lhHttpsData.Length, endpoint);
                    await client.SendAsync(lhHttpsData, lhHttpsData.Length, localEndpoint);
                    
                    byte[] lhHttpData = Encoding.UTF8.GetBytes(localhostHttp);
                    await client.SendAsync(lhHttpData, lhHttpData.Length, endpoint);
                    await client.SendAsync(lhHttpData, lhHttpData.Length, localEndpoint);

                    // 2. Broadcast local IP (for LAN clients)
                    string localIp = GetLocalIPAddress();
                    if (localIp != "127.0.0.1" && localIp != "localhost")
                    {
                        string httpsUrl = $"https://{localIp}:8443/api/";
                        string httpsMessage = $"{DiscoveryMessagePrefix}{httpsUrl}";
                        byte[] httpsData = Encoding.UTF8.GetBytes(httpsMessage);
                        await client.SendAsync(httpsData, httpsData.Length, endpoint);

                        string httpUrl = $"http://{localIp}:8080/api/";
                        string httpMessage = $"{DiscoveryMessagePrefix}{httpUrl}";
                        byte[] httpData = Encoding.UTF8.GetBytes(httpMessage);
                        await client.SendAsync(httpData, httpData.Length, endpoint);
                    }
                    
                    _logger.LogDebug("Broadcasted discovery messages for localhost and {LocalIp}", localIp);
                }
                catch (Exception ex)
                {
                    _logger.LogError(ex, "Error in Discovery Service");
                }

                await Task.Delay(5000, stoppingToken); // Broadcast every 5 seconds
            }
        }

        private string GetLocalIPAddress()
        {
            var host = Dns.GetHostEntry(Dns.GetHostName());
            foreach (var ip in host.AddressList)
            {
                if (ip.AddressFamily == AddressFamily.InterNetwork)
                {
                    return ip.ToString();
                }
            }
            return "localhost";
        }
    }
}
