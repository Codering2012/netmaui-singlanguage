using System.Net;
using System.Net.Sockets;
using System.Net.NetworkInformation;
using System.Text;
using System.Diagnostics;

namespace SignLanguageApp.Services
{
    public class ApiDiscoveryService : IApiDiscoveryService
    {
        private const int DiscoveryPort = 50001;
        private const string DiscoveryMessagePrefix = "SIGN_LANGUAGE_API|";
        
        // Single static HttpClient for resource safety and socket reuse
        private static readonly HttpClient _scanHttpClient = new() { Timeout = TimeSpan.FromMilliseconds(750) };

        public async Task<string?> DiscoverApiUrlAsync(TimeSpan timeout)
        {
            using var udpClient = new UdpClient();
            udpClient.Client.SetSocketOption(SocketOptionLevel.Socket, SocketOptionName.ReuseAddress, true);
            udpClient.ExclusiveAddressUse = false;
            
            try
            {
                udpClient.Client.Bind(new IPEndPoint(IPAddress.Any, DiscoveryPort));
            }
            catch (Exception ex)
            {
                Debug.WriteLine($"Failed to bind to discovery port: {ex.Message}");
                var scanCts = new CancellationTokenSource(timeout);
                return await ScanSubnetAsync(scanCts.Token);
            }

            udpClient.EnableBroadcast = true;
            var cts = new CancellationTokenSource(timeout);
            
            var udpTask = Task.Run(async () =>
            {
                try
                {
                    Debug.WriteLine("Listening for UDP API discovery broadcast...");
                    while (!cts.Token.IsCancellationRequested)
                    {
                        var result = await udpClient.ReceiveAsync(cts.Token);
                        string message = Encoding.UTF8.GetString(result.Buffer);

                        if (message.StartsWith(DiscoveryMessagePrefix))
                        {
                            string url = message.Substring(DiscoveryMessagePrefix.Length);
#if !WINDOWS
                            if (url.Contains("127.0.0.1") || url.Contains("localhost"))
                            {
                                continue;
                            }
#endif
                            if (url.StartsWith("https://") && !cts.Token.IsCancellationRequested)
                            {
                                continue; 
                            }

                            Debug.WriteLine($"API discovered via UDP: {url}");
                            cts.Cancel(); // Cancel the other scanner task
                            return url;
                        }
                    }
                }
                catch (OperationCanceledException) {}
                catch (Exception ex)
                {
                    Debug.WriteLine($"Error during UDP discovery: {ex.Message}");
                }
                return null;
            }, cts.Token);

            var scanTask = Task.Run(async () =>
            {
                // Wait 1.5 seconds for UDP, then fall back to subnet scanning
                await Task.Delay(1500, cts.Token).ContinueWith(_ => {});
                if (cts.Token.IsCancellationRequested) return null;

                Debug.WriteLine("Subnet scanner starting as UDP fallback...");
                var url = await ScanSubnetAsync(cts.Token);
                if (url != null)
                {
                    cts.Cancel();
                }
                return url;
            }, cts.Token);

            var completedTask = await Task.WhenAny(udpTask, scanTask);
            var discoveredUrl = await completedTask;

            if (discoveredUrl == null)
            {
                var otherTask = completedTask == udpTask ? scanTask : udpTask;
                discoveredUrl = await otherTask;
            }

            return discoveredUrl;
        }

        private async Task<string?> ScanSubnetAsync(CancellationToken cancellationToken)
        {
            try
            {
                var localIps = GetLocalIPAddresses();
                if (localIps == null || localIps.Count == 0) return null;

                var tasks = new List<Task<string?>>();
                using var semaphore = new SemaphoreSlim(10); // Limit to 10 concurrent requests to prevent socket exhaustion on mobile

                foreach (var ipStr in localIps)
                {
                    var parts = ipStr.Split('.');
                    if (parts.Length != 4) continue;

                    string subnetPrefix = $"{parts[0]}.{parts[1]}.{parts[2]}.";
                    int deviceLastByte = int.Parse(parts[3]);

                    var ipList = new List<int>();
                    
                    // 1. Gateway (.1)
                    if (deviceLastByte != 1) ipList.Add(1);
                    
                    // 2. Neighboring IPs
                    for (int offset = 1; offset <= 10; offset++)
                    {
                        int next = deviceLastByte + offset;
                        int prev = deviceLastByte - offset;
                        if (next > 0 && next < 255) ipList.Add(next);
                        if (prev > 0 && prev < 255) ipList.Add(prev);
                    }

                    // 3. Complete range
                    for (int i = 1; i < 255; i++)
                    {
                        if (!ipList.Contains(i) && i != deviceLastByte)
                        {
                            ipList.Add(i);
                        }
                    }

                    foreach (var lastByte in ipList)
                    {
                        string targetIp = $"{subnetPrefix}{lastByte}";
                        tasks.Add(Task.Run(async () =>
                        {
                            await semaphore.WaitAsync(cancellationToken);
                            try
                            {
                                return await TestIpAddressAsync(_scanHttpClient, targetIp, cancellationToken);
                            }
                            finally
                            {
                                semaphore.Release();
                            }
                        }, cancellationToken));
                    }
                }

                var results = await Task.WhenAll(tasks);
                return results.FirstOrDefault(r => r != null);
            }
            catch (Exception ex)
            {
                Debug.WriteLine($">>> Subnet scan error: {ex.Message}");
            }
            return null;
        }

        private async Task<string?> TestIpAddressAsync(HttpClient client, string ipAddress, CancellationToken ct)
        {
            try
            {
                string testUrl = $"http://{ipAddress}:5179/api/gesture/health";
                var response = await client.GetAsync(testUrl, ct);
                if (response.IsSuccessStatusCode)
                {
                    var content = await response.Content.ReadAsStringAsync(ct);
                    if (content.Contains("healthy"))
                    {
                        string discoveredUrl = $"http://{ipAddress}:5179/api/";
                        Debug.WriteLine($">>> Automatically discovered server at: {discoveredUrl}");
                        return discoveredUrl;
                    }
                }
            }
            catch
            {
                // Ignored (expected connection failures for offline devices)
            }
            return null;
        }

        private List<string> GetLocalIPAddresses()
        {
            var ipList = new List<string>();
            try
            {
                foreach (var netInterface in NetworkInterface.GetAllNetworkInterfaces())
                {
                    if (netInterface.OperationalStatus != OperationalStatus.Up)
                        continue;

                    var ipProperties = netInterface.GetIPProperties();
                    foreach (var addr in ipProperties.UnicastAddresses)
                    {
                        if (addr.Address.AddressFamily == AddressFamily.InterNetwork)
                        {
                            string ipStr = addr.Address.ToString();
                            if (ipStr != "127.0.0.1" && !ipStr.StartsWith("169.254"))
                            {
                                ipList.Add(ipStr);
                            }
                        }
                    }
                }
            }
            catch (Exception ex)
            {
                Debug.WriteLine($">>> Failed to retrieve local IP addresses: {ex.Message}");
            }
            return ipList;
        }
    }
}
