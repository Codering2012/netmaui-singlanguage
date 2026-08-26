using System;
using System.Threading.Tasks;
using System.Net.Http;
using System.Diagnostics;
using Microsoft.Maui.Networking;

namespace SignLanguageApp.Services
{
    public interface IConnectivityService
    {
        bool IsConnected { get; }
        NetworkAccess NetworkAccess { get; }
        Task<bool> IsServerReachableAsync(string url, int timeoutMs = 3000);
        event EventHandler<ConnectivityChangedEventArgs> ConnectivityChanged;
    }

    public class ConnectivityService : IConnectivityService
    {
        public bool IsConnected
        {
            get
            {
                try
                {
                    var access = Connectivity.Current.NetworkAccess;
                    return access == NetworkAccess.Internet || 
                           access == NetworkAccess.Local || 
                           access == NetworkAccess.ConstrainedInternet;
                }
                catch
                {
                    return true;
                }
            }
        }

        public NetworkAccess NetworkAccess
        {
            get
            {
                try { return Connectivity.Current.NetworkAccess; }
                catch { return NetworkAccess.Local; }
            }
        }

        public event EventHandler<ConnectivityChangedEventArgs> ConnectivityChanged
        {
            add => Connectivity.Current.ConnectivityChanged += value;
            remove => Connectivity.Current.ConnectivityChanged -= value;
        }

        public async Task<bool> IsServerReachableAsync(string url, int timeoutMs = 5000)
        {
            if (string.IsNullOrWhiteSpace(url)) return false;

            bool isLoopback = url.Contains("127.0.0.1") || url.Contains("localhost") || url.Contains("::1") || url.Contains("api.internal");

            if (!isLoopback && !IsConnected) 
            {
                Debug.WriteLine("IsServerReachableAsync: No network connection.");
                return false;
            }

            try
            {
                var handler = new HttpClientHandler
                {
                    ServerCertificateCustomValidationCallback = HttpClientHandler.DangerousAcceptAnyServerCertificateValidator
                };

                using var client = new HttpClient(handler);
                client.Timeout = TimeSpan.FromMilliseconds(timeoutMs);
                
                using var getRequest = new HttpRequestMessage(HttpMethod.Get, url);
                getRequest.Headers.Add("X-API-KEY", "SignLang_Secure_v1_2026");
                
                var getResponse = await client.SendAsync(getRequest);
                return getResponse.IsSuccessStatusCode;
            }
            catch (Exception ex)
            {
                Debug.WriteLine($"IsServerReachableAsync: Failed to reach {url}. Error: {ex.Message}");
                return false;
            }
        }
    }
}
