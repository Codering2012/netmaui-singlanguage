namespace SignLanguageApp.Services
{
    public interface IApiDiscoveryService
    {
        Task<string?> DiscoverApiUrlAsync(TimeSpan timeout);
    }
}
