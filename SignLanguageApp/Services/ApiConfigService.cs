namespace SignLanguageApp.Services
{
    public interface IApiConfigService
    {
        string BaseUrl { get; set; }
    }

    public class ApiConfigService : IApiConfigService
    {
        public string BaseUrl { get; set; } = "http://127.0.0.1:5179/api/";
    }
}
