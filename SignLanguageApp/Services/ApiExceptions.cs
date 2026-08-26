using System;

namespace SignLanguageApp.Services
{
    public class ApiException : Exception
    {
        public int StatusCode { get; }
        public string? ResponseContent { get; }

        public ApiException(string message, int statusCode = 0, string? responseContent = null) 
            : base(message)
        {
            StatusCode = statusCode;
            ResponseContent = responseContent;
        }
    }

    public class NoInternetException : Exception
    {
        public NoInternetException() : base("Can't connect to network. Please check your internet connection.") { }
    }

    public class ServerUnreachableException : Exception
    {
        public ServerUnreachableException(string url) : base($"Can't connect to server. The service at {url} is currently unavailable.") { }
    }

    public class UnauthorizedException : ApiException
    {
        public UnauthorizedException(string message = "Unauthorized") : base(message, 401) { }
    }
}
