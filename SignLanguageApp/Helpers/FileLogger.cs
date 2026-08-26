using System;
using System.IO;

namespace SignLanguageApp.Helpers
{
    public static class FileLogger
    {
        private static readonly string LogPath = @"C:\Users\Windows 10 21H1\source\repos\SignLanguageApp\debug.log";
        private static readonly object _lock = new object();

        public static void Log(string message)
        {
            try
            {
                lock (_lock)
                {
                    File.AppendAllText(LogPath, $"{DateTime.Now:O} - {message}\n");
                }
            }
            catch { }
        }
    }
}
