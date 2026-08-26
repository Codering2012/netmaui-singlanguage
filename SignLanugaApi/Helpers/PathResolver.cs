using System;
using System.IO;

namespace SignLanguageApi.Helpers
{
    public static class PathResolver
    {
        public static string ResolveFolder(string folderName)
        {
            // 1. Check current working directory (e.g. project root when running dotnet run)
            var path1 = Path.Combine(Directory.GetCurrentDirectory(), folderName);
            if (Directory.Exists(path1)) return path1;

            // 2. Check AppDomain BaseDirectory (bin output folder)
            var path2 = Path.Combine(AppDomain.CurrentDomain.BaseDirectory, folderName);
            if (Directory.Exists(path2)) return path2;

            // 3. Search parent directories (climb up 3 levels from bin/Debug/net10.0/)
            var dir = new DirectoryInfo(AppDomain.CurrentDomain.BaseDirectory);
            while (dir != null)
            {
                var candidate = Path.Combine(dir.FullName, folderName);
                if (Directory.Exists(candidate)) return candidate;
                dir = dir.Parent;
            }

            // Fallback: create in current directory
            Directory.CreateDirectory(path1);
            return path1;
        }
    }
}
