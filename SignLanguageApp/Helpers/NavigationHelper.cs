using System;
using System.Threading.Tasks;
using Microsoft.Maui.Controls;

namespace SignLanguageApp.Helpers
{
    public static class NavigationHelper
    {
        public static Task SafeNavigateAsync(string route, System.Collections.Generic.IDictionary<string, object>? parameters = null)
        {
            if (parameters != null)
                return Shell.Current.GoToAsync(route, parameters);
            return Shell.Current.GoToAsync(route);
        }
        
        public static Task SafeNavigateAsync(string route)
        {
            return Shell.Current.GoToAsync(route);
        }
    }
}
