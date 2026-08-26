using System;
using System.Threading.Tasks;
using Microsoft.Maui.ApplicationModel;

namespace SignLanguageApp.Helpers
{
    public static class MainThreadHelper
    {
        public static void SafeInvokeOnMainThread(Action action)
        {
            if (MainThread.IsMainThread)
            {
                action();
            }
            else
            {
                MainThread.BeginInvokeOnMainThread(action);
            }
        }

        public static Task SafeInvokeOnMainThreadAsync(Action action)
        {
            if (MainThread.IsMainThread)
            {
                action();
                return Task.CompletedTask;
            }
            return MainThread.InvokeOnMainThreadAsync(action);
        }
        
        public static Task SafeInvokeOnMainThreadAsync(Func<Task> funcTask)
        {
            if (MainThread.IsMainThread)
            {
                return funcTask();
            }
            return MainThread.InvokeOnMainThreadAsync(funcTask);
        }
    }
}
