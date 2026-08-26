using System;
using System.Diagnostics;
using System.Threading.Tasks;

namespace SignLanguageApp.Helpers
{
    public static class DeadlockWatchdog
    {
        /// <summary>
        /// Executes a task and throws a TimeoutException with a hint if it takes too long.
        /// </summary>
        public static async Task<T> WithTimeout<T>(this Task<T> task, string operationName, int timeoutMs = 5000)
        {
            var timeoutTask = Task.Delay(timeoutMs);
            var completedTask = await Task.WhenAny(task, timeoutTask);

            if (completedTask == timeoutTask)
            {
                // This means the task froze!
                var errorMessage = $"DEADLOCK DETECTED: Operation '{operationName}' took longer than {timeoutMs}ms and is frozen.\n" +
                                   $"This usually means the UI thread is stuck waiting for a Task.Result, Task.Wait(), or an infinite loop.";
                
                FileLogger.Log("=======================================================");
                FileLogger.Log(errorMessage);
                FileLogger.Log("=======================================================");

                throw new TimeoutException(errorMessage);
            }

            return await task;
        }

        public static async Task WithTimeout(this Task task, string operationName, int timeoutMs = 5000)
        {
            var timeoutTask = Task.Delay(timeoutMs);
            var completedTask = await Task.WhenAny(task, timeoutTask);

            if (completedTask == timeoutTask)
            {
                var errorMessage = $"DEADLOCK DETECTED: Operation '{operationName}' took longer than {timeoutMs}ms and is frozen.";
                FileLogger.Log($"[WATCHDOG] {errorMessage}");
                throw new TimeoutException(errorMessage);
            }

            await task;
        }
    }
}
