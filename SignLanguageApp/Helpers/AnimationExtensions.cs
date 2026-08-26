using Microsoft.Maui.Controls;
using System.Threading.Tasks;
using Microsoft.Maui.Graphics;
using System;

namespace SignLanguageApp.Helpers
{
    public static class AnimationExtensions
    {
        public static async Task AnimateStaggeredChildren(this Layout layout, double startY, uint duration, uint stagger)
        {
            if (layout == null) return;
            var tasks = new System.Collections.Generic.List<Task>();
            uint currentDelay = 0;
            foreach (var child in layout.Children)
            {
                if (child is VisualElement view)
                {
                    view.Opacity = 0;
                    view.TranslationY = startY;
                    tasks.Add(Task.Run(async () =>
                    {
                        await Task.Delay((int)currentDelay);
                        MainThreadHelper.SafeInvokeOnMainThread(() =>
                        {
                            view.FadeToAsync(1, duration);
                            view.TranslateToAsync(0, 0, duration, Easing.CubicOut);
                        });
                    }));
                    currentDelay += stagger;
                }
            }
            await Task.WhenAll(tasks);
        }

        public static Task AnimateColorTransition(this VisualElement element, Color fromColor, Color toColor, Action<Color> callback, uint length = 250, Easing? easing = null)
        {
            if (element == null) return Task.CompletedTask;
            var tcs = new TaskCompletionSource<bool>();
            var animation = new Animation(v =>
            {
                var color = Color.FromRgba(
                    fromColor.Red + v * (toColor.Red - fromColor.Red),
                    fromColor.Green + v * (toColor.Green - fromColor.Green),
                    fromColor.Blue + v * (toColor.Blue - fromColor.Blue),
                    fromColor.Alpha + v * (toColor.Alpha - fromColor.Alpha)
                );
                callback(color);
            }, 0, 1);
            
            animation.Commit(element, "ColorTransition", 16, length, easing, (v, c) => tcs.SetResult(c));
            return tcs.Task;
        }
    }
}
