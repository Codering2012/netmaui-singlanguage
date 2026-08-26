# .NET MAUI Development Rules

## 1. Root Navigation in MAUI
- When swapping the root page (e.g., from a loading screen to the main shell), use `Application.Current.MainPage = newRoot;`.
- **CRITICAL WARNING**: You MUST ensure that `newRoot` initializes completely without throwing any exceptions (or rethrows them properly). If `Application.Current.MainPage` is assigned a half-baked page (one that swallowed an exception during `InitializeComponent`), it will trigger a fatal, uncatchable `COMException` that instantly kills the app on Windows.
- **DO NOT** use `window.Page = newRoot` for root navigation. On Windows (WinUI3), this will silently fail and cause the app to freeze forever, leaving the previous page on screen.

## 2. Custom ContentProperty Layouts
When building custom layout controls with `[ContentProperty]`, the bound property changed callback is executed by the XAML deserializer *before* the control's `InitializeComponent()` logic fully resolves inner variables.
**ALWAYS** verify that inner container elements are not null in the property changed callback, and re-apply the content property at the end of the constructor:
```csharp
public CustomLayout()
{
    InitializeComponent();
    if (MyContent != null && InnerContainer != null) InnerContainer.Content = MyContent;
}
```

## 3. UI Animation Safety
Animations (e.g., `FadeToAsync`, `ScaleToAsync`) fired during page navigation hooks (`OnNavigating`, `OnAppearing`) will throw an `ArgumentNullException` if the target view is null during the transition phase.
**ALWAYS** perform an explicit null check on the target view (e.g., `CurrentPage`) before executing asynchronous animations.

## 4. Preventing Layout Deadlocks (WinUI3)
- **CRITICAL WARNING**: Never place a `CollectionView` (or any infinitely expanding layout) inside a `ScrollView` without explicitly setting a `HeightRequest`.
- On Windows (WinUI 3), doing this will cause an infinite layout measurement loop that completely hangs the UI thread and freezes the app (giving an "App is not responding" dialog).
- If you need a vertically expanding list inside a `ScrollView` without a fixed height, use `BindableLayout` on a standard `StackLayout` or `Grid` instead of a `CollectionView`.

## 5. UI Thread Starvation (GraphicsView)
- **CRITICAL WARNING**: Do not continuously call `this.Invalidate()` at high frame rates (e.g., 30fps/60fps) within a `GraphicsView` animation loop without careful throttling.
- On Windows, this can completely peg the UI thread processing layout passes, starving it of the time needed to attach the main window or process user input, leading to a permanent app freeze. 

## 6. Debugging Detached Processes
- When using `dotnet run` for a MAUI Windows application, the app launches as a detached GUI process.
- **DO NOT rely on `Console.WriteLine()`** for debugging, as the output is discarded and invisible in the terminal.
- Always use a local `FileLogger` writing to a `.log` file in the workspace, or attach a native Windows debugger.

## 7. Dynamic SVG Tinting for Light/Dark Mode
When applying the global rule for "pure white, minimalist, flat SVG paths", you MUST ensure these icons remain visible in Light Mode.
Do NOT use dark backgrounds in Light Mode simply to make white icons visible. Do NOT duplicate icons in black and white.

Instead, ALWAYS use `IconTintColorBehavior` from `CommunityToolkit.Maui` combined with `AppThemeBinding` to dynamically tint the white SVGs to a dark color in Light Mode and a white color in Dark Mode. 

Example implementation:
```xml
<ImageButton Source="icon.png" BackgroundColor="Transparent">
    <ImageButton.Behaviors>
        <toolkit:IconTintColorBehavior TintColor="{AppThemeBinding Light={StaticResource PrimaryTextColor}, Dark=White}" />
    </ImageButton.Behaviors>
</ImageButton>
```
