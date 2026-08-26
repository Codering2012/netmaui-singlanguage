using System.Net.WebSockets;
using System.Text;
using System.Text.Json;
using SignLanguageApi.Dtos;
using SignLanguageApi.Services;
using Microsoft.IdentityModel.Tokens;
using System.IdentityModel.Tokens.Jwt;

namespace SignLanguageApi.Hubs;

/// <summary>
/// Handles the /ws/gesture WebSocket endpoint.
/// 
/// Protocol:
///   Client â†’ Server : Binary message = raw JPEG bytes (one frame per message)
///   Server â†’ Client : Text message   = UTF-8 JSON GesturePredictionResponseDto
///
/// Frame drop strategy: latest-wins. A new frame arriving while MediaPipe is busy
/// replaces the previous pending frame â€” the server never queues stale frames.
/// </summary>
public static class GestureStreamingWebSocketHandler
{
    public static async Task HandleAsync(HttpContext context)
    {
        if (!context.WebSockets.IsWebSocketRequest)
        {
            context.Response.StatusCode = StatusCodes.Status400BadRequest;
            await context.Response.WriteAsync("Expected a WebSocket request.");
            return;
        }

        // â”€â”€ Auth: validate JWT from query string (standard SignalR pattern) â”€â”€
        var token = context.Request.Query["access_token"].FirstOrDefault();
        var userId = ValidateToken(context, token);
        if (userId == null)
        {
            context.Response.StatusCode = StatusCodes.Status401Unauthorized;
            return;
        }

        var gestureService = context.RequestServices.GetRequiredService<IGestureRecognitionService>();
        var logger = context.RequestServices.GetRequiredService<ILogger<Program>>();

        using var ws = await context.WebSockets.AcceptWebSocketAsync();
        logger.LogInformation("[WS] Client {User} connected for gesture streaming.", userId);

        // Latest-frame buffer: new frames overwrite unprocessed ones
        byte[]? pendingFrame = null;
        long pendingSequence = 0;
        var frameLock = new object();
        var frameAvailable = new SemaphoreSlim(0, 1);
        using var cts = CancellationTokenSource.CreateLinkedTokenSource(context.RequestAborted);

        // â”€â”€ Receive loop (runs on current Task) â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
        var receiveTask = Task.Run(async () =>
        {
            var buffer = System.Buffers.ArrayPool<byte>.Shared.Rent(512 * 1024); // 512 KB pooled buffer
            try
            {
                while (ws.State == WebSocketState.Open && !cts.Token.IsCancellationRequested)
                {
                    // Accumulate a complete binary message
                    var totalBytes = 0;
                    WebSocketReceiveResult result;
                    do
                    {
                        result = await ws.ReceiveAsync(
                            new ArraySegment<byte>(buffer, totalBytes, buffer.Length - totalBytes),
                            cts.Token);

                        if (result.MessageType == WebSocketMessageType.Close)
                        {
                            await cts.CancelAsync();
                            return;
                        }

                        totalBytes += result.Count;
                    }
                    while (!result.EndOfMessage);

                    if (result.MessageType != WebSocketMessageType.Binary || totalBytes <= 8)
                        continue;

                    // Extract sequence/timestamp (first 8 bytes)
                    var sequence = BitConverter.ToInt64(buffer, 0);
                    var frame = new byte[totalBytes - 8];
                    Buffer.BlockCopy(buffer, 8, frame, 0, totalBytes - 8);

                    // Latest-wins: replace pending frame and sequence
                    lock (frameLock)
                    {
                        pendingFrame = frame;
                        pendingSequence = sequence;
                    }

                    // Signal inference task â€” non-blocking
                    if (frameAvailable.CurrentCount == 0)
                        frameAvailable.Release();
                }
            }
            catch (OperationCanceledException) { }
            catch (WebSocketException wsEx) when (
                wsEx.WebSocketErrorCode == WebSocketError.ConnectionClosedPrematurely ||
                wsEx.WebSocketErrorCode == WebSocketError.InvalidState)
            {
                // Client disconnected without completing the WS close handshake â€” benign.
                await cts.CancelAsync();
            }
            catch (Exception ex) { logger.LogWarning("[WS] Receive error: {Msg}", ex.Message); }
            finally
            {
                System.Buffers.ArrayPool<byte>.Shared.Return(buffer);
            }
        }, cts.Token);

        // â”€â”€ Inference + Send loop (runs concurrently) â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
        var inferenceTask = Task.Run(async () =>
        {
            try
            {
                while (!cts.Token.IsCancellationRequested && ws.State == WebSocketState.Open)
                {
                    await frameAvailable.WaitAsync(cts.Token);

                    byte[]? frame;
                    long sequence;
                    lock (frameLock)
                    {
                        frame = pendingFrame;
                        sequence = pendingSequence;
                        pendingFrame = null;
                    }

                    if (frame == null || frame.Length == 0) continue;

                    GesturePredictionResponseDto result;
                    try
                    {
                        result = await gestureService.PredictGestureAsync(
                            frame, cts.Token);
                        
                        if (result.Data != null)
                        {
                            result.Data.Sequence = (int)sequence;
                        }
                    }
                    catch (OperationCanceledException) { break; }
                    catch (Exception ex)
                    {
                        logger.LogWarning("[WS] Inference error: {Msg}", ex.Message);
                        result = new GesturePredictionResponseDto
                        {
                            Status = "error",
                            Message = "Inference failed"
                        };
                    }

                    logger.LogDebug("[{Time}] [WS] Frame prediction sent to '{User}'. Prediction: '{Letter}' (Conf: {Conf:F2}). Server time: {TimeMs:F1}ms.", 
                        DateTime.Now.ToString("HH:mm:ss.fff"), userId, result.Data?.Letter ?? "none", result.Data?.Confidence ?? 0, result.Data?.ProcessingTimeMs ?? 0);

                    // Send JSON annotation back on the output channel
                    var json = JsonSerializer.Serialize(result, ApiJsonContext.Default.GesturePredictionResponseDto);
                    var bytes = Encoding.UTF8.GetBytes(json);

                    if (ws.State == WebSocketState.Open)
                    {
                        await ws.SendAsync(
                            new ArraySegment<byte>(bytes),
                            WebSocketMessageType.Text,
                            endOfMessage: true,
                            cts.Token);
                    }
                }
            }
            catch (OperationCanceledException) { }
            catch (Exception ex) { logger.LogWarning("[WS] Inference loop error: {Msg}", ex.Message); }
        }, cts.Token);

        await Task.WhenAny(receiveTask, inferenceTask);
        await cts.CancelAsync();

        try { await Task.WhenAll(receiveTask, inferenceTask); } catch (Exception ex) { System.Diagnostics.Debug.WriteLine($"Exception: {ex.Message}"); }

        if (ws.State == WebSocketState.Open)
        {
            try { await ws.CloseAsync(WebSocketCloseStatus.NormalClosure, "Done", CancellationToken.None); }
            catch (Exception ex) { System.Diagnostics.Debug.WriteLine($"Exception: {ex.Message}"); }
        }

        logger.LogInformation("[WS] Client {User} disconnected.", userId);
    }

    private static string? ValidateToken(HttpContext context, string? token)
    {
        if (string.IsNullOrEmpty(token)) return null;

        try
        {
            var config = context.RequestServices.GetRequiredService<IConfiguration>();
            var secretKey  = config["Jwt:SecretKey"]  ?? "SignSpeak_Super_Secret_Demo_Key_2026_!@#";
            var issuer     = config["Jwt:Issuer"]     ?? "SignLanguageApi";
            var audience   = config["Jwt:Audience"]   ?? "SignLanguageApp";

            var handler = new JwtSecurityTokenHandler();
            var principal = handler.ValidateToken(token, new TokenValidationParameters
            {
                ValidateIssuer           = true,
                ValidateAudience         = true,
                ValidateLifetime         = true,
                ValidateIssuerSigningKey = true,
                ValidIssuer              = issuer,
                ValidAudience            = audience,
                IssuerSigningKey         = new SymmetricSecurityKey(Encoding.UTF8.GetBytes(secretKey))
            }, out _);

            return principal.FindFirst(System.Security.Claims.ClaimTypes.NameIdentifier)?.Value;
        }
        catch
        {
            return null;
        }
    }
}
