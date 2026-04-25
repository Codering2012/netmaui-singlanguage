using System.Text;
using Microsoft.AspNetCore.Authentication.JwtBearer;
using Microsoft.EntityFrameworkCore;
using Microsoft.IdentityModel.Tokens;
using SignLanguageApi.Data;
using SignLanguageApi.Services;
using SignLanguageApi.Middleware;

var builder = WebApplication.CreateBuilder(args);

// Raise ThreadPool minimum worker threads for better burst handling under load.
ThreadPool.GetMinThreads(out var minWorkerThreads, out var minCompletionPortThreads);
var targetMinWorkerThreads = Math.Max(minWorkerThreads, Environment.ProcessorCount * 4);
ThreadPool.SetMinThreads(targetMinWorkerThreads, minCompletionPortThreads);

// Configure logging
builder.Logging.ClearProviders();
builder.Logging.AddConsole();
builder.Logging.AddDebug();

// Add services to the container
builder.Services.AddControllers();

// Configure OpenAPI for API documentation
builder.Services.AddOpenApi();

// Register AppDbContext with SQL Server or in-memory database for development
if (builder.Environment.IsDevelopment())
{
    builder.Services.AddDbContext<AppDbContext>(options =>
        options.UseInMemoryDatabase("SignLanguageDb"));
}
else
{
    builder.Services.AddDbContext<AppDbContext>(options =>
        options.UseSqlServer(builder.Configuration.GetConnectionString("DefaultConnection")));
}

// Configure CORS for MAUI and web clients
builder.Services.AddCors(options =>
{
    options.AddPolicy("AllowMauiApp", policy =>
    {
        policy.WithOrigins(
                "https://localhost:7084",      // API HTTPS
                "http://localhost:5179",       // API HTTP
                "http://localhost"             // Add your MAUI app domain here
            )
            .AllowAnyMethod()
            .AllowAnyHeader()
            .AllowCredentials();
    });

    options.AddPolicy("AllowAll", policy =>
    {
        policy.AllowAnyOrigin()
              .AllowAnyMethod()
              .AllowAnyHeader();
    });
});

// Register AuthService for Dependency Injection
builder.Services.AddScoped<IAuthService, AuthService>();

// Register LearnService for Dependency Injection
builder.Services.AddScoped<ILearnService, LearnService>();

// Register UserProgressService for Dependency Injection (handles save/load of user data)
builder.Services.AddScoped<IUserProgressService, UserProgressService>();

// Register GestureRecognitionService for real-time hand gesture detection
builder.Services.AddScoped<IGestureRecognitionService, GestureRecognitionService>();

// Register security services
builder.Services.AddSingleton<ITokenBlacklistService, TokenBlacklistService>();
builder.Services.AddScoped<IPasswordValidator, PasswordValidator>();
builder.Services.AddScoped<IAuditLogger, AuditLogger>();

// Add memory caching for optimization
builder.Services.AddMemoryCache();

// Configure response compression
builder.Services.AddResponseCompression();

// Configure JWT Bearer authentication
var jwtSecretKey = builder.Configuration["Jwt:SecretKey"] ?? throw new InvalidOperationException("JWT SecretKey is not configured");
var jwtIssuer = builder.Configuration["Jwt:Issuer"] ?? throw new InvalidOperationException("JWT Issuer is not configured");
var jwtAudience = builder.Configuration["Jwt:Audience"] ?? throw new InvalidOperationException("JWT Audience is not configured");

builder.Services.AddAuthentication(options =>
{
    options.DefaultAuthenticateScheme = JwtBearerDefaults.AuthenticationScheme;
    options.DefaultChallengeScheme = JwtBearerDefaults.AuthenticationScheme;
})
.AddJwtBearer(options =>
{
    options.TokenValidationParameters = new TokenValidationParameters
    {
        ValidateIssuer = true,
        ValidateAudience = true,
        ValidateLifetime = true,
        ValidateIssuerSigningKey = true,
        ValidIssuer = jwtIssuer,
        ValidAudience = jwtAudience,
        IssuerSigningKey = new SymmetricSecurityKey(Encoding.UTF8.GetBytes(jwtSecretKey))
    };
});

var app = builder.Build();

if (app.Environment.IsDevelopment())
{
    using var scope = app.Services.CreateScope();
    var scopedServices = scope.ServiceProvider;
    var dbContext = scopedServices.GetRequiredService<AppDbContext>();
    var seedLogger = scopedServices.GetRequiredService<ILoggerFactory>().CreateLogger("DemoDataSeeder");
    await DemoDataSeeder.SeedAsync(dbContext, seedLogger);
}

// Add custom error handling middleware
app.UseExceptionHandler(exceptionHandlerApp =>
{
    exceptionHandlerApp.Run(async context =>
    {
        var logger = context.RequestServices.GetRequiredService<ILogger<Program>>();
        var exception = context.Features.Get<Microsoft.AspNetCore.Diagnostics.IExceptionHandlerFeature>();

        if (exception?.Error != null)
        {
            logger.LogError(exception.Error, "Unhandled exception");
        }

        context.Response.StatusCode = StatusCodes.Status500InternalServerError;
        context.Response.ContentType = "application/json";

        await context.Response.WriteAsJsonAsync(new
        {
            message = "An internal server error occurred",
            statusCode = StatusCodes.Status500InternalServerError
        });
    });
});

// Configure the HTTP request pipeline
if (app.Environment.IsDevelopment())
{
    // Enable OpenAPI endpoint
    app.MapOpenApi();
}

// Add HTTPS redirection only outside development
if (!app.Environment.IsDevelopment())
{
    app.UseHttpsRedirection();
}

// Add token blacklist middleware - must be before authentication
app.UseTokenBlacklistMiddleware();

// Add rate limiting middleware
app.UseRateLimitMiddleware();

// Add response compression
app.UseResponseCompression();

// Add CORS - IMPORTANT: Must be before Authentication/Authorization
app.UseCors(builder.Environment.IsDevelopment() ? "AllowAll" : "AllowMauiApp");

// Add security headers middleware
app.Use(async (context, next) =>
{
    context.Response.Headers.Append("X-Content-Type-Options", "nosniff");
    context.Response.Headers.Append("X-Frame-Options", "DENY");
    context.Response.Headers.Append("X-XSS-Protection", "1; mode=block");

    if (!app.Environment.IsDevelopment())
    {
        context.Response.Headers.Append("Strict-Transport-Security", "max-age=31536000; includeSubDomains");
    }

    await next();
});

// Add authentication and authorization
app.UseAuthentication();
app.UseAuthorization();

// Map controllers
app.MapControllers();

// Log startup info
var logger = app.Services.GetRequiredService<ILogger<Program>>();
logger.LogInformation("=== Sign Language Learning API Starting ===");
logger.LogInformation("Environment: {Environment}", app.Environment.EnvironmentName);
logger.LogInformation("Endpoints available:");
logger.LogInformation("  - POST   /api/auth/login      (User login)");
logger.LogInformation("  - POST   /api/auth/register   (User registration)");
logger.LogInformation("  - GET    /api/learn/data               (Get all learn page data)");
logger.LogInformation("  - GET    /api/learn/categories/{{id}}/lessons    (Get lessons by category)");
logger.LogInformation("  - GET    /api/learn/lessons/{{id}}     (Get lesson details)");
logger.LogInformation("  - PUT    /api/learn/lessons/{{id}}/progress    (Update progress)");
logger.LogInformation("  - POST   /api/learn/lessons/{{id}}/complete    (Complete lesson)");
logger.LogInformation("  - GET    /api/learn/daily-reviews     (Get daily reviews)");
logger.LogInformation("  - POST   /api/learn/daily-reviews/{{id}}/review (Submit review)");
logger.LogInformation("  - POST   /api/gesture/predict   (Hand gesture recognition)");
logger.LogInformation("  - GET    /api/gesture/health    (Gesture service health check)");

if (app.Environment.IsDevelopment())
{
    logger.LogInformation("API Documentation: http://localhost:5179/openapi");
}

app.Run();
