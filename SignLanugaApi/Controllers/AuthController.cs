using Microsoft.AspNetCore.Authorization;
using Microsoft.AspNetCore.Mvc;
using SignLanguageApi.Data;
using SignLanguageApi.Dtos;
using SignLanguageApi.Services;
using System.Text;

namespace SignLanguageApi.Controllers
{
    [ApiController]
    [Route("api/[controller]")]
    public class AuthController : ControllerBase
    {
        private readonly AppDbContext _context;
        private readonly IAuthService _authService;
        private readonly IUserProgressService _progressService;
        private readonly IPasswordValidator _passwordValidator;
        private readonly ITokenBlacklistService _tokenBlacklist;
        private readonly IAuditLogger _auditLogger;
        private readonly ILogger<AuthController> _logger;

        public AuthController(
            AppDbContext context, 
            IAuthService authService, 
            IUserProgressService progressService,
            IPasswordValidator passwordValidator,
            ITokenBlacklistService tokenBlacklist,
            IAuditLogger auditLogger,
            ILogger<AuthController> logger)
        {
            _context = context;
            _authService = authService;
            _progressService = progressService;
            _passwordValidator = passwordValidator;
            _tokenBlacklist = tokenBlacklist;
            _auditLogger = auditLogger;
            _logger = logger;
        }

        [HttpPost("register")]
        public async Task<ActionResult<object>> Register([FromBody] RegisterRequest request)
        {
            try
            {
                var remoteIP = HttpContext.Connection.RemoteIpAddress?.ToString() ?? "Unknown";
                
                _logger.LogInformation("========== REGISTER ENDPOINT CALLED ==========");
                _logger.LogInformation("Timestamp: {Timestamp}", DateTime.UtcNow);
                _logger.LogInformation("Request received from IP: {RemoteIP}", remoteIP);
                
                _logger.LogDebug("Registration request received. Email: {Email}, Name: {Name}", 
                    request?.Email ?? "null", request?.Name ?? "null");

                // Validation: Check if all required fields are provided
                if (string.IsNullOrWhiteSpace(request?.Email))
                {
                    _logger.LogWarning("REJECTED - Email is null, empty, or whitespace");
                    return BadRequest(new ApiMessageDto { message = "Email, password, and name are required." });
                }

                if (string.IsNullOrWhiteSpace(request?.Password))
                {
                    _logger.LogWarning("REJECTED - Password is null, empty, or whitespace for email: {Email}", request?.Email ?? "null");
                    return BadRequest(new ApiMessageDto { message = "Email, password, and name are required." });
                }

                if (string.IsNullOrWhiteSpace(request?.Name))
                {
                    _logger.LogWarning("REJECTED - Name is null, empty, or whitespace for email: {Email}", request?.Email ?? "null");
                    return BadRequest(new ApiMessageDto { message = "Email, password, and name are required." });
                }

                _logger.LogInformation("? All required fields validated. Email: {Email}", request.Email);

                // Validate password strength
                var (isValid, errorMessage) = _passwordValidator.ValidatePassword(request.Password);
                if (!isValid)
                {
                    _logger.LogWarning("REJECTED - Password validation failed: {Error}", errorMessage);
                    await _auditLogger.LogRegisterAttemptAsync(request.Email, false, remoteIP);
                    return BadRequest(new ApiMessageDto { message = errorMessage });
                }

                _logger.LogInformation("? Password validation passed");

                // Check if user already exists
                _logger.LogDebug("Checking if user already exists with email: {Email}", request.Email);
                var existingUser = _context.Users.FirstOrDefault(u => u.Email == request.Email);

                if (existingUser != null)
                {
                    _logger.LogWarning("REJECTED - Email already exists: {Email} (User ID: {UserId})", 
                        request.Email, existingUser.Id);
                    return BadRequest(new ApiMessageDto { message = "Email already exists." });
                }

                _logger.LogInformation("? Email is unique. No existing user found.");

                // Hash the password
                _logger.LogDebug("Hashing password for email: {Email}", request.Email);
                var passwordHash = _authService.HashPassword(request.Password);
                _logger.LogDebug("? Password hashed successfully");

                // Create new user
                var newUser = new User
                {
                    Id = Guid.NewGuid().ToString(),
                    Email = request.Email,
                    Name = request.Name,
                    PasswordHash = passwordHash,
                    CreatedAt = DateTime.UtcNow
                };

                _logger.LogInformation("Creating new user account:");
                _logger.LogInformation("  - User ID: {UserId}", newUser.Id);
                _logger.LogInformation("  - Email: {Email}", newUser.Email);
                _logger.LogInformation("  - Name: {Name}", newUser.Name);
                _logger.LogInformation("  - Created At: {CreatedAt}", newUser.CreatedAt);

                // Add to database
                _logger.LogDebug("Adding user to database context...");
                _context.Users.Add(newUser);
                
                _logger.LogDebug("Saving changes to database...");
                await _context.SaveChangesAsync();

                _logger.LogInformation("? User successfully saved to database");

                // Save account creation record to local file
                _logger.LogDebug("Saving account creation record to local file...");
                try
                {
                    await SaveAccountCreationToFileAsync(newUser);
                    _logger.LogInformation("? Account creation record saved to file");
                }
                catch (Exception fileEx)
                {
                    _logger.LogWarning("Warning - Failed to save account creation to file: {Exception}", fileEx.Message);
                    // Don't fail registration if file save fails
                }

                // Save user info to users.json (append or create)
                try
                {
                    string appDataPath = Environment.GetFolderPath(Environment.SpecialFolder.ApplicationData);
                    string accountsDir = Path.Combine(appDataPath, "SignLanguageApp", "Accounts");
                    if (!Directory.Exists(accountsDir))
                        Directory.CreateDirectory(accountsDir);
                    string jsonFile = Path.Combine(accountsDir, "users.json");

                    var userInfo = new {
                        Id = newUser.Id,
                        Email = newUser.Email,
                        Name = newUser.Name,
                        CreatedAt = newUser.CreatedAt,
                        LearningStreak = newUser.LearningStreak
                    };

                    List<object> usersList = new();
                    if (System.IO.File.Exists(jsonFile))
                    {
                        try
                        {
                            var existing = await System.IO.File.ReadAllTextAsync(jsonFile);
                            if (!string.IsNullOrWhiteSpace(existing))
                                usersList = System.Text.Json.JsonSerializer.Deserialize<List<object>>(existing) ?? new();
                        }
                        catch { /* ignore and overwrite if corrupt */ }
                    }
                    usersList.Add(userInfo);
                    var json = System.Text.Json.JsonSerializer.Serialize(usersList, new System.Text.Json.JsonSerializerOptions { WriteIndented = true });
                    await System.IO.File.WriteAllTextAsync(jsonFile, json);
                    _logger.LogInformation("User info appended to users.json");
                }
                catch (Exception fileEx)
                {
                    _logger.LogWarning("Warning - Failed to append user info to users.json: {Exception}", fileEx.Message);
                }

                _logger.LogInformation("========== REGISTRATION SUCCESSFUL ==========");
                _logger.LogInformation("New user registered: Email={Email}, UserId={UserId}", 
                    request.Email, newUser.Id);
                _logger.LogInformation("");

                // Log successful registration
                await _auditLogger.LogRegisterAttemptAsync(request.Email, true, remoteIP);

                return Ok(new ApiMessageDto { message = "User registered successfully." });
            }
            catch (Exception ex)
            {
                _logger.LogError("========== REGISTRATION FAILED WITH EXCEPTION ==========");
                _logger.LogError("Exception Type: {ExceptionType}", ex.GetType().Name);
                _logger.LogError("Exception Message: {Message}", ex.Message);
                _logger.LogError("Stack Trace: {StackTrace}", ex.StackTrace);
                _logger.LogError("Inner Exception: {InnerException}", ex.InnerException?.Message ?? "None");
                _logger.LogError("========== END EXCEPTION ==========");
                _logger.LogError("");

                var remoteIP = HttpContext.Connection.RemoteIpAddress?.ToString() ?? "Unknown";
                // Log failed registration attempt
                var requestEmail = (HttpContext.Request.Form["email"].ToString() ?? "unknown");
                _ = _auditLogger.LogRegisterAttemptAsync(requestEmail, false, remoteIP);

                return StatusCode(StatusCodes.Status500InternalServerError, 
                    new ApiMessageDto { message = "An error occurred during registration." });
            }
        }

        [HttpPost("login")]
        public async Task<ActionResult<object>> Login([FromBody] LoginRequest request)
        {
            try
            {
                var remoteIP = HttpContext.Connection.RemoteIpAddress?.ToString() ?? "Unknown";
                
                _logger.LogInformation("========== LOGIN ENDPOINT CALLED ==========");
                _logger.LogInformation("Timestamp: {Timestamp}", DateTime.UtcNow);
                _logger.LogInformation("Request received from IP: {RemoteIP}", remoteIP);
                
                _logger.LogDebug("Login request received. Email: {Email}", request?.Email ?? "null");

                // Validation: Check if all required fields are provided
                if (string.IsNullOrWhiteSpace(request?.Email))
                {
                    _logger.LogWarning("REJECTED - Email is null, empty, or whitespace");
                    return BadRequest(new ApiMessageDto { message = "Email and password are required." });
                }

                if (string.IsNullOrWhiteSpace(request?.Password))
                {
                    _logger.LogWarning("REJECTED - Password is null, empty, or whitespace for email: {Email}", request?.Email ?? "null");
                    return BadRequest(new ApiMessageDto { message = "Email and password are required." });
                }

                _logger.LogInformation("? All required fields validated. Email: {Email}", request.Email);

                // Look up user
                _logger.LogDebug("Looking up user in database with email: {Email}", request.Email);
                var user = _context.Users.FirstOrDefault(u => u.Email == request.Email);
                
                if (user == null)
                {
                    _logger.LogWarning("REJECTED - No user found with email: {Email}", request.Email);
                    return Unauthorized(new ApiMessageDto { message = "Invalid email or password." });
                }

                _logger.LogInformation("? User found in database:");
                _logger.LogInformation("  - User ID: {UserId}", user.Id);
                _logger.LogInformation("  - Email: {Email}", user.Email);
                _logger.LogInformation("  - Name: {Name}", user.Name);
                _logger.LogInformation("  - Created At: {CreatedAt}", user.CreatedAt);
                _logger.LogInformation("  - Last Login At: {LastLoginAt}", user.LastLoginAt?.ToString("yyyy-MM-dd HH:mm:ss") ?? "Never");

                // Verify password
                _logger.LogDebug("Verifying password for user: {Email}", request.Email);
                if (!_authService.VerifyPassword(request.Password, user.PasswordHash))
                {
                    _logger.LogWarning("REJECTED - Password verification failed for email: {Email} (User ID: {UserId})", 
                        request.Email, user.Id);

                    // Log failed login attempt
                    try
                    {
                        await _auditLogger.LogLoginAttemptAsync(request.Email, false, remoteIP);
                    }
                    catch (Exception auditEx)
                    {
                        _logger.LogWarning(auditEx, "Failed to write failed-login audit event for {Email}", request.Email);
                    }

                    return Unauthorized(new ApiMessageDto { message = "Invalid email or password." });
                }

                _logger.LogInformation("? Password verification successful");

                // Update last login time
                _logger.LogDebug("Updating last login timestamp for user: {Email}", request.Email);
                user.LastLoginAt = DateTime.UtcNow;
                await _context.SaveChangesAsync();
                _logger.LogInformation("? Last login timestamp updated");

                // Load user progress from saved files
                _logger.LogDebug("Loading user progress data for: {Email}", request.Email);
                var savedProgress = await _progressService.LoadUserProgressAsync(user.Id);
                if (savedProgress != null)
                {
                    user.TotalXp = savedProgress.TotalXp;
                    user.LearningStreak = savedProgress.LearningStreak;
                    _logger.LogInformation("? User progress loaded: TotalXp={TotalXp}, Streak={Streak}", 
                        user.TotalXp, user.LearningStreak);
                }

                // Generate JWT token and refresh token
                _logger.LogDebug("Generating JWT token for user: {Email}", request.Email);
                var token = _authService.GenerateJwtToken(user);
                var refreshToken = _authService.GenerateRefreshToken();

                // Save refresh token to user
                user.RefreshToken = refreshToken;
                user.RefreshTokenExpiryTime = DateTime.UtcNow.AddDays(7);
                await _context.SaveChangesAsync();

                _logger.LogInformation("? JWT token and refresh token generated successfully");

                _logger.LogInformation("========== LOGIN SUCCESSFUL ==========");
                _logger.LogInformation("User logged in:");
                _logger.LogInformation("  - User ID: {UserId}", user.Id);
                _logger.LogInformation("  - Email: {Email}", user.Email);
                _logger.LogInformation("  - Name: {Name}", user.Name);
                _logger.LogInformation("  - Total XP: {TotalXp}", user.TotalXp);
                _logger.LogInformation("  - Learning Streak: {Streak}", user.LearningStreak);
                _logger.LogInformation("  - Login Time: {LoginTime}", user.LastLoginAt);
                _logger.LogInformation("  - Token Generated: {TokenLength} characters", token?.Length ?? 0);
                _logger.LogInformation("");

                // Log successful login attempt
                try
                {
                    await _auditLogger.LogLoginAttemptAsync(request.Email, true, remoteIP);
                }
                catch (Exception auditEx)
                {
                    _logger.LogWarning(auditEx, "Failed to write successful-login audit event for {Email}", request.Email);
                }

                return Ok(new AuthTokenResponseDto
                {
                    token = token ?? string.Empty,
                    refreshToken = refreshToken ?? string.Empty,
                    userId = user.Id,
                    name = user.Name
                });
            }
            catch (Exception ex)
            {
                _logger.LogError("========== LOGIN FAILED WITH EXCEPTION ==========");
                _logger.LogError("Exception Type: {ExceptionType}", ex.GetType().Name);
                _logger.LogError("Exception Message: {Message}", ex.Message);
                _logger.LogError("Stack Trace: {StackTrace}", ex.StackTrace);
                _logger.LogError("Inner Exception: {InnerException}", ex.InnerException?.Message ?? "None");
                _logger.LogError("========== END EXCEPTION ==========");
                _logger.LogError("");

                var remoteIP = HttpContext.Connection.RemoteIpAddress?.ToString() ?? "Unknown";
                var requestEmail = (HttpContext.Request.Form["email"].ToString() ?? "unknown");
                _ = _auditLogger.LogLoginAttemptAsync(requestEmail, false, remoteIP);

                return StatusCode(StatusCodes.Status500InternalServerError, 
                    new ApiMessageDto { message = "An error occurred during login." });
            }
        }

        [Authorize]
        [HttpPost("logout")]
        public async Task<ActionResult<object>> Logout()
        {
            try
            {
                var remoteIP = HttpContext.Connection.RemoteIpAddress?.ToString() ?? "Unknown";

                _logger.LogInformation("========== LOGOUT ENDPOINT CALLED ==========");
                _logger.LogInformation("Timestamp: {Timestamp}", DateTime.UtcNow);
                _logger.LogInformation("Request received from IP: {RemoteIP}", remoteIP);

                // Extract user info from JWT token (if present)
                var userIdClaim = User.FindFirst(System.Security.Claims.ClaimTypes.NameIdentifier);
                var emailClaim = User.FindFirst(System.Security.Claims.ClaimTypes.Email);

                var userId = userIdClaim?.Value;
                var email = emailClaim?.Value;

                if (userIdClaim != null || emailClaim != null)
                {
                    _logger.LogInformation("User logout requested:");
                    _logger.LogInformation("  - User ID: {UserId}", userId ?? "Unknown");
                    _logger.LogInformation("  - Email: {Email}", email ?? "Unknown");
                    _logger.LogInformation("  - Logout Time: {LogoutTime}", DateTime.UtcNow);
                }
                else
                {
                    _logger.LogWarning("Logout requested but no valid token found");
                }

                // Get the token from the Authorization header
                var authHeader = Request.Headers["Authorization"].ToString();
                if (!string.IsNullOrEmpty(authHeader) && authHeader.StartsWith("Bearer "))
                {
                    var token = authHeader.Substring("Bearer ".Length).Trim();

                    // Blacklist the token
                    var expiryTime = DateTime.UtcNow.AddHours(1);
                    await _tokenBlacklist.BlacklistTokenAsync(token, expiryTime);
                    _logger.LogInformation("? Token added to blacklist");
                }

                if (!string.IsNullOrEmpty(userId) && !string.IsNullOrEmpty(email))
                {
                    await _auditLogger.LogLogoutAsync(userId, email, remoteIP);
                }

                _logger.LogInformation("? Logout successful");
                _logger.LogInformation("========== LOGOUT SUCCESSFUL ==========");
                _logger.LogInformation("");

                return Ok(new ApiMessageDto { message = "Logout successful." });
            }
            catch (Exception ex)
            {
                _logger.LogError("========== LOGOUT FAILED WITH EXCEPTION ==========");
                _logger.LogError("Exception Type: {ExceptionType}", ex.GetType().Name);
                _logger.LogError("Exception Message: {Message}", ex.Message);
                _logger.LogError("Stack Trace: {StackTrace}", ex.StackTrace);
                _logger.LogError("========== END EXCEPTION ==========");
                _logger.LogError("");

                return StatusCode(StatusCodes.Status500InternalServerError, 
                    new ApiMessageDto { message = "An error occurred during logout." });
            }
        }

        [HttpDelete("account/{userId}")]
        public async Task<ActionResult<object>> DeleteAccount(string userId)
        {
            try
            {
                var remoteIP = HttpContext.Connection.RemoteIpAddress?.ToString() ?? "Unknown";
                
                _logger.LogInformation("========== DELETE ACCOUNT ENDPOINT CALLED ==========");
                _logger.LogInformation("Timestamp: {Timestamp}", DateTime.UtcNow);
                _logger.LogInformation("Request received from IP: {RemoteIP}", remoteIP);
                _logger.LogInformation("User ID to delete: {UserId}", userId);

                // Validation: Check if userId is provided
                if (string.IsNullOrWhiteSpace(userId))
                {
                    _logger.LogWarning("REJECTED - User ID is null, empty, or whitespace");
                    return BadRequest(new ApiMessageDto { message = "User ID is required." });
                }

                _logger.LogInformation("? Validation passed. User ID: {UserId}", userId);

                // Look up user
                _logger.LogDebug("Looking up user in database with ID: {UserId}", userId);
                var user = _context.Users.FirstOrDefault(u => u.Id == userId);
                
                if (user == null)
                {
                    _logger.LogWarning("REJECTED - No user found with ID: {UserId}", userId);
                    return NotFound(new ApiMessageDto { message = "User not found." });
                }

                _logger.LogInformation("? User found in database:");
                _logger.LogInformation("  - User ID: {UserId}", user.Id);
                _logger.LogInformation("  - Email: {Email}", user.Email);
                _logger.LogInformation("  - Name: {Name}", user.Name);
                _logger.LogInformation("  - Created At: {CreatedAt}", user.CreatedAt);

                // Save deletion record to file before deleting
                _logger.LogDebug("Saving account deletion record to local file...");
                try
                {
                    await SaveAccountDeletionToFileAsync(user);
                    _logger.LogInformation("? Account deletion record saved to file");
                }
                catch (Exception fileEx)
                {
                    _logger.LogWarning("Warning - Failed to save account deletion to file: {Exception}", fileEx.Message);
                    // Don't fail deletion if file save fails
                }

                // Delete user progress data files
                _logger.LogDebug("Deleting user progress data files...");
                try
                {
                    await _progressService.DeleteUserDataAsync(user.Id);
                    _logger.LogInformation("? User progress data files deleted");
                }
                catch (Exception fileEx)
                {
                    _logger.LogWarning("Warning - Failed to delete user progress data: {Exception}", fileEx.Message);
                    // Don't fail deletion if file delete fails
                }

                // Delete the user from database
                _logger.LogDebug("Deleting user from database...");
                _context.Users.Remove(user);
                await _context.SaveChangesAsync();
                _logger.LogInformation("? User successfully deleted from database");

                _logger.LogInformation("========== ACCOUNT DELETION SUCCESSFUL ==========");
                _logger.LogInformation("Account deleted: Email={Email}, UserId={UserId}", 
                    user.Email, user.Id);
                _logger.LogInformation("");

                return Ok(new ApiMessageDto { message = "Account deleted successfully." });
            }
            catch (Exception ex)
            {
                _logger.LogError("========== DELETE ACCOUNT FAILED WITH EXCEPTION ==========");
                _logger.LogError("Exception Type: {ExceptionType}", ex.GetType().Name);
                _logger.LogError("Exception Message: {Message}", ex.Message);
                _logger.LogError("Stack Trace: {StackTrace}", ex.StackTrace);
                _logger.LogError("Inner Exception: {InnerException}", ex.InnerException?.Message ?? "None");
                _logger.LogError("========== END EXCEPTION ==========");
                _logger.LogError("");
                
                return StatusCode(StatusCodes.Status500InternalServerError, 
                    new ApiMessageDto { message = "An error occurred during account deletion." });
            }
        }

        private async Task SaveAccountCreationToFileAsync(User user)
        {
            try
            {
                // Create accounts directory in AppData
                string appDataPath = Environment.GetFolderPath(Environment.SpecialFolder.ApplicationData);
                string accountsDir = Path.Combine(appDataPath, "SignLanguageApp", "Accounts");
                
                // Ensure directory exists
                if (!Directory.Exists(accountsDir))
                {
                    Directory.CreateDirectory(accountsDir);
                    _logger.LogDebug("Created accounts directory: {Directory}", accountsDir);
                }

                // Create file name with user ID and timestamp
                string fileName = $"account_created_{user.Id}_{DateTime.UtcNow:yyyyMMdd_HHmmss}.txt";
                string filePath = Path.Combine(accountsDir, fileName);

                // Prepare account creation record
                var record = new StringBuilder();
                record.AppendLine("========== ACCOUNT CREATION RECORD ==========");
                record.AppendLine($"Created Date: {DateTime.UtcNow:yyyy-MM-dd HH:mm:ss} UTC");
                record.AppendLine($"User ID: {user.Id}");
                record.AppendLine($"Email: {user.Email}");
                record.AppendLine($"Name: {user.Name}");
                record.AppendLine($"Account Created At: {user.CreatedAt:yyyy-MM-dd HH:mm:ss} UTC");
                record.AppendLine($"Learning Streak: {user.LearningStreak}");
                record.AppendLine("==========================================");

                // Write to file
                await System.IO.File.WriteAllTextAsync(filePath, record.ToString());
                _logger.LogDebug("Account creation record saved to: {FilePath}", filePath);
            }
            catch (Exception ex)
            {
                _logger.LogError("Error saving account creation to file: {Exception}", ex.Message);
                throw;
            }
        }

        private async Task SaveAccountDeletionToFileAsync(User user)
        {
            try
            {
                // Create accounts directory in AppData
                string appDataPath = Environment.GetFolderPath(Environment.SpecialFolder.ApplicationData);
                string accountsDir = Path.Combine(appDataPath, "SignLanguageApp", "Accounts");
                
                // Ensure directory exists
                if (!Directory.Exists(accountsDir))
                {
                    Directory.CreateDirectory(accountsDir);
                    _logger.LogDebug("Created accounts directory: {Directory}", accountsDir);
                }

                // Create file name with user ID and timestamp
                string fileName = $"account_deleted_{user.Id}_{DateTime.UtcNow:yyyyMMdd_HHmmss}.txt";
                string filePath = Path.Combine(accountsDir, fileName);

                // Prepare account deletion record
                var record = new StringBuilder();
                record.AppendLine("========== ACCOUNT DELETION RECORD ==========");
                record.AppendLine($"Deleted Date: {DateTime.UtcNow:yyyy-MM-dd HH:mm:ss} UTC");
                record.AppendLine($"User ID: {user.Id}");
                record.AppendLine($"Email: {user.Email}");
                record.AppendLine($"Name: {user.Name}");
                record.AppendLine($"Account Created At: {user.CreatedAt:yyyy-MM-dd HH:mm:ss} UTC");
                record.AppendLine($"Account Deleted At: {DateTime.UtcNow:yyyy-MM-dd HH:mm:ss} UTC");
                record.AppendLine($"Account Duration: {(DateTime.UtcNow - user.CreatedAt).TotalDays:F2} days");
                record.AppendLine($"Last Login: {(user.LastLoginAt?.ToString("yyyy-MM-dd HH:mm:ss") ?? "Never")} UTC");
                record.AppendLine("===========================================");

                // Write to file
                await System.IO.File.WriteAllTextAsync(filePath, record.ToString());
                _logger.LogDebug("Account deletion record saved to: {FilePath}", filePath);
            }
            catch (Exception ex)
            {
                _logger.LogError("Error saving account deletion to file: {Exception}", ex.Message);
                throw;
            }
        }

        [HttpPost("refresh")]
        public async Task<ActionResult<object>> RefreshToken([FromBody] RefreshTokenRequest request)
        {
            try
            {
                _logger.LogInformation("========== REFRESH TOKEN ENDPOINT CALLED ==========");
                _logger.LogInformation("Timestamp: {Timestamp}", DateTime.UtcNow);

                if (string.IsNullOrWhiteSpace(request?.RefreshToken))
                {
                    _logger.LogWarning("REJECTED - Refresh token is null or empty");
                    return BadRequest(new ApiMessageDto { message = "Refresh token is required." });
                }

                // Find user by refresh token
                _logger.LogDebug("Looking up user with refresh token");
                var user = _context.Users.FirstOrDefault(u => u.RefreshToken == request.RefreshToken);

                if (user == null)
                {
                    _logger.LogWarning("REJECTED - No user found with provided refresh token");
                    return Unauthorized(new ApiMessageDto { message = "Invalid refresh token." });
                }

                // Check if refresh token has expired
                if (user.RefreshTokenExpiryTime == null || user.RefreshTokenExpiryTime < DateTime.UtcNow)
                {
                    _logger.LogWarning("REJECTED - Refresh token expired for user: {UserId}", user.Id);
                    return Unauthorized(new ApiMessageDto { message = "Refresh token has expired." });
                }

                _logger.LogInformation("? Refresh token validation successful for user: {UserId}", user.Id);

                // Generate new JWT token and refresh token
                _logger.LogDebug("Generating new JWT token for user: {UserId}", user.Id);
                var newToken = _authService.GenerateJwtToken(user);
                var newRefreshToken = _authService.GenerateRefreshToken();

                // Update refresh token in database
                user.RefreshToken = newRefreshToken;
                user.RefreshTokenExpiryTime = DateTime.UtcNow.AddDays(7);
                _context.Users.Update(user);
                await _context.SaveChangesAsync();

                _logger.LogInformation("========== REFRESH TOKEN SUCCESSFUL ==========");
                _logger.LogInformation("New token generated for user: {UserId}", user.Id);
                _logger.LogInformation("");

                return Ok(new AuthTokenResponseDto
                {
                    token = newToken,
                    refreshToken = newRefreshToken,
                    userId = user.Id,
                    name = user.Name
                });
            }
            catch (Exception ex)
            {
                _logger.LogError("========== REFRESH TOKEN FAILED WITH EXCEPTION ==========");
                _logger.LogError("Exception Type: {ExceptionType}", ex.GetType().Name);
                _logger.LogError("Exception Message: {Message}", ex.Message);
                _logger.LogError("========== END EXCEPTION ==========");
                _logger.LogError("");

                return StatusCode(StatusCodes.Status500InternalServerError,
                    new ApiMessageDto { message = "An error occurred during token refresh." });
            }
        }
    }
}

