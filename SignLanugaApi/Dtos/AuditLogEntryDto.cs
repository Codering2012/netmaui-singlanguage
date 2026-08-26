using System;

namespace SignLanguageApi.Dtos
{
    public class AuditLogEntryDto
    {
        public string EventType { get; set; } = string.Empty;
        public string? Email { get; set; }
        public string? UserId { get; set; }
        public bool? Success { get; set; }
        public string IpAddress { get; set; } = string.Empty;
        public string Timestamp { get; set; } = string.Empty;
        public string Status { get; set; } = string.Empty;
        public string? Endpoint { get; set; }
    }
}
