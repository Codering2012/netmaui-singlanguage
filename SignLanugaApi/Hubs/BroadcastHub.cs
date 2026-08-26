using Microsoft.AspNetCore.SignalR;

namespace SignLanguageApi.Hubs
{
    public class BroadcastHub : Hub
    {
        public async Task SendGlobalMessage(string user, string message)
        {
            await Clients.All.SendAsync("ReceiveGlobalMessage", user, message);
        }

        public async Task NotifyNewMedia(string type, string fileName, string url)
        {
            await Clients.All.SendAsync("NewMediaAdded", type, fileName, url);
        }

        public override async Task OnConnectedAsync()
        {
            await Groups.AddToGroupAsync(Context.ConnectionId, "AllUsers");
            await base.OnConnectedAsync();
        }
    }
}
