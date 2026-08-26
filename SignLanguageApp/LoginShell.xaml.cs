namespace SignLanguageApp;

public partial class LoginShell : Shell
{
    public LoginShell()
    {
        try
        {
            InitializeComponent();
        }
        catch (System.Exception ex)
        {
            SignLanguageApp.Helpers.GlobalExceptionHandler.HandleException(ex);
            throw;
        }
    }
}
