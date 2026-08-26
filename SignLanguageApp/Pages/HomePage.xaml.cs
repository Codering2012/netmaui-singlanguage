using Microsoft.Maui.Controls;
using SignLanguageApp.ViewModels;

namespace SignLanguageApp.Pages
{
    public partial class HomePage : ContentPage
    {
        public HomePage()
        {
            try
            {
                InitializeComponent();
                BindingContext = App.Services.GetService<HomeViewModel>();
            }
            catch (System.Exception ex)
            {
                SignLanguageApp.Helpers.GlobalExceptionHandler.HandleException(ex);
                throw;
            }
        }

        public HomePage(HomeViewModel viewModel)
        {
            try
            {
                InitializeComponent();
                BindingContext = viewModel;
            }
            catch (System.Exception ex)
            {
                SignLanguageApp.Helpers.GlobalExceptionHandler.HandleException(ex);
                throw;
            }
        }
    }
}