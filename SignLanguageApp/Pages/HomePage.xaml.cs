using Microsoft.Maui.Controls;
using SignLanguageApp.ViewModels;

namespace SignLanguageApp.Pages
{
    public partial class HomePage : ContentPage
    {
        public HomePage(HomeViewModel viewModel)
        {
            InitializeComponent();
            BindingContext = viewModel;
        }
    }
}