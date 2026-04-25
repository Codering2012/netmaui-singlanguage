using Microsoft.Maui.Controls;
using SignLanguageApp.ViewModels;

namespace SignLanguageApp.Pages;

public partial class RegisterPage : ContentPage
{
    public RegisterPage(RegisterViewModel viewModel)
    {
        InitializeComponent();
        BindingContext = viewModel;
    }
}
