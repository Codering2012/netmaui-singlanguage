using SignLanguageApp.ViewModels;

namespace SignLanguageApp.Pages;

public partial class SentenceBuilderPage : ContentPage
{
    public SentenceBuilderPage(SentenceBuilderViewModel viewModel)
    {
        InitializeComponent();
        BindingContext = viewModel;
    }
}
