using SignLanguageApp.ViewModels;

namespace SignLanguageApp.Pages;

public partial class DictionaryPage : ContentPage
{
    private readonly DictionaryViewModel _viewModel;

    public DictionaryPage(DictionaryViewModel viewModel)
    {
        InitializeComponent();
        _viewModel = viewModel;
        BindingContext = _viewModel;
    }
}
