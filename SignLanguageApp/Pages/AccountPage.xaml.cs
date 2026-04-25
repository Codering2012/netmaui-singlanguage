using Microsoft.Maui.Controls;
using SignLanguageApp.ViewModels;
using SignLanguageApp.Services;

namespace SignLanguageApp.Pages
{
    public partial class AccountPage : ContentPage
    {
        private readonly AccountViewModel _viewModel;

        public AccountPage(AccountViewModel viewModel)
        {
            InitializeComponent();
            _viewModel = viewModel;
            BindingContext = _viewModel;
        }

        protected override async void OnAppearing()
        {
            base.OnAppearing();
            await _viewModel.LoadDataCommand.ExecuteAsync(null);
        }

        protected override void OnDisappearing()
        {
            base.OnDisappearing();
            _viewModel.OnDisappearing();
        }
    }
}
