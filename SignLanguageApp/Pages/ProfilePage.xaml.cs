using SignLanguageApp.ViewModels;

namespace SignLanguageApp.Pages;

public partial class ProfilePage : ContentPage
{
    public ProfilePage(AccountViewModel viewModel)
    {
        InitializeComponent();
        BindingContext = viewModel;
    }

    protected override async void OnAppearing()
    {
        base.OnAppearing();
        if (BindingContext is AccountViewModel vm)
        {
            await vm.LoadDataCommand.ExecuteAsync(null);
        }
    }

    protected override void OnDisappearing()
    {
        base.OnDisappearing();
        if (BindingContext is AccountViewModel vm)
        {
            vm.OnDisappearing();
        }
    }
}
