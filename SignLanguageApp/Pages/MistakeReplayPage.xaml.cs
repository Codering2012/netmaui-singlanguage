using Microsoft.Maui.Controls;
using SignLanguageApp.ViewModels;

namespace SignLanguageApp.Pages
{
    public partial class MistakeReplayPage : ContentPage
    {
        public MistakeReplayPage(MistakeReplayViewModel viewModel)
        {
            InitializeComponent();
            BindingContext = viewModel;
        }
    }
}
