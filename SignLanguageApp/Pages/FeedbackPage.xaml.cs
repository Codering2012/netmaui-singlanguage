using SignLanguageApp.ViewModels;

namespace SignLanguageApp.Pages
{
    public partial class FeedbackPage : ContentPage
    {
        public FeedbackPage(FeedbackViewModel viewModel)
        {
            InitializeComponent();
            BindingContext = viewModel;
        }
    }
}
