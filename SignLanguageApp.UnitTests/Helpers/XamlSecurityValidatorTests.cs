using System.Security;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using SignLanguageApp.Helpers;

namespace SignLanguageApp.UnitTests.Helpers;

[TestClass]
public class XamlSecurityValidatorTests
{
    [TestMethod]
    public void EnsureSafeXaml_ValidXaml_DoesNotThrow()
    {
        string safeXaml = @"<StackLayout xmlns=""http://schemas.microsoft.com/dotnet/2021/maui"">
            <Label Text=""Hello World"" />
        </StackLayout>";

        XamlSecurityValidator.EnsureSafeXaml(safeXaml);
    }

    [TestMethod]
    public void EnsureSafeXaml_ObjectDataProvider_ThrowsSecurityException()
    {
        string unsafeXaml = @"<StackLayout xmlns=""http://schemas.microsoft.com/dotnet/2021/maui"">
            <ObjectDataProvider MethodName=""Start"" />
        </StackLayout>";

        Assert.Throws<SecurityException>(() => XamlSecurityValidator.EnsureSafeXaml(unsafeXaml));
    }

    [TestMethod]
    public void EnsureSafeXaml_ClrNamespaceSystem_ThrowsSecurityException()
    {
        string unsafeXaml = @"<ContentPage xmlns:sys=""clr-namespace:System;assembly=mscorlib"">
            <Label Text=""Test"" />
        </ContentPage>";

        Assert.Throws<SecurityException>(() => XamlSecurityValidator.EnsureSafeXaml(unsafeXaml));
    }
}
