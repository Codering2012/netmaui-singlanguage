using System.Text.RegularExpressions;
using System.Xml.Linq;

namespace SignLanguageApp.Helpers;

public static class BinaryXamlIsolator
{
    public static List<string> SplitIntoElements(string xaml)
    {
        try
        {
            var doc = XDocument.Parse(xaml);
            var root = doc.Root;
            if (root == null) return new List<string> { xaml };

            var elements = new List<string>();
            foreach (var node in root.Elements())
            {
                elements.Add(node.ToString());
            }

            return elements;
        }
        catch
        {
            // Fallback to regex if XML is malformed
            var matches = Regex.Matches(xaml, @"<[A-Za-z]+[^>]*>.*?</[A-Za-z]+>|<[A-Za-z]+[^>]*/>", RegexOptions.Singleline);
            return matches.Cast<Match>().Select(m => m.Value).ToList();
        }
    }

    public static string ReconstructXaml(string originalRoot, List<string> elements)
    {
        try
        {
            var rootMatch = Regex.Match(originalRoot, @"^<([A-Za-z]+[^>]*)>", RegexOptions.Singleline);
            if (!rootMatch.Success) return string.Join("\n", elements);

            var rootStart = rootMatch.Value;
            var rootName = Regex.Match(rootStart, @"<([A-Za-z:]+)").Groups[1].Value;
            
            return $"{rootStart}\n{string.Join("\n", elements)}\n</{rootName}>";
        }
        catch
        {
            return string.Join("\n", elements);
        }
    }
}
