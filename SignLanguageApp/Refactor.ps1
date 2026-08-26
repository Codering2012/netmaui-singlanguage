$pattern = '(?s)(\[ObservableProperty\](?:(?!\bprivate\b).)*?)private\s+([a-zA-Z0-9_<>\[\],\?]+)\s+_([a-zA-Z])(\w*)\s*;'

$files = Get-ChildItem -Path "C:\Users\Windows 10 21H1\source\repos\SignLanguageApp\ViewModels" -Filter "*.cs" -Recurse

foreach ($file in $files) {
    $content = Get-Content $file.FullName -Raw
    
    $newContent = [regex]::Replace($content, $pattern, {
        param($match)
        $attrs = $match.Groups[1].Value
        $type = $match.Groups[2].Value
        $firstChar = $match.Groups[3].Value.ToUpper()
        $rest = $match.Groups[4].Value
        
        return "${attrs}public partial $type $firstChar$rest { get; set; }"
    })
    
    if ($content -cne $newContent) {
        Set-Content -Path $file.FullName -Value $newContent -NoNewline
        Write-Host "Updated $($file.Name)"
    }
}
