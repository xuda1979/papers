$currentDir = Get-Location
$fileName = "2026工作计划_许达.docx"
$filePath = Join-Path $currentDir $fileName

Write-Host "Target file: $filePath"

try {
    $word = New-Object -ComObject Word.Application
    $word.Visible = $false
    
    if (Test-Path $filePath) {
        $doc = $word.Documents.Open($filePath)
        
        # 1. Set Global Font to FangSong (Chinese) and Color to Black
        $doc.Content.Font.Color = 0 # wdColorBlack
        $doc.Content.Font.NameFarEast = "仿宋"
        $doc.Content.Font.Name = "仿宋" 
        
        # 2. Use Wildcards to set English/Numbers to Times New Roman
        $find = $doc.Content.Find
        $find.ClearFormatting()
        $find.Replacement.ClearFormatting()
        
        # Match alphanumeric and common punctuation
        $find.Text = "[0-9a-zA-Z\.\-\(\)]{1,}" 
        $find.MatchWildcards = $true
        $find.Replacement.Font.Name = "Times New Roman"
        
        # Execute Replace All (wdReplaceAll = 2)
        $find.Execute($null, $null, $null, $null, $null, $null, $null, $null, $null, $null, 2)
        
        $doc.Save()
        $doc.Close()
        Write-Host "Successfully formatted the document."
    } else {
        Write-Error "File not found at: $filePath"
    }
} catch {
    Write-Error "Failed to format document. Error: $_"
} finally {
    if ($word) {
        $word.Quit()
        [System.Runtime.Interopservices.Marshal]::ReleaseComObject($word) | Out-Null
    }
}
