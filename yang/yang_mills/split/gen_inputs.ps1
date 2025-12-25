$files = Get-ChildItem *.tex
$secs = $files | Where-Object { $_.Name -match '^sec(\d+)_' } | Sort-Object { [int]($_.Name -replace '^sec(\d+)_.*', '$1') }
$apps = $files | Where-Object { $_.Name -match '^app(\d+)_' } | Sort-Object { [int]($_.Name -replace '^app(\d+)_.*', '$1') }

Write-Output "% Sections"
foreach ($f in $secs) { Write-Output "\input{$($f.BaseName)}" }
Write-Output ""
Write-Output "\appendix"
Write-Output "% Appendices"
foreach ($f in $apps) { Write-Output "\input{$($f.BaseName)}" }