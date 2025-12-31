$sections = Get-ChildItem sec*.tex
$appendices = Get-ChildItem app*.tex

$secDict = @{}
foreach ($f in $sections) {
    if ($f.Name -match "sec(\d+)") {
        $num = [int]$matches[1]
        if (-not $secDict.ContainsKey($num)) {
            $secDict[$num] = @()
        }
        $secDict[$num] += $f.Name
    }
}

$finalSections = @()
foreach ($num in ($secDict.Keys | Sort-Object)) {
    $candidates = $secDict[$num]
    $chosen = $candidates[0]
    # Preference logic
    foreach ($pref in @("_new", "_revised", "_rigorous")) {
        foreach ($c in $candidates) {
            if ($c -match $pref) {
                $chosen = $c
                break
            }
        }
        if ($chosen -ne $candidates[0]) { break }
    }
    $finalSections += $chosen
}

$finalAppendices = $appendices | Sort-Object Name | ForEach-Object { $_.Name }

$content = @"
\documentclass[11pt,a4paper]{article}
\input{preamble}
\input{document-info}

\begin{document}
\maketitle
\begin{abstract}
\input{abstract_revised}
\end{abstract}
\tableofcontents
\newpage

\part{Main Text}
"@

foreach ($s in $finalSections) {
    $content += "\input{$s}`n"
}

$content += @"

\part{Appendices}
\appendix
"@

foreach ($a in $finalAppendices) {
    $content += "\input{$a}`n"
}

$content += "\end{document}"

$content | Out-File -Encoding utf8 yang_mills_complete.tex
Write-Host "Created yang_mills_complete.tex"
