# PowerShell script to split paper_clean.tex into separate section files

$inputFile = "paper_clean.tex"
$outputDir = "split"

# Read all lines
Write-Host "Reading paper_clean.tex..."
$lines = Get-Content $inputFile -Encoding UTF8

# Find \begin{document} and \end{document}
$docBegin = -1
$docEnd = -1
for ($i = 0; $i -lt $lines.Count; $i++) {
    if ($lines[$i] -match '\\begin\{document\}') {
        $docBegin = $i
    }
    if ($lines[$i] -match '\\end\{document\}') {
        $docEnd = $i
        break
    }
}

Write-Host "Document begins at line $docBegin, ends at line $docEnd"

# Extract preamble and ending
$preamble = $lines[0..$docBegin]
$ending = $lines[$docEnd..($lines.Count-1)]

# Find all section markers
Write-Host "Finding sections..."
$sections = @()
for ($i = $docBegin; $i -lt $docEnd; $i++) {
    if ($lines[$i] -match '^\\section(\*?)\{') {
        $sections += $i
    }
}

Write-Host "Found $($sections.Count) sections"

# Find front matter (before first section)
$firstSectionLine = $sections[0]
$frontMatter = $lines[($docBegin+1)..($firstSectionLine-1)]

# Function to sanitize filename
function Get-SafeFilename {
    param($text, $index)
    
    # Remove LaTeX commands
    $clean = $text -replace '\\texorpdfstring\{[^}]+\}\{[^}]*\}', ''
    $clean = $clean -replace '\\[a-zA-Z]+\{([^}]+)\}', '$1'
    $clean = $clean -replace '[\\$\{\}^_]', ''
    $clean = $clean -replace '[^\w\s-]', ''
    $clean = $clean -replace '[-\s]+', '_'
    $clean = $clean.Trim('_').ToLower()
    
    # Truncate and add number
    if ($clean.Length -gt 50) {
        $clean = $clean.Substring(0, 50)
    }
    
    return "sec_{0:D2}_{1}.tex" -f ($index + 1), $clean
}

# Extract each section
Write-Host "Extracting sections..."
for ($i = 0; $i -lt $sections.Count; $i++) {
    $startLine = $sections[$i]
    $endLine = if ($i -lt $sections.Count - 1) { $sections[$i+1] - 1 } else { $docEnd - 1 }
    
    # Get section title
    $sectionLine = $lines[$startLine]
    if ($sectionLine -match '\\section(\*?)\{([^}]+)\}') {
        $title = $matches[2]
    } else {
        $title = "section_$i"
    }
    
    $filename = Get-SafeFilename $title $i
    $filepath = Join-Path $outputDir $filename
    
    # Extract section content
    $sectionContent = $lines[$startLine..$endLine]
    
    # Write to file
    $sectionContent | Out-File -FilePath $filepath -Encoding UTF8
    
    Write-Host "  Created: $filename"
}

# Create main.tex
Write-Host "Creating main.tex..."
$mainPath = Join-Path $outputDir "main.tex"
$mainLines = @()

# Add preamble
$mainLines += $preamble

# Add front matter
$mainLines += ""
$mainLines += $frontMatter

# Add include statements for each section
$mainLines += ""
for ($i = 0; $i -lt $sections.Count; $i++) {
    $sectionLine = $lines[$sections[$i]]
    if ($sectionLine -match '\\section(\*?)\{([^}]+)\}') {
        $title = $matches[2]
    } else {
        $title = "section_$i"
    }
    
    $filename = Get-SafeFilename $title $i
    $filebase = $filename -replace '\.tex$', ''
    $mainLines += "\input{$filebase}  % $title"
}

# Add ending
$mainLines += ""
$mainLines += $ending

# Write main.tex
$mainLines | Out-File -FilePath $mainPath -Encoding UTF8

Write-Host "`n==== DONE ===="
Write-Host "Created main.tex and $($sections.Count) section files"
Write-Host "To compile: cd split && pdflatex main.tex"
