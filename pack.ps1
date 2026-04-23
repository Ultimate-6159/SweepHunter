# ============================================================
#  SweepHunter - Portable packer (called by pack.bat)
# ============================================================
$ErrorActionPreference = 'Stop'

$src = $PSScriptRoot
$ts  = Get-Date -Format 'yyyyMMdd_HHmmss'
$out = Join-Path (Split-Path $src -Parent) ("SweepHunter_$ts.zip")

Write-Host "Output : $out"

if (Test-Path $out) { Remove-Item $out -Force }

$tmp  = Join-Path $env:TEMP ("SH_pack_" + [guid]::NewGuid().ToString('N'))
$dest = Join-Path $tmp 'SweepHunter'
$null = New-Item -ItemType Directory -Path $dest -Force

$excludeDirs  = @('data', '__pycache__', '.vs', '.git', '.idea')
$excludeFiles = @('*.pyc', '*.log', '*.sqlite', '*.zip')

$srcLen = $src.Length
Get-ChildItem -Path $src -Recurse -Force | ForEach-Object {
    $rel = $_.FullName.Substring($srcLen).TrimStart('\')
    if (-not $rel) { return }

    # skip if any path segment is an excluded directory
    $parts = $rel -split '\\'
    foreach ($p in $parts) {
        if ($excludeDirs -contains $p) { return }
    }

    if ($_.PSIsContainer) {
        $target = Join-Path $dest $rel
        $null = New-Item -ItemType Directory -Path $target -Force
    } else {
        # skip excluded file patterns
        foreach ($pat in $excludeFiles) {
            if ($_.Name -like $pat) { return }
        }
        $target = Join-Path $dest $rel
        $td = Split-Path $target -Parent
        if (-not (Test-Path $td)) { $null = New-Item -ItemType Directory -Path $td -Force }
        Copy-Item $_.FullName $target -Force
    }
}

Compress-Archive -Path $dest -DestinationPath $out -Force
Remove-Item $tmp -Recurse -Force

$sizeKB = [math]::Round((Get-Item $out).Length / 1KB, 1)
Write-Host "Packed : $out  ($sizeKB KB)" -ForegroundColor Green
