# Push the repo (code only, no data/models/outputs) from this laptop to the cluster home.
# Usage (PowerShell, VPN on):  .\cluster\sync.ps1
$ErrorActionPreference = "Stop"
$repo = Split-Path -Parent $PSScriptRoot
$remote = "nischit001@10.96.189.12"
$tmp = Join-Path $env:TEMP "DL-Project-sync.tar"
Push-Location $repo
tar --exclude=.git --exclude=data --exclude=out --exclude=logs --exclude=__pycache__ -cf $tmp .
Pop-Location
ssh $remote "mkdir -p ~/DL-Project"
scp $tmp "${remote}:~/DL-Project-sync.tar"
ssh $remote "cd ~/DL-Project && tar -xf ~/DL-Project-sync.tar && rm ~/DL-Project-sync.tar && mkdir -p logs out data && echo SYNC_OK"
Remove-Item $tmp
