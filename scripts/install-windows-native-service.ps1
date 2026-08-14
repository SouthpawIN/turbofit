[CmdletBinding()]
param(
    [Parameter(Mandatory = $false)] [string] $Binary,
    [Parameter(Mandatory = $false)] [string] $Model,
    [ValidateRange(1, 65535)] [int] $Port = 8092,
    [ValidateRange(1, 65535)] [int] $GatewayPort = 8091,
    [ValidateSet("cuda", "cpu")] [string] $Backend = "cuda",
    [ValidateRange(4096, 1048576)] [int] $Context = 131072,
    [string] $InstallRoot = "$env:LOCALAPPDATA\Turbofit",
    [switch] $Uninstall
)

$ErrorActionPreference = "Stop"
$TaskName = "TurbofitRuntime"
$GatewayTaskName = "TurbofitGateway"

if ($Uninstall) {
    Unregister-ScheduledTask -TaskName $TaskName -Confirm:$false -ErrorAction SilentlyContinue
    Unregister-ScheduledTask -TaskName $GatewayTaskName -Confirm:$false -ErrorAction SilentlyContinue
    Write-Host "Removed $TaskName and $GatewayTaskName"
    exit 0
}

$Python = (Get-Command python.exe -ErrorAction SilentlyContinue).Source
if (-not $Python) { $Python = (Get-Command python -ErrorAction SilentlyContinue).Source }
if (-not $Python) { throw "Python 3 is required for the runtime and gateway" }
if (-not $Binary) {
    $Installer = Join-Path $PSScriptRoot "install-dspark-runtime"
    & $Python $Installer install --backend $Backend
    if ($LASTEXITCODE -ne 0) { throw "Pinned llama.cpp runtime installation failed" }
    $Binary = (& $Python $Installer print-binary --backend $Backend).Trim()
}
if (-not (Test-Path -LiteralPath $Binary -PathType Leaf)) {
    throw "Pinned llama-server.exe is missing: $Binary"
}
if (-not $Model) {
    $ModelRoot = if ($env:TURBOFIT_MODEL_ROOT) { $env:TURBOFIT_MODEL_ROOT } else { Join-Path $env:USERPROFILE "Models\storage\gguf" }
    $Model = Join-Path $ModelRoot "Bonsai-27B\Bonsai-27B-Q1_0.gguf"
}
if (-not (Test-Path -LiteralPath $Model -PathType Leaf)) {
    throw "Model is missing; run scripts/download-artifacts --family bonsai-27b first: $Model"
}

New-Item -ItemType Directory -Force -Path $InstallRoot | Out-Null
$ConfigPath = Join-Path $InstallRoot "runtime.json"
$LauncherPath = Join-Path $InstallRoot "launch-runtime.ps1"
$GatewayLauncherPath = Join-Path $InstallRoot "launch-gateway.ps1"
$RoutePath = Join-Path $InstallRoot "runtime-state.json"
$LogPath = Join-Path $InstallRoot "runtime.log"
$GatewayLogPath = Join-Path $InstallRoot "gateway.log"

$config = [ordered]@{
    binary = (Resolve-Path -LiteralPath $Binary).Path
    model = (Resolve-Path -LiteralPath $Model).Path
    port = $Port
    context = $Context
    backend = $Backend
    log = $LogPath
}
$config | ConvertTo-Json | Set-Content -LiteralPath $ConfigPath -Encoding UTF8

$launcher = @'
$ErrorActionPreference = "Stop"
$config = Get-Content -Raw -LiteralPath "__CONFIG__" | ConvertFrom-Json
$arguments = @(
    "--model", $config.model,
    "--host", "127.0.0.1",
    "--port", [string]$config.port,
    "--alias", "bonsai-27b-1bit-128k-main",
    "--ctx-size", [string]$config.context,
    "--gpu-layers", $(if ($config.backend -eq "cpu") { "0" } else { "auto" }),
    "--fit", "on",
    "--flash-attn", "auto",
    "--jinja",
    "--metrics"
)
& $config.binary @arguments *>> $config.log
exit $LASTEXITCODE
'@
$launcher = $launcher.Replace("__CONFIG__", $ConfigPath.Replace("'", "''"))
Set-Content -LiteralPath $LauncherPath -Value $launcher -Encoding UTF8

$routes = [ordered]@{
    schema = "turbofit.runtime-routes/v1"
    active = "hardware-windows-native"
    rung_id = "local-bonsai-131072"
    rung_index = 0
    routes = [ordered]@{
        main = [ordered]@{ kind = "local"; alias = "bonsai-27b-1bit-128k-main"; port = $Port }
        aux = [ordered]@{ kind = "shared-main" }
    }
}
$routes | ConvertTo-Json -Depth 6 | Set-Content -LiteralPath $RoutePath -Encoding UTF8

$GatewayScript = Join-Path $PSScriptRoot "turbofit-gateway.py"
$gatewayLauncher = @'
$ErrorActionPreference = "Stop"
$env:TURBOFIT_RUNTIME_STATE = "__ROUTES__"
$env:TURBOFIT_GATEWAY_PORT = "__PORT__"
$env:TURBOFIT_ALLOW_API = "0"
& "__PYTHON__" "__GATEWAY__" *>> "__LOG__"
exit $LASTEXITCODE
'@
$gatewayLauncher = $gatewayLauncher.Replace("__ROUTES__", $RoutePath).Replace("__PORT__", [string]$GatewayPort).Replace("__PYTHON__", $Python).Replace("__GATEWAY__", $GatewayScript).Replace("__LOG__", $GatewayLogPath)
Set-Content -LiteralPath $GatewayLauncherPath -Value $gatewayLauncher -Encoding UTF8

$PowerShell = (Get-Command powershell.exe).Source
$Action = New-ScheduledTaskAction -Execute $PowerShell -Argument "-NoProfile -ExecutionPolicy Bypass -File `"$LauncherPath`""
$Trigger = New-ScheduledTaskTrigger -AtLogOn -User $env:USERNAME
$Settings = New-ScheduledTaskSettingsSet -RestartCount 5 -RestartInterval (New-TimeSpan -Minutes 1) -ExecutionTimeLimit (New-TimeSpan -Days 3650)
$Principal = New-ScheduledTaskPrincipal -UserId "$env:USERDOMAIN\$env:USERNAME" -LogonType Interactive -RunLevel Limited
Register-ScheduledTask -TaskName $TaskName -Action $Action -Trigger $Trigger -Settings $Settings -Principal $Principal -Force | Out-Null
$GatewayAction = New-ScheduledTaskAction -Execute $PowerShell -Argument "-NoProfile -ExecutionPolicy Bypass -File `"$GatewayLauncherPath`""
Register-ScheduledTask -TaskName $GatewayTaskName -Action $GatewayAction -Trigger $Trigger -Settings $Settings -Principal $Principal -Force | Out-Null
Start-ScheduledTask -TaskName $TaskName
Start-ScheduledTask -TaskName $GatewayTaskName

$Deadline = (Get-Date).AddMinutes(10)
do {
    try {
        $Health = Invoke-RestMethod -Uri "http://127.0.0.1:$Port/health" -TimeoutSec 3
        if ($Health.status -eq "ok") {
            $GatewayHealth = Invoke-RestMethod -Uri "http://127.0.0.1:$GatewayPort/health" -TimeoutSec 3
            if ($GatewayHealth.status -eq "ok") {
                Write-Host "Turbofit Windows provider verified on http://127.0.0.1:$GatewayPort/v1"
                exit 0
            }
        }
    } catch {
    }
    Start-Sleep -Seconds 2
} while ((Get-Date) -lt $Deadline)

throw "Turbofit runtime did not become healthy; inspect $LogPath"
