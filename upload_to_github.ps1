<#
一键上传到 GitHub。

用法（在项目根目录执行）：
    powershell -ExecutionPolicy Bypass -File .\upload_to_github.ps1
    powershell -ExecutionPolicy Bypass -File .\upload_to_github.ps1 -Message "修复裁边补偿"

行为：
  1. 没有 git 仓库就自动 git init 并切到 main 分支
  2. 自动配置/更新 origin 远端
  3. git add -A 之后逐个检查暂存文件大小，超过 1MB 的自动移出暂存区（不会上传）
  4. 提交并 push

不会做的事：不会 force push、不会改 git 全局配置、不会删除任何本地文件。
#>
[CmdletBinding()]
param(
    [string]$Remote = "https://github.com/baideji521/c.git",
    [string]$Branch = "main",
    [string]$Message = "",
    [int]$MaxFileKB = 1024
)

# git 会把普通提示写到 stderr，这里不能用 Stop，否则会被当成终止错误
$ErrorActionPreference = "Continue"
Set-Location -Path $PSScriptRoot

function Info($m) { Write-Host "[INFO] $m" -ForegroundColor Cyan }
function Warn($m) { Write-Host "[WARN] $m" -ForegroundColor Yellow }
function Fail($m) { Write-Host "[FAIL] $m" -ForegroundColor Red; exit 1 }

# 统一封装 git 调用：合并 stderr，返回 (输出, 退出码)
function Invoke-Git {
    param([Parameter(ValueFromRemainingArguments = $true)][string[]]$GitArgs)
    $out = & git @GitArgs 2>&1 | Out-String
    return [pscustomobject]@{ Out = $out.Trim(); Code = $LASTEXITCODE }
}

# ---------- 0. 环境检查 ----------
if (-not (Get-Command git -ErrorAction SilentlyContinue)) {
    Fail "未找到 git，请先安装 Git for Windows：https://git-scm.com/download/win"
}
if (-not (Test-Path ".gitignore")) {
    Warn ".gitignore 不存在，大文件可能被误提交，建议先创建"
}

# ---------- 1. 初始化仓库 ----------
if (-not (Test-Path ".git")) {
    Info "初始化 git 仓库..."
    (Invoke-Git init -q) | Out-Null
    (Invoke-Git symbolic-ref HEAD "refs/heads/$Branch") | Out-Null
} else {
    $cur = (Invoke-Git rev-parse --abbrev-ref HEAD).Out
    if ($cur -and $cur -ne $Branch -and $cur -ne "HEAD") {
        Info "当前分支 $cur，切换到 $Branch"
        $sw = Invoke-Git switch -c $Branch
        if ($sw.Code -ne 0) { (Invoke-Git switch $Branch) | Out-Null }
    }
}

# ---------- 2. 配置远端 ----------
$remotes = (Invoke-Git remote).Out -split "\r?\n" | ForEach-Object { $_.Trim() }
if ($remotes -contains "origin") {
    $existing = (Invoke-Git remote get-url origin).Out
    if ($existing -ne $Remote) {
        Info "更新远端 origin：$existing -> $Remote"
        (Invoke-Git remote set-url origin $Remote) | Out-Null
    }
} else {
    Info "添加远端 origin -> $Remote"
    (Invoke-Git remote add origin $Remote) | Out-Null
}

# 本仓库没有提交者身份时给个兜底（不动全局配置）
if (-not (Invoke-Git config user.name).Out) {
    (Invoke-Git config user.name "baideji521") | Out-Null
}
if (-not (Invoke-Git config user.email).Out) {
    (Invoke-Git config user.email "baideji521@users.noreply.github.com") | Out-Null
}

# ---------- 3. 暂存 ----------
Info "扫描改动..."
(Invoke-Git add -A) | Out-Null
$staged = @((Invoke-Git diff --cached --name-only --diff-filter=ACMR).Out -split "\r?\n" |
    Where-Object { $_ -and $_.Trim() })
if ($staged.Count -eq 0) {
    Info "没有需要提交的改动，退出。"
    exit 0
}

# ---------- 4. 大文件拦截 ----------
$limit = $MaxFileKB * 1KB
$tooBig = @()
foreach ($rel in $staged) {
    $p = Join-Path $PSScriptRoot $rel
    if (Test-Path -LiteralPath $p) {
        $len = (Get-Item -LiteralPath $p).Length
        if ($len -gt $limit) {
            $tooBig += [pscustomobject]@{ Path = $rel; MB = [math]::Round($len / 1MB, 2) }
        }
    }
}
if ($tooBig.Count -gt 0) {
    Warn "以下文件超过 $MaxFileKB KB，已移出暂存区（不会上传）:"
    foreach ($f in $tooBig) {
        Write-Host ("       {0,8:N2} MB  {1}" -f $f.MB, $f.Path) -ForegroundColor Yellow
        (Invoke-Git restore --staged -- $f.Path) | Out-Null
    }
    Warn "如需长期忽略，请把它们加进 .gitignore"
    $staged = @((Invoke-Git diff --cached --name-only --diff-filter=ACMR).Out -split "\r?\n" |
        Where-Object { $_ -and $_.Trim() })
    if ($staged.Count -eq 0) {
        Info "剔除大文件后没有可提交内容，退出。"
        exit 0
    }
}

$total = 0
foreach ($rel in $staged) {
    $p = Join-Path $PSScriptRoot $rel
    if (Test-Path -LiteralPath $p) { $total += (Get-Item -LiteralPath $p).Length }
}
Info ("待提交 {0} 个文件，共 {1:N1} KB" -f $staged.Count, ($total / 1KB))

# ---------- 5. 提交 ----------
if (-not $Message) {
    $Message = "chore: sync " + (Get-Date -Format "yyyy-MM-dd HH:mm")
}
$commit = Invoke-Git commit -q -m $Message
if ($commit.Code -ne 0) { Fail "提交失败：$($commit.Out)" }
Info "已提交：$Message"

# ---------- 6. 推送 ----------
Info "推送到 $Remote ($Branch) ..."
$push = Invoke-Git push -u origin $Branch
Write-Host $push.Out
if ($push.Code -ne 0) {
    Warn "推送失败。常见原因:"
    Warn "  1) 没有登录凭据: 用 Personal Access Token 当密码，或先执行 gh auth login"
    Warn "  2) 远端已有提交且历史不同: 先执行 git pull --rebase origin $Branch 再重跑本脚本"
    exit 1
}
Info "上传完成: https://github.com/baideji521/c"
