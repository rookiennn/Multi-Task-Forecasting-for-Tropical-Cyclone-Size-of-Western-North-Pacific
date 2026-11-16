<#
.SYNOPSIS
Commit this project and push it to an existing GitHub repository.
.DESCRIPTION
Run from any directory. Uses Git credentials already available on this machine.
Respects .gitignore and never force-pushes or replaces an existing origin.
Use -DryRun to preview the steps without changing files or contacting a remote.
.EXAMPLE
.\upload_github.ps1 -RepositoryUrl https://github.com/YOUR_USERNAME/TCSF.git -DryRun
.EXAMPLE
.\upload_github.ps1 -RepositoryUrl https://github.com/YOUR_USERNAME/TCSF.git
.EXAMPLE
.\upload_github.ps1 -RepositoryUrl git@github.com:YOUR_USERNAME/TCSF.git -CommitMessage "Update project"
#>
[CmdletBinding()]
param(
    [Parameter(Mandatory = $true)]
    [ValidateNotNullOrEmpty()]
    [string]$RepositoryUrl,

    [ValidateNotNullOrEmpty()]
    [string]$Branch = 'main',

    [ValidateNotNullOrEmpty()]
    [string]$CommitMessage = 'Upload TCSF project',

    [ValidateNotNullOrEmpty()]
    [string]$ProjectPath,

    [switch]$DryRun
)

Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'

function Invoke-ProjectGit {
    param(
        [Parameter(Mandatory = $true)]
        [string[]]$GitArguments,
        [switch]$AllowFailure,
        [switch]$ShowOutput
    )

    # Inspect exit codes explicitly, including commands that legitimately fail
    # when no repository, remote, or first commit exists yet.
    $ErrorActionPreference = 'Continue'
    $PSNativeCommandUseErrorActionPreference = $false
    $lines = @(& $script:GitExecutable -C $script:ProjectDirectory @GitArguments 2>&1)
    $gitExitCode = $LASTEXITCODE
    $message = ($lines | ForEach-Object { $_.ToString() }) -join [Environment]::NewLine
    if ($ShowOutput -and $message) {
        Write-Host $message
    }
    if ($gitExitCode -ne 0 -and -not $AllowFailure) {
        throw "Git command failed (exit $gitExitCode): git $($GitArguments -join ' ')`n$message"
    }
    return [pscustomobject]@{ ExitCode = $gitExitCode; Text = $message }
}

function Get-ComparableRemote {
    param([string]$Address)
    $value = $Address.Trim().TrimEnd('/') -replace '\.git$', ''
    # Treat GitHub HTTPS and SSH addresses for the same repository as equal.
    $value = $value -replace '^https://github\.com/', 'github.com/'
    $value = $value -replace '^git@github\.com:', 'github.com/'
    $value = $value -replace '^ssh://git@github\.com/', 'github.com/'
    return $value
}

try {
    if (-not $PSBoundParameters.ContainsKey('ProjectPath')) {
        $ProjectPath = $PSScriptRoot
    }
    $gitCommand = Get-Command git -CommandType Application -ErrorAction SilentlyContinue | Select-Object -First 1
    if (-not $gitCommand) {
        throw 'Git is required. Install Git and open a new PowerShell session.'
    }
    $script:GitExecutable = $gitCommand.Source
    $projectItem = Get-Item -LiteralPath $ProjectPath
    if (-not $projectItem.PSIsContainer) {
        throw 'ProjectPath must point to a project directory.'
    }
    $script:ProjectDirectory = $projectItem.FullName
    $RepositoryUrl = $RepositoryUrl.Trim()
    if (-not $RepositoryUrl -or $RepositoryUrl.StartsWith('-')) {
        throw 'RepositoryUrl must be a Git repository address.'
    }
    if (-not (Test-Path -LiteralPath (Join-Path $script:ProjectDirectory '.gitignore') -PathType Leaf)) {
        throw 'The project needs a .gitignore file before it can be uploaded.'
    }

    $null = Invoke-ProjectGit -GitArguments @('check-ref-format', '--branch', $Branch)
    $root = Invoke-ProjectGit -GitArguments @('rev-parse', '--show-toplevel') -AllowFailure
    $hasRepository = $root.ExitCode -eq 0
    $hasOrigin = $false
    if ($hasRepository) {
        $existingRoot = [IO.Path]::GetFullPath($root.Text.Trim()).TrimEnd([char[]]'\/')
        $expectedRoot = [IO.Path]::GetFullPath($script:ProjectDirectory).TrimEnd([char[]]'\/')
        if ($existingRoot -ne $expectedRoot) {
            throw "ProjectPath is inside a different Git repository: $existingRoot. Select the repository root."
        }
        $currentBranch = Invoke-ProjectGit -GitArguments @('symbolic-ref', '--quiet', '--short', 'HEAD') -AllowFailure
        if ($currentBranch.ExitCode -ne 0) {
            throw 'The repository has a detached HEAD. Check out a local branch before uploading.'
        }
        $conflicts = Invoke-ProjectGit -GitArguments @('ls-files', '--unmerged')
        if ($conflicts.Text) {
            throw 'Resolve the existing merge conflicts before uploading.'
        }
        $origin = Invoke-ProjectGit -GitArguments @('remote', 'get-url', 'origin') -AllowFailure
        $hasOrigin = $origin.ExitCode -eq 0
        if ($hasOrigin) {
            $pushOrigin = Invoke-ProjectGit -GitArguments @('remote', 'get-url', '--push', '--all', 'origin')
            $pushAddresses = @($pushOrigin.Text -split '\r?\n' | Where-Object { $_.Trim() })
            $expectedRemote = Get-ComparableRemote $RepositoryUrl
            if ($pushAddresses.Count -ne 1 -or
                (Get-ComparableRemote $origin.Text) -ne $expectedRemote -or
                (Get-ComparableRemote $pushAddresses[0]) -ne $expectedRemote) {
                throw 'Existing origin does not match RepositoryUrl, or has multiple push addresses. Check git remote -v before retrying.'
            }
        }
    }

    Write-Host "Project: $script:ProjectDirectory"
    Write-Host "Remote:  $RepositoryUrl"
    Write-Host "Target:  $Branch"
    if ($DryRun) {
        Write-Host ''
        Write-Host 'Dry run: no files, Git history, or remote repositories will be changed.'
        if (-not $hasRepository) { Write-Host "1. Initialize a Git repository with branch '$Branch'." }
        if (-not $hasOrigin) { Write-Host '2. Add the supplied address as origin.' }
        Write-Host '3. Stage project changes using .gitignore.'
        Write-Host "4. Commit staged changes with message: $CommitMessage"
        Write-Host "5. Push the current commit to origin/$Branch without force."
        Write-Host 'The real run also requires Git author details and permission to push to the remote.'
        exit 0
    }

    $identity = Invoke-ProjectGit -GitArguments @('var', 'GIT_AUTHOR_IDENT') -AllowFailure
    if ($identity.ExitCode -ne 0) {
        throw 'Git author details are missing. Set git config --global user.name "Your Name" and git config --global user.email "your-email@example.com", then retry.'
    }
    if (-not $hasRepository) {
        $null = Invoke-ProjectGit -GitArguments @('init', "--initial-branch=$Branch") -ShowOutput
    }
    if (-not $hasOrigin) {
        $null = Invoke-ProjectGit -GitArguments @('remote', 'add', 'origin', $RepositoryUrl)
    }
    $null = Invoke-ProjectGit -GitArguments @('add', '--all')
    $staged = Invoke-ProjectGit -GitArguments @('diff', '--cached', '--name-only')
    if ($staged.Text) {
        Write-Host "Files staged for commit:`n$($staged.Text)"
        $null = Invoke-ProjectGit -GitArguments @('commit', '-m', $CommitMessage) -ShowOutput
    }
    else {
        Write-Host 'No new changes to commit.'
    }
    $head = Invoke-ProjectGit -GitArguments @('rev-parse', '--verify', 'HEAD') -AllowFailure
    if ($head.ExitCode -ne 0) {
        throw 'There is no commit to push. Check which project files are excluded by .gitignore.'
    }
    $null = Invoke-ProjectGit -GitArguments @('push', '--set-upstream', 'origin', "HEAD:refs/heads/$Branch") -ShowOutput
    Write-Host "Upload complete: $RepositoryUrl (branch: $Branch)."
}
catch {
    [Console]::Error.WriteLine("Upload failed: $($_.Exception.Message)")
    [Console]::Error.WriteLine('No force push was attempted. Local commits and staged files are kept for review.')
    exit 1
}
