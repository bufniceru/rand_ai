pipeline {
    agent any

    parameters {
        gitParameter(
            name: 'SOURCE_BRANCH',
            type: 'PT_BRANCH',
            branchFilter: 'origin/(.*)',
            defaultValue: 'master',
            selectedValue: 'DEFAULT',
            sortMode: 'ASCENDING_SMART',
            quickFilterEnabled: true,
            description: 'Select the remote Git branch to build'
        )
    }

    options {
        skipDefaultCheckout(true)
        disableConcurrentBuilds()
    }

    environment {
        UV_INSTALL_DIR = 'C:\\ProgramData\\Jenkins\\.jenkins\\tools\\uv'
        UV_NO_MODIFY_PATH = '1'
        UV_PYTHON_INSTALL_DIR = 'C:\\ProgramData\\Jenkins\\.jenkins\\tools\\uv-python'
        UV_CACHE_DIR = 'C:\\ProgramData\\Jenkins\\.jenkins\\cache\\uv'
    }

    tools {
        nodejs 'NodeJS-26'
    }

    stages {
        stage('Check Internet Access') {
            steps {
                powershell '''
                    $ErrorActionPreference = 'Stop'

                    $addresses = [System.Net.Dns]::GetHostAddresses('astral.sh')
                    Write-Host "DNS resolution succeeded:"
                    $addresses | ForEach-Object { Write-Host "  $_" }

                    $connection = Test-NetConnection `
                        -ComputerName astral.sh `
                        -Port 443 `
                        -InformationLevel Detailed

                    if (-not $connection.TcpTestSucceeded) {
                        throw 'Cannot connect to astral.sh on TCP port 443'
                    }

                    Write-Host 'TCP connection to astral.sh:443 succeeded'

                    $response = Invoke-WebRequest `
                        -Uri 'https://astral.sh/uv/install.ps1' `
                        -UseBasicParsing `
                        -TimeoutSec 30

                    if ($response.StatusCode -ne 200) {
                        throw "Astral returned HTTP status $($response.StatusCode)"
                    }

                    Write-Host "Astral HTTPS request succeeded: HTTP $($response.StatusCode)"

                    $githubResponse = Invoke-WebRequest `
                        -Uri 'https://github.com' `
                        -UseBasicParsing `
                        -TimeoutSec 30

                    Write-Host "GitHub HTTPS request succeeded: HTTP $($githubResponse.StatusCode)"
                '''
            }
        }
        stage('Install uv') {
            steps {
                powershell '''
                    $ErrorActionPreference = 'Stop'
                    $uvExecutable = Join-Path $env:UV_INSTALL_DIR 'uv.exe'

                    if (-not (Test-Path -LiteralPath $uvExecutable)) {
                        Write-Host 'Downloading uv from Astral...'

                        New-Item `
                            -ItemType Directory `
                            -Force `
                            -Path $env:UV_INSTALL_DIR | Out-Null

                        Invoke-RestMethod https://astral.sh/uv/install.ps1 |
                            Invoke-Expression
                    }
                    else {
                        Write-Host "Using existing uv installation: $uvExecutable"
                    }

                    & $uvExecutable --version
                '''
            }
        }
        stage('Verify Tools') {
            steps {
                withEnv(["PATH+UV=${env.UV_INSTALL_DIR}"]) {
                    bat 'whoami'
                    bat 'where git'
                    bat 'git --version'
                    bat 'where uv'
                    bat 'uv --version'
                }
            }
        }
        stage('Checkout') {
            steps {
                script {
                    def requestedBranch = params.SOURCE_BRANCH
                    def sourceBranch = requestedBranch == null ? 'master' : requestedBranch.trim()
                    if (!sourceBranch) {
                        error('SOURCE_BRANCH must not be empty')
                    }

                    echo "Checking out remote branch: ${sourceBranch}"
                    deleteDir()

                    def checkoutResult = checkout scmGit(
                        branches: [[name: "*/${sourceBranch}"]],
                        userRemoteConfigs: [[
                            credentialsId: 'github-rand-ai-ssh',
                            url: 'git@github.com:bufniceru/rand_ai.git'
                        ]]
                    )

                    echo "Building ${checkoutResult.GIT_BRANCH} at ${checkoutResult.GIT_COMMIT}"
                }
            }
        }
        stage('Display project version') {
            steps {
                withEnv(["PATH+UV=${env.UV_INSTALL_DIR}"]) {
                    script {
                        def projectVersion = bat(
                            script: '@uv version --short',
                            returnStdout: true
                        ).trim()

                        echo "Project version: ${projectVersion}"
                    }
                }
            }
        }
        stage('Sync Python environment') {
            steps {
                withEnv(["PATH+UV=${env.UV_INSTALL_DIR}"]) {
                    bat 'uv sync --locked --group docs'
                }
            }
        }
        stage('Build application documentation') {
            steps {
                withEnv(["PATH+UV=${env.UV_INSTALL_DIR}"]) {
                    bat 'uv run --group docs sphinx-build -W --keep-going -b html docs/app docs/_build/html'
                }
            }
        }
        stage('Verify Node') {
            steps {
                bat 'node --version'
                bat 'npm --version'
            }
        }

        stage('Install dependencies') {
            steps {
                dir('web') {
                    bat 'npm ci'
                }
            }
        }
        stage('Build portable Electron executable') {
            steps {
                dir('web') {
                    bat 'npm run electron:build'
                }

            }
        }
        stage('ArchiveArtifacts') {
            steps {
                archiveArtifacts(
                    artifacts: 'web/electron-package/*.exe',
                    fingerprint: true,
                    onlyIfSuccessful: true
                )
            }
        }
        stage('Publish GitHub Pages') {
            when {
                expression {
                    def selectedBranch = (params.SOURCE_BRANCH ?: 'master').trim()
                    return selectedBranch.replaceFirst('^origin/', '') == 'master'
                }
            }
            steps {
                sshagent(credentials: ['github-rand-ai-ssh']) {
                    powershell '''
                        $ErrorActionPreference = 'Stop'

                        function Invoke-GitCommand {
                            param([string[]] $Arguments)

                            & git @Arguments
                            if ($LASTEXITCODE -ne 0) {
                                throw "git $($Arguments -join ' ') failed with exit code $LASTEXITCODE"
                            }
                        }

                        $repositoryUrl = 'git@github.com:bufniceru/rand_ai.git'
                        $pagesBranch = 'gh-pages'
                        $workspaceRoot = [System.IO.Path]::GetFullPath($env:WORKSPACE).TrimEnd('\\')
                        $workspacePrefix = $workspaceRoot + [System.IO.Path]::DirectorySeparatorChar
                        $publishRoot = [System.IO.Path]::GetFullPath(
                            (Join-Path $workspaceRoot '.jenkins-pages')
                        )

                        if (-not $publishRoot.StartsWith(
                            $workspacePrefix,
                            [System.StringComparison]::OrdinalIgnoreCase
                        )) {
                            throw "Pages staging path is outside the Jenkins workspace: $publishRoot"
                        }

                        $htmlRoot = (Resolve-Path -LiteralPath (
                            Join-Path $workspaceRoot 'docs\\_build\\html'
                        )).Path
                        $htmlIndex = Join-Path $htmlRoot 'index.html'
                        if (-not (Test-Path -LiteralPath $htmlIndex -PathType Leaf)) {
                            throw "Sphinx output is missing its index: $htmlIndex"
                        }

                        $sourceCommitOutput = & git -C $workspaceRoot rev-parse HEAD
                        if ($LASTEXITCODE -ne 0) {
                            throw 'Cannot resolve the source commit for the Pages publication'
                        }
                        $sourceCommit = ($sourceCommitOutput | Select-Object -First 1).Trim()

                        try {
                            if (Test-Path -LiteralPath $publishRoot) {
                                Remove-Item -LiteralPath $publishRoot -Recurse -Force
                            }

                            & git ls-remote `
                                --exit-code `
                                --heads `
                                $repositoryUrl `
                                "refs/heads/$pagesBranch" | Out-Null
                            $remoteBranchStatus = $LASTEXITCODE

                            if ($remoteBranchStatus -eq 0) {
                                Invoke-GitCommand @(
                                    'clone',
                                    '--depth', '1',
                                    '--branch', $pagesBranch,
                                    '--single-branch',
                                    $repositoryUrl,
                                    $publishRoot
                                )
                            }
                            elseif ($remoteBranchStatus -eq 2) {
                                New-Item `
                                    -ItemType Directory `
                                    -Path $publishRoot `
                                    -Force | Out-Null
                                Invoke-GitCommand @('-C', $publishRoot, 'init')
                                Invoke-GitCommand @(
                                    '-C', $publishRoot,
                                    'remote', 'add', 'origin', $repositoryUrl
                                )
                                Invoke-GitCommand @(
                                    '-C', $publishRoot,
                                    'checkout', '--orphan', $pagesBranch
                                )
                            }
                            else {
                                throw "Cannot inspect remote $pagesBranch branch (git exit $remoteBranchStatus)"
                            }

                            Get-ChildItem -LiteralPath $publishRoot -Force |
                                Where-Object { $_.Name -ne '.git' } |
                                ForEach-Object {
                                    Remove-Item `
                                        -LiteralPath $_.FullName `
                                        -Recurse `
                                        -Force
                                }

                            Get-ChildItem -LiteralPath $htmlRoot -Force |
                                ForEach-Object {
                                    Copy-Item `
                                        -LiteralPath $_.FullName `
                                        -Destination $publishRoot `
                                        -Recurse `
                                        -Force
                                }

                            New-Item `
                                -ItemType File `
                                -Path (Join-Path $publishRoot '.nojekyll') `
                                -Force | Out-Null

                            if (-not (Test-Path -LiteralPath (
                                Join-Path $publishRoot 'index.html'
                            ) -PathType Leaf)) {
                                throw 'Published Pages tree does not contain index.html'
                            }

                            Invoke-GitCommand @(
                                '-C', $publishRoot,
                                'config', 'user.name', 'Rand AI Jenkins'
                            )
                            Invoke-GitCommand @(
                                '-C', $publishRoot,
                                'config', 'user.email', 'jenkins@rand-ai.local'
                            )
                            Invoke-GitCommand @('-C', $publishRoot, 'add', '-A')

                            & git -C $publishRoot diff --cached --quiet
                            $diffStatus = $LASTEXITCODE
                            if ($diffStatus -eq 0) {
                                Write-Host 'GitHub Pages content is unchanged; skipping commit and push.'
                            }
                            elseif ($diffStatus -eq 1) {
                                $shortCommit = $sourceCommit.Substring(
                                    0,
                                    [Math]::Min(12, $sourceCommit.Length)
                                )
                                $buildIdentity = if (
                                    [string]::IsNullOrWhiteSpace($env:BUILD_TAG)
                                ) {
                                    "Jenkins build $env:BUILD_NUMBER"
                                }
                                else {
                                    $env:BUILD_TAG
                                }
                                $commitMessage = "Publish documentation from $shortCommit ($buildIdentity)"

                                Invoke-GitCommand @(
                                    '-C', $publishRoot,
                                    'commit', '-m', $commitMessage
                                )
                                Invoke-GitCommand @(
                                    '-C', $publishRoot,
                                    'push', 'origin', $pagesBranch
                                )
                                Write-Host 'GitHub Pages documentation published successfully.'
                            }
                            else {
                                throw "Cannot inspect staged Pages changes (git exit $diffStatus)"
                            }
                        }
                        finally {
                            if (Test-Path -LiteralPath $publishRoot) {
                                try {
                                    Remove-Item `
                                        -LiteralPath $publishRoot `
                                        -Recurse `
                                        -Force
                                }
                                catch {
                                    Write-Warning "Could not clean Pages staging directory: $_"
                                }
                            }
                        }
                    '''
                }
            }
        }
    }
}
