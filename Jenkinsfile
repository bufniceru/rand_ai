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
                    bat 'uv sync --locked'
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
    }
}
