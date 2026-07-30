pipeline {
    agent any

    options {
        skipDefaultCheckout(true)
    }

    stages {
        stage('Checkout') {
            steps {
                checkout scm
            }
        }

        stage('Display project version') {
            steps {
                script {
                    def projectVersion = null
                    def insideProjectSection = false

                    for (String rawLine : readFile(file: 'pyproject.toml').readLines()) {
                        def line = rawLine.trim()

                        if (line == '[project]') {
                            insideProjectSection = true
                            continue
                        }

                        if (insideProjectSection && line.startsWith('[')) {
                            break
                        }

                        def equalsIndex = line.indexOf('=')
                        if (insideProjectSection && equalsIndex > 0) {
                            def key = line.substring(0, equalsIndex).trim()
                            def value = line.substring(equalsIndex + 1).trim()

                            if (key == 'version' && value.length() >= 2) {
                                def quote = value.substring(0, 1)
                                if ((quote == '"' || quote == "'") && value.endsWith(quote)) {
                                    projectVersion = value.substring(1, value.length() - 1)
                                    break
                                }
                            }
                        }
                    }

                    if (!projectVersion) {
                        error('Could not find project.version in pyproject.toml')
                    }

                    echo "Project version from pyproject.toml: ${projectVersion}"
                }
            }
        }
    }
}
