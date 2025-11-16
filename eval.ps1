$VENV_DIR = ".venv"
$REQUIREMENTS_FILE = "requirements.txt"

# Check if .venv exists
if (Test-Path $VENV_DIR) {
    Write-Host "Activating existing virtual environment..."
} else {
    Write-Host "Creating a new virtual environment..."
    python -m venv $VENV_DIR
}

# Activate the virtual environment
& (Join-Path $VENV_DIR "Scripts\Activate.ps1")

# Install dependencies in quiet mode
if (Test-Path $REQUIREMENTS_FILE) {
    Write-Host "Installing dependencies from $REQUIREMENTS_FILE..."
    pip install -q -r $REQUIREMENTS_FILE
} else {
    Write-Host "Warning: $REQUIREMENTS_FILE not found, skipping installation."
}