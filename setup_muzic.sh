#!/bin/bash
set -e

# Original setup
echo "Setting up original environment..."
source setup.sh

# Setup Python 3.6 environment for Muzic
echo "Setting up Python 3.6 environment for Microsoft Muzic..."

# Create a directory for Muzic
mkdir -p muzic_env

# Check if pyenv is installed
if ! command -v pyenv &> /dev/null; then
    echo "Installing pyenv..."
    curl https://pyenv.run | bash
    
    # Add pyenv to shell configuration
    echo 'export PYENV_ROOT="$HOME/.pyenv"' >> ~/.bashrc
    echo 'export PATH="$PYENV_ROOT/bin:$PATH"' >> ~/.bashrc
    echo 'eval "$(pyenv init --path)"' >> ~/.bashrc
    echo 'eval "$(pyenv virtualenv-init -)"' >> ~/.bashrc
    
    # Source the updated bashrc
    export PYENV_ROOT="$HOME/.pyenv"
    export PATH="$PYENV_ROOT/bin:$PATH"
    eval "$(pyenv init --path)"
    eval "$(pyenv virtualenv-init -)"
fi

# Install Python 3.6.15
echo "Installing Python 3.6.15..."
pyenv install -s 3.6.15

# Create a dedicated virtual environment
echo "Creating muzic_env virtual environment..."
pyenv virtualenv 3.6.15 muzic_env 2>/dev/null || true
cd muzic_env
pyenv local muzic_env

# Clone Microsoft Muzic repository if not already cloned
if [ ! -d "muzic" ]; then
    echo "Cloning Microsoft Muzic repository..."
    git clone https://github.com/microsoft/muzic.git
    cd muzic
    
    # Install dependencies
    echo "Installing Muzic dependencies..."
    pip install --upgrade pip
    sed -i.bak '/^secrets$/d' requirements.txt
    
    # Install PyTorch 1.7.1
    if [ "$(uname)" == "Darwin" ]; then
        # For macOS
        pip install torch==1.7.1 torchvision==0.8.2 torchaudio==0.7.2
    else
        # For Linux/other
        pip install torch==1.7.1+cpu torchvision==0.8.2+cpu torchaudio==0.7.2 -f https://download.pytorch.org/whl/torch_stable.html
    fi
    
    # Install remaining dependencies
    pip install -r requirements.txt
    cd ..
fi

echo "Muzic environment setup complete!"
cd ..

# Create symbolic links to the main code
echo "Creating symbolic links..."
ln -sf $(pwd)/model.py muzic_env/
ln -sf $(pwd)/generate.py muzic_env/
ln -sf $(pwd)/api.py muzic_env/

echo "Setup complete!"
