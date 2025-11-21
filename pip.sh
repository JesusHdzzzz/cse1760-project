#!/bin/bash

# Script to install necessary Python packages for machine learning project

echo "Installing required Python packages..."

# Update pip to the latest version
pip3 install --upgrade pip

# Install core packages from your import statements
pip3 install scipy
pip3 install numpy
pip3 install scikit-learn

# Additional commonly needed packages for data science/machine learning
pip3 install pandas
pip3 install matplotlib
pip3 install seaborn

# Install Jupyter for interactive development (optional but recommended)
pip3 install jupyter

echo "All packages installed successfully!"
echo ""
echo "Installed packages:"
echo "- scipy"
echo "- numpy" 
echo "- scikit-learn"
echo "- pandas"
echo "- matplotlib"
echo "- seaborn"
echo "- jupyter"