#!/bin/bash
# Script to set up your fork of cs5342-fall2025

echo "Setting up fork configuration..."
echo ""

# Check if fork exists
FORK_URL="https://github.com/JoyJEZhang/cs5342-fall2025.git"
echo "Checking if fork exists at $FORK_URL..."

if git ls-remote "$FORK_URL" &> /dev/null; then
    echo "✓ Fork found!"
    echo ""
    
    # Rename current origin to upstream
    echo "Renaming 'origin' to 'upstream' (original repository)..."
    git remote rename origin upstream
    
    # Add your fork as origin
    echo "Adding your fork as 'origin'..."
    git remote add origin "$FORK_URL"
    
    # Verify setup
    echo ""
    echo "✓ Fork setup complete!"
    echo ""
    echo "Current remotes:"
    git remote -v
    echo ""
    echo "To push to your fork, use: git push origin main"
    echo "To pull updates from original, use: git pull upstream main"
else
    echo "✗ Fork not found yet."
    echo ""
    echo "Please create the fork first:"
    echo "1. Go to https://github.com/tomrist/cs5342-fall2025"
    echo "2. Click the 'Fork' button (top right)"
    echo "3. Select your account (JoyJEZhang)"
    echo "4. Then run this script again: bash setup-fork.sh"
fi

