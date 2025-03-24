#!/bin/bash
set -e

# Install rclone if not already installed
if ! command -v rclone &> /dev/null
then
    echo "rclone not found. Installing..."
    curl https://rclone.org/install.sh | sudo bash
fi

# Check if rclone is configured
rclone listremotes | grep mygdrive &> /dev/null
if [ $? -ne 0 ]; then
    echo "Remote 'mygdrive' not found in rclone config. Please run:"
    echo "    rclone config"
    echo "to set up your Google Drive. Then re-run this script."
    exit 1
fi

# Copy the folder from Google Drive to local
rclone copy mygdrive:"MyDatasets/Pokemon" /workspace/dataset -v

echo "Copied dataset from Google Drive to /workspace/dataset"
