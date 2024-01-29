#!/bin/bash

# Install Python
apt-get -y update
apt-get install -y python3-pip python3-venv

# Create empty virtual environment
python3 -m venv titanic
source titanic/bin/activate

# Install project dependencies
pip install -r requirements.txt