#!/bin/bash

# Load environment variables from .env if it exists
if [ -f .env ]; then
    echo "Loading .env..."
    export $(cat .env | grep -v '#' | awk '/=/ {print $1}')
fi

# Check if SILICONFLOW_API_KEY is set
if [ -z "$SILICONFLOW_API_KEY" ]; then
    echo "Error: SILICONFLOW_API_KEY is not set."
    echo "Please set it in .env or export it."
    exit 1
fi

echo "Running debug_kg_build.py..."
python3 debug_kg_build.py
