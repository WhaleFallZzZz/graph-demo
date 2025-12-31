#!/bin/bash

# Exit on error
set -e

# Load environment variables from .env if it exists
if [ -f .env ]; then
    export $(cat .env | grep -v '#' | awk '/=/ {print $1}')
fi

# Check for required environment variables
required_vars=("SILICONFLOW_API_KEY" "NEO4J_PASSWORD" "COS_SECRET_ID" "COS_SECRET_KEY")
missing_vars=()

for var in "${required_vars[@]}"; do
    if [ -z "${!var}" ]; then
        missing_vars+=("$var")
    fi
done

DAEMONIZE_GUNICORN=""

# Parse arguments
while getopts "d" opt; do
  case $opt in
    d)
      DAEMONIZE_GUNICORN="--daemon"
      ;;
    \?)
      echo "Usage: $0 [-d]"
      exit 1
      ;;
  esac
done
shift $((OPTIND-1)) # Remove parsed options from arguments list

echo "Checking server configuration..."

if command -v gunicorn &> /dev/null; then
    echo "Starting server with Gunicorn..."
    gunicorn --workers 1 --bind 0.0.0.0:8001 --timeout 0 --access-logfile - --error-logfile - $DAEMONIZE_GUNICORN llama.server:app
elif command -v waitress-serve &> /dev/null; then
    echo "Starting server with Waitress (Windows)..."
    # Waitress does not have a direct daemonize flag.
    # For background, user would need to use shell-specific commands like 'start /B ...' or '&'
    waitress-serve --port=8001 --call llama.server:app
else
    echo "Warning: No production server (gunicorn/waitress) found."
    echo "Falling back to Flask development server..."
    # For Flask dev server, use '&' for background execution
    python -m llama.server &
    echo "Flask development server running in background. PID: $!"
fi
