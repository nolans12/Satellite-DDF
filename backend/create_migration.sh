#!/bin/bash

set -e

function usage() {
    # Display usage information
    echo "Usage: $0 <migration message>"
    echo "Create a new Alembic migration with the given message."
    exit 1
}

# Print usage if no arguments are provided
if [ $# -eq 0 ]; then
    usage
fi

venv/bin/alembic revision --autogenerate -m "$1"
