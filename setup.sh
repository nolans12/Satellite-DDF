#!/bin/bash

set -e

function yes_or_no() {
    # Prompt user for yes or no input
    while true; do
        read -p "$1 [y/n]: " yn
        case $yn in
            [Yy]* ) return 0;;
            [Nn]* ) return 1;;
            * ) echo "Please answer yes or no.";;
        esac
    done
}

function setup_python() {
    # Check if deadsnakes PPA is installed
    if ! apt-cache policy | grep -q deadsnakes; then
        # Prompt user to install deadsnakes PPA
        echo "deadsnakes PPA is not installed. Install it?"
        if yes_or_no; then
            sudo add-apt-repository ppa:deadsnakes/ppa
            sudo apt update
        else
            echo "Please install deadsnakes PPA before continuing."
            exit 1
        fi
    fi

    # Check that `python3.10`, `python3.10-venv`, and `python3-pip` are installed via apt
    for pkg in python3.10 python3.10-venv python3-pip; do
        if ! dpkg -l | grep -q $pkg; then
            # Prompt user to install missing package
            echo "$pkg is not installed. Install it?"
            if yes_or_no; then
                sudo apt update
                sudo apt install $pkg
            else
                echo "Please install $pkg before continuing."
                exit 1
            fi
        fi
    done
}

function setup_venv() {
    # Create a virtual environment
    if [ ! -d "venv" ]; then
        python3.10 -m venv venv
    fi
    # Source the virtual environment, if not activated
    if [ -z "$VIRTUAL_ENV" ]; then
        source venv/bin/activate
    fi
}

function update_requirements() {
    # Update the requirements lock file
    echo "requirements.txt has changed. Update requirements_lock.txt?"
    if yes_or_no; then
        pip-compile -o requirements_lock.txt pyproject.toml
    fi

    # Append `# <hash of requirements.txt>` to `requirements_lock.txt`
    hash=$(sha256sum requirements.txt)
    echo "# $hash" >> requirements_lock.txt

    # Re-install requirements but don't bother recursing
    pip install -r requirements_lock.txt
}

function install_requirements() {
    # Install requirements
    pip install -r requirements_lock.txt
    pip install -e .

    # Ensure the requirements lock file is up-to-date;
    # use the hash of `requirements.txt` stored at the end
    # of `requirements_lock.txt` to determine if an update
    # is needed
    hash=$(tail -n 1 requirements_lock.txt | cut -d ' ' -f 2)
    if [ "$hash" != "$(sha256sum requirements.txt | cut -d ' ' -f 1)" ]; then
        update_requirements
    fi
}

function success() {
    echo -e "\e[32m"
    echo "(っ◔◡◔)っ ♥success♥"
    echo -e "\e[0m"
}

function main() {
    setup_python
    setup_venv
    install_requirements
    success
}

main
