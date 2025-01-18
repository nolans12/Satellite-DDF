#!/bin/bash

set -e

venv/bin/alembic upgrade head
