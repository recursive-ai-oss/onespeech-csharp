#!/usr/bin/env bash
set -euo pipefail
# Install only released binary packages, never build or inspect an oracle module.
prefix=${1:?environment directory}
manager=${MICROMAMBA:?path to micromamba 2.3.3}
"$manager" create -y -p "$prefix" --file "$(dirname "$0")/mfa-linux-64.explicit.txt"
