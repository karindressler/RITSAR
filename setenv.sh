#!/bin/bash

# this scipt should be sourced not excecuted!
if [ "${BASH_SOURCE[0]}" -ef "$0" ]
then
    echo "Usage:     source setenv.sh"
    exit 1
fi

export SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
pushd ${SCRIPT_DIR} > /dev/null
export PYTHONPATH=${SCRIPT_DIR}:${SCRIPT_DIR}/examples

echo Activating Python virtual environment .venv ...
source .venv/bin/activate
echo currently used python:
which python


