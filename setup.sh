#!/bin/bash

die () {
  echo "${PROGNAME}: ${1:-"Unknown Error, Abort."}" 1>&2
  exit 1
}

echo "RITSAR Setup"
echo "==============="

SCRIPT_DIR=$( cd "$( dirname "$0" )" && pwd )
pushd ${SCRIPT_DIR} > /dev/null

# setup path to Python and proxy for pip
export PATH=/opt/data/software/prefix/bin:${PATH}
export https_proxy="http://ds-proxy-bc-fdh01.mmain.m.corp:8080"


# install virtual python environment
if [ ! -d .venv ]
then
    echo creating new virtualenv .venv...
    python -m venv .venv
    source .venv/bin/activate
    pip install --upgrade pip
    pip install wheel
    #pip --cache-dir=.pip install -r requirements.txt
	pip --cache-dir=.pip install -e .
    deactivate
fi

popd > /dev/null

echo "Setup... done."