#!/bin/bash

# To recover the original peft, change this to the backed up peft folder 
TUCKA_DIR="./peft"

# Aquire site-packages path
PY_VER=$(python -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")
SITE_PACKAGES="$CONDA_PREFIX/lib/python$PY_VER/site-packages"
echo $SITE_PACKAGES

if [ ! -d "$SITE_PACKAGES/peft-0.14.0.dist-info" ];then
  echo "Requires peft==0.14.0"
fi

# Copy files
cp -Rf "$TUCKA_DIR"/. "$SITE_PACKAGES/peft"
echo "Installation complete"