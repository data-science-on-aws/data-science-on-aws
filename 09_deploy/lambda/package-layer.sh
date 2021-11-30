#!/usr/bin/env bash
  
# Builds a lambda package from a single Python 3 module with pip dependencies.
# This is a modified version of the AWS packaging instructions:
# https://docs.aws.amazon.com/lambda/latest/dg/lambda-python-how-to-create-deployment-package.html#python-package-dependencies
  
# https://stackoverflow.com/a/246128
SCRIPT_DIRECTORY="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
  
pushd $SCRIPT_DIRECTORY > /dev/null
  
rm -rf .package layer.zip
mkdir .package/python
  
pip install --target .package/python --requirement requirements.txt
  
pushd .package > /dev/null
zip --recurse-paths ${SCRIPT_DIRECTORY}/layer.zip .
popd > /dev/null
  
#zip --grow function.zip lambda_function.py
  
chmod u=rwx,go=r layer.zip

popd > /dev/null
