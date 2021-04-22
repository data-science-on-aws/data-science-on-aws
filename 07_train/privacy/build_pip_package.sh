#!/usr/bin/env bash
# Copyright 2020, The TensorFlow Privacy Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# Tool to build the TensorFlow Privacy pip package.
#
# Usage:
#   bazel run //tensorflow_privacy:build_pip_package -- \
#       "/tmp/tensorflow_privacy"
#
# Arguments:
#   output_dir: An output directory.
set -e

die() {
  echo >&2 "$@"
  exit 1
}

main() {
  local output_dir="$1"

  if [[ ! -d "${output_dir}" ]]; then
    die "The output directory '${output_dir}' does not exist."
  fi

  # Create a virtual environment
  virtualenv --python=python3 "venv"
  source "venv/bin/activate"
  pip install --upgrade pip

  # Build pip package
  pip install --upgrade setuptools wheel
  python "setup.py" sdist bdist_wheel

  cp "dist/"* "${output_dir}"
}

main "$@"
