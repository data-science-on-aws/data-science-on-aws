#!/usr/bin/env bash

# update location of Git hooks from default (.git/hooks) to the versioned folder .devtools/githooks
git config core.hooksPath ".devtools/githooks"

# make the githook executable
chmod +x ".devtools/githooks/pre-commit"
