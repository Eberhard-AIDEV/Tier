#!/bin/bash
# а1
exec "${TIER_PY:-python3}" "$(dirname "$0")/hwprobe.py" --deps
