#!/usr/bin/env python3
"""
================================================================================
ASL V3 FLAGSHIP TRAINING ORCHESTRATOR ENTRYPOINT ALIAS
================================================================================
Convenience entrypoint alias for train_all_in_one_tpu.py.
Enables launching ASL V3 via:
    python train_all_in_one.py [args]
or
    python train_all_in_one_tpu.py [args]
================================================================================
"""

import sys
import os

_current_dir = os.path.dirname(os.path.abspath(__file__))
if _current_dir not in sys.path:
    sys.path.insert(0, _current_dir)

from train_all_in_one_tpu import main, parse_args

build_v3_parser = parse_args

if __name__ == "__main__":
    main()
