#!/usr/bin/env python
"""便捷入口：``python reverse_trace.py Final.mp4 Sources/``"""

import sys

from source_trace.cli import main

if __name__ == "__main__":
    sys.exit(main())
