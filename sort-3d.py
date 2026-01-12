#!/usr/bin/env python3
"""
SORT websocket bridge entrypoint.

Run this alongside `replay_ben_6001_distance_heading.py`.

This process:
  - connects to the replay websocket servers (video/nmea/control),
  - runs SORT on streamed detections (bbox + optional distance/heading),
  - re-broadcasts the same streams on new ports, with `obj_id` replaced by tracker IDs.
"""

from sort_ws.bridge import cli_main


if __name__ == "__main__":
    cli_main()


