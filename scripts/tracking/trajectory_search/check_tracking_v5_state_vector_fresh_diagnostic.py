"""Replay the original 122 responses once, retaining strict decision diagnostics."""

import argparse

from scripts.tracking.trajectory_search.check_tracking_v5_state_vector_dual_recovery import main
from scripts.tracking.trajectory_search.state_vector_fresh_diagnostic_admission import CONFIG

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry", action="store_true")
    main(parser.parse_args().dry, diagnostic_config=CONFIG)
