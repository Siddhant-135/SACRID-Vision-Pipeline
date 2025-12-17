#!/usr/bin/env python3

import time
from visionNode import visionNode

def main():
    print("\n========== VISION TEST ==========\n")

    vision = visionNode()
    
    # ---------------------------------------------------------
    # Test 1: Normal single-angle scan
    # ---------------------------------------------------------
    print("\n--- TEST 1: Normal Scan ---")
    vision.capture(x=1.0, y=0.5, isTop=False, isBottom=True)
    time.sleep(1)

    # ---------------------------------------------------------
    # Test 2: Top-only scan
    # ---------------------------------------------------------
    print("\n--- TEST 2: TOP Scan ---")
    vision.capture(x=1.0, y=0.6, isTop=True, isBottom=False)
    time.sleep(1)

    # ---------------------------------------------------------
    # Test 3: Bottom-only scan
    # ---------------------------------------------------------
    print("\n--- TEST 3: BOTTOM Scan ---")
    vision.capture(x=1.0, y=0.4, isTop=False, isBottom=True)
    time.sleep(1)

    # ---------------------------------------------------------
    # Test 4: Full sweep
    # ---------------------------------------------------------
    print("\n--- TEST 4: FULL SWEEP (Bottom → Normal → Top) ---")
    vision.capture(x=1.0, y=0.5, isTop=True, isBottom=True)
    time.sleep(1)

    # ---------------------------------------------------------
    # Test 5: Random repeated calls to test orientation tracking
    # ---------------------------------------------------------
    print("\n--- TEST 5: Mixed Calls (Orientation Tracking Test) ---")
    vision.capture(1.2, 0.5, False, True)
    time.sleep(0.8)

    vision.capture(1.2, 0.5, True, False)
    time.sleep(0.8)

    vision.capture(1.2, 0.5, False, False)
    time.sleep(0.8)

    print("\nWaiting 3 seconds for background processing to finish...\n")
    time.sleep(3)

    print("\nShutting down VisionManager...")
    vision.shutdown()

    print("\n========== TEST COMPLETE ==========\n")


if __name__ == "__main__":
    main()
