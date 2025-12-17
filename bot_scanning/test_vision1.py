import time
from visionNode import visionNode

# Get singleton instance
vision = visionNode.get()

"""
We generate 20 positions, each 1 unit apart:
(0,0), (1,1), (2,2), ... (19,19)
You can change pattern easily later.
"""
NUM_IMAGES = 20
WAIT_TIME = 1.0  # seconds

positions = [(i * 1.0, i * 1.0) for i in range(NUM_IMAGES)]

print("Starting continuous capture...\n")

for i, (x, y) in enumerate(positions):
    print(f"=== IMAGE {i+1}/{NUM_IMAGES} ===")
    print(f"Robot moved to X={x}, Y={y}")

    # Capture 1 image at orientation=1 (normal)
    vision.capture_once(x=x, y=y, orientation=1)

    # Wait before next capture
    time.sleep(WAIT_TIME)

print("\nAll capture requests sent. Waiting for processing...")

# Give worker time to finish processing
time.sleep(30)

vision.wait_until_done()
vision.shutdown()


print("Capture session finished.")
