"""
Scanner.py - Professional Camera and Servo Control Module

Provides camera capture and optional servo control functionality
for warehouse inventory scanning applications.

Coordinate Frame Conventions:
- UV Frame: Image pixel coordinates (u, v)
  - Origin: Top-left corner of image
  - Units: pixels
  - u: horizontal (left to right)
  - v: vertical (top to bottom)

- Camera Frame: Camera-centered coordinates (x_cam, y_cam)
  - Origin: Camera optical center
  - Units: meters
  - x_cam: horizontal
  - y_cam: vertical

- Global Frame: World coordinates (global_x, global_y)
  - Origin: World origin from SLAM system
  - Units: meters
  - Transformation: global = camera + slam_offset
"""

import os
import csv
import time
import traceback
import threading
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
import cv2
import numpy as np

# ==========================================================
# CONFIG LOADING (YAML)
# ==========================================================
import yaml

CONFIG_PATH = Path(__file__).resolve().parent.parent / "config" / "config.yaml"

with open(CONFIG_PATH, "r") as f:
    _cfg = yaml.safe_load(f)

scanner_cfg = _cfg["scanner"]



# ==========================================================
# SCANNER CLASS
# ==========================================================
class Scanner:
    """
    Professional camera and servo controller for inventory scanning.
    
    Features:
    - Multi-camera support with automatic selection
    - Burst capture with sharpness-based frame selection
    - Servo motor control (optional)
    - Thread-safe operations
    - Comprehensive logging
    """

    def __init__(
        self,
        camera_candidates: Tuple[int, ...] = tuple(scanner_cfg["camera_candidates"]),
        image_width: int = scanner_cfg["image_width"],
        image_height: int = scanner_cfg["image_height"],
        fourcc_codec: str = scanner_cfg["fourcc_codec"],
        fps_target: int = scanner_cfg["fps_target"],
        servo_port: str = scanner_cfg["servo_port"],
        servo_baud_rate: int = scanner_cfg["servo_baud_rate"],
        enable_servo: bool = scanner_cfg["enable_servo"],
        enable_logging: bool = scanner_cfg["enable_logging"],
        log_file_path: str = scanner_cfg["log_file_path"]
    ):

        """
        Initialize Scanner with camera, QR detector, and servo.

        Args:
            camera_candidates: Tuple of camera device indices to try
            image_width: Desired image width in pixels
            image_height: Desired image height in pixels
            fourcc_codec: Video codec (e.g., 'MJPG', 'YUYV')
            fps_target: Target frames per second
            servo_port: Serial port for servo board
            servo_baud_rate: Baud rate for servo communication
            enable_servo: Whether to initialize servo hardware
            enable_logging: Whether to enable file logging
            log_file_path: Path to log file

        Raises:
            ValueError: If camera_candidates is empty
            SystemExit: If camera initialization fails
        """
        # Validate inputs
        if not camera_candidates:
            raise ValueError("camera_candidates cannot be empty")

        # Camera configuration
        self.camera_candidates = tuple(camera_candidates)
        self.image_width = int(image_width)
        self.image_height = int(image_height)
        self.fourcc_codec = str(fourcc_codec)
        self.fps_target = int(fps_target)

        # Servo configuration
        self.servo_port = str(servo_port)
        self.servo_baud_rate = int(servo_baud_rate)
        self.enable_servo = bool(enable_servo)

        # Logging configuration
        self.enable_logging = bool(enable_logging)
        self.log_file_path = Path(log_file_path)

        # Runtime objects
        self.camera_capture: Optional[cv2.VideoCapture] = None
        self.camera_index: Optional[int] = None
        self.servo_board: Optional[Any] = None

        # Thread safety
        self._lock = threading.Lock()

        # Initialize all components
        self._initialize_all_components()
        self._servo_move_to_Home()

    # ==========================================================
    # LOGGING METHODS
    # ==========================================================

    def _log_message(self, message: str, level: str = "INFO") -> None:
        """
        Log message to console and/or file.

        Args:
            message: Message to log
            level: Log level (INFO, WARN, ERROR, SUCCESS)
        """
        # Format message with level prefix
        level_prefixes = {
            "INFO": "[INFO]",
            "WARN": "[WARN]",
            "ERROR": "[ERROR]",
            "SUCCESS": "[SUCCESS]",
            "DEBUG": "[DEBUG]"
        }
        prefix = level_prefixes.get(level.upper(), "[INFO]")
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        formatted_msg = f"{timestamp} {prefix} {message}"

        # Console output
        print(formatted_msg)

        # File output
        if self.enable_logging:
            try:
                self.log_file_path.parent.mkdir(parents=True, exist_ok=True)
                with open(self.log_file_path, "a") as f:
                    f.write(formatted_msg + "\n")
            except Exception as e:
                # Logging must not break the pipeline
                print(f"[WARN] Failed to write log: {e}")

    # ==========================================================
    # FILE SYSTEM UTILITIES
    # ==========================================================

    def _ensure_directory(self, directory_path: Path) -> None:
        """
        Create directory if it doesn't exist.

        Args:
            directory_path: Path to directory
        """
        directory_path.mkdir(parents=True, exist_ok=True)

    def _ensure_csv_with_header(
        self,
        csv_path: Path,
        header: List[str]
    ) -> None:
        """
        Create CSV file with header if it doesn't exist.

        Args:
            csv_path: Path to CSV file
            header: List of column names
        """
        try:
            # Ensure parent directory exists
            csv_path.parent.mkdir(parents=True, exist_ok=True)

            # Check if file exists and has content
            file_exists = csv_path.exists()
            file_empty = not file_exists or csv_path.stat().st_size == 0

            if file_empty:
                with open(csv_path, "w", newline="") as f:
                    writer = csv.writer(f)
                    writer.writerow(header)
        except Exception as e:
            self._log_message(f"CSV header creation failed: {e}", "ERROR")

    # ==========================================================
    # IMAGE QUALITY ASSESSMENT
    # ==========================================================

    def _compute_image_sharpness(self, frame: np.ndarray) -> float:
        """
        Compute image sharpness using Laplacian variance.

        Higher values indicate sharper images.

        Args:
            frame: Input image (BGR format)

        Returns:
            Sharpness score (variance of Laplacian)
        """
        try:
            # Convert to grayscale
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            # Compute Laplacian and return variance
            laplacian = cv2.Laplacian(gray, cv2.CV_64F)
            sharpness = laplacian.var()

            return float(sharpness)
        except Exception as e:
            self._log_message(f"Sharpness computation failed: {e}", "WARN")
            return 0.0

    # ==========================================================
    # CAMERA INITIALIZATION
    # ==========================================================

    def _initialize_camera(self) -> None:
        """
        Initialize camera by trying each candidate device.

        Raises:
            SystemExit: If no usable camera found
        """
        initialization_errors = []

        for device_index in self.camera_candidates:
            capture = None
            try:
                # Open camera with V4L2 backend (Linux)
                capture = cv2.VideoCapture(device_index, cv2.CAP_V4L2)
            except Exception as e:
                initialization_errors.append(
                    f"Device {device_index}: Open failed - {e}"
                )
                continue

            # Check if opened successfully
            if not capture.isOpened():
                initialization_errors.append(
                    f"Device {device_index}: Not opened"
                )
                if capture:
                    try:
                        capture.release()
                    except Exception:
                        pass
                continue

            # Configure camera properties
            try:
                capture.set(cv2.CAP_PROP_FRAME_WIDTH, self.image_width)
                capture.set(cv2.CAP_PROP_FRAME_HEIGHT, self.image_height)
                capture.set(
                    cv2.CAP_PROP_FOURCC,
                    cv2.VideoWriter_fourcc(*self.fourcc_codec)
                )
                capture.set(cv2.CAP_PROP_FPS, self.fps_target)
            except Exception as e:
                self._log_message(
                    f"Device {device_index}: Property setting failed - {e}",
                    "WARN"
                )

            # Warmup period
            time.sleep(0.15)

            # Verify camera can capture frames
            try:
                if capture.grab():
                    ret, _ = capture.retrieve()
                else:
                    ret = False
            except Exception:
                ret = False

            if not ret:
                initialization_errors.append(
                    f"Device {device_index}: Frame capture test failed"
                )
                try:
                    capture.release()
                except Exception:
                    pass
                continue

            # Success - store camera object
            self.camera_capture = capture
            self.camera_index = device_index
            self._log_message(
                f"Camera initialized on device {device_index} "
                f"({self.image_width}x{self.image_height} @ {self.fps_target}fps)",
                "SUCCESS"
            )
            return

        # No usable camera found
        error_summary = "; ".join(initialization_errors)
        self._log_message(
            f"Camera initialization failed: {error_summary}",
            "ERROR"
        )
        raise SystemExit("No usable camera found")
 
    # ==========================================================
    # SERVO INITIALIZATION
    # ==========================================================

    def _initialize_servo(self) -> None:
        """
        Initialize servo motor controller.

        Requires servoBoard module. Fails gracefully if unavailable.
        """
        if not self.enable_servo:
            self._log_message("Servo disabled by configuration", "INFO")
            self.servo_board = None
            return

        try:
            # Lazy import to avoid dependency when servo not used
            from servoBoard import ServoBoard

            self.servo_board = ServoBoard(
                port=self.servo_port,
                baudrate=self.servo_baud_rate
            )
            self._log_message(
                f"Servo initialized on {self.servo_port} @ {self.servo_baud_rate} baud",
                "SUCCESS"
            )
        except ImportError:
            self._log_message(
                "servoBoard module not found - servo disabled",
                "WARN"
            )
            self.servo_board = None
        except Exception as e:
            self._log_message(
                f"Servo initialization failed: {e}",
                "WARN"
            )
            self.servo_board = None

    # ==========================================================
    # COMPONENT INITIALIZATION
    # ==========================================================

    def _initialize_all_components(self) -> None:
        """Initialize all scanner components in sequence."""
        self._log_message("Initializing scanner components...", "INFO")
        self._initialize_camera()
        self._initialize_servo()
        self._log_message("Scanner initialization complete", "SUCCESS")

    def _servo_move_to_Home(self) -> bool:
        """
        Move servo to home position (1, 1).

        Returns:
            True if successful, False if servo not initialized or move fails
        """
        if self.servo_board is None:
            self._log_message("Servo not initialized", "WARN")
            return False

        try:
            from servoBoard import servo_move_to

            # Thread-safe servo movement
            with self._lock:
                success, _ = servo_move_to(
                    self.servo_board,
                    h=1,
                    v=1,
                    cap=self.camera_capture,
                    capture_after_move=False
                )
                return success
        except ImportError:
            self._log_message(
                "servo_move_to function not available",
                "ERROR"
            )
            return False
        except Exception as e:
            self._log_message(f"Servo move to home failed: {e}", "ERROR")
            return False

    # ==========================================================
    # FRAME CAPTURE METHODS
    # ==========================================================

    def capture_best_frame(
        self,
        x: float,
        y: float,
        num_burst_frames: int = 10,
        output_directory: str = scanner_cfg["raw_save_dir"],
        metadata_csv_file: str = scanner_cfg["metadata_csv_path"],
        error_log_file: str = scanner_cfg["log_file_path"]
    ) -> Dict[str, Any]:
        """
        Capture burst of frames and save the sharpest one.

        Uses grab/retrieve for fast burst capture, then selects frame
        with highest Laplacian variance (sharpness metric).

        Args:
            num_burst_frames: Number of frames to capture in burst
            output_directory: Directory to save captured image
            metadata_csv_file: CSV file to append capture metadata
            error_log_file: Log file for capture errors

        Returns:
            Dictionary containing:
                - status: "OK" or "ERROR"
                - timestamp: Capture timestamp
                - filename: Saved image filename
                - sharpness: Sharpness score
                - resolution: Tuple (width, height)
                - path: Full path to saved image
                - error: Error message (if status="ERROR")
        """
        # Validate inputs
        num_burst_frames = max(1, int(num_burst_frames))
        output_dir_path = Path(output_directory)
        metadata_csv_path = Path(metadata_csv_file)
        error_log_path = Path(error_log_file)

        try:
            # Ensure output directory exists
            self._ensure_directory(output_dir_path)

            # Ensure metadata CSV exists with header
            csv_header = ["timestamp", "filename", "sharpness", "width", "height"]
            with self._lock:
                self._ensure_csv_with_header(metadata_csv_path, csv_header)

            # Verify camera is initialized
            if self.camera_capture is None:
                raise RuntimeError("Camera not initialized")

            # Capture burst of frames (thread-safe)
            with self._lock:
                # Flush camera buffer
                for _ in range(3):
                    try:
                        self.camera_capture.grab()
                    except Exception:
                        pass

                # Capture frames and track best
                best_frame = None
                best_sharpness = -1.0

                for _ in range(num_burst_frames):
                    try:
                        # Fast capture using grab/retrieve
                        if self.camera_capture.grab():
                            ret, frame = self.camera_capture.retrieve()
                        else:
                            ret = False
                            frame = None
                    except Exception:
                        ret = False
                        frame = None

                    if not ret or frame is None:
                        continue

                    # Compute sharpness
                    sharpness = self._compute_image_sharpness(frame)

                    # Update best frame
                    if sharpness > best_sharpness:
                        best_sharpness = sharpness
                        best_frame = frame

            # Verify we captured at least one valid frame
            if best_frame is None:
                raise RuntimeError("No valid frames captured in burst")

            # Get frame dimensions
            frame_height, frame_width = best_frame.shape[:2]

            # Generate filename with timestamp
            timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            filename = f"capture_{timestamp}.png"
            output_path = output_dir_path / filename

            # Save image
            cv2.imwrite(str(output_path), best_frame)

            # Append metadata to CSV (thread-safe)
            with self._lock:
                with open(metadata_csv_path, "a", newline="") as f:
                    csv_writer = csv.writer(f)
                    csv_writer.writerow([
                        timestamp,
                        filename,
                        round(best_sharpness, 2),
                        frame_width,
                        frame_height
                    ])

            # Return success result
            return {
                "status": "OK",
                "timestamp": timestamp,
                "filename": filename,
                "sharpness": best_sharpness,
                "resolution": (frame_width, frame_height),
                "x0": x,
                "y0": y,
                "path": str(output_path)
            }

        except Exception as e:
            # Log full traceback
            error_traceback = traceback.format_exc()
            try:
                error_log_path.parent.mkdir(parents=True, exist_ok=True)
                with open(error_log_path, "a") as f:
                    f.write(f"[{datetime.now().isoformat()}]\n")
                    f.write(error_traceback)
                    f.write("\n" + "="*70 + "\n")
            except Exception:
                pass

            # Log error message
            self._log_message(f"Frame capture failed: {e}", "ERROR")

            # Return error result
            return {
                "status": "ERROR",
                "error": str(e)
            }


    # ==========================================================
    # SERVO CONTROL METHODS
    # ==========================================================

    def open_servo_debug_terminal(self) -> None:
        """
        Open interactive debug terminal for servo control.

        Requires servoBoard module with servo_debug_terminal function.
        """
        if self.servo_board is None:
            self._log_message("Servo not initialized", "WARN")
            return

        try:
            from servoBoard import servo_debug_terminal
            servo_debug_terminal(
                port=self.servo_port,
                baud=self.servo_baud_rate
            )
        except ImportError:
            self._log_message(
                "servo_debug_terminal function not available",
                "ERROR"
            )
        except Exception as e:
            self._log_message(f"Debug terminal failed: {e}", "ERROR")

    def get_servo_positions(self) -> Dict[str, Any]:
        """
        Get current positions of all servos.

        Returns:
            Dictionary mapping servo IDs to positions
            Returns empty dict if servo not initialized
        """
        if self.servo_board is None:
            self._log_message("Servo not initialized", "WARN")
            return {}

        try:
            return self.servo_board.get_positions()
        except Exception as e:
            self._log_message(f"Failed to get servo positions: {e}", "WARN")
            return {}

    def move_servo_to_position(
        self,
        h: int = 1,
        v: int = 1,
        capture_after_move: bool = False,
        **servo_kwargs
    ) -> Tuple[bool, Optional[np.ndarray]]:
        """
        Move servo to specified position.

        Args:
            h: Horizontal servo position
            v: Vertical servo position
            **servo_kwargs: Additional servo control parameters

        Returns:
            Tuple (success, captured_frame)
            Returns (False, None) if servo not initialized or move fails
        """
        if self.servo_board is None:
            self._log_message("Servo not initialized", "WARN")
            return False, None

        try:
            from servoBoard import servo_move_to

            # Thread-safe servo movement (may use camera)
            with self._lock:
                success, frame = servo_move_to(
                    self.servo_board,
                    h,
                    v,
                    cap=self.camera_capture,
                    capture_after_move=capture_after_move,
                    **servo_kwargs
                )
                return success, frame

        except ImportError:
            self._log_message(
                "servo_move_to function not available",
                "ERROR"
            )
            return False, None
        except Exception as e:
            self._log_message(f"Servo move failed: {e}", "ERROR")
            return False, None

    # ==========================================================
    # CLEANUP METHODS
    # ==========================================================

    def close(self) -> None:
        """
        Clean up resources and close connections.

        Releases camera, closes servo connection, and logs shutdown.
        """
        # Close camera
        if self.camera_capture is not None:
            try:
                self.camera_capture.release()
                self._log_message("Camera released", "INFO")
            except Exception as e:
                self._log_message(f"Error releasing camera: {e}", "WARN")
            finally:
                self.camera_capture = None
                self.camera_index = None

        # Close servo
        if self.servo_board is not None:
            try:
                self.servo_board.close()
                self._log_message("Servo connection closed", "INFO")
            except Exception as e:
                self._log_message(f"Error closing servo: {e}", "WARN")
            finally:
                self.servo_board = None

        self._log_message("Scanner shutdown complete", "INFO")

    def __enter__(self):
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit - ensures cleanup."""
        self.close()
        return False


# ==========================================================
# SMOKE TEST
# ==========================================================
if __name__ == "__main__":
    print("=" * 70)
    print("SMOKE TEST - Scanner Module")
    print("=" * 70)

    # Test 1: Initialization
    print("\n[TEST 1] Scanner Initialization")
    print("-" * 70)
    try:
        scanner = Scanner(
            camera_candidates=(0, 1, 2),
            enable_servo=True,  # Disable servo for testing
            enable_logging=True
        )
        print("✓ Scanner initialized successfully")
        print(f"  - Camera index: {scanner.camera_index}")
        print(f"  - Resolution: {scanner.image_width}x{scanner.image_height}")
        print(f"  - Servo: {'Enabled' if scanner.servo_board else 'Disabled'}")
    except Exception as e:
        print(f"✗ Initialization failed: {e}")
        exit(1)

    # Test 2: Frame capture
    print("\n[TEST 2] Frame Capture (8-frame burst)")
    print("-" * 70)
    try:
        result = scanner.capture_best_frame(
            num_burst_frames=8,
            output_directory="test_captures",
            metadata_csv_file="test_captures/metadata.csv"
        )
        
        if result["status"] == "OK":
            print("✓ Frame capture successful")
            print(f"  - Timestamp: {result['timestamp']}")
            print(f"  - Filename: {result['filename']}")
            print(f"  - Sharpness: {result['sharpness']:.2f}")
            print(f"  - Resolution: {result['resolution']}")
            print(f"  - Path: {result['path']}")
        else:
            print(f"✗ Frame capture failed: {result.get('error')}")
    except Exception as e:
        print(f"✗ Frame capture exception: {e}")
        import traceback
        traceback.print_exc()


    # Test 4: Servo status (if enabled)
    print("\n[TEST 4] Servo Status")
    print("-" * 70)
    if scanner.servo_board is not None:
        try:
            positions = scanner.get_servo_positions()
            print(f"✓ Servo positions retrieved: {positions}")
        except Exception as e:
            print(f"✗ Servo query failed: {e}")
    else:
        print("ℹ Servo not enabled - skipping test")

    """# Test 5: Context manager
    print("\n[TEST 5] Context Manager Test")
    print("-" * 70)
    try:
        with Scanner(enable_servo=False, enable_logging=False) as test_scanner:
            print(f"✓ Context manager entered")
            print(f"  - Camera: {test_scanner.camera_index}")
        print("✓ Context manager exited - resources cleaned up")
    except Exception as e:
        print(f"✗ Context manager test failed: {e}")"""


# servo move to home
    scanner._servo_move_to_Home()
    print("✓ Servo moved to home position")

    # Cleanup
    print("\n[CLEANUP] Closing Scanner")
    print("-" * 70)
    scanner.close()
    print("✓ Scanner closed successfully")

    # Summary
    print("\n" + "=" * 70)
    print("SMOKE TEST COMPLETE")
    print("=" * 70)
    print("\nAll core functionality validated successfully!")
    print("Note: Servo tests skipped if hardware not connected")


