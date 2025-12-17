# vision_manager.py
import cv2
import os
from pathlib import Path
import yaml,time
from datetime import datetime
from multiprocessing import Process, Queue, Event, get_context

from bot_scanning.bot_scanning.Scanner import Scanner
from bot_scanning.bot_scanning.Processor import Processor

# Resolve config.yaml relative to this file
CONFIG_PATH = Path(__file__).resolve().parent.parent / "config" / "config.yaml"
SESSION_ID_PATH = Path(__file__).resolve().parent.parent / "config" / "session_id.yaml"


with open(CONFIG_PATH, "r") as f:
    _cfg = yaml.safe_load(f)
with open(SESSION_ID_PATH, "r") as f:
    _session_cfg = yaml.safe_load(f)

vision_node_cfg = _cfg["vision_node"]


class visionNode:
    _instance = None

    # LED brightness presets
    BRIGHTNESS_MAP = {
        "low": int(255 * 0.25),
        "med": int(255 * 0.5),
        "high": int(255 * 0.8)
    }

    # ==========================================================
    # INIT
    # ==========================================================
    def __init__(self):
        if visionNode._instance is not None:
            raise RuntimeError("Use visionNode.get() instead")
        visionNode._instance = self

        # ------------------------------------------------------
        # Load config
        # ------------------------------------------------------
        self.log_file = vision_node_cfg["log_file"]
        os.makedirs(os.path.dirname(self.log_file), exist_ok=True)

        # ------------------------------------------------------
        # Logger warm start
        # ------------------------------------------------------
        self._log("Initializing visionNode...", "INFO")

        # ------------------------------------------------------
        # Scanner Init
        # ------------------------------------------------------
        self._log("Initializing Scanner...", "INFO")
        self.scanner = Scanner(enable_servo=True, enable_logging=True)

        self.curr_h = 1         # horizontal
        self.curr_v = 1         # normal orientation

        self._log("Scanner initialized successfully", "INFO")

        # ------------------------------------------------------
        # Processor Worker Setup
        # ------------------------------------------------------
        self.ctx = get_context("spawn")
        self.queue = self.ctx.Queue(maxsize=50)
        self.stop_event = self.ctx.Event()

        self.delete_raw_flag = vision_node_cfg["delete_raw_flag"]

        self.proc = Process(
            target=self._process_worker,
            args=(self.queue, self.stop_event),
            daemon=True
        )
        self.proc.start()

        self._log("Processor worker started", "INFO")
        self._log("visionNode ready.", "INFO")

    # ==========================================================
    # LOGGER
    # ==========================================================
    def _log(self, msg, level="DEBUG"):
        """
        Unified logger:
        - Console: INFO, WARN, ERROR
        - File: ALL
        """
        ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        formatted = f"{ts} [{level}] {msg}"

        # Print only important logs to console
        if level in ("INFO", "WARN", "ERROR"):
            print(formatted)

        # Always write to log file
        with open(self.log_file, "a") as f:
            f.write(formatted + "\n")

    # ==========================================================
    # GET SINGLETON
    # ==========================================================
    @staticmethod
    def get():
        if visionNode._instance is None:
            visionNode()
        return visionNode._instance

    # ==========================================================
    # PUBLIC SCAN API
    # ==========================================================
    def capture(self, x, y, isTop, isBottom):
        self._log(f"capture(x={x}, y={y}, isTop={isTop}, isBottom={isBottom})", "INFO")
        self._log(f"Current orientation H={self.curr_h}, V={self.curr_v}", "DEBUG")

        # ------------ internal helper ------------
        def click(vpos):
            self._log(f"Request: Move to V={vpos}", "DEBUG")
            self._move_to_v(self.curr_h, vpos)
            self._log(f"Capturing at V={vpos}", "DEBUG")
            self.capture_once(x=x, y=y, orientation=vpos)

        # ------------------------------------------------------
        # CASE 1: No flags → single-angle capture
        # ------------------------------------------------------
        if not isTop and not isBottom:
            self._log("Normal single-angle capture", "INFO")
            click(self.curr_v)
            return

        # ------------------------------------------------------
        # CASE 2: Full sweep
        # ------------------------------------------------------
        if isTop and isBottom:
            self._log("FULL SWEEP: bottom → normal → top → normal", "INFO")
            click(0)
            click(1)
            click(2)
            self._move_to_v(self.curr_h, 1)
            return

        # ------------------------------------------------------
        # CASE 3: Top-only
        # ------------------------------------------------------
        if isTop:
            self._log("TOP-ONLY scan", "INFO")

            if self.curr_v == 2:
                click(2)
                click(1)
            else:
                click(1)
                click(2)
                self._move_to_v(self.curr_h, 1)
            return

        # ------------------------------------------------------
        # CASE 4: Bottom-only
        # ------------------------------------------------------
        if isBottom:
            self._log("BOTTOM-ONLY scan", "INFO")

            if self.curr_v == 0:
                click(0)
                click(1)
            else:
                click(1)
                click(0)
                self._move_to_v(self.curr_h, 1)
            return

    # ==========================================================
    # CAPTURE SINGLE FRAME
    # ==========================================================
    def capture_once(self, x=0.0, y=0.0, orientation=1):
        self._log(f"Capture request → (x={x}, y={y}, ori={orientation})", "INFO")

        res = self.scanner.capture_best_frame(
            x=x,
            y=y,
            num_burst_frames=6,
        )

        if res["status"] != "OK":
            self._log(f"Capture failed: {res.get('error')}", "ERROR")
            return

        img_path = res["path"]
        self._log(f"Image saved: {img_path}", "DEBUG")

        self.queue.put({
            "path": img_path,
            "x0": x,
            "y0": y,
            "orientation": orientation
        })


    # ==========================================================
    # LED CONTROL (PUBLIC API)
    # ==========================================================
    def led(self, color="white", level="low", index=0):
        """
        ROS-accessible LED trigger
        color: "red", "green", "yellow", etc.
        level: "low", "med", "high"
        """
        brightness = self.BRIGHTNESS_MAP.get(level, int(255 * 0.25))

        sb = self.scanner.servo_board
        if sb is None:
            self._log("LED command ignored: servo board not initialized", "WARN")
            return False

        try:
            sb.led(index, color, brightness)   # from servoBoard.py
            self._log(f"LED → color={color}, level={level}, index={index}", "INFO")
            return True
        except Exception as e:
            self._log(f"LED command failed: {e}", "ERROR")
            return False
        

    def led_off(self, index=0):
        sb = self.scanner.servo_board
        if sb is None:
            self._log("LED off ignored: servo board not initialized", "WARN")
            return False

        try:
            sb.led_off(index)
            self._log(f"LED OFF (index={index})", "INFO")
            return True
        except Exception as e:
            self._log(f"LED OFF failed: {e}", "ERROR")
            return False

    # ==========================================================
    # SERVO MOVEMENT
    # ==========================================================
    def _move_to_v(self, target_h, target_v):
        if target_v == self.curr_v and target_h == self.curr_h:
            self._log("Servo already in correct orientation", "DEBUG")
            return True

        ok, _ = self.scanner.move_servo_to_position(
            h=target_h,
            v=target_v,
            capture_after_move=False
        )

        if ok:
            self._log(f"Servo moved to V={target_v}", "DEBUG")
            self.curr_v = target_v
            self.curr_h = target_h
        else:
            self._log("Servo movement FAILED!", "ERROR")

        return ok

    # ==========================================================
    # BACKGROUND WORKER
    # ==========================================================
    def _process_worker(self, queue, stop_event):
        self._log("Processor worker started", "INFO")
        processor = Processor()

        while not stop_event.is_set():
            try:
                item = queue.get(timeout=0.5)
            except:
                continue

            if item is None:
                break

            img_path = item["path"]
            x = item["x0"]
            y = item["y0"]
            ori = item["orientation"]

            self._log(f"Worker processing: {img_path}", "DEBUG")

            img = cv2.imread(img_path)
            if img is None:
                self._log(f"Failed to read image: {img_path}", "ERROR")
                continue

            session_id = _session_cfg["session_id"]

            result = processor.process_image(
                img=img,
                x_offset=x,
                y_offset=y,
                orientation=ori,
                session_id= session_id
            )
            if self.delete_raw_flag:
                try:
                    os.remove(img_path)
                    self._log(f"Deleted raw image: {img_path}", "DEBUG")
                except Exception as e:
                    self._log(f"Failed to delete raw image: {e}", "ERROR")

            self._log(f"Processing finished: {result}", "DEBUG")

        self._log("Processor worker exiting", "INFO")

        # ==========================================================
    # WAIT UNTIL ALL PROCESSING IS DONE (NEW)
    # ==========================================================
    def wait_until_done(self, check_interval=0.5, quiet=False):
        """
        Dynamically waits until the processing queue is empty.
        Prevents shutting down before all images are processed.
        """
        q = self.queue
        proc = self.proc

        if not quiet:
            self._log("Waiting for all processing to finish...", "INFO")

        while True:
            try:
                size = q.qsize()
            except NotImplementedError:
                size = "unknown"

            if not quiet:
                self._log(f"Pending items in queue: {size}", "DEBUG")

            # CASE 1: qsize works → wait until == 0
            if isinstance(size, int) and size == 0:
                break

            # CASE 2: unknown qsize → check if queue is empty via non-blocking get
            if size == "unknown":
                try:
                    q.get_nowait()
                except:
                    break  # Queue empty
                continue

            time.sleep(check_interval)

        if not quiet:
            self._log("All processing completed.", "INFO")

    # ==========================================================
    # SHUTDOWN
    # ==========================================================
    def shutdown(self):
        self._log("Shutting down visionNode...", "INFO")
        self.stop_event.set()
        self.queue.put(None)
        self.proc.join(timeout=2)
        self.scanner.close()
        self._log("Shutdown complete.", "INFO")
