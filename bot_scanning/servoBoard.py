import serial
import time
import re
import yaml
from pathlib import Path

# Resolve config.yaml relative to this file
CONFIG_PATH = Path(__file__).resolve().parent.parent / "config" / "config.yaml"

with open(CONFIG_PATH, "r") as f:
    _cfg = yaml.safe_load(f)

DEFAULT_H_THETA = _cfg['servo']['h_theta']
DEFAULT_V_THETA = _cfg['servo']['v_theta']

H_NEUTRAL = _cfg['servo']['h_neutral']
V_NEUTRAL = _cfg['servo']['v_neutral']

SERVO_H_ID = _cfg['servo_ids']['horizontal']
SERVO_V_ID = _cfg['servo_ids']['vertical']
wait_after_move = _cfg['servo']['wait_after_move']
timeout = _cfg['servo']['timeout']

# ==========================================
# PART 1: DRIVER CLASS + LED API
# ==========================================
class ServoBoard:
    # -------------------------------------------------
    # Predefined colors for easy use
    # -------------------------------------------------
    COLOR_MAP = {
        # --- Basic Colors ---
        'r': (255, 0, 0), 'red': (255, 0, 0),
        'g': (0, 255, 0), 'green': (0, 255, 0),
        'b': (0, 0, 255), 'blue': (0, 0, 255),
        'y': (255, 255, 0), 'yellow': (255, 255, 0),
        'w': (255, 255, 255), 'white': (255, 255, 255),
        'cyan': (0, 255, 255),
        'magenta': (255, 0, 255),
        'orange': (255, 165, 0),
        'off': (0, 0, 0),

        # -------------------------------------------------
        # OFFICIAL SIGNAL COLORS (HEX → RGB)
        # -------------------------------------------------

        # Signal Yellow (Warning) #F9A800
        'signal_yellow': (249, 168, 0),
        'sy': (249, 168, 0),

        # Signal Red (Danger / Prohibition) #9B2423
        'signal_red': (155, 36, 35),
        'sr': (155, 36, 35),

        # Signal Blue (Mandatory) #005387
        'signal_blue': (0, 83, 135),
        'sb': (0, 83, 135),

        # Signal Green (Safe Condition / Exit) #237F52
        'signal_green': (35, 127, 82),
        'sg': (35, 127, 82),

        # White background #ECECE7
        'signal_white': (236, 236, 231),
        'sw': (236, 236, 231),

        # Black contrast #2B2B2C
        'signal_black': (43, 43, 44),
        'sblk': (43, 43, 44),
    }

    def __init__(self, port, baudrate=115200, timeout=timeout):
        try:
            self.ser = serial.Serial(port, baudrate, timeout=timeout)
            print(f"Connected to {port} at {baudrate} baud.")
            time.sleep(2)
            self.clear_buffer()
        except serial.SerialException as e:
            print(f"Error connecting to serial port: {e}")
            self.ser = None

    def clear_buffer(self):
        if self.ser and self.ser.in_waiting:
            self.ser.read(self.ser.in_waiting)

    # ---------------------------------------------------------
    # SERVO MOVEMENT
    # ---------------------------------------------------------
    def move(self, servo_id, angle):
        if not self.ser:
            return
        angle = max(0, min(180, angle))
        cmd = f"{servo_id} {angle}\n"
        self.ser.write(cmd.encode())
        time.sleep(0.01)

    def calibrate(self, servo_id):
        if not self.ser:
            return
        cmd = f"{servo_id} -1\n"
        self.ser.write(cmd.encode())
        time.sleep(0.5)

    def get_positions(self):
        if not self.ser:
            return {}
        self.clear_buffer()
        self.ser.write(b"P\n")

        angles = {}
        start = time.time()

        while time.time() - start < 1.0:
            if self.ser.in_waiting:
                try:
                    line = self.ser.readline().decode().strip()
                    match = re.search(r"ID (\d+): ([\d\.-]+) deg", line)
                    if match:
                        s_id = int(match.group(1))
                        s_ang = float(match.group(2))
                        angles[s_id] = s_ang

                    if 0 in angles and 1 in angles:
                        break
                except:
                    pass
        return angles

    # ---------------------------------------------------------
    # LOW-LEVEL LED COMMAND
    # ---------------------------------------------------------
    def LED(self, index=0, r=255, g=255, b=255, brightness=int(255*0.7)):
        if not self.ser:
            print("[ERROR] Servo board not initialized.")
            return

        cmd = f"L {index} {r} {g} {b} {brightness}\n"
        print("Sending LED command:", repr(cmd))
        self.ser.write(cmd.encode())
        time.sleep(0.05)

    # ---------------------------------------------------------
    # NEW HIGH-LEVEL LED API
    # ---------------------------------------------------------
    def _normalize_color(self, color):
        if isinstance(color, (tuple, list)) and len(color) == 3:
            return tuple(map(int, color))
        if isinstance(color, str):
            c = color.strip().lower()
            if c in self.COLOR_MAP:
                return self.COLOR_MAP[c]
        raise ValueError(f"Unknown color: {color}")

    def led(self, index, color, brightness=int(255*0.6)):
        """High-level simple LED control."""
        if not self.ser:
            print("[ERROR] Board not initialized")
            return

        try:
            r, g, b = self._normalize_color(color)
            self.LED(index, r, g, b, int(brightness))
        except Exception as e:
            print("[LED ERROR]", e)

    def led_off(self, index):
        """Turn off LED completely."""
        self.LED(index, 0, 0, 0, 0)

    def led_flash_on(self, index):
        """Bright flashlight color."""
        self.LED(index, 20, 255, 222, 230)

    def led_alert(self, index, color='red', brightness=int(255*0.6)):
        """Alert mode (NO blinking)."""
        try:
            r, g, b = self._normalize_color(color)
            self.LED(index, r, g, b, int(brightness))
        except Exception as e:
            print("[ALERT ERROR]", e)

    def set_led_manual(self, index, r, g, b, brightness=int(255*0.6)):
        """Raw manual control."""
        r = max(0, min(255, int(r)))
        g = max(0, min(255, int(g)))
        b = max(0, min(255, int(b)))
        brightness = max(0, min(255, int(brightness)))
        self.LED(index, r, g, b, brightness)

    # ---------------------------------------------------------
    # PRESET SIGNAL FUNCTIONS
    # ---------------------------------------------------------
    def warning(self, index):
        self.led(index, 'signal_yellow')

    def danger(self, index):
        self.led(index, 'signal_red')

    def mandatory(self, index):
        self.led(index, 'signal_blue')

    def safe(self, index):
        self.led(index, 'signal_green')

    def bg(self, index):
        self.led(index, 'signal_white')

    def contrast(self, index):
        self.led(index, 'signal_black')

    # ---------------------------------------------------------
    def close(self):
        if self.ser:
            self.ser.close()
            print("Connection closed.")


# ==========================================
# PART 2: CAMERA SERVO HELPERS (UNCHANGED)
# ==========================================



def set_camera_orientation(board, h, v):
    if board is None or getattr(board, "ser", None) is None:
        print("[ERROR] Servo board not initialized.")
        return False

    try:
        if h not in [0, 1] or v not in [0, 1]:
            print("[ERROR] h and v must be 0 or 1.")
            return False

        h_angle = H_NEUTRAL - (DEFAULT_H_THETA * (1 - h))
        v_angle = V_NEUTRAL - (DEFAULT_V_THETA * (1 - v))

        board.move(SERVO_H_ID, h_angle)
        board.move(SERVO_V_ID, v_angle)

        return True

    except Exception as e:
        print(f"[ERROR] Servo move failed: {e}")
        return False


def servo_move_to(board, h=1, v=1,
                  h_delta=DEFAULT_H_THETA, v_delta=DEFAULT_V_THETA,
                  neutral_h=H_NEUTRAL, neutral_v=V_NEUTRAL,
                  wait_after_move=wait_after_move, cap=None, capture_after_move=False):

    if board is None or getattr(board, "ser", None) is None:
        print("[WARN] servo_move_to: board not initialized (no-servo).")
        return False, None

    if h not in (0, 1, 2) or v not in (0, 1, 2):
        print("[ERROR] servo_move_to: h and v must be 0,1,2")
        return False, None

    def map_angle(idx, neutral, delta):
        if idx == 1:
            return float(neutral)
        elif idx == 0:
            return float(max(0, neutral - delta))
        else:
            return float(min(180, neutral + delta))

    h_angle = map_angle(h, neutral_h, h_delta)
    v_angle = map_angle(v, neutral_v, v_delta)

    try:
        board.move(SERVO_H_ID, h_angle)
        board.move(SERVO_V_ID, v_angle)

        time.sleep(wait_after_move)

        captured = None
        if capture_after_move and cap is not None:
            ret, frame = cap.read()
            if ret and frame is not None:
                captured = frame.copy()

        return True, captured

    except Exception as e:
        print(f"[ERROR] servo_move_to failed: {e}")
        return False, None


def get_pose_list(is_top=False, is_bottom=False):
    Hs = [0, 1, 2]
    if not is_top and not is_bottom:
        return [(h, 1) for h in Hs]
    if is_top:
        return [(h, v) for h in Hs for v in (1, 2)]
    if is_bottom:
        return [(h, v) for h in Hs for v in (0, 1)]
    return [(h, 1) for h in Hs]


# ==========================================
# PART 3: DEBUG TERMINAL (UNCHANGED)
# ==========================================
def servo_debug_terminal(port, baud=115200):
    board = ServoBoard(port, baud)
    if not board.ser:
        print("[ERROR] Could not open servo terminal.")
        return

    print("\n====================== SERVO DEBUG TERMINAL ======================")
    print("SERVE / CALIBRATE:")
    print("  <id> <angle>        Move servo <id> to <angle> (0–180)")
    print("  <id> -1             Calibrate servo <id>")
    print("  P                   Read current servo positions")
    print("")
    print("LED CONTROL (RAW RGB):")
    print("  LED <i> R G B [brightness]")
    print("     Example: LED 0 255 0 0       # red")
    print("     Example: LED 1 0 255 0 150   # green @ brightness 150")
    print("")
    # print("LED CONTROL (COLOR NAMES):")
    # print("  LED <i> <color_name> [brightness]")
    # print("     Available colors:")
    # print("       r, g, b, y, w, cyan, magenta, orange, off")
    # print("       signal_yellow, signal_red, signal_blue, signal_green")
    # print("       signal_white, signal_black")
    # print("     Example: LED 0 red")
    # print("     Example: LED 1 signal_yellow 200")
    # print("")
    print("OTHER:")
    print("  Q                   Quit terminal")
    print("-------------------------------------------------------------------\n")

    while True:
        cmd = input("debug > ").strip().upper()

        if cmd in ["Q", "QUIT", "EXIT"]:
            break

        elif cmd == "P":
            print(board.get_positions())
            continue

        elif cmd.startswith("LED"):
            parts = cmd.split()
            if len(parts) not in (5, 6):
                print("Usage: LED index R G B [brightness]")
                continue

            try:
                index = int(parts[1])
                r = int(parts[2])
                g = int(parts[3])
                b = int(parts[4])
                brightness = int(parts[5]) if len(parts) == 6 else int(255*0.6)
                board.LED(index, r, g, b, brightness)
            except:
                print("[ERROR] LED command invalid.")
            continue

        else:
            parts = cmd.split()
            if len(parts) == 2:
                try:
                    s_id = int(parts[0])
                    angle = float(parts[1])
                    if angle == -1:
                        board.calibrate(s_id)
                    else:
                        board.move(s_id, angle)
                except:
                    print("[ERROR] Invalid command.")
            else:
                print("[ERROR] Format: ID ANGLE")

    board.close()
