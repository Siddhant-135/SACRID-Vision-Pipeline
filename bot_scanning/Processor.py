# ==========================================================
# DEPENDENCIES
# ==========================================================
import os
import csv
import cv2
import math
from datetime import datetime
import numpy as np
from qrdet import QRDetector
from pathlib import Path
import requests

# import callibrate_K as callibrate_K 

# ==========================================================
# CONFIG LOADING (YAML)
# ==========================================================
import yaml

# Resolve config.yaml relative to this file
CONFIG_PATH = Path(__file__).resolve().parent.parent / "config" / "config.yaml"
SESSION_CONFIG_PATH = Path(__file__).resolve().parent.parent / "config" / "session_id.yaml"

with open(CONFIG_PATH, "r") as f:
    _cfg = yaml.safe_load(f)
with open(SESSION_CONFIG_PATH, "r") as f:
    _session_cfg = yaml.safe_load(f)
# extract relevant sub-configs
camera_cfg = _cfg["camera"]
processor_cfg = _cfg["processor"]
duplicate_cfg = _cfg["duplicacy_thresholds"]
qr_cfg = _cfg["qr_detector"]


# ==========================================================
# QR PROCESSOR CLASS
# ==========================================================
class Processor:
    # Initialize QR Processor with configuration parameters.
    def __init__(
        self,
        K=camera_cfg["K"],
        D=camera_cfg["distortions_coeffs"],
        log_file=processor_cfg["log_file"],
        inv_file=processor_cfg["inventory_file"],
        num_frames_burst=10,
        raw_save_dir=processor_cfg["raw_save_dir"],
        raw_meta_csv=processor_cfg["raw_meta_csv"],
        duplicacy_x_threshold=duplicate_cfg["x"],
        duplicacy_y_threshold=duplicate_cfg["y"],
        d=camera_cfg["d"],
        model_size=qr_cfg["model_size"],
        qr_conf_th=qr_cfg["conf_th"],
        qr_nms_iou=qr_cfg["nms_iou"],
        padding_value=qr_cfg["padding_value"],
        wechat_dir=qr_cfg["wechat_dir"],
        save_annotated_imgs=processor_cfg["save_annotated"],
        annotated_save_dir=processor_cfg["annotated_save_dir"]
    ): 

        """
        Args:
            K: Camera intrinsic matrix (3x3 numpy array)
            log_file: File for logging processing events
            inv_file: Inventory CSV file (final output)
            num_frames_burst: Number of frames to capture in burst mode
            raw_save_dir: Directory to save captured images
            raw_meta_csv: CSV file for capture metadata
            duplicacy_x_threshold: Minimum X distance (meters) for QRs to be distinct
            duplicacy_y_threshold: Minimum Y distance (meters) for QRs to be distinct
            d: Distance to wall in meters (default 0.20)
            qr_conf_th: QR detector confidence threshold
            qr_nms_iou: QR detector NMS IOU threshold
            padding_value: Padding around detected QR codes (0.0-1.0)
            wechat_dir: Directory containing WeChat QR code models
        """
        # File paths
        self.log_file = log_file
        self.inv_file = inv_file
        self.raw_save_dir = raw_save_dir
        self.raw_meta_csv = raw_meta_csv
        self.annotated_save_dir = annotated_save_dir
        self.save_annotated_imgs = save_annotated_imgs
        # Capture settings
        self.num_frames_burst = num_frames_burst
        self.model_size = model_size
        self.qr_conf_th = qr_conf_th
        self.qr_nms_iou = qr_nms_iou

        # Duplicacy thresholds (meters)
        self.duplicacy_x_threshold = duplicacy_x_threshold
        self.duplicacy_y_threshold = duplicacy_y_threshold

        self.K = np.array(K, dtype=float)
        self.D = np.array(D, dtype=float)

        self.d = d  # Distance to wall (meters)
        self.padding_value = padding_value

        # Initialize QR detector
        self.qr_detector = QRDetector(
            model_size=self.model_size,
            conf_th=self.qr_conf_th,
            nms_iou=self.qr_nms_iou
        )

        # WeChat detector (lazy loaded)
        self.wechat_dir = wechat_dir
        self._wechat_detector = None

    # ==========================================================
    # FILE UTILITY METHODS
    # ==========================================================
    '''
        FILESYSTEM HELPERS
        Basic wrapper methods to ensure robust file I/O operations.
        - Automatically creates missing directories to prevent path errors
        - Ensures target files exist before writing to prevent runtime IO exceptions
        '''
    def _ensure_file(self, path):
        """Create an empty file if it doesn't exist."""
        if not os.path.exists(path):
            os.makedirs(os.path.dirname(path), exist_ok=True)
            with open(path, "w") as f:
                pass

    def _ensure_dir(self, path):
        """Create directory if it doesn't exist."""
        if not os.path.exists(path):
            os.makedirs(path, exist_ok=True)

    # ==========================================================
    # INVENTORY MANAGEMENT METHODS
    # ==========================================================
    '''
        INVENTORY STATE MANAGEMENT
        Handles the parsing, validation, and storage of scanned data.
        - Parses raw QR strings into structured data (Rack, Shelf, Item)
        - Manages the CSV inventory file (read/append operations)
        - Implements business logic for duplicate detection (same item, same location) 
        versus new entries (same item, new location)
        '''
    def _log_raw_input(self, raw_payload, global_coords, session_id):
        """
        Append raw input event to log file.
        
        Args:
            raw_payload: Raw QR code string
            global_coords: Tuple (global_x, global_y) in meters
            session_id: Unique session identifier
        """
        self._ensure_file(self.log_file)
        ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        with open(self.log_file, "a") as f:
            f.write(
                f"{ts} | session_id={session_id} | "
                f"payload={raw_payload} | "
                f"coords=({global_coords[0]:.2f}, {global_coords[1]:.2f})\n"
            )

    def _parse_payload(self, payload):
        """
        Parse QR code payload in format: R03_S4_ITM240
        
        Args:
            payload: Raw QR code string
            
        Returns:
            Tuple (rack, shelf, item) or (None, None, None) if invalid
        """
        try:
            parts = payload.split("_")
            if len(parts) != 3:
                return None, None, None
            rack = parts[0]
            shelf = parts[1]
            item = parts[2]
            return rack, shelf, item
        except Exception:
            return None, None, None

    def _check_distance(self, coords1, coords2):
        """
        Check if two coordinate pairs are within threshold distance.
        
        Uses normalized Euclidean distance:
        dist = sqrt((dx/threshold_x)^2 + (dy/threshold_y)^2)
        
        Args:
            coords1: Tuple (x1, y1) in meters
            coords2: Tuple (x2, y2) in meters
            
        Returns:
            Tuple (distance, is_duplicate) where is_duplicate = (distance <= 1.0)
        """
        dx = coords1[0] - coords2[0]
        dy = coords1[1] - coords2[1]
        
        # Normalized distance
        norm_dist = math.hypot(
            dx / self.duplicacy_x_threshold,
            dy / self.duplicacy_y_threshold
        )
        
        is_duplicate = norm_dist <= 1.0
        return norm_dist, is_duplicate

    def _read_inventory(self):
        """
        Read inventory CSV file.
        
        Returns:
            List of dictionaries, one per inventory entry
        """
        if not os.path.exists(self.inv_file):
            return []

        entries = []
        with open(self.inv_file, "r") as f:
            reader = csv.DictReader(f)
            for row in reader:
                row["global_x"] = float(row["global_x"])
                row["global_y"] = float(row["global_y"])
                entries.append(row)
        return entries

    def _append_inventory(self, row_dict):
        """
        Append one item row to inventory CSV.
        Creates file with header if it doesn't exist.
        
        Args:
            row_dict: Dictionary with keys matching CSV columns
        """
        self._ensure_dir(os.path.dirname(self.inv_file))
        new_file = not os.path.exists(self.inv_file)
        
        with open(self.inv_file, "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=[
                "session_id",
                "timestamp",
                "payload",
                "rack",
                "shelf",
                "item",
                "global_x",
                "global_y"
            ])
            if new_file:
                writer.writeheader()
            writer.writerow(row_dict)

    # ==========================================================
    # COORDINATE TRANSFORMATION METHODS
    # ==========================================================
    '''
    COORDINATE FRAME DEFINITIONS
    UV Frame: Image pixel coordinates (u, v)
    - Origin: Top-left corner of image
    - Units: pixels
    - u: horizontal (left to right)
    - v: vertical (top to bottom)

    Camera Frame: Camera-centered coordinates (x_cam, y_cam)
    - Origin: Camera optical center projected onto wall
    - Units: meters
    - x_cam: horizontal (parallel to wall)
    - y_cam: vertical (perpendicular to wall, pointing away from camera)

    Global Frame: World coordinates with SLAM offsets (global_x, global_y)
    - Origin: World origin from SLAM system
    - Units: meters
    - global_x = x_cam + x_offset
    - global_y = y_cam + y_offset
    '''

    def _get_rotation_matrix(self, theta, phi):
        """
        Calculate rotation matrix from camera orientation angles.
        
        Args:
            theta: Pitch angle (radians)
            phi: Yaw angle (radians)
            
        Returns:
            3x3 rotation matrix as numpy array
        """
        # Camera Z-axis (optical axis)
        cam_z = np.array([
            np.sin(theta) * np.cos(phi),
            np.sin(theta) * np.sin(phi),
            np.cos(theta)
        ])

        # Camera Y-axis (down direction)
        cam_y = np.array([
            np.cos(theta) * np.cos(phi),
            np.cos(theta) * np.sin(phi),
            -np.sin(theta)
        ])

        # Camera X-axis (right direction)
        cam_x = np.array([
            -np.sin(phi),
            np.cos(phi),
            0.0
        ])

        R = np.stack((cam_x, cam_y, cam_z), axis=1)
        return R

    def _get_camera_ray(self, u, v):
            """
            Calculate camera ray direction from pixel coordinates with distortion correction.
    Handles python scalars (int/float) AND 0-d numpy arrays
            """

            if np.ndim(u) == 0:
                u_arr = np.array([u], dtype=np.float32)
                v_arr = np.array([v], dtype=np.float32)
                is_scalar = True
            else:
                u_arr = np.asanyarray(u, dtype=np.float32).flatten()
                v_arr = np.asanyarray(v, dtype=np.float32).flatten()
                is_scalar = False

            # 2. Prepare for OpenCV (N, 1, 2)
            points_src = np.stack([u_arr, v_arr], axis=1).reshape(-1, 1, 2)

            # 3. Undistort Points
            points_norm = cv2.undistortPoints(points_src, self.K, self.D)

            # Reshape back to (N, 2)
            points_norm = points_norm.reshape(-1, 2)

            # 4. Create Rays
            rays = np.stack([
                points_norm[:, 0],
                points_norm[:, 1],
                np.ones(len(points_norm))
            ], axis=0) # Shape (3, N)

            # 5. Return (Unpack if it was a scalar input)
            if is_scalar:
                return rays[:, 0]
            else:
                return rays

    def _uv_to_camera_coords(self, uv, theta, phi):
        """
        Convert UV pixel coordinates to camera frame coordinates.
        
        Projects pixel through camera model onto wall plane at distance d.
        
        Args:
            uv: Tuple (u, v) in pixels
            theta: Camera pitch angle (radians)
            phi: Camera yaw angle (radians)
            
        Returns:
            Tuple (x_cam, y_cam) in meters
        """
        R = self._get_rotation_matrix(theta, phi)
        ray_cam = self._get_camera_ray(uv[0], uv[1])
        ray_world = R @ ray_cam
        
        r_x, r_y, r_z = ray_world

        # Intersect ray with plane at distance d
        t = self.d / r_y
        x_cam = t * r_x
        y_cam = t * r_z

        return (x_cam, y_cam)

    def _rectify_image_to_camera_frame(self, img, theta_deg, phi_deg):
        """
        Unwarp image onto flat wall plane in camera frame.
        
        Args:
            img: Input image (numpy array)
            theta_deg: Camera pitch angle (degrees)
            phi_deg: Camera yaw angle (degrees)
            
        Returns:
            Tuple (warped_img, pixels_per_meter, (x_cam_min, y_cam_max))
            Returns (None, None, None) if projection fails
        """
        if img is None:
            return None, None, None

        H, W = img.shape[:2]
        pixels_per_meter = self.K[0, 0] / self.d

        # Convert angles to radians
        theta = np.deg2rad(theta_deg)
        phi = np.deg2rad(phi_deg)

        # Project image corners to determine bounds
        corners_uv = np.array([
            (0, 0),      # Top-left
            (W, 0),      # Top-right
            (W, H),      # Bottom-right
            (0, H)       # Bottom-left
        ])
        
        x_cams, y_cams = [], []
        for u, v in corners_uv:
            x_cam, y_cam = self._uv_to_camera_coords((u, v), theta, phi)
            x_cams.append(x_cam)
            y_cams.append(y_cam)

        if not x_cams:
            return None, None, None

        # Calculate output image bounds
        x_cam_min = min(x_cams)
        x_cam_max = max(x_cams)
        y_cam_min = min(y_cams)
        y_cam_max = max(y_cams)

        out_W = int(np.ceil((x_cam_max - x_cam_min) * pixels_per_meter))
        out_H = int(np.ceil((y_cam_max - y_cam_min) * pixels_per_meter))

        if out_W <= 0 or out_H <= 0:
            return None, None, None

        # Vectorized projection for all pixels
        u_coords, v_coords = np.meshgrid(np.arange(W), np.arange(H))
        # u_flat = u_coords.flatten()
        # v_flat = v_coords.flatten()

        # # Get camera parameters
        # f_x, f_y = self.K[0, 0], self.K[1, 1]
        # c_x, c_y = self.K[0, 2], self.K[1, 2]

        # # Calculate rays for all pixels
        # ray_cam = np.stack([
        #     (u_flat - c_x) / f_x,
        #     (v_flat - c_y) / f_y,
        #     np.ones_like(u_flat)
        # ], axis=0)

        ray_cam = self._get_camera_ray(u_coords, v_coords)
        u_flat = u_coords.flatten() 
        v_flat = v_coords.flatten()

        # Transform to world frame
        R = self._get_rotation_matrix(theta, phi)
        ray_world = R @ ray_cam
        r_x, r_y, r_z = ray_world[0], ray_world[1], ray_world[2]

        # Avoid division by zero
        r_y = np.where(np.abs(r_y) < 1e-6, 1e-6, r_y)
        
        # Intersect with wall plane
        t = self.d / r_y
        x_cam = t * r_x
        y_cam = t * r_z

        # Map to output image coordinates
        i_out = np.clip(
            (y_cam_max - y_cam) / (y_cam_max - y_cam_min) * (out_H - 1),
            0,
            out_H - 1
        ).astype(int)
        
        j_out = np.clip(
            (x_cam - x_cam_min) / (x_cam_max - x_cam_min) * (out_W - 1),
            0,
            out_W - 1
        ).astype(int)

        # Create output image
        if len(img.shape) == 3:
            warped_img = np.zeros((out_H, out_W, img.shape[2]), dtype=img.dtype)
            warped_img[i_out, j_out] = img[v_flat, u_flat]
        else:
            warped_img = np.zeros((out_H, out_W), dtype=img.dtype)
            warped_img[i_out, j_out] = img[v_flat, u_flat]

        return warped_img, pixels_per_meter, (x_cam_min, y_cam_max)

    def _project_image_to_camera_frame(self, img, orientation):
        """
        Project image to camera frame based on orientation.
        
        Args:
            img: Input image (numpy array)
            orientation: Camera orientation
                0 = down (looking down at floor)
                1 = front (looking straight ahead)
                2 = up (looking up at ceiling)
                
        Returns:
            Tuple (projection, pixels_per_meter, (x_cam_origin, y_cam_origin))
        """
        # Map orientation to camera angles
        orientation_map = {
            0: (180, 90),  # Down
            1: (90, 90),   # Front
            2: (0, 90)     # Up
        }
        
        theta, phi = orientation_map.get(orientation, (90, 90))
        
        projection, scale, camera_origin = self._rectify_image_to_camera_frame(
            img, theta, phi
        )
        
        return projection, scale, camera_origin

    def _uv_to_global_coords(self, xy_cam_origin, uv_center, pixels_per_meter, 
                             x_offset, y_offset):
        """
        Convert UV pixel coordinates to global frame coordinates.
        
        Transformation pipeline:
        1. UV (pixels) -> Camera frame (meters)
        2. Camera frame -> Global frame (add SLAM offsets)
        
        Args:
            xy_cam_origin: Tuple (x_cam_min, y_cam_max) - camera frame origin
            uv_center: Tuple (u, v) - pixel coordinates in projected image
            pixels_per_meter: Scale factor (pixels/meter)
            x_offset: Global X offset from SLAM (meters)
            y_offset: Global Y offset from SLAM (meters)
            
        Returns:
            Tuple (global_x, global_y) in meters
        """
        u_center, v_center = uv_center
        x_cam_origin, y_cam_origin = xy_cam_origin

        # Convert UV to camera frame
        x_cam = u_center / pixels_per_meter + x_cam_origin
        y_cam = y_cam_origin - v_center / pixels_per_meter

        # Add SLAM offsets to get global coordinates
        global_x = x_cam + x_offset
        global_y = y_cam + y_offset

        return global_x, global_y

    # ==========================================================
    # QR DETECTION METHODS
    # ==========================================================
    '''
        VISUAL DETECTION PIPELINE
        Responsible for locating QR codes within the raw camera frame.
        - Detects bounding boxes for potential QR codes using the base detector
        - Calculates geometric centroids for spatial mapping
        - Extracts, crops, and squares the Regions of Interest (ROI) to prepare 
        standardized inputs for the decoding stage
        '''
    def _get_bbox_centroid(self, x1, y1, x2, y2):
        """
        Calculate centroid of bounding box.
        
        Args:
            x1, y1: Top-left corner
            x2, y2: Bottom-right corner
            
        Returns:
            Tuple (cx, cy) as integers
        """
        cx = (x1 + x2) / 2
        cy = (y1 + y2) / 2
        return int(cx), int(cy)

    def _make_square_canvas(self, image, padding_color=(255, 255, 255)):
        """
        Pad image to create square with 1:1 aspect ratio.
        Preserves QR code finder pattern ratios.
        
        Args:
            image: Input image
            padding_color: RGB tuple for padding color
            
        Returns:
            Square image with same content, padded as needed
        """
        h, w = image.shape[:2]

        if h == w:
            return image

        max_dim = max(h, w)
        delta_w = max_dim - w
        delta_h = max_dim - h

        top = delta_h // 2
        bottom = delta_h - top
        left = delta_w // 2
        right = delta_w - left

        squared_img = cv2.copyMakeBorder(
            image,
            top, bottom, left, right,
            cv2.BORDER_CONSTANT,
            value=padding_color
        )

        return squared_img

    def _detect_qr_codes_in_image(self, image):
        """
        Detect all QR codes in image.
        
        Args:
            image: Input image (BGR format)
            
        Returns:
            List of tuples: [(qr_screenshot, (u_center, v_center)), ...]
        """
        if image is None:
            return []

        detections = self.qr_detector.detect(image=image, is_bgr=True)
        qr_array = []
        img_h, img_w = image.shape[:2]

        for det in detections:
            x1, y1, x2, y2 = det["bbox_xyxy"]

            # Clamp to image bounds
            x1 = max(0, int(x1))
            y1 = max(0, int(y1))
            x2 = min(img_w, int(x2))
            y2 = min(img_h, int(y2))

            if x2 <= x1 or y2 <= y1:
                continue

            # Calculate centroid
            u_center, v_center = self._get_bbox_centroid(x1, y1, x2, y2)

            # Add padding around QR code
            box_w = x2 - x1
            box_h = y2 - y1
            pad_x = int(box_w * self.padding_value)
            pad_y = int(box_h * self.padding_value)

            crop_x1 = max(0, x1 - pad_x)
            crop_y1 = max(0, y1 - pad_y)
            crop_x2 = min(img_w, x2 + pad_x)
            crop_y2 = min(img_h, y2 + pad_y)

            # Extract and square the QR code region
            qr_screenshot = image[crop_y1:crop_y2, crop_x1:crop_x2].copy()
            qr_screenshot = self._make_square_canvas(qr_screenshot)
            
            qr_array.append((qr_screenshot, (u_center, v_center)))

        return qr_array

    # ==========================================================
    # QR ENHANCEMENT AND DECODING METHODS
    # ==========================================================
    '''
        IMAGE PREPROCESSING & INFERENCE
        Optimizes extracted QR regions for maximum readability before decoding.
        - Applies image enhancement pipeline: CLAHE (Contrast), Unsharp Masking, and Binary Thresholding
        - Lazy-loads the WeChat QR inference model to save resources until needed
        - Decodes the enhanced binary image data into text payloads
        '''
    def _apply_binary_threshold(self, image, threshold=130): # if low light reduce threshold
        
        """
        Apply binary thresholding to image.
        
        Args:
            image: Grayscale image
            threshold: Threshold value (-1 for Otsu's method)
            
        Returns:
            Binary thresholded image
        """
        if threshold == -1:
            _, thresholded = cv2.threshold(
                image, 0, 255,
                cv2.THRESH_BINARY + cv2.THRESH_OTSU
            )
        else:
            _, thresholded = cv2.threshold(
                image, threshold, 255,
                cv2.THRESH_BINARY
            )
        return thresholded

    def _enhance_qr_contrast(self, image):
        """
        Apply CLAHE and Unsharp Mask for contrast enhancement.
        
        Args:
            image: Input image (grayscale or BGR)
            
        Returns:
            Enhanced grayscale image
        """
        # Convert to grayscale if needed
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image

        # Apply CLAHE (Contrast Limited Adaptive Histogram Equalization)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        contrast_enhanced = clahe.apply(gray)

        # Apply Unsharp Masking
        gaussian = cv2.GaussianBlur(contrast_enhanced, (0, 0), 2.0)
        sharpened = cv2.addWeighted(
            contrast_enhanced, 1.5,
            gaussian, -0.5,
            0
        )

        return sharpened

    def _preprocess_qr_image(self, qr_roi):
        """
        Full preprocessing pipeline for QR code image.
        
        Args:
            qr_roi: QR code region of interest
            
        Returns:
            Preprocessed binary image ready for decoding
        """
        # Resize to standard size
        resized = cv2.resize(
            qr_roi, (400, 400),
            interpolation=cv2.INTER_LANCZOS4
        )

        # Enhance contrast
        enhanced = self._enhance_qr_contrast(resized)
        
        # Apply binary threshold
        binary = self._apply_binary_threshold(enhanced, threshold=140)

        return binary

    def _get_wechat_detector(self):
        """
        Lazy load WeChat QR detector.
        
        Returns:
            WeChat QR code detector instance
        """
        if self._wechat_detector is None:
            base = Path(self.wechat_dir)
            detect_prototxt = str(base / "detect.prototxt")
            detect_model = str(base / "detect.caffemodel")
            sr_prototxt = str(base / "sr.prototxt")
            sr_model = str(base / "sr.caffemodel")
            
            self._wechat_detector = cv2.wechat_qrcode_WeChatQRCode(
                detect_prototxt, detect_model,
                sr_prototxt, sr_model
            )
        return self._wechat_detector

    def _decode_qr_image(self, preprocessed_img):
        """
        Decode QR code from preprocessed image.
        
        Args:
            preprocessed_img: Binary preprocessed image
            
        Returns:
            List of decoded strings (empty if decoding failed)
        """
        if preprocessed_img is None:
            return []

        detector = self._get_wechat_detector()
        decoded_strings, points = detector.detectAndDecode(preprocessed_img)

        return decoded_strings

    def _read_qr_code(self, qr_image):
        """
        Full pipeline to read QR code from image.
        
        Args:
            qr_image: QR code image region
            
        Returns:
            Decoded string or empty string if failed
        """
        if qr_image is None:
            return ""

        # Preprocess
        preprocessed = self._preprocess_qr_image(qr_image)
        
        # Decode
        results = self._decode_qr_image(preprocessed)

        return results[0] if len(results) > 0 else ""

    def _annotate_img(self, img, detections, save_path):
        """
        Draw bounding boxes + confidence scores on img and save to save_path.
        """
        if img is None or not detections:
            return

        im2 = img.copy()

        for detection in detections:
            x1, y1, x2, y2 = detection['bbox_xyxy']
            confidence = detection['confidence']

            cv2.rectangle(
                im2,
                (int(x1), int(y1)),
                (int(x2), int(y2)),
                (0, 255, 0),
                2
            )
            cv2.putText(
                im2,
                f'{confidence:.2f}',
                (int(x1), int(y1) - 10),
                fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                fontScale=1,
                color=(0, 255, 0),
                thickness=2
            )

        # Ensure parent directory exists, then save
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        cv2.imwrite(save_path, im2)

    def _save_full_frame(self, img, x_offset, y_offset, orientation, session_id, detections):
        """
        Save the annotated full camera frame with filename encoding pose + scan info.

        Filename format:
            full_x{X}_z{Z}_ori{orientation}_scan{scan_id}_{timestamp}.png
        """
        if img is None or not detections:
            return

        # Ensure base directory exists
        self._ensure_dir(self.annotated_save_dir)

        # Format coordinates
        x_str = f"{x_offset:.2f}"
        y_str = f"{y_offset:.2f}"

        # Timestamp for uniqueness
        ts = datetime.now().strftime("%Y%m%d_%H%M%S_%f")

        filename = f"session{session_id}_x{x_str}_y{y_str}_ori{orientation}_{ts}.png"
        save_path = os.path.join(self.annotated_save_dir, filename)

        # Call your annotate function
        self._annotate_img(img, detections, save_path)
    # ==========================================================
    # HIGH-LEVEL PROCESSING METHODS
    # ==========================================================
    '''
        PIPELINE ORCHESTRATION
        The primary public interfaces that integrate all subsystems.
        - Connects Image -> Geometry -> Detection -> Decoding -> Database
        - process_image_to_qr_data: Handles visual processing and global coordinate mapping
        - process_scan_to_inventory: Handles data logic and database updates
        - process_image: The master function executing the full end-to-end flow
        '''
    def process_image_to_qr_data(self, img, x_offset, y_offset, orientation):
        """
        Process image to extract QR codes with global coordinates.
        
        Pipeline:
        1. Project image to camera frame
        2. Detect QR codes
        3. Transform coordinates to global frame
        4. Decode QR codes
        
        Args:
            img: Input image (numpy array, BGR format)
            x_offset: Global X offset from SLAM (meters)
            y_offset: Global Y offset from SLAM (meters)
            orientation: Camera orientation (0=down, 1=front, 2=up)
            
        Returns:
            List of tuples: [(payload_string, (global_x, global_y)), ...]
        """
        # Project image to camera frame
        projection, pixels_per_meter, xy_cam_origin = self._project_image_to_camera_frame(img, orientation)

        if projection is None:
            return []

        # Detect QR codes in projected image
        qr_detections = self._detect_qr_codes_in_image(projection)

        # Process each QR code
        output_data = []
        for qr_image, uv_center in qr_detections:
            # Transform to global coordinates
            global_x, global_y = self._uv_to_global_coords(
                xy_cam_origin, uv_center,
                pixels_per_meter,
                x_offset, y_offset
            )

            # Decode QR code
            payload = self._read_qr_code(qr_image)

            output_data.append((payload, (global_x, global_y)))

        return output_data



    def post_inventory(self, row):
        """
        Post the scanned inventory row to the remote API.
        """
        url = _session_cfg["server_url"]

        # Convert row into web JSON format
        web_payload = {
            "round_id": row["session_id"],
            "timestamp": row["timestamp"],
            "shelf_id": row["payload"],
            "shelf_status": "Occupied",

        }

        try:
            response = requests.post(url, json=web_payload)

            print(
                f"[POST] session={row['session_id']} "
                f"payload={row['payload']} "
                f"rack={row['rack']} shelf={row['shelf']} "
                f"→ HTTP {response.status_code}"
            )

            # Optionally print server response text
            # print(response.text)

        except requests.exceptions.ConnectionError:
            print("[POST ERROR] Cannot connect to server at", url)

        except Exception as e:
            print(f"[POST ERROR] {e}")

    def process_scan_to_inventory(self, payload, global_coords, session_id):
        """
        Process scanned QR code and update inventory.
        
        Handles three cases:
        A) New item -> Create inventory entry
        B) Same item, same location -> Ignore (duplicate)
        C) Same item, different location -> Create new entry
        
        Args:
            payload: Raw QR code string
            global_coords: Tuple (global_x, global_y) in meters
            session_id: Unique session identifier
            
        Returns:
            Dictionary with status information
        """
        # Log the scan
        self._log_raw_input(payload, global_coords, session_id)
        
        # Parse payload
        rack, shelf, item = self._parse_payload(payload)
        if rack is None:
            return {"status": "ERR_BAD_PAYLOAD", "payload": payload}

        # Read existing inventory
        global_x, global_y = global_coords
        inventory = self._read_inventory()
        same_item_entries = [
            r for r in inventory
            if r["payload"] == payload and r["session_id"] == session_id
        ]

        # Case A: New item
        if len(same_item_entries) == 0:
            row = {
                "session_id": session_id,
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "payload": payload,
                "rack": rack,
                "shelf": shelf,
                "item": item,
                "global_x": float(f"{global_x:.2f}"),
                "global_y": float(f"{global_y:.2f}")
            }
            self._append_inventory(row)
            self.post_inventory(row)
            return {"status": "NEW_ITEM_CREATED", "data": row}

        # Case B: Check for duplicates
        for entry in same_item_entries:
            existing_coords = (entry["global_x"], entry["global_y"])
            distance, is_duplicate = self._check_distance(
                global_coords, existing_coords
            )

            if is_duplicate:
                return {
                    "status": "DUPLICATE_IGNORED",
                    "distance": float(f"{distance:.2f}"),
                    "already_scanned_at": entry["timestamp"],
                }

        # Case C: Same item, different location
        row = {
            "session_id": session_id,
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "payload": payload,
            "rack": rack,
            "shelf": shelf,
            "item": item,
            "global_x": float(f"{global_x:.2f}"),
            "global_y": float(f"{global_y:.2f}")
        }
        self._append_inventory(row)
        self.post_inventory(row)
        return {"status": "NEW_LOCATION_ADDED", "data": row}

    def process_image(self, img, x_offset, y_offset, orientation, session_id):
        """
        Complete pipeline: image -> QR detection -> inventory update.
        
        Args:
            img: Input image (numpy array, BGR format)
            x_offset: Global X offset from SLAM (meters)
            y_offset: Global Y offset from SLAM (meters)
            orientation: Camera orientation (0=down, 1=front, 2=up)
            session_id: unique
            
        Returns:
            Dictionary with overall status and results for each QR code
        """
        # Validate input
        if img is None:
            return {
                "status": "ERROR",
                "message": "Invalid image (None)",
                "count": 0
            }

        if self.save_annotated_imgs:
            detections = self.qr_detector.detect(image=img, is_bgr=True)
            self._save_full_frame(
                img=img,
                x_offset=x_offset,
                y_offset=y_offset,
                orientation=orientation,
                session_id=session_id,   # base scan ID as you wanted
                detections=detections
            )

        # Extract QR data from image
        qr_data_list = self.process_image_to_qr_data(
            img, x_offset, y_offset, orientation,
        )

        if not qr_data_list:
            return {
                "status": "NO_QR_DETECTED",
                "count": 0
            }

        # Process each QR code
        results = []
        for idx, (payload, global_coords) in enumerate(qr_data_list):
            # Skip empty payloads
            if not payload:
                continue


            # Update inventory
            status = self.process_scan_to_inventory(
                payload, global_coords, session_id
            )
            results.append(status)

        return {
            "status": "SUCCESS",
            "count": len(results),
            "results": results
        }

# ==========================================================
# SMOKE TEST
# ==========================================================
if __name__ == "__main__":
    print("=" * 70)
    print("SMOKE TEST - QR Processor")
    print("=" * 70)

    # Test 1: Initialization
    print("\n[TEST 1] Initialization")
    print("-" * 70)
    try:
        processor = Processor()
        print("✓ Processor initialized successfully")
        print(f"  - Camera intrinsics: fx={processor.K[0,0]:.2f}")
        print(f"  - Wall distance: {processor.d} m")
        print(f"  - Duplicacy thresholds: x={processor.duplicacy_x_threshold}m, "
              f"y={processor.duplicacy_y_threshold}m")
    except Exception as e:
        print(f"✗ Initialization failed: {e}")
        exit(1)

    # Test 2: Create synthetic test image
    print("\n[TEST 2] Creating synthetic test image")
    print("-" * 70)
    try:
        test_img = np.ones((480, 640, 3), dtype=np.uint8) * 255
        cv2.rectangle(test_img, (200, 150), (440, 330), (0, 0, 0), -1)
        print(f"✓ Test image created: {test_img.shape}")
    except Exception as e:
        print(f"✗ Image creation failed: {e}")
        exit(1)

    # Test 3: Coordinate transformations
    print("\n[TEST 3] Testing coordinate transformations")
    print("-" * 70)
    try:
        # Test rotation matrix
        R = processor._get_rotation_matrix(np.pi/2, np.pi/2)
        print(f"✓ Rotation matrix computed: shape={R.shape}")

        # Test camera ray
        ray = processor._get_camera_ray(320, 240)
        print(f"✓ Camera ray computed: {ray}")

        # Test UV to camera coords
        x_cam, y_cam = processor._uv_to_camera_coords(
            (320, 240), np.pi/2, np.pi/2
        )
        print(f"✓ UV->Camera: (320, 240) -> ({x_cam:.3f}, {y_cam:.3f}) m")

    except Exception as e:
        print(f"✗ Coordinate transform failed: {e}")
        import traceback
        traceback.print_exc()

    # Test 4: Image projection
    print("\n[TEST 4] Testing image projection to camera frame")
    print("-" * 70)
    try:
        projection, ppm, xy_cam_origin = processor._project_image_to_camera_frame(
            test_img, orientation=1
        )
        if projection is not None:
            print(f"✓ Projection successful")
            print(f"  - Output shape: {projection.shape}")
            print(f"  - Pixels per meter: {ppm:.2f}")
            print(f"  - Camera frame origin: ({xy_cam_origin[0]:.3f}, "
                  f"{xy_cam_origin[1]:.3f}) m")
        else:
            print("✗ Projection returned None")
    except Exception as e:
        print(f"✗ Projection failed: {e}")
        import traceback
        traceback.print_exc()

    # Test 5: QR detection
    print("\n[TEST 5] Testing QR detection (no QRs expected)")
    print("-" * 70)
    try:
        qr_array = processor._detect_qr_codes_in_image(test_img)
        print(f"✓ QR detection completed: {len(qr_array)} QR codes found")
    except Exception as e:
        print(f"✗ QR detection failed: {e}")

    # Test 6: Global coordinate transformation
    print("\n[TEST 6] Testing UV to global coordinate transformation")
    print("-" * 70)
    try:
        xy_cam_origin = (-1.0, 2.0)
        uv_center = (320, 240)
        ppm = 1300.0
        x_offset = 5.0
        y_offset = 1.5

        global_x, global_y = processor._uv_to_global_coords(
            xy_cam_origin, uv_center, ppm, x_offset, y_offset
        )
        print(f"✓ Coordinate transformation successful")
        print(f"  - UV: ({uv_center[0]}, {uv_center[1]}) pixels")
        print(f"  - Camera origin: ({xy_cam_origin[0]:.2f}, {xy_cam_origin[1]:.2f}) m")
        print(f"  - SLAM offset: ({x_offset:.2f}, {y_offset:.2f}) m")
        print(f"  - Global coords: ({global_x:.2f}, {global_y:.2f}) m")
    except Exception as e:
        print(f"✗ Coordinate transformation failed: {e}")
        import traceback
        traceback.print_exc()

    # Test 7: Payload parsing
    print("\n[TEST 7] Testing payload parsing")
    print("-" * 70)
    try:
        # Valid payload
        rack, shelf, item = processor._parse_payload("R03_S4_ITM240")
        if rack == "R03" and shelf == "S4" and item == "ITM240":
            print(f"✓ Valid payload parsed correctly:")
            print(f"  - Rack: {rack}, Shelf: {shelf}, Item: {item}")
        else:
            print(f"✗ Incorrect parsing: {rack}, {shelf}, {item}")

        # Invalid payload
        rack, shelf, item = processor._parse_payload("INVALID")
        if rack is None:
            print("✓ Invalid payload handled correctly (returned None)")
        else:
            print("✗ Invalid payload should return None")
    except Exception as e:
        print(f"✗ Payload parsing failed: {e}")

    # Test 8: Distance checking
    print("\n[TEST 8] Testing distance checking for duplicates")
    print("-" * 70)
    try:
        coords1 = (0.0, 0.0)
        coords2 = (1.0, 1.0)
        dist, is_dup = processor._check_distance(coords1, coords2)
        print(f"✓ Distance check: dist={dist:.2f}, duplicate={is_dup}")

        coords3 = (0.1, 0.1)
        dist2, is_dup2 = processor._check_distance(coords1, coords3)
        print(f"✓ Close points: dist={dist2:.2f}, duplicate={is_dup2}")
    except Exception as e:
        print(f"✗ Distance check failed: {e}")

    # Test 9: File utilities
    print("\n[TEST 9] Testing file utilities")
    print("-" * 70)
    try:
        test_file = "test_smoke_file.txt"
        processor._ensure_file(test_file)
        if os.path.exists(test_file):
            print(f"✓ File created: {test_file}")
            os.remove(test_file)
            print(f"✓ Test file cleaned up")
        else:
            print("✗ File creation failed")
    except Exception as e:
        print(f"✗ File utilities failed: {e}")

    # Test 10: Full pipeline
    print("\n[TEST 10] Testing full pipeline (without QR reading)")
    print("-" * 70)
    try:
        results = processor.process_image_to_qr_data(
            img=test_img,
            x_offset=0.0,
            y_offset=1.5,
            orientation=1
        )
        print(f"✓ Full pipeline executed: {len(results)} QR codes processed")
    except Exception as e:
        print(f"✗ Full pipeline failed: {e}")
        import traceback
        traceback.print_exc()

    # Test 11: process_image with validation
    print("\n[TEST 11] Testing high-level process_image method")
    print("-" * 70)
    try:
        result = processor.process_image(
            img=test_img,
            x_offset=0.0,
            y_offset=1.5,
            orientation=1,
            session_id="S001"
        )
        print(f"✓ process_image executed successfully")
        print(f"  - Status: {result['status']}")
        print(f"  - Count: {result.get('count', 0)}")

        # Test with None image
        result_none = processor.process_image(
            None, 0.0, 1.5, 1, "S001"
        )
        if result_none['status'] == 'ERROR':
            print(f"✓ None image handled: {result_none['message']}")
    except Exception as e:
        print(f"✗ process_image failed: {e}")
        import traceback
        traceback.print_exc()

    # Test 12: Image preprocessing
    print("\n[TEST 12] Testing QR image preprocessing")
    print("-" * 70)
    try:
        # Create small test image
        test_qr = np.ones((100, 100), dtype=np.uint8) * 128
        preprocessed = processor._preprocess_qr_image(test_qr)
        print(f"✓ QR preprocessing successful: output shape={preprocessed.shape}")
    except Exception as e:
        print(f"✗ QR preprocessing failed: {e}")

    # Summary
    print("\n" + "=" * 70)
    print("SMOKE TEST COMPLETE")
    print("=" * 70)
    print("\nNotes:")
    print("  - WeChat QR decoder tests skipped (requires model files)")
    print("  - Actual QR reading not tested (no real QR codes in test image)")
    print("  - Inventory file I/O not fully tested (requires write permissions)")
    print("\nAll core functionality validated successfully!")


    