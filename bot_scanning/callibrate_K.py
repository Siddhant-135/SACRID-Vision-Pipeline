import numpy as np
import cv2 as cv
import glob
import sys
import csv

# termination criteria for cornerSubPix
criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)

# YOUR BOARD: 9 x 7 squares => inner corners = (8, 6)
pattern_size = (8, 6)   # (cols, rows) = inner corners

# prepare object points based on pattern_size (0..7, 0..5)
square_size = 20.0  # e.g., 25 mm
objp = np.zeros((pattern_size[0] * pattern_size[1], 3), np.float32)
objp[:, :2] = np.mgrid[0:pattern_size[0], 0:pattern_size[1]].T.reshape(-1, 2)
objp[:, :2] *= square_size

# storage for 3D points (objpoints) and 2D image points (imgpoints)
objpoints = []
imgpoints = []
successful_filenames = []  # To track which files contributed to calibration

images = glob.glob('calibration_images/*.jpg')
if len(images) == 0:
    print("No images found in calibration_images/*.jpg — check path or working directory.")
    sys.exit(1)

print(f"Found {len(images)} files. Running detection using pattern_size={pattern_size}...")

successful_image_shape = None
count = 0

for fname in images:
    img = cv.imread(fname)
    if img is None:
        print("WARN: Failed to read", fname)
        continue

    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

    # Try the standard finder first. If it fails often, consider findChessboardCornersSB.
    ret, corners = cv.findChessboardCorners(
        gray, pattern_size,
        flags=cv.CALIB_CB_ADAPTIVE_THRESH + cv.CALIB_CB_NORMALIZE_IMAGE
    )

    if not ret:
        # optional: try the SB (more robust) variant
        ret_sb, corners_sb = cv.findChessboardCornersSB(gray, pattern_size)
        if ret_sb:
            ret = True
            corners = corners_sb

    if ret:
        count += 1
        # append a copy of objp so shapes line up exactly
        objpoints.append(objp.copy())

        # refine to subpixel accuracy
        corners2 = cv.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
        imgpoints.append(corners2)
        
        # Save the filename to match with errors later
        successful_filenames.append(fname)

        # draw and show for debug
        cv.drawChessboardCorners(img, pattern_size, corners2, ret)
        cv.imshow('detected', img)
        cv.waitKey(150)

        # store one successful image shape for calibration
        if successful_image_shape is None:
            successful_image_shape = gray.shape[::-1]  # (width, height)
    else:
        # optional: save/debug a failing image filename
        print("No corners in:", fname)

cv.destroyAllWindows()
print("Number of successful detections:", count)

if count < 3:
    print("Need at least a few successful detections (usually 8-15+) for good calibration.")
    sys.exit(1)

# Use the shape from a successful image
if successful_image_shape is None:
    print("No successful images to calibrate.")
    sys.exit(1)

w, h = successful_image_shape

# debug: check shapes
print("objpoints[0].shape:", objpoints[0].shape)
print("imgpoints[0].shape:", imgpoints[0].shape)

print("\nRunning calibration...")
# calibrate
ret, mtx, dist, rvecs, tvecs = cv.calibrateCamera(objpoints, imgpoints, (w, h), None, None)
print("Calibration success flag (RMS):", ret)
print("Camera matrix (intrinsic):\n", mtx)
print("\nDistortion coefficients:\n", dist.ravel())

# optional: get optimal new camera matrix and undistort one image to inspect result
newcameramtx, roi = cv.getOptimalNewCameraMatrix(mtx, dist, (w, h), 1, (w, h))

# undistort the first image that was successfully read
test_img_path = successful_filenames[0]
test_img = cv.imread(test_img_path)
if test_img is not None:
    dst = cv.undistort(test_img, mtx, dist, None, newcameramtx)
    x, y, wi, hi = roi
    dst_cropped = dst[y:y+hi, x:x+wi]
    cv.imwrite('calibresult.png', dst_cropped)
    print("Wrote sample undistorted image to calibresult.png")

# ---------------------------------------------------------
# Reprojection Error Calculation & CSV Output
# ---------------------------------------------------------
csv_filename = "calibration_results.csv"
print(f"\nCalculating reprojection errors and writing to {csv_filename}...")

total_error = 0
per_image_errors = []

# Open CSV for writing (overwrites existing file)
with open(csv_filename, mode='w', newline='') as csv_file:
    writer = csv.writer(csv_file)
    
    # Write Header
    writer.writerow(["Image Index", "Image Name", "Reprojection Error (px)"])

    for i in range(len(objpoints)):
        imgpoints2, _ = cv.projectPoints(objpoints[i], rvecs[i], tvecs[i], mtx, dist)
        error = cv.norm(imgpoints[i], imgpoints2, cv.NORM_L2) / len(imgpoints2)
        
        # Accumulate total
        total_error += error
        
        # Print to console (optional logging)
        print(f"Image: {successful_filenames[i]} | Error: {error:.4f}")
        
        # Write row to CSV
        writer.writerow([i, successful_filenames[i], error])

    mean_error = total_error / len(objpoints)
    print("\nMean reprojection error (pixels):", mean_error)

    # Write separator and Important Parameters at the bottom
    writer.writerow([])
    writer.writerow([])
    writer.writerow(["--- GLOBAL PARAMETERS ---"])
    writer.writerow(["RMS (Calibration Success Flag)", ret])
    writer.writerow(["Mean Reprojection Error", mean_error])
    
    # Write Matrix nicely
    writer.writerow(["Camera Matrix (Intrinsics)"])
    for row in mtx:
        writer.writerow(row)
        
    # Write Distortion Coefficients
    writer.writerow(["Distortion Coefficients"])
    writer.writerow(dist.ravel())

print(f"Complete. check {csv_filename} for detailed report.")