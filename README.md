### PROJECT SACRID
Working in a team of 11, we developed an automated inventory management bot at SAC, Robotics club, IIT Delhi (Hence the name, SACRID) that can currently read QR codes and log their data. <br>
We optimised it for dark, narrow alleyways by adding robust QR enhancement pipelines and Homographic corrections based on the camera's and robot's orientation (input from a separate SLAM code). The code also includes smoke tests, detailed error logs and fail-safes, and a dedicated debugging mode for Human-Machine Interaction. <br>
As for the exact structure of the code, our VisionNode.py (part of larger ROS2 framework) implements a Multiprocessing Queue running hardware aspects (Camera Motion, Image capture and filtering, LEDs and Flashlight) and Image Analysis(Homography, Localisation, Payload detection and Logging results) in parallel <br>
(Clarif: By localisation, I mean we store the location in global coordinate frame of each detection. This allows us to have redundancy in our images as failsafe against poor QR codes or unexpected blurring, while avoiding duplicacy in the logged data.) <br>

#### Contributors:
**Mahesh Pareek** : Github Mahesh-pareek @ https://github.com/Mahesh-pareek <br>
**Preesha Agrawal** : Github Preesha07 @  https://github.com/Preesha07 <br>
(Me) **Siddhant Agrawal** : Github Siddhant-135 @ https://github.com/Siddhant-135 <br>

### Further Enhancements Planned (Contributions welcome)
While the current code is specific to a particular set of Hardware, we have some generalisations underway <br>
- Separate out the localisation and detection pipelines to enable users to choose between general OCR , QR, Aruko markers or custom object detections
- Implement absolute general motion of camera: Minor modification, currently we have precomputed matrices for the angles we required from our camera, we'll make it easier to input your configurations by taking it directly from the config file than the code
- Autocompute d, the distance of camera from the rack, for use with autofocus camera's: Currently, distance to the measure objects is hardcoded, but we can use dimensions of object to be detected / SLAM to adjust to varying distances. This will require multiple callibrations of the camera (K,D values) for different distances and accordingly more code.
