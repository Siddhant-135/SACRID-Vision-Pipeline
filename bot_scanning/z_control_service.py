#!/usr/bin/env python3
from example_interfaces.srv import SetBool

import rclpy
from rclpy.node import Node
# from rclpy.executors import SingleThreadedExecutor

# import time

from bot_scanning.bot_scanning.visionNode import visionNode
from nav_msgs.msg import Odometry
from std_msgs.msg import Float32, Float32MultiArray


class ZControlService(Node):

    def __init__(self):
        super().__init__('z_control_service')

        # ---------------- PARAMETERS ----------------
        self.declare_parameter('height_topic', '/height')
        self.declare_parameter('z_camera_topic', '/z_camera')
        self.declare_parameter('checkpoints', [0.0, 22.5, 45.0, 67.5, 90.0])
        self.declare_parameter('threshold_cm', 1.5)
        self.declare_parameter('loop_hz', 10.0)

        self.height_topic = (
            self.get_parameter('height_topic')
            .get_parameter_value()
            .string_value
        )
        self.z_camera_topic = (
            self.get_parameter('z_camera_topic')
            .get_parameter_value()
            .string_value
        )
        self.checkpoints = list(
            self.get_parameter('checkpoints')
            .get_parameter_value()
            .double_array_value
        )
        self.threshold_cm = float(
            self.get_parameter('threshold_cm')
            .get_parameter_value()
            .double_value
        )
        self.loop_hz = float(
            self.get_parameter('loop_hz')
            .get_parameter_value()
            .double_value
        )

        # ---------------- INTERNAL STATE ----------------
        self.current_cm = 0.0
        self.x = 0.0
        self.y = 0.0

        self.direction_up = True      # False => downward scanning
        self.active = False
        self.finished = False
        self.state = "IDLE"           # MOVING / CAPTURE / DONE

        self._target_index = 0
        self._target_cm = self.checkpoints[0]

        # ---------------- CAMERA ----------------
        self.camera = visionNode.get()

        # ---------------- PUB/SUB ----------------
        self.pub_pwm = self.create_publisher(Float32, self.z_camera_topic, 10)
        self.sub_height = self.create_subscription(
            Float32MultiArray, self.height_topic, self.height_cb, 10
        )
        self.sub_odom = self.create_subscription(
            Odometry, "/odom", self.odom_cb, 10
        )

        # ---------------- SERVICE ----------------
        self.srv = self.create_service(
            SetBool, "z_control", self.service_callback
        )

        # ---------------- TIMER ----------------
        period = 1.0 / max(1.0, self.loop_hz)
        self.create_timer(period, self.control_loop)

        self._publish_pwm(0.0)
        self.get_logger().info(
            f"[Z] Controller READY. Checkpoints = {self.checkpoints}"
        )

    # ============================================================
    # SERVICE CALLBACK — BLOCKS UNTIL FULL SCAN IS COMPLETED
    # ============================================================
    def service_callback(self, request, response):

        # request.data FALSE => upward (normal)
        # request.data TRUE  => downward
        self.direction_up = not request.data

        # Reset sequence state
        if self.direction_up:
            self._target_index = 0
        else:
            self._target_index = len(self.checkpoints) - 1

        self._target_cm = self.checkpoints[self._target_index]
        self.active = True
        self.finished = False
        self.state = "MOVING"

        self.get_logger().info("[Z] Service triggered — starting scan.")
        self.get_logger().info(f"[Z] Direction UP = {self.direction_up}")
        self.get_logger().info(f"[Z] First target = {self._target_cm} cm")

        # -----------------------------------------------------------
        # BLOCK UNTIL FINISHED
        # -----------------------------------------------------------
        while not self.finished:
            rclpy.spin_once(self, timeout_sec=0.1)

        # scanning finished → return service response
        response.success = True
        response.message = "Scan completed successfully."

        self.get_logger().info("[Z] Scan finished — service response returned.")
        return response

    # ============================================================
    # SUBSCRIBERS
    # ============================================================
    def height_cb(self, msg):
        self.current_cm = msg.data[0]

    def odom_cb(self, msg):
        self.x = msg.pose.pose.position.x
        self.y = msg.pose.pose.position.y

    # ============================================================
    # PWM PUBLISHER
    # ============================================================
    def _publish_pwm(self, value):
        msg = Float32()
        msg.data = float(value)
        self.pub_pwm.publish(msg)

    # ============================================================
    # MAIN CONTROL LOOP (runs continuously)
    # ============================================================
    def control_loop(self):

        if not self.active:
            return

        # -------------------------------------------
        # STATE: MOVING
        # -------------------------------------------
        if self.state == "MOVING":

            error = self._target_cm - self.current_cm

            if abs(error) <= self.threshold_cm:
                self.get_logger().info(
                    f"[Z] Reached {self._target_cm} cm (current={self.current_cm:.2f})"
                )
                self._publish_pwm(0.0)
                self.state = "CAPTURE"
                return

            pwm = 150 if error > 0 else -150
            self._publish_pwm(pwm)

        # -------------------------------------------
        # STATE: CAPTURE
        # -------------------------------------------
        elif self.state == "CAPTURE":

            isTop = (self._target_index == len(self.checkpoints) - 1)
            isBottom = (self._target_index == 0)

            self.get_logger().info(
                f"[Z] Capturing at {self._target_cm}cm | (x={self.x:.2f}, y={self.y:.2f})"
            )

            self.camera.capture(
                self.x,
                self.y,
                isTop=isTop,
                isBottom=isBottom
            )

            # Advance to next checkpoint
            if self.direction_up:
                self._target_index += 1
            else:
                self._target_index -= 1

            # Check if sequence finished
            if self._target_index < 0 or self._target_index >= len(self.checkpoints):
                self.get_logger().info("[Z] All checkpoints done — stopping.")
                self._publish_pwm(0.0)
                self.finished = True
                self.active = False
                self.state = "DONE"
                return

            self._target_cm = self.checkpoints[self._target_index]
            self.get_logger().info(
                f"[Z] Next target = {self._target_cm} cm"
            )
            self.state = "MOVING"

        # -------------------------------------------
        # STATE: DONE
        # -------------------------------------------
        elif self.state == "DONE":
            self._publish_pwm(0.0)
            self.finished = True
            self.active = False
            return


# ============================================================
# MAIN
# ============================================================
def main(args=None):
    rclpy.init(args=args)
    node = ZControlService()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
