#!/usr/bin/env python3
import rospy
import cv2
from cv_bridge import CvBridge
from sensor_msgs.msg import Image, CameraInfo


class EERgbdReader:
    def __init__(self):
        self.bridge = CvBridge()
        self.color = None
        self.depth = None
        self.camera_info = None

        color_topic = rospy.get_param("~color_topic", "/camera/color/image_raw")
        depth_topic = rospy.get_param("~depth_topic", "/camera/depth/image_rect_raw")
        info_topic = rospy.get_param("~info_topic", "/camera/color/camera_info")

        rospy.Subscriber(color_topic, Image, self.color_cb, queue_size=1)
        rospy.Subscriber(depth_topic, Image, self.depth_cb, queue_size=1)
        rospy.Subscriber(info_topic, CameraInfo, self.info_cb, queue_size=1)

        rospy.loginfo("Subscribed color: %s", color_topic)
        rospy.loginfo("Subscribed depth: %s", depth_topic)
        rospy.loginfo("Subscribed info : %s", info_topic)

    def color_cb(self, msg):
        self.color = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")

    def depth_cb(self, msg):
        self.depth = self.bridge.imgmsg_to_cv2(msg, desired_encoding="passthrough")

    def info_cb(self, msg):
        self.camera_info = msg

    def run(self):
        rate = rospy.Rate(20)
        while not rospy.is_shutdown():
            if self.color is not None:
                vis = self.color.copy()
                h, w = vis.shape[:2]
                cx = w // 2
                cy = h // 2
                cv2.circle(vis, (cx, cy), 4, (0, 255, 0), -1)

                if self.depth is not None:
                    d = self.depth[cy, cx]
                    if hasattr(d, "item"):
                        d = d.item()
                    text = "center depth: {:.4f}".format(float(d))
                    cv2.putText(vis, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

                cv2.imshow("ee_rgb", vis)

            if self.depth is not None:
                depth_vis = self.depth.copy()
                depth_show = cv2.normalize(depth_vis, None, 0, 255, cv2.NORM_MINMAX)
                depth_show = depth_show.astype("uint8")
                depth_show = cv2.applyColorMap(depth_show, cv2.COLORMAP_JET)
                cv2.imshow("ee_depth", depth_show)

            key = cv2.waitKey(1)
            if key == 27 or key == ord('q'):
                break
            rate.sleep()

        cv2.destroyAllWindows()


if __name__ == "__main__":
    rospy.init_node("ee_rgbd_reader", anonymous=True)
    node = EERgbdReader()
    node.run()
