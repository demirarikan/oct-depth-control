import rospy
from cv_bridge import CvBridge
from sensor_msgs.msg import Image

from leica_engine import LeicaEngine

if __name__ == "__main__":

    leica_reader = LeicaEngine(
        ip_address="192.168.1.75",
        n_bscans=n_bscans,
        xd=dims[0],
        yd=dims[1],
        zd=3.379,
    )

    b_scan_publisher = rospy.Publisher("oct_b_scan", Image, queue_size=1)
    cv_bridge = CvBridge()
    rospy.init_node("b_scan_publisher", anonymous=True)
    print("B scan publisher initialized")

    try:
        while not rospy.is_shutdown():
            b_scan_img = leica_reader.get_b_scan(frame_to_save=0)
            image_message = cv_bridge.cv2_to_imgmsg(b_scan_img * 255)
    except KeyboardInterrupt:
        print("Shutting down b scan publisher")