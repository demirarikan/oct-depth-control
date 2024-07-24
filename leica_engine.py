import socket
import struct

import cv2
import matplotlib.pyplot as plt
import numpy as np
import rospy
from geometry_msgs.msg import Transform


class LeicaEngine(object):

    def __init__(
        self,
        ip_address="192.168.1.75",
        port_num=2000,
        n_bscans=100,
        xd=2.5,
        yd=2.5,
        zd=3.379,
        scale=1,
        save_dir=None,
        save_robot_pos=False,
    ):

        # x: n_Ascan in Bsacn dir
        # y: n_Bscans dir
        # z: Ascan dir
        # output: n_bscans*len_ascan*width

        self.max_bytes = 2**16
        self.server_address = (ip_address, port_num)
        self.b_scan_reading = None
        self.n_bscans = n_bscans
        self.xd = xd
        self.yd = yd
        self.zd = zd
        self.scale = scale
        self.save_dir = save_dir
        if save_robot_pos:
            self.x = None
            self.y = None
            self.z = None
            self.rx = None
            self.ry = None
            self.rz = None
            self.rw = None
            self.robot_pos_sub = rospy.Subscriber(
                "/eye_robot/FrameEE", Transform, self.__update_robot_pos
            )

        self.__connect__()
        self.active = True
        self.latest_complete_scans = None
        self.latest_spacing = None

    def __get_b_scans_volume_continously__(self):

        while self.active:

            latest_volume, latest_spacing = self.__get_b_scans_volume__()

            self.latest_complete_scans = latest_volume
            self.latest_spacing = latest_spacing

    
    def fast_get_b_scan_volume(self):
        start = None

        buf = self.__get_buffer__()
        _, frame = self.__parse_data__(buf)
        latest_scans = np.zeros((self.n_bscans, frame.shape[0], frame.shape[1]))
        # resized shape is (n_bscans, frame0, frame1)
        # resized_shape = (np.array(latest_scans.shape) * self.scale).astype(int)
        # resized 1 = (n_bscans, frame0, frame1) BUT N_BSCAN NOT SCALED
        # latest_scans_resized_1 = np.zeros(
        #     [self.n_bscans, resized_shape[1], resized_shape[2]]
        # )
        # resized 2 = n_bscans, frame0, frame1 BUT N_BSCAN SCALED
        # latest_scans_resized_2 = np.zeros(resized_shape)

        spacing = self.__calculate_spacing(latest_scans.shape)

        while True:
            buf = self.__get_buffer__()
            frame_number, frame = self.__parse_data__(buf)

            if start is None:
                start = frame_number

            latest_scans[frame_number, :, :] = frame
            # latest_scans_resized_1[frame_number, :, :] = cv2.resize(
            #     frame, (resized_shape[2], resized_shape[1])
            # )

            if frame_number == (start - 1) % self.n_bscans:
                break

        # for i in range(resized_shape[2]):
        #     latest_scans_resized_2[:, :, i] = cv2.resize(
        #         latest_scans_resized_1[:, :, i], (resized_shape[1], resized_shape[0])
        #     )

        # latest_scans_resized_2 = np.transpose(latest_scans_resized_2, (2, 0, 1))
        # latest_scans_resized_2 = np.flip(latest_scans_resized_2, 1)
        # latest_scans_resized_2 = np.flip(latest_scans_resized_2, 2)
        spacing = spacing[[2, 0, 1]]

        return latest_scans, spacing


    
    def __get_b_scans_volume__(self):
        start = None

        buf = self.__get_buffer__()
        _, frame = self.__parse_data__(buf)
        latest_scans = np.zeros((self.n_bscans, frame.shape[0], frame.shape[1]))
        # resized shape is (n_bscans, frame0, frame1)
        resized_shape = (np.array(latest_scans.shape) * self.scale).astype(int)
        # resized 1 = (n_bscans, frame0, frame1) BUT N_BSCAN NOT SCALED
        latest_scans_resized_1 = np.zeros(
            [self.n_bscans, resized_shape[1], resized_shape[2]]
        )
        # resized 2 = n_bscans, frame0, frame1 BUT N_BSCAN SCALED
        latest_scans_resized_2 = np.zeros(resized_shape)

        spacing = self.__calculate_spacing(latest_scans_resized_2.shape)

        while True:
            buf = self.__get_buffer__()
            frame_number, frame = self.__parse_data__(buf)

            if start is None:
                start = frame_number

            latest_scans[frame_number, :, :] = frame
            latest_scans_resized_1[frame_number, :, :] = cv2.resize(
                frame, (resized_shape[2], resized_shape[1])
            )

            if frame_number == (start - 1) % self.n_bscans:
                break

        for i in range(resized_shape[2]):
            latest_scans_resized_2[:, :, i] = cv2.resize(
                latest_scans_resized_1[:, :, i], (resized_shape[1], resized_shape[0])
            )

        latest_scans_resized_2 = np.transpose(latest_scans_resized_2, (2, 0, 1))
        latest_scans_resized_2 = np.flip(latest_scans_resized_2, 1)
        latest_scans_resized_2 = np.flip(latest_scans_resized_2, 2)
        spacing = spacing[[2, 0, 1]]

        return latest_scans_resized_2, spacing

    def __update_robot_pos(self, data):
        self.x = data.translation.x
        self.y = data.translation.y
        self.z = data.translation.z
        self.rx = data.rotation.x
        self.ry = data.rotation.y
        self.rz = data.rotation.z
        self.rw = data.rotation.w

    def __calculate_spacing(self, shape):
        t = np.array(shape)
        spacing = np.array([self.xd, self.zd, self.yd]) / t

        return spacing

    def __disconnect__(self):

        self.active = False
        self.sock.close()

    def __connect__(self):

        print(
            f"Connecting to {self.server_address[0]} and port {self.server_address[1]}"
        )

        tries = 0
        connected = False
        while tries < 10 and not connected:
            try:
                self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                self.sock.connect(self.server_address)
                connected = True
            except Exception as e:
                print(f"No connection. Waiting on server. {tries} Attempts.")
                tries += 1

        self.active = True

        if connected:
            print(f"Connection Successful")
        else:
            print("Connection Failed")

    def __get_buffer__(self):

        buf = None
        num_expected_bytes = 0

        while True:
            try:
                data = self.sock.recv(self.max_bytes)
            except Exception as e:
                print("Connection error. Trying to re-establish connection.")
                break

            if buf is None:
                if len(data) == 0:
                    break

                if len(data) < 10:
                    message = "Waiting for new frame"
                    message_bytes = str.encode(message)
                    self.sock.sendall(message_bytes)
                    continue

                buf = data

                start_pos = 0
                end_pos = 4
                dataLabelSize = struct.unpack("I", buf[start_pos:end_pos])[0]
                dataLabel = struct.unpack(
                    "B" * int(dataLabelSize),
                    buf[end_pos : end_pos + int(dataLabelSize)],
                )
                dataLabel = "".join([chr(L) for L in dataLabel])

                start_pos = end_pos + int(dataLabelSize)
                end_pos = start_pos + 4

                if dataLabel == "EXPECTEDBYTES":
                    val_length = struct.unpack("I", buf[start_pos:end_pos])[0]
                    num_expected_bytes = struct.unpack(
                        "I", buf[end_pos : end_pos + val_length]
                    )[0]
                    start_pos = end_pos + 4
                    end_pos = start_pos + 4
            else:
                buf = buf + data

            if buf is not None and len(buf) >= num_expected_bytes:
                break

        message = "Received frame"
        message_bytes = str.encode(message)
        self.sock.sendall(message_bytes)

        return buf

    def __parse_data__(self, buf):

        dataLabel = None
        start_pos = 0
        end_pos = 4

        while dataLabel != "ENDFRAMEHEADER":
            dataLabelSize = struct.unpack("I", buf[start_pos:end_pos])[0]
            dataLabel = struct.unpack(
                "B" * int(dataLabelSize), buf[end_pos : end_pos + int(dataLabelSize)]
            )
            start_pos = end_pos + int(dataLabelSize)
            end_pos = start_pos + 4

            dataLabel = "".join([chr(L) for L in dataLabel])

            if dataLabel == "ENDFRAMEHEADER":
                data_start_pos = start_pos + 8
                break
            else:
                val_length = struct.unpack("I", buf[start_pos:end_pos])[0]
                if val_length <= 4:
                    val = struct.unpack("I", buf[end_pos : end_pos + 4])[0]
                    val_pos = end_pos
                    start_pos = end_pos + 4
                    end_pos = start_pos + 4
                else:
                    val = struct.unpack("d", buf[end_pos : end_pos + 8])[0]
                    val_pos = end_pos
                    start_pos = end_pos + 8
                    end_pos = start_pos + 4

                if dataLabel == "FRAMENUMBER":
                    frame_number = val
                    frame_number_pos = val_pos
                if dataLabel == "FRAMECOUNT":
                    frame_count = val
                if dataLabel == "LINECOUNT":
                    line_count = val
                if dataLabel == "LINELENGTH":
                    line_length = val
                if dataLabel == "AIMFRAMES":
                    aim_frames = val

        frameData = np.zeros((line_length, line_count))

        frame_number = frame_number % frame_count

        for i in range(0, line_count):
            start = data_start_pos + i * line_length * 2
            frameData[:, i] = np.frombuffer(
                buf[start : start + line_length * 2], dtype="u2", count=line_length
            )

        frame = frameData / self.max_bytes

        return frame_number, frame

    def get_robot_pos(self):
        return [self.x, self.y, self.z, self.rx, self.ry, self.rz, self.rw]

    def get_b_scan(self, frame_to_save):  # frame_to_save(0 for upper, 1 for lower)
        buf = self.__get_buffer__()
        frame_number, frame = self.__parse_data__(buf)
        if frame_number % 2 == frame_to_save:
            return frame


if __name__ == "__main__":
    import time
    le = LeicaEngine()
    num_iterations = 50 

    avg_time1 = []
    avg_time2 = []
    for i in range(num_iterations):
        start = time.perf_counter()
        volume1 = le.__get_b_scans_volume__()
        end = time.perf_counter()
        print(f"Time taken: {end-start} seconds")
        avg_time1.append(end-start)

    for i in range(num_iterations):
        start = time.perf_counter()
        volume2 = le.fast_get_b_scan_volume()
        end = time.perf_counter()
        print(f"Time taken: {end-start} seconds")
        avg_time2.append(end-start)

    print(f'shape for method 1: {volume1.shape}')
    print(f'shape for method 2: {volume2.shape}')

    print(f"Average time for method 1: {sum(avg_time1)/num_iterations}")
    print(f"Average time for method 2: {sum(avg_time2)/num_iterations}")


