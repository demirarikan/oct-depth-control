import robot_controller
import multiprocessing
import time

def move():
    print("moving")
    while not stop_event.is_set():
        robot_controller.move_forward_needle_axis(
            kp_linear_vel=1, linear_vel=0.1, duration_sec=1
        )
    print("stopping")
    robot_controller.stop()

def do_stuff():
    for _ in range(10):
        print("doing stuff")
        time.sleep(0.5)
    stop_event.set()

if __name__ == "__main__":
    stop_event = multiprocessing.Event()
    p1 = multiprocessing.Process(target=move)
    p1.start()
    do_stuff()
    p1.join()
    print("done")