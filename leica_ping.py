from leica_engine import LeicaEngine
import time

NUM_ITER = 10
N_BSCANS = 5
DIMS = (0.1, 4)

leica_reader = LeicaEngine(
            ip_address="192.168.1.75",
            n_bscans=N_BSCANS,
            xd=DIMS[0],
            yd=DIMS[1],
            zd=3.379,
        )

times = []

for i in range(NUM_ITER):
    start_time = time.perf_counter()
    leica_reader.__get_b_scans_volume__()
    duration = time.perf_counter() - start_time
    print(f"Duration: {duration:.2f} seconds")
    times.append(time.perf_counter() - start_time)

print(f"Average duration: {sum(times) / len(times):.2f} seconds")
