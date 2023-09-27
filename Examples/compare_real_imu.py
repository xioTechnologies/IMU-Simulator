import imusim
import matplotlib.pyplot as pyplot
import numpy

# Simulate IMU measurements using ground truth
simulator = imusim.Simulator("indoor_forward_10_davis_with_gt/groundtruth.txt", delimiter=" ", xyzw=True, bandwidth=100)  # use bandwidth to filter noise in ground truth measurements

simulator.set_gyroscope(offset=[-3.5, -0.2, -1.13])  # offsets to match real IMU measurements
simulator.set_accelerometer(offset=[0.068, 0.025, 0.102])

# Import real IMU measurements
imu = numpy.genfromtxt("indoor_forward_10_davis_with_gt/imu.txt", skip_header=1)

time = imu[:, 1] - simulator.start_time
gyroscope = numpy.degrees(imu[:, 2:5])
accelerometer = imu[:, 5:8] / imusim.Simulator.GRAVITY

# Compare simulated and real IMU measurements
_, axes = pyplot.subplots(nrows=3, ncols=2, sharex=True)

axes[0, 0].set_title("Gyroscope")
axes[2, 0].set_xlabel("Time (s)")

for axis in imusim.Axis:
    axes[axis.index, 0].plot(time, gyroscope[:, axis.index], imusim.LIGHT_GREY, label="Real")
    axes[axis.index, 0].plot(simulator.time, simulator.gyroscope[:, axis.index], axis.colour, label="Simulated")
    axes[axis.index, 0].set_ylabel(str(axis) + " (°/s)")
    axes[axis.index, 0].grid()
    axes[axis.index, 0].legend()

axes[0, 1].set_title("Accelerometer")
axes[2, 1].set_xlabel("Time (s)")

for axis in imusim.Axis:
    axes[axis.index, 1].plot(time, accelerometer[:, axis.index], imusim.LIGHT_GREY, label="Real")
    axes[axis.index, 1].plot(simulator.time, simulator.accelerometer[:, axis.index], axis.colour, label="Simulated")
    axes[axis.index, 1].set_ylabel(str(axis) + " (g)")
    axes[axis.index, 1].grid()
    axes[axis.index, 1].legend()

pyplot.show()
