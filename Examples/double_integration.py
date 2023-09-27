import imufusion
import imusim
import matplotlib.pyplot as pyplot
import numpy

simulator = imusim.Simulator("indoor_forward_10_davis_with_gt/groundtruth.txt", delimiter=" ", xyzw=True, sample_rate=100)

# Uncomment the following line to see how small sensor errors have a big effect on the double integrated result
# simulator.set_accelerometer(offset=numpy.ones(3) * 0.01)  # 10 mg

# Process IMU measurements using Fusion
ahrs = imufusion.Ahrs()

ahrs.settings = imufusion.Settings(imufusion.CONVENTION_NWU, 0, 0, 0, 0, 0)  # 0 gain so that only gyroscope is used
ahrs.quaternion = imufusion.Quaternion(simulator.quaternion[0])

euler = numpy.empty_like(simulator.euler)
acceleration = numpy.empty_like(simulator.acceleration)

for index, (gyroscope, accelerometer) in enumerate(zip(simulator.gyroscope, simulator.accelerometer)):
    ahrs.update_no_magnetometer(gyroscope, accelerometer, 1 / simulator.sample_rate)

    euler[index] = ahrs.quaternion.to_euler()
    acceleration[index] = ahrs.earth_acceleration * imusim.Simulator.GRAVITY

# Calculate velocity snd position (double integration)
velocity = numpy.cumsum(acceleration, axis=0) / simulator.sample_rate
position = (numpy.cumsum(velocity, axis=0) / simulator.sample_rate) + simulator.position[0, :]

# Plot Fusion error
imusim.plot_euler_error(simulator.time, simulator.euler, euler, actual_label="Simulated", measured_label="Fusion")

imusim.plot_xyz_error(simulator.time, simulator.acceleration, acceleration, "Acceleration", "m/s/s", "Simulated", "Fusion")

imusim.plot_xyz_error(simulator.time, simulator.velocity, velocity, "Velocity", "m/s", "Simulated", "Calculated")

imusim.plot_xyz_error(simulator.time, simulator.position, position, "Position", "m", "Simulated", "Calculated")

# 3D animation
imusim.plot_3d(position, euler, label="Calculated",
               ref_b=simulator.position, ref_b_label="Actual",
               quiver_length=1, samples_per_quiver=1,
               animate=True, sample_rate=simulator.sample_rate)

pyplot.show()
