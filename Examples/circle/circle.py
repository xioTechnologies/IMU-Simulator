import matplotlib.pyplot as pyplot
import numpy

# Calculate angle and position
SAMPLE_RATE = 100
ANGULAR_VELOCITY = 45
RADIUS = 1

starting_stationary = numpy.zeros(2 * SAMPLE_RATE)  # 2 seconds
accelerating = numpy.linspace(0, ANGULAR_VELOCITY, 5 * SAMPLE_RATE)  # 5 seconds
constant_velocity = numpy.full(3 * SAMPLE_RATE, ANGULAR_VELOCITY)  # 3 seconds
decelerating = numpy.linspace(ANGULAR_VELOCITY, 0, 5 * SAMPLE_RATE)  # 5 seconds
ending_stationary = numpy.zeros(2 * SAMPLE_RATE)  # 2 seconds

omega = numpy.concatenate((starting_stationary, accelerating, constant_velocity, decelerating, ending_stationary))

time = numpy.linspace(0, (len(omega) - 1) / SAMPLE_RATE, len(omega))

theta = (numpy.cumsum(omega) / SAMPLE_RATE) % 360

x = RADIUS * numpy.cos(numpy.radians(theta))
y = RADIUS * numpy.sin(numpy.radians(theta))

# Write to file
with open("circle.csv", "w") as file:
    file.write("Time (s),X (m),Y (m),Z (m),X (deg),Y (deg),Z (deg)\n")

    for index, _ in enumerate(time):
        file.write(str(time[index]) + "," + str(x[index]) + "," + str(y[index]) + ",0,0,0," + str(theta[index]) + "\n")

# Plot
_, axes = pyplot.subplots(nrows=3, sharex=True)

axes[0].plot(time, omega, label="omega")
axes[0].set_ylabel("Angular velocity (°/s)")
axes[0].grid()
axes[0].legend()

axes[1].plot(time, theta, label="theta")
axes[1].set_ylabel("Angle (°)")
axes[1].grid()
axes[1].legend()

axes[2].plot(time, x, label="x")
axes[2].plot(time, y, label="y")
axes[2].set_ylabel("Position (m)")
axes[2].set_xlabel("Time (s)")
axes[2].grid()
axes[2].legend()

pyplot.show()
