import allantools
import imusim
import matplotlib.pyplot as pyplot
import numpy

# This script may take over a minute to run

simulator = imusim.Simulator("stationary/stationary.csv", sample_rate=10)

noise_density = numpy.array([0.1, 0.01, 0.001])  # noise density can be read from an Allan variance plot as the tau=1 value

bias_instability = numpy.array([0.01, 0.001, 0.0001])  # bias instability multiplied by 0.664 is the lowest value of an Allan variance plot

random_walk = numpy.array([0.0003, 0.00003, 0.000003])  # random walk is the intercept of tau=3 and the 0.5 gradient of the ramp of an Allan variance plot

simulator.set_gyroscope(noise_density=noise_density,
                        bias_instability=bias_instability,
                        number_of_poles=100,  # large differences between noise density and random walk require more poles to accurately model bias instability
                        random_walk=random_walk)

# Calculate Allan variance
tau_x, gyroscope_x, _, _ = allantools.oadev(simulator.gyroscope[:, 0], simulator.sample_rate, "freq")
tau_y, gyroscope_y, _, _ = allantools.oadev(simulator.gyroscope[:, 1], simulator.sample_rate, "freq")
tau_z, gyroscope_z, _, _ = allantools.oadev(simulator.gyroscope[:, 2], simulator.sample_rate, "freq")

# Plot Allan variance
pyplot.loglog(tau_x, gyroscope_x, imusim.RED, label="X")
pyplot.loglog(tau_y, gyroscope_y, imusim.GREEN, label="Y")
pyplot.loglog(tau_z, gyroscope_z, imusim.BLUE, label="Z")

# Annotate noise density (equivalent to angle random walk)
pyplot.loglog(1, noise_density[0], imusim.RED, linestyle="", marker="o")
pyplot.loglog(1, noise_density[1], imusim.GREEN, linestyle="", marker="o")
pyplot.loglog(1, noise_density[2], imusim.BLUE, linestyle="", marker="o")

pyplot.text(1, noise_density[0], "noise density = {:.0e}".format(noise_density[0]), verticalalignment="bottom")
pyplot.text(1, noise_density[1], "noise density = {:.0e}".format(noise_density[1]), verticalalignment="bottom")
pyplot.text(1, noise_density[2], "noise density = {:.0e}".format(noise_density[2]), verticalalignment="bottom")

# Annotate bias instability
factor = numpy.sqrt((2 * numpy.log(2)) / numpy.pi)  # 0.664

pyplot.loglog([tau_x[0], tau_x[-1]], factor * numpy.array([bias_instability[0], bias_instability[0]]), imusim.RED, linestyle="dashed")
pyplot.loglog([tau_y[0], tau_y[-1]], factor * numpy.array([bias_instability[1], bias_instability[1]]), imusim.GREEN, linestyle="dashed")
pyplot.loglog([tau_z[0], tau_z[-1]], factor * numpy.array([bias_instability[2], bias_instability[2]]), imusim.BLUE, linestyle="dashed")

pyplot.text(1E3, factor * bias_instability[0], "0.664 * bias instability = {:.2e}".format(factor * bias_instability[0]), verticalalignment="top")
pyplot.text(1E3, factor * bias_instability[1], "0.664 * bias instability = {:.2e}".format(factor * bias_instability[1]), verticalalignment="top")
pyplot.text(1E3, factor * bias_instability[2], "0.664 * bias instability = {:.2e}".format(factor * bias_instability[2]), verticalalignment="top")

# Annotate random walk
pyplot.loglog(tau_x, numpy.sqrt(tau_x) * (random_walk[0] / numpy.sqrt(3)), imusim.RED, linestyle="dotted")
pyplot.loglog(tau_y, numpy.sqrt(tau_y) * (random_walk[1] / numpy.sqrt(3)), imusim.GREEN, linestyle="dotted")
pyplot.loglog(tau_z, numpy.sqrt(tau_z) * (random_walk[2] / numpy.sqrt(3)), imusim.BLUE, linestyle="dotted")

pyplot.loglog(3, random_walk[0], imusim.RED, linestyle="", marker="o")
pyplot.loglog(3, random_walk[1], imusim.GREEN, linestyle="", marker="o")
pyplot.loglog(3, random_walk[2], imusim.BLUE, linestyle="", marker="o")

pyplot.text(3, random_walk[0], "random walk = {:.0e}".format(random_walk[0]), verticalalignment="top")
pyplot.text(3, random_walk[1], "random walk = {:.0e}".format(random_walk[1]), verticalalignment="top")
pyplot.text(3, random_walk[2], "random walk = {:.0e}".format(random_walk[2]), verticalalignment="top")

# Axes and legend
pyplot.xlabel("Tau")
pyplot.ylabel("Allan deviation")
pyplot.grid(True, which="both")
pyplot.legend()

pyplot.show()
