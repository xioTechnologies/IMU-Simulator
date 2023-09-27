import imusim
import numpy
import matplotlib.pyplot as pyplot

simulator = imusim.Simulator("circle/circle.csv")

# Minimum example
imusim.plot_3d(simulator.position)

# Include orientation
imusim.plot_3d(simulator.position, euler=simulator.euler)

# Plot against references
actual = simulator.position
gps = actual + numpy.random.normal(0, 0.1, simulator.position.shape)
ins = actual + numpy.random.normal(0, 0.01, simulator.position.shape)

gps[~numpy.arange(gps.shape[0]) % 10 != 0] = numpy.nan  # reduce GPS sample rate

imusim.plot_3d(ins, label="INS",
               ref_a=gps, ref_a_label="GPS",
               ref_b=actual, ref_b_label="Actual",
               title="GPS vs. INS")

pyplot.show()
