import imusim
import numpy

simulator = imusim.Simulator("circle/circle.csv")

# Minimum example
imusim.plot_3d(simulator.position, animate=True, sample_rate=simulator.sample_rate)

# Include orientation
imusim.plot_3d(simulator.position, euler=simulator.euler, animate=True, sample_rate=simulator.sample_rate)

# Plot against references
actual = simulator.position
gps = actual + numpy.random.normal(0, 0.1, simulator.position.shape)
ins = actual + numpy.random.normal(0, 0.01, simulator.position.shape)

gps[~numpy.arange(gps.shape[0]) % 10 != 0] = numpy.nan  # reduce GPS sample rate

imusim.plot_3d(ins, label="INS",
               ref_a=gps, ref_a_label="GPS",
               ref_b=actual, ref_b_label="Actual",
               title="GPS vs. INS",
               animate=True, sample_rate=simulator.sample_rate)

# Save as .gif (1920 x 1080, 30 FPS)
imusim.plot_3d(simulator.position, euler=simulator.euler, samples_per_quiver=1,
               animate=True, sample_rate=simulator.sample_rate,
               file_name="animation.gif", fps=30, figsize=(16, 9), dpi=120)
