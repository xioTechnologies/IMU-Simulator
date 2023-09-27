import imusim
import matplotlib.pyplot as pyplot
import numpy

simulator = imusim.Simulator("circle/circle.csv")

simulator.set_gyroscope(range=2000,  # 2000°/s
                        bandwidth=50,  # 50 Hz
                        offset=numpy.ones(3) * 0.1,  # 0.1 °/s
                        sensitivity_error=numpy.ones(3) * 1,  # 1%
                        noise_density=numpy.ones(3) * 0.03,  # 0.03°/s/√Hz
                        cross_axis=numpy.ones(3) * 0.1,  # 0.1°
                        misalignment=1)  # 1°

simulator.set_accelerometer(range=16,  # 16 g
                            bandwidth=50,  # 50 Hz
                            offset=numpy.ones(3) * 1E-3,  # 1 mg
                            sensitivity_error=numpy.ones(3) * 1,  # 1%
                            noise_density=numpy.ones(3) * 200E-6,  # 200 ug/√Hz
                            cross_axis=numpy.ones(3) * 0.1,  # 0.1°
                            misalignment=1)  # 1°

simulator.plot_gyroscope()

simulator.plot_accelerometer()

pyplot.show()
