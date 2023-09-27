import imusim
import matplotlib.pyplot as pyplot

simulator = imusim.Simulator("circle/circle.csv")

simulator.plot_kinematics()

simulator.plot_imu()

pyplot.show()
