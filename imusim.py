import fractalcoef
import matplotlib.pyplot as pyplot
import numpy
import scipy
from enum import Enum, auto
from matplotlib import animation
from quaternion import Quaternion


class Simulator:
    GRAVITY = 9.8  # m/s/s

    def __init__(self, file_name, delimiter=",", xyzw=False, sample_rate=None, bandwidth=None):

        # Import CSV
        csv = numpy.genfromtxt(file_name, delimiter=delimiter, skip_header=1)

        csv = csv[numpy.unique(csv[:, 0], return_index=True)[1]]  # remove duplicate timestamps

        self.__start_time = csv[0, 0]

        self.__original_time = csv[:, 0] - self.__start_time

        self.__original_position = csv[:, 1:4]

        if csv.shape[1] == 7:  # CSV columns are: time, x, y, z, roll, pitch, yaw
            self.__original_euler = (csv[:, 4:7] + 180) % 360 - 180  # wrap to ±180

            self.__original_quaternion = Quaternion.fix_continuity(numpy.array([Quaternion(euler=e).wxyz for e in self.__original_euler]))
        else:
            if xyzw:  # CSV columns are: time, x, y, z, qx, qy, qz, qw
                self.__original_quaternion = Quaternion.fix_continuity(numpy.column_stack((csv[:, 7], csv[:, 4:7])))
            else:  # CSV columns are: time, x, y, z, qw, qx, qy, qz
                self.__original_quaternion = Quaternion.fix_continuity(csv[:, 4:8])

            self.__original_euler = numpy.array([Quaternion(q).to_euler() for q in self.__original_quaternion])

        # Sample rate
        original_sample_rate = 1 / numpy.mean(numpy.diff(self.__original_time))

        if sample_rate is None:
            self.__sample_rate = original_sample_rate
        else:
            self.__sample_rate = sample_rate

        # Bandwidth
        self.__bandwidth = 0.499 * numpy.min((original_sample_rate, self.__sample_rate))  # default bandwidth is half sample rate

        if bandwidth is not None:
            self.__bandwidth = numpy.min((self.__bandwidth, bandwidth))

        # Resample
        self.__time = numpy.arange(self.__original_time[0], self.__original_time[-1], 1 / self.sample_rate)

        position = self.__filter(self.__interpolate(self.__original_position))

        quaternion = numpy.array([Quaternion(q).normalise().wxyz for q in self.__filter(self.__interpolate(self.__original_quaternion))])

        # Velocity from position
        velocity = Simulator.__diff(position) * self.sample_rate

        # Acceleration from velocity
        self.__acceleration = Simulator.__diff(velocity) * self.sample_rate

        # Velocity from acceleration
        self.__velocity = numpy.cumsum(self.acceleration, axis=0) / self.sample_rate  # assumes initial actual velocity is zero

        # Position from velocity
        self.__position = (numpy.cumsum(self.velocity, axis=0) / self.sample_rate) + self.__original_position[0, :]

        # Gyroscope from quaternion
        quaternion_rate = Simulator.__diff(quaternion) * self.sample_rate

        self.__ideal_gyroscope = numpy.degrees(numpy.array([(2 * Quaternion(q).conjugate() * Quaternion(r)).wxyz for q, r in zip(quaternion, quaternion_rate)])[:, 1:])

        self.__gyroscope = self.__ideal_gyroscope

        # Quaternion from gyroscope
        self.__quaternion = numpy.empty_like(quaternion)

        for index, gyroscope in enumerate(numpy.radians(self.__ideal_gyroscope)):
            previous_quaternion = Quaternion(self.__original_quaternion[0] if index == 0 else self.__quaternion[index - 1])

            quaternion_rate = 0.5 * previous_quaternion * Quaternion([0, gyroscope[0], gyroscope[1], gyroscope[2]])

            self.__quaternion[index] = (previous_quaternion + (quaternion_rate / self.sample_rate)).normalise().wxyz

        # Euler from quaternion
        self.__euler = numpy.array([Quaternion(q).to_euler() for q in self.quaternion])

        # Accelerometer from acceleration and quaternion
        self.__ideal_accelerometer = numpy.array([(Quaternion(q).conjugate() * Quaternion([0, a[0], a[1], a[2] + Simulator.GRAVITY]) * Quaternion(q)).wxyz[1:] for q, a in zip(self.quaternion, self.acceleration)]) / Simulator.GRAVITY

        self.__accelerometer = self.__ideal_accelerometer

    def __interpolate(self, x):
        return scipy.interpolate.interp1d(self.__original_time, x, kind="cubic", axis=0)(self.time)

    def __filter(self, x):
        b, a = scipy.signal.butter(1, self.__bandwidth / (0.5 * self.__sample_rate), btype="low")

        return scipy.signal.filtfilt(b, a, x, axis=0)

    @staticmethod
    def __diff(a):
        return numpy.diff(a, axis=0, prepend=[a[0]])

    def set_gyroscope(self,
                      range=0,  # °/s (0 = unlimited)
                      bandwidth=0,  # Hz (0 = unlimited)
                      offset=numpy.array([0, 0, 0]),  # °/s
                      sensitivity_error=numpy.array([0, 0, 0]),  # %
                      noise_density=numpy.array([0, 0, 0]),  # °/s/√Hz
                      cross_axis=numpy.array([0, 0, 0]),  # %
                      misalignment=0,  # °
                      bias_instability=numpy.array([0, 0, 0]),  # °/s
                      number_of_poles=0,  # used for bias instability noise generation
                      random_walk=numpy.array([0, 0, 0])):  # °/s/√Hz
        self.__gyroscope = Simulator.__sensor_model(self.__ideal_gyroscope, self.sample_rate, range, bandwidth, offset, sensitivity_error, noise_density, cross_axis, misalignment, bias_instability, number_of_poles, random_walk)

    def set_accelerometer(self,
                          range=0,  # g (0 = unlimited)
                          bandwidth=0,  # Hz (0 = unlimited)
                          offset=numpy.array([0, 0, 0]),  # g
                          sensitivity_error=numpy.array([0, 0, 0]),  # %
                          noise_density=numpy.array([0, 0, 0]),  # g/√Hz
                          cross_axis=numpy.array([0, 0, 0]),  # %
                          misalignment=0,  # °
                          bias_instability=numpy.array([0, 0, 0]),  # g
                          number_of_poles=0,  # used for bias instability noise generation
                          random_walk=numpy.array([0, 0, 0])):  # g/√Hz
        self.__accelerometer = Simulator.__sensor_model(self.__ideal_accelerometer, self.sample_rate, range, bandwidth, offset, sensitivity_error, noise_density, cross_axis, misalignment, bias_instability, number_of_poles, random_walk)

    @staticmethod
    def __sensor_model(sensor, sample_rate, range, bandwidth, offset, sensitivity_error, noise_density, cross_axis, misalignment, bias_instability, number_of_poles, random_walk):

        # Offset
        sensor += offset

        # Noise density
        if numpy.any(noise_density):
            sensor += numpy.random.normal(0, noise_density * numpy.sqrt(sample_rate), sensor.shape)

        # Bias instability
        if numpy.any(bias_instability) and number_of_poles > 0:
            end_index = number_of_poles + 1

            end_index = numpy.min((end_index, len(fractalcoef.DENOMINATOR)))

            sensor += scipy.signal.lfilter([1], fractalcoef.DENOMINATOR[:end_index], numpy.random.normal(0, bias_instability, sensor.shape), axis=0)

        # Rate random walk
        if numpy.any(bias_instability):
            sensor += numpy.cumsum(numpy.random.normal(0, random_walk / numpy.sqrt(sample_rate), sensor.shape), axis=0)

        # Cross-axis sensitivity and misalignment
        if numpy.any(cross_axis) or misalignment != 0:
            cross_axis_matrix = numpy.matrix([[1, cross_axis[1] / 100, cross_axis[2] / 100],
                                              [cross_axis[0] / 100, 1, cross_axis[2] / 100],
                                              [cross_axis[0] / 100, cross_axis[1] / 100, 1]])

            misalignment_matrix = Quaternion(axis=[1, 1, 1], angle=misalignment).to_matrix()

            combined_matrix = cross_axis_matrix * misalignment_matrix

            sensor = numpy.array([(combined_matrix * numpy.matrix(a).T).A1 for a in sensor])

        # Sensitivity error
        sensor *= 1 + (sensitivity_error / 100)

        # Bandwidth
        if bandwidth > 0:
            bandwidth = numpy.min((bandwidth, 0.499 * sample_rate))

            wn = bandwidth / (0.5 * sample_rate)

            b, a = scipy.signal.butter(1, wn, btype="low")

            sensor = scipy.signal.lfilter(b, a, sensor, axis=0)

        # Range
        if range != 0:
            sensor = numpy.clip(sensor, -range, +range)

        return sensor

    @property
    def start_time(self):
        return self.__start_time

    @property
    def sample_rate(self):
        return self.__sample_rate

    @property
    def time(self):
        return self.__time

    @property
    def position(self):
        return self.__position

    @property
    def velocity(self):
        return self.__velocity

    @property
    def acceleration(self):
        return self.__acceleration

    @property
    def euler(self):
        return self.__euler

    @property
    def quaternion(self):
        return self.__quaternion

    @property
    def gyroscope(self):
        return self.__gyroscope

    @property
    def accelerometer(self):
        return self.__accelerometer

    def plot_kinematics(self):
        plot_kinematics(self.__original_time, self.__original_position, self.__original_euler, self.time, self.position, self.velocity, self.acceleration, self.euler)

    def plot_imu(self):
        plot_imu(self.time, self.gyroscope, self.accelerometer)

    def plot_gyroscope(self):
        plot_xyz_error(self.time, self.__ideal_gyroscope, self.gyroscope, "Gyroscope", "°/s", "Ideal", "Model")

    def plot_accelerometer(self):
        plot_xyz_error(self.time, self.__ideal_accelerometer, self.accelerometer, "Accelerometer", "g", "Ideal", "Model")


RED = "tab:red"
GREEN = "tab:green"
BLUE = "tab:blue"
LIGHT_GREY = "silver"
DARK_GREY = "tab:gray"


class Axis(Enum):
    X = auto()
    Y = auto()
    Z = auto()

    @property
    def index(self):
        match self:
            case Axis.X:
                return 0
            case Axis.Y:
                return 1
            case Axis.Z:
                return 2

    @property
    def colour(self):
        match self:
            case Axis.X:
                return RED
            case Axis.Y:
                return GREEN
            case Axis.Z:
                return BLUE

    def __str__(self):
        match self:
            case Axis.X:
                return "X"
            case Axis.Y:
                return "Y"
            case Axis.Z:
                return "Z"


def plot_kinematics(original_time, original_position, original_euler, time, position, velocity, acceleration, euler):
    figure, axes = pyplot.subplots(nrows=4, ncols=3, sharex=True)

    figure.suptitle("Kinematics")

    for axis in Axis:
        axes[0, axis.index].set_title(axis)
        axes[3, axis.index].set_xlabel("Time (s)")

    axes[0, 0].set_ylabel("Position (m)")
    axes[1, 0].set_ylabel("Velocity (m/s)")
    axes[2, 0].set_ylabel("Acceleration (m/s/s)")
    axes[3, 0].set_ylabel("Euler (°)")

    __subplot_kinematics(axes, 0, original_time, original_position, time, position)
    __subplot_kinematics(axes, 1, None, None, time, velocity)
    __subplot_kinematics(axes, 2, None, None, time, acceleration)
    __subplot_kinematics(axes, 3, original_time, original_euler, time, euler)


def __subplot_kinematics(axes, row, original_time, original, time, simulated):
    for axis in Axis:
        if original is not None:
            axes[row, axis.index].plot(original_time, original[:, axis.index], LIGHT_GREY, marker="o", label="Original")

        axes[row, axis.index].plot(time, simulated[:, axis.index], axis.colour, label="Simulated")

        axes[row, axis.index].grid()
        axes[row, axis.index].legend()


def plot_imu(time, gyroscope, accelerometer):
    _, axes = pyplot.subplots(nrows=2, sharex=True)

    axes[0].set_title("IMU")

    for axis in Axis:
        axes[0].plot(time, gyroscope[:, axis.index], axis.colour, label=str(axis))

    axes[0].set_ylabel("Gyroscope (°/s)")
    axes[0].grid()
    axes[0].legend()

    for axis in Axis:
        axes[1].plot(time, accelerometer[:, axis.index], axis.colour, label=str(axis))

    axes[1].set_ylabel("Accelerometer (g)")
    axes[1].grid()
    axes[1].legend()

    axes[1].set_xlabel("Time (s)")


def plot_xyz_error(time, actual, measured, title="Acceleration", units="m/s/s", actual_label="Actual", measured_label="Measured"):
    _, axes = pyplot.subplots(nrows=4, sharex=True)

    axes[0].set_title(title)

    for axis in Axis:
        axes[axis.index].plot(time, actual[:, axis.index], LIGHT_GREY, label=actual_label)
        axes[axis.index].plot(time, measured[:, axis.index], axis.colour, label=measured_label)
        axes[axis.index].set_ylabel(str(axis) + " (" + units + ")")
        axes[axis.index].grid()
        axes[axis.index].legend()

    axes[3].plot(time, numpy.linalg.norm(measured - actual, ord=2, axis=1), DARK_GREY)
    axes[3].set_ylabel("Error" + " (" + units + ")")
    axes[3].grid()

    axes[3].set_xlabel("Time (s)")


def plot_euler_error(time, actual, measured, title="Euler", actual_label="Actual", measured_label="Measured"):
    _, axes = pyplot.subplots(nrows=4, sharex=True)

    axes[0].set_title(title)

    for axis in Axis:
        axes[axis.index].plot(time, actual[:, axis.index], LIGHT_GREY, label=actual_label)
        axes[axis.index].plot(time, measured[:, axis.index], axis.colour, label=measured_label)
        axes[axis.index].set_ylabel(str(axis) + " (°)")
        axes[axis.index].grid()
        axes[axis.index].legend()

    error = numpy.array([(Quaternion(euler=a).conjugate() * Quaternion(euler=m)).wxyz for a, m in zip(actual, measured)])

    error = numpy.array([Quaternion(q).to_axis_angle()[1] for q in Quaternion.fix_continuity(error)])

    axes[3].plot(time, error, DARK_GREY)
    axes[3].set_ylabel("Error" + " (°)")
    axes[3].grid()

    axes[3].set_xlabel("Time (s)")


def plot_3d(position,  # nx3 array (XYZ in meters)
            euler=None,  # nx3 array (XYZ in °)
            quaternion=None,  # nx4 array (WXYZ)
            matrix=None,  # nx3x3 array
            ref_a=None,  # nx3 array (XYZ in meters)
            ref_b=None,  # nx3 array (XYZ in meters)
            label="Position",
            ref_a_label="Ref A",
            ref_b_label="Ref B",
            title="3D",
            quiver_length=0.1,  # in meters
            samples_per_quiver=10,  # 1 = trail disabled
            azim=None,  # see mpl_toolkits.mplot3d.axes3d.Axes3D.view_init
            animate=False,
            sample_rate=None,
            auto_rotate=True,  # rotate azimuth during animation
            file_name="",  # must be .gif
            fps=15,  # animation frames per second
            figsize=None,  # see matplotlib.pyplot.figure
            dpi=None):  # see matplotlib.pyplot.figure

    # Calculate rotation matrix
    if euler is not None:
        matrix = [Quaternion(euler=e).to_matrix() for e in euler]

    if quaternion is not None:
        matrix = [Quaternion(q).to_matrix() for q in quaternion]

    if matrix is not None:
        scaled_matrix = [quiver_length * m for m in matrix]

        x_quiver_segments = [[(p[0], p[1], p[2]), (p[0] + m[0, 0], p[1] + m[1, 0], p[2] + m[2, 0])] for i, (p, m) in enumerate(zip(position, scaled_matrix)) if not i % samples_per_quiver]
        y_quiver_segments = [[(p[0], p[1], p[2]), (p[0] + m[0, 1], p[1] + m[1, 1], p[2] + m[2, 1])] for i, (p, m) in enumerate(zip(position, scaled_matrix)) if not i % samples_per_quiver]
        z_quiver_segments = [[(p[0], p[1], p[2]), (p[0] + m[0, 2], p[1] + m[1, 2], p[2] + m[2, 2])] for i, (p, m) in enumerate(zip(position, scaled_matrix)) if not i % samples_per_quiver]

        origin = numpy.array([p for i, p in enumerate(position) if not i % samples_per_quiver])

    # Create figure
    figure = pyplot.figure(figsize=figsize, dpi=dpi)

    axis = pyplot.axes(projection="3d")

    pyplot.subplots_adjust(top=0.95, bottom=0, left=0, right=1)

    # Create lines
    if ref_a is not None:
        ref_a_line, = axis.plot([], [], [], LIGHT_GREY, linestyle="None", marker="+", label=ref_a_label)

    if ref_b is not None:
        ref_b_line, = axis.plot([], [], [], LIGHT_GREY if matrix else DARK_GREY, label=ref_b_label)

    position_line, = axis.plot([], [], [], DARK_GREY if matrix else BLUE, label=None if (ref_a is None) and (ref_b is None) else label)

    # Create quivers
    if matrix is not None:
        x_quiver = axis.quiver([], [], [], [], [], [], color=RED, label="X")
        y_quiver = axis.quiver([], [], [], [], [], [], color=GREEN, label="Y")
        z_quiver = axis.quiver([], [], [], [], [], [], color=BLUE, label="Z")

        origin_line, = axis.plot([], [], [], "ko", markersize=2, zorder=numpy.inf)

    # Create time text
    if animate:
        time_text = pyplot.figtext(0.99, 0.01, "", horizontalalignment="right")

    # Set labels
    axis.set_xlabel("X (m)", labelpad=10)
    axis.set_ylabel("Y (m)", labelpad=10)
    axis.set_zlabel("Z (m)", labelpad=10)

    # Show legend
    if (ref_a is not None) or (ref_b is not None) or (matrix is not None):
        axis.legend(loc="upper left", frameon=0)

    # Set title
    axis.set_title(title)

    # Set view
    axis.view_init(azim=azim)

    # Animation variables
    if animate:
        fps = numpy.min((fps, sample_rate))
        samples_per_frame = int(sample_rate / fps)

        if auto_rotate:
            quaternion = [Quaternion(axis=[0, 0, 1], angle=axis.azim + numpy.degrees(numpy.arctan2(p[1], p[0]))).wxyz for p in position]
            quaternion = Quaternion.fix_continuity(quaternion)

            b, a = scipy.signal.butter(1, 0.1 / (0.5 * sample_rate), btype="low")  # 0.1 Hz bandwidth
            quaternion = scipy.signal.filtfilt(b, a, quaternion, axis=0)  # filter quaternion instead of scalar angle to avoid discontinuities

            azimuth = [Quaternion(q).to_euler()[2] for q in quaternion]

    # Update plot
    def update(frame):

        # Calculate index
        if animate:
            index = frame * samples_per_frame
        else:
            index = len(position) - 1

        # Set lines
        end_index = index + 1

        position_line.set_data(position[:end_index, :2].T)
        position_line.set_3d_properties(position[:end_index, 2])

        if ref_a is not None:
            ref_a_line.set_data(ref_a[:end_index, :2].T)
            ref_a_line.set_3d_properties(ref_a[:end_index, 2])

        if ref_b is not None:
            ref_b_line.set_data(ref_b[:end_index, :2].T)
            ref_b_line.set_3d_properties(ref_b[:end_index, 2])

        # Set quivers
        if matrix is not None:
            quiver_start_index = index if samples_per_quiver == 1 else 0
            quiver_end_index = int(index / samples_per_quiver) + 1

            x_quiver.set_segments(x_quiver_segments[quiver_start_index:quiver_end_index])
            y_quiver.set_segments(y_quiver_segments[quiver_start_index:quiver_end_index])
            z_quiver.set_segments(z_quiver_segments[quiver_start_index:quiver_end_index])

            origin_line.set_data(origin[quiver_start_index:quiver_end_index, :2].T)
            origin_line.set_3d_properties(origin[quiver_start_index:quiver_end_index, 2])

        # Set time text
        if animate:
            time_text.set_text("{:.3f}".format(index * (1 / sample_rate)) + " s")

        # Set limits
        all_xyz = position[:end_index, :]

        if ref_a is not None:
            all_xyz = numpy.concatenate((all_xyz, ref_a[:end_index, :]))

        if ref_b is not None:
            all_xyz = numpy.concatenate((all_xyz, ref_b[:end_index, :]))

        if matrix is not None:
            all_xyz = numpy.concatenate((all_xyz, [x[1] for x in x_quiver_segments[:quiver_end_index]], [y[1] for y in y_quiver_segments[:quiver_end_index]], [z[1] for z in z_quiver_segments[:quiver_end_index]]))

        all_xyz = all_xyz[~numpy.isnan(all_xyz).any(axis=1)]  # remove rows containing nan

        MIN_DISTANCE = 0.001

        axis.set_xlim3d(numpy.min(all_xyz[:, 0]) - MIN_DISTANCE, numpy.max(all_xyz[:, 0]) + MIN_DISTANCE)
        axis.set_ylim3d(numpy.min(all_xyz[:, 1]) - MIN_DISTANCE, numpy.max(all_xyz[:, 1]) + MIN_DISTANCE)
        axis.set_zlim3d(numpy.min(all_xyz[:, 2]) - MIN_DISTANCE, numpy.max(all_xyz[:, 2]) + MIN_DISTANCE)

        axis.set_box_aspect(numpy.max(([MIN_DISTANCE, MIN_DISTANCE, MIN_DISTANCE], numpy.ptp(all_xyz, axis=0)), axis=0))

        # Set view
        if animate and auto_rotate:
            axis.view_init(azim=azimuth[index])

    # Static plot
    if not animate:
        update(None)
        return

    # Animation
    anim = animation.FuncAnimation(figure, update, frames=int(len(position) / samples_per_frame), interval=1000 / fps, repeat=False, blit=False)

    if file_name:
        anim.save(file_name, writer=animation.PillowWriter(fps), dpi="figure", progress_callback=lambda i, n: print(f"Saving frame {i + 1} of {n}"))
    else:
        pyplot.show()  # play animation
