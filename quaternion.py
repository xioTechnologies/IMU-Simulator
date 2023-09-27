import numpy


class Quaternion:
    def __init__(self, wxyz=None, euler=None, axis=None, angle=None):
        if wxyz is not None:
            self.__wxyz = numpy.array(wxyz).squeeze()
            return

        if euler is not None:
            self.__wxyz = Quaternion.__from_euler(euler[0], euler[1], euler[2])
            return

        if (axis is not None) and (angle is not None):
            self.__wxyz = Quaternion.__from_axis_angle(axis, angle)
            return

        self.__wxyz = numpy.array([1, 0, 0, 0])

    @staticmethod
    def __from_euler(roll, pitch, yaw):  # Quaternions and Rotation Sequence by Jack B. Kuipers, ISBN 0-691-10298-8, Page 166
        psi = numpy.radians(yaw)
        theta = numpy.radians(pitch)
        phi = numpy.radians(roll)

        cos_half_psi = numpy.cos(0.5 * psi)
        sin_half_psi = numpy.sin(0.5 * psi)

        cos_half_theta = numpy.cos(0.5 * theta)
        sin_half_theta = numpy.sin(0.5 * theta)

        cos_half_phi = numpy.cos(0.5 * phi)
        sin_half_phi = numpy.sin(0.5 * phi)

        return numpy.array([cos_half_psi * cos_half_theta * cos_half_phi + sin_half_psi * sin_half_theta * sin_half_phi,
                            cos_half_psi * cos_half_theta * sin_half_phi - sin_half_psi * sin_half_theta * cos_half_phi,
                            cos_half_psi * sin_half_theta * cos_half_phi + sin_half_psi * cos_half_theta * sin_half_phi,
                            sin_half_psi * cos_half_theta * cos_half_phi - cos_half_psi * sin_half_theta * sin_half_phi])

    @staticmethod
    def __from_axis_angle(axis, angle):
        axis = axis / numpy.linalg.norm(axis)

        half_angle = 0.5 * numpy.radians(angle)

        return numpy.array([numpy.cos(half_angle), axis[0] * numpy.sin(half_angle), axis[1] * numpy.sin(half_angle), axis[2] * numpy.sin(half_angle)])

    @property
    def wxyz(self):
        return self.__wxyz

    @property
    def w(self):
        return self.__wxyz[0]

    @property
    def x(self):
        return self.__wxyz[1]

    @property
    def y(self):
        return self.__wxyz[2]

    @property
    def z(self):
        return self.__wxyz[3]

    def to_matrix(self):  # Quaternions and Rotation Sequence by Jack B. Kuipers, ISBN 0-691-10298-8, Page 167
        q = self

        return numpy.matrix([[2 * (q.w * q.w - 0.5 + q.x * q.x), 2 * (q.x * q.y - q.w * q.z), 2 * (q.x * q.z + q.w * q.y)],
                             [2 * (q.x * q.y + q.w * q.z), 2 * (q.w * q.w - 0.5 + q.y * q.y), 2 * (q.y * q.z - q.w * q.x)],
                             [2 * (q.x * q.z - q.w * q.y), 2 * (q.y * q.z + q.w * q.x), 2 * (q.w * q.w - 0.5 + q.z * q.z)]])

    def to_euler(self):  # Quaternions and Rotation Sequence by Jack B. Kuipers, ISBN 0-691-10298-8, Page 168
        q = self

        roll = numpy.degrees(numpy.arctan2(2 * (q.y * q.z + q.w * q.x), 2 * (q.w * q.w - 0.5 + q.z * q.z)))

        pitch = numpy.degrees(-1 * numpy.arcsin(numpy.clip(2 * (q.x * q.z - q.w * q.y), -1, 1)))

        yaw = numpy.degrees(numpy.arctan2(2 * (q.x * q.y + q.w * q.z), 2 * (q.w * q.w - 0.5 + q.x * q.x)))

        return numpy.array([roll, pitch, yaw])

    def to_axis_angle(self):
        angle = 2 * numpy.arccos(numpy.clip(self.w, -1, 1))

        if angle == 0:
            axis = numpy.array([1, 0, 0])
        else:
            sin_half_angle = numpy.sin(0.5 * angle)

            axis = self.wxyz[1:] / sin_half_angle

        return axis, numpy.degrees(angle)

    def conjugate(self):
        return Quaternion(self.wxyz * numpy.array([1, -1, -1, -1]))

    def normalise(self):
        return Quaternion(self.wxyz / numpy.linalg.norm(self.wxyz))

    def __add__(self, other):
        return Quaternion(self.wxyz + other.wxyz)

    def __sub__(self, other):
        return Quaternion(self.wxyz - other.wxyz)

    def __mul__(self, other):
        if isinstance(other, Quaternion):
            a = self
            b = other

            return Quaternion([a.w * b.w - a.x * b.x - a.y * b.y - a.z * b.z,
                               a.w * b.x + a.x * b.w + a.y * b.z - a.z * b.y,
                               a.w * b.y - a.x * b.z + a.y * b.w + a.z * b.x,
                               a.w * b.z + a.x * b.y - a.y * b.x + a.z * b.w])
        else:
            return Quaternion(self.wxyz * other)

    def __rmul__(self, other):
        return self.__mul__(other)

    def __truediv__(self, other):
        return Quaternion(self.wxyz / other)

    def __str__(self):
        return str(self.wxyz)

    @staticmethod
    def fix_continuity(wxyz):
        for index, _ in enumerate(wxyz):
            if index == 0:
                continue

            a = numpy.linalg.norm(wxyz[index] - wxyz[index - 1])
            b = numpy.linalg.norm((-1 * wxyz[index]) - wxyz[index - 1])

            if b < a:  # use negated/non-negated quaternion corresponding to the smallest change in euclidean distance
                wxyz[index] *= -1

        return wxyz
