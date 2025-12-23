'''
点和线的几何处理模块。提供了 Points 类用于处理点集（如聚类、求均值），Lines 类用于处理直线（如霍夫变换检测到的线），以及一系列坐标转换和几何计算函数。
'''

import numpy as np
from scipy import optimize

from .utils import area_pad


class Points:
    """
    作用：点集包装类。
    封装了 numpy 数组，提供方便的点集操作方法。
    """

    def __init__(self, points):
        if points is None or len(points) == 0:
            self._bool = False
            self.points = None
        else:
            self._bool = True
            self.points = np.array(points)
            if len(self.points.shape) == 1:
                self.points = np.array([self.points])
            self.x, self.y = self.points.T

    def __str__(self):
        return str(self.points)

    __repr__ = __str__

    def __iter__(self):
        return iter(self.points)

    def __getitem__(self, item):
        return self.points[item]

    def __len__(self):
        if self:
            return len(self.points)
        else:
            return 0

    def __bool__(self):
        return self._bool

    def link(self, point, is_horizontal=False):
        """
        作用：将点集中的每个点与指定点连接，生成直线集合。

        Args:
            point: 目标点 (x, y)。
            is_horizontal: 是否生成水平线。
        """
        if is_horizontal:
            lines = [[y, np.pi / 2] for y in self.y]
            return Lines(lines, is_horizontal=True)
        else:
            x, y = point
            # 计算极坐标参数 rho, theta
            theta = -np.arctan((self.x - x) / (self.y - y))
            rho = self.x * np.cos(theta) + self.y * np.sin(theta)
            lines = np.array([rho, theta]).T
            return Lines(lines, is_horizontal=False)

    def mean(self):
        """
        计算点集的中心点（均值）。
        """
        if not self:
            return None

        return np.round(np.mean(self.points, axis=0)).astype(int)

    def group(self, threshold=3):
        """
        作用：将距离相近的点聚类，返回每组的中心点。

        Args:
            threshold: 距离阈值。

        Returns:
            np.array: 聚类后的点集。
        """
        if not self:
            return np.array([])
        groups = []
        points = self.points
        if len(points) == 1:
            return np.array([points[0]])

        while len(points):
            p0, p1 = points[0], points[1:]
            # 计算距离
            distance = np.sum(np.abs(p1 - p0), axis=1)
            # 找到距离小于阈值的点，归为一组，计算均值
            new = Points(np.append(p1[distance <= threshold], [p0], axis=0)).mean().tolist()
            groups.append(new)
            # 剩余的点继续处理
            points = p1[distance > threshold]

        return np.array(groups)


class Lines:
    """
    作用：直线集合包装类。
    使用极坐标 (rho, theta) 表示直线。
    """
    MID_Y = 360  # 屏幕中心 Y 坐标，用于计算中点

    def __init__(self, lines, is_horizontal):
        if lines is None or len(lines) == 0:
            self._bool = False
            self.lines = None
        else:
            self._bool = True
            self.lines = np.array(lines)
            if len(self.lines.shape) == 1:
                self.lines = np.array([self.lines])
            self.rho, self.theta = self.lines.T
        self.is_horizontal = is_horizontal

    def __str__(self):
        return str(self.lines)

    __repr__ = __str__

    def __iter__(self):
        return iter(self.lines)

    def __getitem__(self, item):
        return Lines(self.lines[item], is_horizontal=self.is_horizontal)

    def __len__(self):
        if self:
            return len(self.lines)
        else:
            return 0

    def __bool__(self):
        return self._bool

    @property
    def sin(self):
        return np.sin(self.theta)

    @property
    def cos(self):
        return np.cos(self.theta)

    @property
    def mean(self):
        """
        计算直线的平均值。
        """
        if not self:
            return None
        if self.is_horizontal:
            return np.mean(self.lines, axis=0)
        else:
            # 对于非水平线，先计算中点 X 坐标的均值，再还原回 rho
            x = np.mean(self.mid)
            theta = np.mean(self.theta)
            rho = x * np.cos(theta) + self.MID_Y * np.sin(theta)
            return np.array((rho, theta))

    @property
    def mid(self):
        """
        计算直线在 MID_Y 处的 X 坐标（如果是水平线则为 rho）。
        """
        if not self:
            return np.array([])
        if self.is_horizontal:
            return self.rho
        else:
            return (self.rho - self.MID_Y * self.sin) / self.cos

    def get_x(self, y):
        """根据 Y 计算 X"""
        return (self.rho - y * self.sin) / self.cos

    def get_y(self, x):
        """根据 X 计算 Y"""
        return (self.rho - x * self.cos) / self.sin

    def add(self, other):
        """合并两个直线集合"""
        if not other:
            return self
        if not self:
            return other
        lines = np.append(self.lines, other.lines, axis=0)
        return Lines(lines, is_horizontal=self.is_horizontal)

    def move(self, x, y):
        """平移直线"""
        if not self:
            return self
        if self.is_horizontal:
            self.lines[:, 0] += y
        else:
            self.lines[:, 0] += x * self.cos + y * self.sin
        return Lines(self.lines, is_horizontal=self.is_horizontal)

    def sort(self):
        """根据中点位置排序"""
        if not self:
            return self
        lines = self.lines[np.argsort(self.mid)]
        return Lines(lines, is_horizontal=self.is_horizontal)

    def group(self, threshold=3):
        """
        将相近的直线聚类合并。
        """
        if not self:
            return self
        lines = self.sort()
        prev = 0
        regrouped = []
        group = []
        for mid, line in zip(lines.mid, lines.lines):
            line = line.tolist()
            if mid - prev > threshold:
                if len(regrouped) == 0:
                    if len(group) != 0:
                        regrouped = [group]
                else:
                    regrouped += [group]
                group = [line]
            else:
                group.append(line)
            prev = mid
        regrouped += [group]
        regrouped = np.vstack([Lines(r, is_horizontal=self.is_horizontal).mean for r in regrouped])
        return Lines(regrouped, is_horizontal=self.is_horizontal)

    def distance_to_point(self, point):
        """计算点到直线的距离"""
        x, y = point
        return self.rho - x * self.cos - y * self.sin

    @staticmethod
    def cross_two_lines(lines1, lines2):
        """计算两组直线的交点"""
        for rho1, sin1, cos1 in zip(lines1.rho, lines1.sin, lines1.cos):
            for rho2, sin2, cos2 in zip(lines2.rho, lines2.sin, lines2.cos):
                a = np.array([[cos1, sin1], [cos2, sin2]])
                b = np.array([rho1, rho2])
                yield np.linalg.solve(a, b)

    def cross(self, other):
        """计算与另一组直线的交点"""
        points = np.vstack(self.cross_two_lines(self, other))
        points = Points(points)
        return points

    def delete(self, other, threshold=3):
        """删除与 other 中直线相近的直线"""
        if not self:
            return self

        other_mid = other.mid
        lines = []
        for mid, line in zip(self.mid, self.lines):
            if np.any(np.abs(other_mid - mid) < threshold):
                continue
            lines.append(line)

        return Lines(lines, is_horizontal=self.is_horizontal)


def area2corner(area):
    """
    Args:
        area: (x1, y1, x2, y2)

    Returns:
        np.ndarray: [左上, 右上, 左下, 右下] 四个角点坐标
    """
    return np.array([[area[0], area[1]], [area[2], area[1]], [area[0], area[3]], [area[2], area[3]]])


def corner2area(corner):
    """
    Args:
        corner: [左上, 右上, 左下, 右下]

    Returns:
        np.ndarray: (x1, y1, x2, y2) 包围盒
    """
    x, y = np.array(corner).T
    return np.rint([np.min(x), np.min(y), np.max(x), np.max(y)]).astype(int)


def corner2inner(corner):
    """
    获取梯形内的最大内接矩形。

    Args:
        corner: ((x0, y0), (x1, y1), (x2, y2), (x3, y3))

    Returns:
        tuple[int]: (upper_left_x, upper_left_y, bottom_right_x, bottom_right_y).
    """
    x0, y0, x1, y1, x2, y2, x3, y3 = np.array(corner).flatten()
    area = tuple(np.rint((max(x0, x2), max(y0, y1), min(x1, x3), min(y2, y3))).astype(int))
    return area


def corner2outer(corner):
    """
    获取梯形的最小外接矩形。

    Args:
        corner: ((x0, y0), (x1, y1), (x2, y2), (x3, y3))

    Returns:
        tuple[int]: (upper_left_x, upper_left_y, bottom_right_x, bottom_right_y).
    """
    x0, y0, x1, y1, x2, y2, x3, y3 = np.array(corner).flatten()
    area = tuple(np.rint((min(x0, x2), min(y0, y1), max(x1, x3), max(y2, y3))).astype(int))
    return area


def trapezoid2area(corner, pad=0):
    """
    将梯形角点转换为矩形区域。

    Args:
        corner: ((x0, y0), (x1, y1), (x2, y2), (x3, y3))
        pad (int):
            正值：内接矩形并内缩 pad。
            负值或0：外接矩形并内缩 pad。

    Returns:
        tuple[int]: (upper_left_x, upper_left_y, bottom_right_x, bottom_right_y).
    """
    if pad > 0:
        return area_pad(corner2inner(corner), pad=pad)
    elif pad < 0:
        return area_pad(corner2outer(corner), pad=pad)
    else:
        return area_pad(corner2area(corner), pad=pad)


def points_to_area_generator(points, shape):
    """
    将网格点转换为区域生成器。

    Args:
        points (np.ndarray): N x 2 数组。
        shape (tuple): (x, y) 网格形状。

    Yields:
        tuple, np.ndarray: ((x, y), [四个角点])
    """
    points = points.reshape(*shape[::-1], 2)
    for y in range(shape[1] - 1):
        for x in range(shape[0] - 1):
            area = np.array([points[y, x], points[y, x + 1], points[y + 1, x], points[y + 1, x + 1]])
            yield ((x, y), area)


def get_map_inner(points):
    """
    获取点集的中心点。
    """
    points = np.array(points)
    if len(points.shape) == 1:
        points = np.array([points])

    return np.mean(points, axis=0)


def separate_edges(edges, inner):
    """
    根据内部点将边缘分为上下两部分。

    Args:
        edges: 边缘坐标列表。
        inner (float, int): 分隔点。

    Returns:
        float, float: 下边缘和上边缘。
    """
    if len(edges) == 0:
        return None, None
    elif len(edges) == 1:
        edge = edges[0]
        return (None, edge) if edge > inner else (edge, None)
    else:
        lower = [edge for edge in edges if edge < inner]
        upper = [edge for edge in edges if edge > inner]
        lower = lower[0] if len(lower) else None
        upper = upper[-1] if len(upper) else None
        return lower, upper


def perspective_transform(points, data):
    """
    透视变换。

    Args:
        points: 2D 数组 (n, 2)
        data: 透视变换矩阵 (3, 3)

    Returns:
        np.ndarray: 变换后的点集
    """
    points = np.pad(np.array(points), ((0, 0), (0, 1)), mode='constant', constant_values=1)
    matrix = data.dot(points.T)
    x, y = matrix[0] / matrix[2], matrix[1] / matrix[2]
    points = np.array([x, y]).T
    return points


def fit_points(points, mod, encourage=1):
    """
    将一组点拟合到网格上。
    用于在图像上找到最符合网格分布的点。

    Args:
        points: 图像上的点集 (n, 2)
        mod: 点的公差/间距 (x, y)。
        encourage (int, float): 拟合紧密度参数。

    Returns:
        np.ndarray: 最佳拟合偏移量 (x, y)
    """
    encourage = np.square(encourage)
    mod = np.array(mod)
    points = np.array(points) % mod
    points = np.append(points - mod, points, axis=0)

    def cal_distance(point):
        distance = np.linalg.norm(points - point, axis=1)
        return np.sum(1 / (1 + np.exp(encourage / distance) / distance))

    # 暴力搜索全局最小值
    area = np.append(-mod - 10, mod + 10)
    result = optimize.brute(cal_distance, ((area[0], area[2]), (area[1], area[3])))
    return result % mod