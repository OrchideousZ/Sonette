'''
网格处理模块。定义了 SelectedGrids 类，用于管理一组网格对象（Grid），提供筛选、索引、排序和集合运算功能。这在处理地图格子、背包物品格子等场景中非常有用。
'''

import operator
import typing as t


class SelectedGrids:
    """
    作用：网格集合类。
    用于包装一组 Grid 对象，提供类似 SQL 的查询、筛选、排序和集合操作。
    """

    def __init__(self, grids):
        self.grids = grids
        # 索引缓存，用于加速查询
        self.indexes: t.Dict[tuple, SelectedGrids] = {}

    def __iter__(self):
        return iter(self.grids)

    def __getitem__(self, item):
        if isinstance(item, int):
            return self.grids[item]
        else:
            # 支持切片操作，返回新的 SelectedGrids 对象
            return SelectedGrids(self.grids[item])

    def __contains__(self, item):
        return item in self.grids

    def __str__(self):
        # return str([str(grid) for grid in self])
        return '[' + ', '.join([str(grid) for grid in self]) + ']'

    def __len__(self):
        return len(self.grids)

    def __bool__(self):
        return self.count > 0

    # def __getattr__(self, item):
    #     return [grid.__getattribute__(item) for grid in self.grids]

    @property
    def location(self):
        """
        Returns:
            list[tuple]: 返回所有网格的坐标列表。
        """
        return [grid.location for grid in self.grids]

    @property
    def cost(self):
        """
        Returns:
            list[int]: 返回所有网格的代价列表（用于寻路）。
        """
        return [grid.cost for grid in self.grids]

    @property
    def weight(self):
        """
        Returns:
            list[int]: 返回所有网格的权重列表。
        """
        return [grid.weight for grid in self.grids]

    @property
    def count(self):
        """
        Returns:
            int: 网格数量。
        """
        return len(self.grids)

    def select(self, **kwargs):
        """
        作用：根据属性筛选网格。

        Args:
            **kwargs: Grid 对象的属性名和目标值。

        Returns:
            SelectedGrids: 包含符合条件网格的新对象。

        示例：
            grids.select(is_enemy=True, type='boss')
        """

        def matched(obj):
            flag = True
            for k, v in kwargs.items():
                obj_v = obj.__getattribute__(k)
                # 检查类型和值是否相等
                if type(obj_v) != type(v) or obj_v != v:
                    flag = False
            return flag

        return SelectedGrids([grid for grid in self.grids if matched(grid)])

    def create_index(self, *attrs):
        """
        作用：为网格集合创建索引，加速后续查询。

        Args:
            *attrs: 用于建立索引的属性名。
        """
        indexes = {}
        # index_keys = [(grid.__getattribute__(attr) for attr in attrs) for grid in self.grids]
        for grid in self.grids:
            # 创建索引键（元组）
            k = tuple(grid.__getattribute__(attr) for attr in attrs)
            try:
                indexes[k].append(grid)
            except KeyError:
                indexes[k] = [grid]

        # 将列表转换为 SelectedGrids 对象
        indexes = {k: SelectedGrids(v) for k, v in indexes.items()}
        self.indexes = indexes
        return indexes

    def indexed_select(self, *values):
        """
        作用：使用已创建的索引进行快速查询。
        必须先调用 create_index。
        """
        return self.indexes.get(values, SelectedGrids([]))

    def left_join(self, right, on_attr, set_attr, default=None):
        """
        作用：左连接操作。
        将另一个 SelectedGrids (right) 中的属性合并到当前对象中。

        Args:
            right (SelectedGrids): 右表。
            on_attr: 连接键（属性名列表）。
            set_attr: 需要复制的属性名列表。
            default: 如果未匹配到，设置的默认值。

        Returns:
            SelectedGrids: 返回自身（链式调用）。
        """
        right.create_index(*on_attr)
        for grid in self:
            attr_value = tuple([grid.__getattribute__(attr) for attr in on_attr])
            right_grid = right.indexed_select(*attr_value).first_or_none()
            if right_grid is not None:
                for attr in set_attr:
                    grid.__setattr__(attr, right_grid.__getattribute__(attr))
            else:
                for attr in set_attr:
                    grid.__setattr__(attr, default)

        return self

    def filter(self, func):
        """
        作用：使用自定义函数筛选网格。

        Args:
            func (callable): 接收 grid 对象，返回 bool。

        Returns:
            SelectedGrids:
        """
        return SelectedGrids([grid for grid in self if func(grid)])

    def set(self, **kwargs):
        """
        作用：批量设置网格属性。

        Args:
            **kwargs: 属性名和值。
        """
        for grid in self:
            for key, value in kwargs.items():
                grid.__setattr__(key, value)

    def get(self, attr):
        """
        作用：获取所有网格的指定属性值。

        Args:
            attr: 属性名。

        Returns:
            list: 属性值列表。
        """
        return [grid.__getattribute__(attr) for grid in self.grids]

    def call(self, func, **kwargs):
        """
        作用：对每个网格调用指定方法。

        Args:
            func (str): 方法名。
            **kwargs: 方法参数。

        Returns:
            list: 方法返回值列表。
        """
        return [grid.__getattribute__(func)(**kwargs) for grid in self]

    def first_or_none(self):
        """
        Returns:
            Grid | None: 返回第一个网格，如果为空则返回 None。
        """
        try:
            return self.grids[0]
        except IndexError:
            return None

    def add(self, grids):
        """
        作用：合并两个网格集合（去重）。
        使用 set 去重，要求 Grid 对象可哈希。

        Args:
            grids(SelectedGrids):

        Returns:
            SelectedGrids:
        """
        return SelectedGrids(list(set(self.grids + grids.grids)))

    def add_by_eq(self, grids):
        """
        作用：合并两个网格集合（去重）。
        使用 __eq__ 去重，适用于不可哈希的对象。

        Args:
            grids(SelectedGrids):

        Returns:
            SelectedGrids:
        """
        new = []
        for grid in self.grids + grids.grids:
            if grid not in new:
                new.append(grid)

        return SelectedGrids(new)

    def intersect(self, grids):
        """
        作用：取交集。

        Args:
            grids(SelectedGrids):

        Returns:
            SelectedGrids:
        """
        return SelectedGrids(list(set(self.grids).intersection(set(grids.grids))))

    def intersect_by_eq(self, grids):
        """
        作用：取交集（使用 __eq__）。

        Args:
            grids(SelectedGrids):

        Returns:
            SelectedGrids:
        """
        new = []
        for grid in self.grids:
            if grid in grids.grids:
                new.append(grid)

        return SelectedGrids(new)

    def delete(self, grids):
        """
        作用：从当前集合中删除指定的网格。

        Args:
            grids(SelectedGrids):

        Returns:
            SelectedGrids:
        """
        g = [grid for grid in self.grids if grid not in grids]
        return SelectedGrids(g)

    def sort(self, *args):
        """
        作用：根据属性排序。

        Args:
            args (str): 排序的属性名。

        Returns:
            SelectedGrids:
        """
        if not self:
            return self
        if len(args):
            grids = sorted(self.grids, key=operator.attrgetter(*args))
            return SelectedGrids(grids)
        else:
            return self

    def sort_by_camera_distance(self, camera):
        """
        作用：根据距离摄像机（中心点）的距离排序。

        Args:
            camera (tuple): 摄像机坐标 (x, y)。

        Returns:
            SelectedGrids:
        """
        import numpy as np
        if not self:
            return self
        location = np.array(self.location)
        # 计算曼哈顿距离或欧几里得距离的近似值（这里直接求和绝对差值）
        diff = np.sum(np.abs(location - camera), axis=1)
        # grids = [x for _, x in sorted(zip(diff, self.grids))]
        grids = tuple(np.array(self.grids)[np.argsort(diff)])
        return SelectedGrids(grids)

    def sort_by_clock_degree(self, center=(0, 0), start=(0, 1), clockwise=True):
        """
        作用：根据相对于中心点的角度排序（顺时针或逆时针）。

        Args:
            center (tuple): 中心点坐标。
            start (tuple): 起始向量，定义 0 度角。
            clockwise (bool): True 为顺时针。

        Returns:
            SelectedGrids:
        """
        import numpy as np
        if not self:
            return self
        vector = np.subtract(self.location, center)
        # 计算角度
        theta = np.arctan2(vector[:, 1], vector[:, 0]) / np.pi * 180
        vector = np.subtract(start, center)
        # 减去起始角度
        theta = theta - np.arctan2(vector[1], vector[0]) / np.pi * 180
        if not clockwise:
            theta = -theta
        # 归一化到 0-360
        theta[theta < 0] += 360
        grids = tuple(np.array(self.grids)[np.argsort(theta)])
        return SelectedGrids(grids)


class RoadGrids:
    """
    作用：道路网格类。
    用于处理路径规划中的路块。
    """

    def __init__(self, grids):
        """
        Args:
            grids (list): 网格列表的列表（二维结构）。
        """
        self.grids = []
        for grid in grids:
            if isinstance(grid, list):
                self.grids.append(SelectedGrids(grids=grid))
            else:
                self.grids.append(SelectedGrids(grids=[grid]))

    def __str__(self):
        return str(' - '.join([str(grid) for grid in self.grids]))

    def roadblocks(self):
        """
        Returns:
            SelectedGrids: 返回所有完全被敌人阻挡的路块。
        """
        grids = []
        for block in self.grids:
            if block.count == block.select(is_enemy=True).count:
                grids += block.grids
        return SelectedGrids(grids)

    def potential_roadblocks(self):
        """
        Returns:
            SelectedGrids: 返回潜在的路障（除了一个敌人外全是空的）。
        """
        grids = []
        for block in self.grids:
            if any([grid.is_fleet for grid in block]):
                continue
            if any([grid.is_cleared for grid in block]):
                continue
            if block.count - block.select(is_enemy=True).count == 1:
                grids += block.select(is_enemy=True).grids
        return SelectedGrids(grids)

    def first_roadblocks(self):
        """
        Returns:
            SelectedGrids: 返回第一个遇到的路障。
        """
        grids = []
        for block in self.grids:
            if any([grid.is_fleet for grid in block]):
                continue
            if any([grid.is_cleared for grid in block]):
                continue
            if block.select(is_enemy=True).count >= 1:
                grids += block.select(is_enemy=True).grids
        return SelectedGrids(grids)

    def combine(self, road):
        """
        作用：合并两条道路。

        Args:
            road (RoadGrids):

        Returns:
            RoadGrids:
        """
        out = RoadGrids([])
        for select_1 in self.grids:
            for select_2 in road.grids:
                select = select_1.add(select_2)
                out.grids.append(select)

        return out