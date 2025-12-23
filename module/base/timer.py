'''
计时器模块。提供了一个 Timer 类，用于处理超时逻辑。
它不仅支持基于时间（秒）的超时，还支持基于调用次数（count）的超时，这在处理游戏帧率不稳定或操作响应延迟时非常有用（例如：截图 10 次或等待 5 秒，满足其一即视为超时）。
'''

from datetime import datetime
from functools import wraps
from time import sleep, time


def timer(function):
    """
    装饰器：用于计算函数执行时间，仅用于调试。
    """

    @wraps(function)
    def function_timer(*args, **kwargs):
        start = time()
        result = function(*args, **kwargs)
        cost = time() - start
        print(f'{function.__name__}: {cost:.10f} s')
        return result

    return function_timer


def now():
    """
    获取当前时间（不带时区信息），去除微秒部分。
    """
    return datetime.now().replace(microsecond=0)


def nowtz():
    """
    获取当前时间（带时区信息），去除微秒部分。
    """
    return datetime.now().replace(microsecond=0).astimezone()


class Timer:
    def __init__(self, limit, count=0):
        """
        双重限制计时器：时间限制和访问次数限制。
        访问次数限制可以提高在慢速设备上的鲁棒性（例如截图耗时很长，导致实际检查次数很少）。

        Args:
            limit (int | float): 时间限制（秒）。
            count (int): 访问次数限制。默认为 0（不限制次数）。
        """
        self.limit = limit
        self.count = count
        self._start = 0.  # 计时开始时间戳
        self._access = 0  # 当前访问次数

    @classmethod
    def from_seconds(cls, limit, speed=0.5):
        """
        根据给定的秒数创建计时器，并自动估算需要的 count。

        Args:
            limit (int | float): 时间限制。
            speed (int | float): 预估每次操作（如截图）的耗时。
                如果耗时 > 0.5s，通常认为设备较慢。
        """
        count = int(limit / speed)
        return cls(limit, count=count)

    def start(self):
        """
        启动计时器。
        如果计时器尚未启动，reached() 会始终返回 True。
        这样设计是为了支持类似 "while 1: if interval.reached(): ..." 的快速首次执行逻辑。
        """
        if self._start <= 0:
            self._start = time()
            self._access = 0

        return self

    def started(self):
        """
        Returns:
            bool: 计时器是否已启动。
        """
        return self._start > 0

    def current_time(self):
        """
        Returns:
            float: 已经过去的时间（秒）。
        """
        if self._start > 0:
            diff = time() - self._start
            if diff < 0:
                diff = 0.
            return diff
        else:
            return 0.

    def current_count(self):
        """
        Returns:
            int: 已经访问的次数。
        """
        return self._access

    def set(self, current=None, count=None, speed=0.5):
        """
        直接设置计时器的内部状态。
        通常用于在特定逻辑中手动调整计时器进度。

        Args:
            current (int, float): 已过去的时间。
            count (int): 已访问的次数。
            speed (int, float): 用于根据时间估算次数的速度系数。
        """
        if current is not None:
            if count is not None:
                # 同时设置时间和次数
                self._start = time() - current
                self._access = count
            else:
                # 仅设置时间，根据速度估算次数
                count = int(current / speed)
                self._start = time() - current
                self._access = count
        else:
            if count is not None:
                # 仅设置次数
                self._access = count
            else:
                # 什么都不做
                pass
        return self

    def add_count(self):
        """
        手动增加一次访问计数。
        """
        self._access += 1
        return self

    def reached(self):
        """
        检查是否达到限制（超时或超次）。
        每次调用此方法都会增加一次访问计数。

        Returns:
            bool: 如果超时或超次，返回 True。
        """
        # 每次调用 reached() 视为一次访问
        self._access += 1
        if self._start > 0:
            # 必须同时满足：访问次数 > 限制次数 且 耗时 > 限制时间
            # 这样设计是为了防止在极慢的设备上，时间到了但只执行了很少几次检查
            return self._access > self.count and time() - self._start > self.limit
        else:
            # 未启动，返回 True 以便快速执行第一次
            return True

    def reset(self):
        """
        重置计时器，相当于重新开始计时。
        """
        self._start = time()
        self._access = 0
        return self

    def clear(self):
        """
        清除计时器状态，使其变为“未启动”且“已超时”状态。
        """
        self._start = 0.
        self._access = self.count
        return self

    def reached_and_reset(self):
        """
        检查是否到达限制，如果是，则自动重置。
        常用于周期性任务。

        Returns:
            bool: 是否到达限制。
        """
        if self.reached():
            self.reset()
            return True
        else:
            return False

    def wait(self):
        """
        阻塞等待，直到计时器到达时间限制。
        """
        diff = self._start + self.limit - time()
        if diff > 0:
            sleep(diff)

    def show(self):
        from module.logger import logger
        logger.info(str(self))

    def __str__(self):
        # 格式示例: Timer(limit=2.351/3, count=4/6)
        return f'Timer(limit={round(self.current_time(), 3)}/{self.limit}, count={self._access}/{self.count})'

    __repr__ = __str__