'''
定义了几个关键的装饰器（Decorators），用于增强函数和方法的行为，是项目核心框架的一部分。
    Config 类/@Config.when 装饰器： 这是最核心的装饰器。它的作用是实现运行时多态（Runtime Polymorphism）或功能分支。它允许定义多个同名函数，
                                  但根据配置文件（AzurLaneConfig）中的不同选项值，在运行时只调用符合条件的那个函数版本。这极大地提高了代码的灵活性和可维护性。
    cached_property 类： 一个属性装饰器。它将一个方法的计算结果缓存起来，使其在第一次访问后表现得像一个普通的属性，后续访问将直接返回缓存值，避免重复计算。
    @function_drop 装饰器： 这是一个调试/测试专用的装饰器。它以给定的概率（rate）随机跳过函数的实际执行，并记录日志。这用于模拟模拟器卡顿或操作失败等随机异常情况，以测试脚本的鲁棒性。
    @run_once 装饰器： 确保一个函数或方法在整个程序生命周期中只执行一次。

    调用的模块及位置：
        random：用于 @function_drop 装饰器生成随机数。
        re：用于 @function_drop 装饰器处理日志中的类名和参数。
        functools.wraps：用于在装饰器中保留原始函数的元数据。
        module.logger：用于记录日志信息。
        module.exception.ScriptError：虽然未直接导入，但用于 @Config.when 内部逻辑可能涉及配置错误判断。
'''

import random
import re
from functools import wraps
from typing import Callable, Generic, TypeVar

T = TypeVar("T")


class Config:
    """
    装饰器类，根据配置文件的值调用同名函数中不同的实现。
    实现运行时多态和功能分支。
    """
    # 类属性，存储所有被 @Config.when 装饰的函数定义
    # 结构: {'函数名': [{'options': {...}, 'func': <function>}, ...]}
    func_list = {}

    @classmethod
    def when(cls, **kwargs):
        """
        装饰器工厂：根据指定的配置选项值决定是否执行该函数体。

        Args:
            **kwargs: AzurLaneConfig 中的任意配置项和目标值。
        """
        from module.logger import logger
        options = kwargs

        def decorate(func):
            name = func.__name__
            data = {'options': options, 'func': func}

            # 将该函数及其条件添加到 func_list
            if name not in cls.func_list:
                cls.func_list[name] = [data]
            else:
                override = False
                for record in cls.func_list[name]:
                    # 如果配置条件完全相同，则覆盖旧的函数实现
                    if record['options'] == data['options']:
                        record['func'] = data['func']
                        override = True
                if not override:
                    # 否则，添加新的函数实现和条件
                    cls.func_list[name].append(data)

            @wraps(func)
            def wrapper(self, *args, **kwargs):
                """
                包装函数：在运行时检查配置，并调用匹配的函数。

                Args:
                    self: ModuleBase 实例 (包含 self.config)。
                """
                for record in cls.func_list[name]:

                    # 检查当前配置是否满足 record['options'] 中的所有条件
                    flag = [value is None or self.config.__getattribute__(key) == value
                            for key, value in record['options'].items()]

                    # 如果所有条件都满足
                    if all(flag):
                        # 调用匹配到的函数实现并返回结果
                        return record['func'](self, *args, **kwargs)

                # 如果所有带有条件的函数版本都不匹配
                logger.warning(f'No option fits for {name}, using the last define func.')
                # 则退回到原始函数（即最后定义的那个版本，通常是默认/通用逻辑）
                return func(self, *args, **kwargs)

            return wrapper

        return decorate


class cached_property(Generic[T]):
    """
    属性缓存装饰器。
    计算一次后，将结果作为实例属性存储，后续访问直接返回属性值。
    """

    def __init__(self, func: Callable[..., T]):
        self.func = func

    def __get__(self, obj, cls) -> T:
        if obj is None:
            return self

        # 首次调用时：执行函数计算值，并将结果存储到实例的 __dict__ 中
        value = obj.__dict__[self.func.__name__] = self.func(obj)
        return value


def del_cached_property(obj, name):
    """
    安全地删除一个被缓存的属性。

    Args:
        obj: 实例对象。
        name (str): 属性名称。
    """
    try:
        # 尝试从实例的字典中删除缓存值
        del obj.__dict__[name]
    except KeyError:
        # 如果不存在，则忽略
        pass


def has_cached_property(obj, name):
    """
    检查一个属性是否已经被缓存。

    Args:
        obj: 实例对象。
        name (str): 属性名称。
    """
    return name in obj.__dict__


def set_cached_property(obj, name, value):
    """
    手动设置一个缓存属性的值。

    Args:
        obj: 实例对象。
        name (str): 属性名称。
        value: 要设置的值。
    """
    obj.__dict__[name] = value


def function_drop(rate=0.5, default=None):
    """
    函数丢弃装饰器：以指定概率跳过函数调用，用于测试脚本鲁棒性。

    Args:
        rate (float): 丢弃概率（0到1）。
        default: 被丢弃时函数的返回值。
    """
    from module.logger import logger

    def decorate(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            # 随机数大于丢弃率时，正常执行
            if random.uniform(0, 1) > rate:
                return func(*args, **kwargs)
            else:
                # 否则，执行丢弃操作
                cls = ''
                # 格式化参数列表用于日志输出
                arguments = [str(arg) for arg in args]
                if len(arguments):
                    # 尝试从第一个参数（通常是 self）中提取类名
                    matched = re.search('<(.*?) object at', arguments[0])
                    if matched:
                        cls = matched.group(1) + '.'
                        arguments.pop(0)
                # 添加关键字参数
                arguments += [f'{k}={v}' for k, v in kwargs.items()]
                arguments = ', '.join(arguments)
                # 记录丢弃日志
                logger.info(f'Dropped: {cls}{func.__name__}({arguments})')
                # 返回默认值
                return default

        return wrapper

    return decorate


def run_once(f):
    """
    执行一次装饰器：确保函数只运行一次。
    """

    def wrapper(*args, **kwargs):
        if not wrapper.has_run:
            wrapper.has_run = True
            return f(*args, **kwargs)

    # 在包装函数上添加一个状态标志
    wrapper.has_run = False
    return wrapper