'''
单例模式实现模块。提供了两种线程安全的单例元类：Singleton（全局唯一）和 SingletonNamed（按名称唯一）。这在管理配置、设备连接等全局状态时非常有用。
'''

import threading
from typing import Type, TypeVar

T = TypeVar('T')


class Singleton(type):
    """
    用于创建全局单例的元类 (Metaclass)。

    任何使用此元类的类，在整个程序生命周期中只会有一个实例。
    子类会有它们自己独立的单例实例。
    此实现是线程安全的。
    """

    def __init__(cls, name, bases, dct):
        super().__init__(name, bases, dct)
        # 存储单例实例
        cls.__instances = None
        # 线程锁，确保创建实例时的线程安全
        cls.__lock = threading.Lock()

    def __call__(cls: Type[T], *args, **kwargs) -> T:
        # 如果实例已存在，直接返回
        instance = cls.__instances
        if instance is not None:
            return instance

        # 如果不存在，加锁创建
        with cls.__lock:
            # 双重检查锁定 (Double-checked locking)
            # 防止在等待锁的过程中其他线程已经创建了实例
            instance = cls.__instances
            if instance is not None:
                return instance

            # 创建新实例
            instance = super().__call__(*args, **kwargs)
            cls.__instances = instance
            return instance

    def singleton_clear_all(cls):
        """
        清除单例实例，下次调用时会重新创建。
        """
        with cls.__lock:
            cls.__instances = None


class SingletonNamed(type):
    """
    用于创建命名单例的元类。

    实例是基于构造函数的第一个参数（name）来缓存的。
    不同的 name 会对应不同的实例，相同的 name 返回同一个实例。
    每个使用此元类的类都有自己独立的缓存字典。
    此实现是线程安全的。
    """

    def __init__(cls, name, bases, dct):
        super().__init__(name, bases, dct)
        # 存储实例的字典，Key 为 name
        cls.__instances = {}
        cls.__lock = threading.Lock()

    def __call__(cls: Type[T], name, *args, **kwargs) -> T:
        # 尝试直接获取缓存实例
        try:
            return cls.__instances[name]
        except KeyError:
            pass

        # 加锁创建
        with cls.__lock:
            # 双重检查
            try:
                return cls.__instances[name]
            except KeyError:
                pass

            # 创建新实例并存入字典
            instance = super().__call__(name, *args, **kwargs)
            cls.__instances[name] = instance
            return instance

    def singleton_remove(cls, name):
        """
        移除指定名称的单例实例。
        下次请求该名称时将重新创建。

        Returns:
            bool: 是否成功移除（如果存在则移除并返回 True）。
        """
        # 字典删除操作本身是线程安全的（在 CPython 中）
        try:
            del cls.__instances[name]
            return True
        except KeyError:
            return False

    def singleton_clear_all(cls):
        """
        移除所有实例。
        """
        with cls.__lock:
            cls.__instances.clear()

    def singleton_instances(cls):
        """
        直接访问实例字典。

        Returns:
            dict: 包含所有命名实例的字典。
        """
        return cls.__instances