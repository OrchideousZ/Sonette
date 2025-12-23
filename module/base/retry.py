'''
重试机制模块。提供了一个装饰器 retry 和一个函数 retry_call，用于在代码执行失败（抛出异常）时自动重试。
支持设置重试次数、延时、最大延时、指数退避（backoff）和随机抖动（jitter），非常适合处理不稳定的网络请求或模拟器操作。
'''

import functools
import random
import time
from functools import partial

from module.logger import logger as logging_logger

"""
本模块修改自 Python 的 `retry` 库，增加了一些定制化功能。
"""

try:
    from decorator import decorator
except ImportError:
    # 如果没有安装 decorator 库，提供一个简单的替代实现
    def decorator(caller):
        """ Turns caller into a decorator.
        Unlike decorator module, function signature is not preserved.

        :param caller: caller(f, *args, **kwargs)
        """

        def decor(f):
            @functools.wraps(f)
            def wrapper(*args, **kwargs):
                return caller(f, *args, **kwargs)

            return wrapper

        return decor


def __retry_internal(f, exceptions=Exception, tries=-1, delay=0, max_delay=None, backoff=1, jitter=0,
                     logger=logging_logger):
    """
    执行函数并在失败时重试的核心逻辑。

    :param f: 要执行的函数。
    :param exceptions: 需要捕获并重试的异常类型（或元组）。默认捕获所有 Exception。
    :param tries: 最大尝试次数。默认 -1 表示无限重试。
    :param delay: 初始重试延时（秒）。
    :param max_delay: 最大延时上限。
    :param backoff: 延时倍数（指数退避）。每次重试 delay = delay * backoff。
    :param jitter: 随机抖动时间。可以是固定秒数，也可以是 (min, max) 元组。
    :param logger: 用于记录重试警告的日志对象。
    :returns: 函数 f 的执行结果。
    """
    _tries, _delay = tries, delay
    while _tries:
        try:
            return f()
        except exceptions as e:
            _tries -= 1
            if not _tries:
                # 如果尝试次数用尽，抛出最后一次捕获的异常
                raise e

            if logger is not None:
                # 记录异常堆栈和重试警告
                logger.exception(e)
                logger.warning(f'{type(e).__name__}({e}), retrying in {_delay} seconds...')

            # 等待延时
            time.sleep(_delay)
            # 计算下一次延时
            _delay *= backoff

            # 添加随机抖动
            if isinstance(jitter, tuple):
                _delay += random.uniform(*jitter)
            else:
                _delay += jitter

            # 限制最大延时
            if max_delay is not None:
                _delay = min(_delay, max_delay)


def retry(exceptions=Exception, tries=-1, delay=0, max_delay=None, backoff=1, jitter=0, logger=logging_logger):
    """
    返回一个重试装饰器。

    用法示例：
    @retry(tries=3, delay=1)
    def connect():
        ...

    :param exceptions: 需要捕获的异常。
    :param tries: 最大尝试次数。
    :param delay: 初始延时。
    :param max_delay: 最大延时。
    :param backoff: 退避系数。
    :param jitter: 抖动。
    :param logger: 日志记录器。
    :returns: 装饰器函数。
    """

    @decorator
    def retry_decorator(f, *fargs, **fkwargs):
        args = fargs if fargs else list()
        kwargs = fkwargs if fkwargs else dict()
        # 使用 partial 绑定参数，然后调用内部重试逻辑
        return __retry_internal(partial(f, *args, **kwargs), exceptions, tries, delay, max_delay, backoff, jitter,
                                logger)

    return retry_decorator


def retry_call(f, fargs=None, fkwargs=None, exceptions=Exception, tries=-1, delay=0, max_delay=None, backoff=1,
               jitter=0,
               logger=logging_logger):
    """
    调用一个函数并在失败时重试（非装饰器用法）。

    :param f: 要执行的函数。
    :param fargs: 位置参数列表。
    :param fkwargs: 关键字参数字典。
    ... (其他参数同上)
    :returns: 函数 f 的结果。
    """
    args = fargs if fargs else list()
    kwargs = fkwargs if fkwargs else dict()
    return __retry_internal(partial(f, *args, **kwargs), exceptions, tries, delay, max_delay, backoff, jitter, logger)