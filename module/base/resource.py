'''
资源管理模块。主要负责管理图片资源（Assets）和 OCR 模型。
它维护了一个全局的资源注册表，并提供了释放资源（release_resources）的功能，以防止内存泄漏或占用过多内存，特别是在切换任务时清理不必要的缓存。
'''

import re

from module.base.decorator import cached_property


def get_assets_from_file(file):
    """
    作用：从 Python 代码文件中解析出定义的资源文件路径。
    通常用于静态分析哪些资源正在被使用。

    Args:
        file (str): Python 文件路径。

    Returns:
        set: 包含所有匹配到的资源文件路径的集合。
    """
    assets = set()
    # 匹配形如 file='./assets/...' 的字符串
    regex = re.compile(r"file='(.*?)'")
    with open(file, 'r', encoding='utf-8') as f:
        for row in f.readlines():
            result = regex.search(row)
            if result:
                assets.add(result.group(1))
    return assets


class PreservedAssets:
    """
    作用：定义需要保留的资源集合。
    这些资源通常是基础 UI 元素（如返回按钮、主页检查点），在任务切换时不应被释放。
    """

    @cached_property
    def ui(self):
        """
        返回基础 UI 相关的资源路径集合。
        """
        assets = set()
        # 从基础 UI 定义文件中提取资源路径
        assets |= get_assets_from_file(
            file='./tasks/base/assets/assets_base_page.py',
        )
        assets |= get_assets_from_file(
            file='./tasks/base/assets/assets_base_popup.py',
        )
        assets |= get_assets_from_file(
            file='./tasks/base/assets/assets_base_main_page.py',
        )
        return assets


# 全局单例，用于访问保留资源列表
_preserved_assets = PreservedAssets()


class Resource:
    """
    作用：资源基类。
    所有需要被管理的资源类（如 Button, Template）都应继承此类或将其实例注册到这里。
    """
    # 类属性，记录所有 Resource 的实例
    # Key: 资源标识符（通常是文件名或名称）, Value: Resource 实例对象
    instances = {}

    def resource_add(self, key):
        """
        将当前实例注册到全局资源池中。
        """
        Resource.instances[key] = self

    def resource_release(self):
        """
        释放资源的具体实现。
        子类（如 Button）需要重写此方法来清理缓存（如 self.image）。
        """
        pass

    @classmethod
    def is_loaded(cls, obj):
        """
        检查资源对象是否已加载数据（占用内存）。
        """
        if hasattr(obj, '_image') and obj._image is not None:
            return True
        if hasattr(obj, 'image') and obj.image is not None:
            return True
        if hasattr(obj, 'buttons') and obj.buttons is not None:
            return True
        return False

    @classmethod
    def resource_show(cls):
        """
        调试用：打印当前所有已加载的资源。
        """
        from module.logger import logger
        logger.hr('Show resource')
        for key, obj in cls.instances.items():
            if cls.is_loaded(obj):
                logger.info(f'{obj}: {key}')


def release_resources(next_task=''):
    """
    作用：释放内存中的资源。
    通常在任务切换时调用。

    Args:
        next_task (str): 下一个要执行的任务名称。
                         如果为空，则释放所有非保留资源（包括 OCR 模型）。
    """
    # 释放所有 OCR 模型
    # 检测模型 (det models) 占用约 400MB 内存，如果不立即进行下一个任务，应该释放
    if not next_task:
        from module.ocr.models import OCR_MODEL
        OCR_MODEL.resource_release()

    # 释放图片资源缓存
    # module.ui 大约有 80 个资源，占用 3MB
    # 整个项目有 800+ 资源，模板图片每个约 6MB，如果不释放会占用大量内存
    for key, obj in Resource.instances.items():
        # 如果资源属于基础 UI（保留资源），则跳过释放
        if next_task and str(obj) in _preserved_assets.ui:
            continue
        # 调试日志：显示释放了哪些资源
        # if Resource.is_loaded(obj):
        #     logger.info(f'Release {obj}')

        # 调用对象的释放方法（通常是清除 cached_property）
        obj.resource_release()

    # 如果没有下一个任务，重置语言检查状态
    # 因为用户可能在脚本停止期间更改了游戏语言
    if not next_task:
        from tasks.base.main_page import MainPage
        MainPage._lang_checked = False

    # 手动调用垃圾回收（通常不需要，但在内存紧张环境下可能有帮助）
    # gc.collect()