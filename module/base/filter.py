'''
定义了两个用于对象过滤的核心类：Filter 和 MultiLangFilter。它们是脚本用于从游戏数据中选择特定目标（例如：选择要刷的副本、要升级的遗器、要完成的任务等）的关键工具。
    Filter 类： 实现了通用的过滤逻辑。它接受一个正则表达式（regex）和一组属性名（attr），用于解析用户输入的复杂过滤字符串（例如 "属性1值>属性2值"）。
               它支持预设值（preset）和多重过滤条件（通过 > 符号分隔）。
    MultiLangFilter 类： 继承自 Filter，专门用于处理多语言数据。它允许对象的属性是一个列表（例如，一个副本名称的多种语言表达），只要过滤器中的值与列表中的任一元素匹配，就视为成功。

    调用的模块及位置：
        re：Python 的正则表达式模块，用于解析和处理过滤字符串。
        module.logger：日志记录模块，用于报告无效的过滤器字符串。
'''

import re

from module.logger import logger


# Filter 类：用于解析和应用基于正则表达式和属性的过滤条件
class Filter:
    def __init__(self, regex, attr, preset=()):
        """
        Args:
            regex: 正则表达式，用于解析过滤字符串的结构。
            attr: 属性名称列表（例如 ['名称', '稀有度']），与 regex 的捕获组对应。
            preset: 内置的字符串预设值（例如 'all', 'reset'）。
        """
        # 如果 regex 是字符串，将其编译为正则表达式对象
        if isinstance(regex, str):
            regex = re.compile(regex)
        self.regex = regex
        self.attr = attr
        # 将预设值转换为小写并存储为元组
        self.preset = tuple(list(p.lower() for p in preset))
        # 原始过滤字符串列表（按 > 分隔）
        self.filter_raw = []
        # 解析后的过滤条件列表
        self.filter = []

    def load(self, string):
        """
        加载一个过滤字符串，过滤条件使用 ">" 连接。
        例: "副本名称1>副本名称2>reset"

        Args:
            string (str): 原始过滤字符串。
        """
        string = str(string)
        # 移除空格、制表符、回车和换行符
        string = re.sub(r'[ \t\r\n]', '', string)
        # 将各种类似 ">" 的 Unicode 字符替换为标准的 ">"
        string = re.sub(r'[＞﹥›˃ᐳ❯]', '>', string)
        # 按 ">" 分割成原始过滤条件
        self.filter_raw = string.split('>')
        # 解析每个原始过滤条件
        self.filter = [self.parse_filter(f) for f in self.filter_raw]

    # 检查一个过滤字符串是否是预设值
    def is_preset(self, filter):
        return len(filter) and filter.lower() in self.preset

    def apply(self, objs, func=None):
        """
        应用加载的过滤条件到对象列表。

        Args:
            objs (list): 待过滤的对象列表。
            func (callable): 可选的二次过滤函数，对匹配后的对象进行额外处理。

        Returns:
            list: 匹配成功的对象和预设字符串（例如 [object, 'reset']）。
        """
        out = []
        # 遍历原始过滤条件和解析后的条件
        for raw, filter in zip(self.filter_raw, self.filter):
            # 1. 处理预设值
            if self.is_preset(raw):
                raw = raw.lower()
                # 避免重复添加预设值
                if raw not in out:
                    out.append(raw)
            # 2. 处理对象过滤
            else:
                for index, obj in enumerate(objs):
                    # 如果对象满足过滤条件，且尚未在输出列表中
                    if self.apply_filter_to_obj(obj=obj, filter=filter) and obj not in out:
                        out.append(obj)

        # 3. 应用可选的二次过滤函数
        if func is not None:
            objs, out = out, []
            for obj in objs:
                if isinstance(obj, str):
                    # 保留预设字符串
                    out.append(obj)
                elif func(obj):
                    # 对象满足二次过滤条件
                    out.append(obj)
                else:
                    # 不满足，丢弃
                    pass

        return out

    # 将解析后的过滤条件应用到单个对象
    def apply_filter_to_obj(self, obj, filter):
        """
        Args:
            obj (object): 待检查的对象。
            filter (list[str]): 解析后的属性值列表，与 self.attr 对应。

        Returns:
            bool: 对象是否满足所有条件。
        """
        # 遍历属性名和对应的值
        for attr, value in zip(self.attr, filter):
            if not value:
                # 如果过滤条件值为 None 或空字符串，则跳过此属性检查
                continue
            # 检查对象的属性值（转换为小写字符串）是否与过滤值匹配
            if str(obj.__getattribute__(attr)).lower() != str(value):
                return False

        return True

    # 解析原始过滤字符串
    def parse_filter(self, string):
        """
        将原始字符串解析为属性值列表。

        Args:
            string (str): 原始过滤字符串。

        Returns:
            list[str|None]: 匹配到的属性值列表。
        """
        string = string.replace(' ', '').lower()
        # 尝试使用正则表达式匹配
        result = re.search(self.regex, string)

        # 如果是预设值，直接返回
        if self.is_preset(string):
            return [string]

        # 如果匹配成功且字符串非空
        if result and len(string) and result.span()[1]:
            # 返回每个捕获组的内容
            return [result.group(index + 1) for index, attr in enumerate(self.attr)]
        else:
            # 过滤字符串无效
            logger.warning(f'Invalid filter: "{string}". This selector does not match the regex, nor a preset.')
            # 返回一个不可能匹配到的值，使其在 apply_filter_to_obj 中失败
            return ['1nVa1d'] + [None] * (len(self.attr) - 1)


# MultiLangFilter 类：支持多语言属性匹配的过滤器
class MultiLangFilter(Filter):
    """
    为了支持多语言，对象的属性值可能是数组（包含多个语言的名称），只要匹配其中任何一个元素即可。
    """

    # 重写 apply_filter_to_obj 以支持属性值是列表的情况
    def apply_filter_to_obj(self, obj, filter):
        """
        Args:
            obj (object): 待检查的对象。
            filter (list[str]): 解析后的属性值列表。

        Returns:
            bool: 对象是否满足所有条件。
        """
        for attr, value in zip(self.attr, filter):
            if not value:
                continue
            # 确保对象有该属性
            if not hasattr(obj, attr):
                continue

            obj_value = obj.__getattribute__(attr)

            # 1. 如果属性是字符串或整数（单值）
            if isinstance(obj_value, (str, int)):
                if str(obj_value).lower() != str(value):
                    return False
            # 2. 如果属性是列表（多值，如多语言名称）
            if isinstance(obj_value, list):
                # 检查过滤值是否在属性值列表中
                if value not in obj_value:
                    return False

        return True