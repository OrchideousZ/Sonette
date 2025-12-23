'''
定义了两个主要类：CodeGenerator 和 MarkdownGenerator，它们是用于生成格式化文本的工具类，常用于项目中的自动化生成代码文件或文档表格。
    CodeGenerator 类： 实现了 Python 代码的自动化生成。它通过维护一个缩进级别（tab_count）和行列表（lines），提供了一系列方法来方便地生成 Python 语法结构，
                      如变量赋值（Value）、列表（List）、字典（Dict）、类（Class）和函数（Def）。它特别使用了 TabWrapper 类作为上下文管理器来自动处理代码块的缩进。
    MarkdownGenerator 类： 用于生成 Markdown 格式的表格。它根据输入的数据行自动计算每列的最大宽度，以确保生成的 Markdown 表格在渲染时具有良好的对齐效果。

    调用的模块及位置：
        typing as t：用于类型提示。
        concurrent.futures.ThreadPoolExecutor：虽然在 CodeGenerator 中没有直接导入，但 TabWrapper 和 VariableWrapper 的设计是通用的。
        numpy as np：在 MarkdownGenerator 中使用，用于计算表格的列最大宽度。
'''

import typing as t


# TabWrapper 类：上下文管理器，用于自动处理代码块的缩进
class TabWrapper:
    def __init__(self, generator, prefix='', suffix='', newline=True):
        """
        Args:
            generator (CodeGenerator): 关联的代码生成器实例。
            prefix (str): 进入代码块时添加的前缀（例如 'class Name('）。
            suffix (str): 退出代码块时添加的后缀（例如 ':' 或 ')'）。
            newline (bool): 前缀后是否添加换行符。
        """
        self.generator = generator
        self.prefix = prefix
        self.suffix = suffix
        self.newline = newline

        # 标志是否为嵌套调用（用于 List/Dict/Object 内部元素）
        self.nested = False

    # 进入上下文时执行
    def __enter__(self):
        # 如果不是嵌套调用且有前缀，则添加前缀行
        if not self.nested and self.prefix:
            self.generator.add(self.prefix, newline=self.newline)
        # 增加缩进计数
        self.generator.tab_count += 1
        return self

    # 退出上下文时执行
    def __exit__(self, exc_type, exc_val, exc_tb):
        # 减少缩进计数
        self.generator.tab_count -= 1
        # 如果有后缀，则添加后缀行
        if self.suffix:
            self.generator.add(self.suffix)

    def __repr__(self):
        return self.prefix

    # 设置嵌套标志，用于在生成列表/字典/对象内部元素时，将逗号添加到后缀
    def set_nested(self, suffix=''):
        self.nested = True
        self.suffix += suffix


# VariableWrapper 类：简单的封装，用于区分生成的代码中的变量名和字符串字面量
class VariableWrapper:
    def __init__(self, name):
        self.name = name

    def __repr__(self):
        return str(self.name)

    __str__ = __repr__


# CodeGenerator 类：核心代码生成器
class CodeGenerator:
    def __init__(self):
        # 当前的缩进级别
        self.tab_count = 0
        # 存储生成的代码行
        self.lines = []

    # 添加一行代码
    def add(self, line, comment=False, newline=True):
        self.lines.append(self._line_with_tabs(line, comment=comment, newline=newline))

    # 生成完整的代码字符串
    def generate(self):
        return ''.join(self.lines)

    # 打印生成的代码
    def print(self):
        lines = self.generate()
        print(lines)

    # 将生成的代码写入文件
    def write(self, file: str = None):
        lines = self.generate()
        # 使用 utf-8 编码，确保换行符是 '\n'
        with open(file, 'w', encoding='utf-8', newline='') as f:
            f.write(lines)

    # 内部方法：添加缩进和可选的注释前缀
    def _line_with_tabs(self, line, comment=False, newline=True):
        if comment:
            line = '# ' + line
        # 插入缩进（4个空格）
        out = '    ' * self.tab_count + line
        if newline:
            out += '\n'
        return out

    # 内部方法：获取对象的 Python 表示形式（处理多行字符串）
    def _repr(self, obj):
        if isinstance(obj, str):
            # 如果是多行字符串，则使用三重引号 """
            if '\n' in obj:
                out = '"""\n'
                with self.tab():
                    for line in obj.strip().split('\n'):
                        line = line.strip()
                        out += self._line_with_tabs(line)
                # 在退出缩进前添加结束三重引号
                out += self._line_with_tabs('"""', newline=False)
                return out
        # 对其他类型使用标准的 repr()
        return repr(obj)

    # 创建 TabWrapper 上下文管理器，用于缩进
    def tab(self):
        return TabWrapper(self)

    # 添加空行
    def Empty(self):
        self.lines.append('\n')

    # 添加 pass 语句
    def Pass(self):
        self.add('pass')

    # 添加 import 语句
    def Import(self, text, empty=2):
        for line in text.strip().split('\n'):
            line = line.strip()
            self.add(line)
        # 导入块后添加空行
        for _ in range(empty):
            self.Empty()

    # 创建 VariableWrapper 实例
    def Variable(self, name):
        return VariableWrapper(name)

    # 添加变量赋值语句
    def Value(self, key=None, value=None, type_=None, **kwargs):
        if key is not None:
            if type_ is not None:
                # 带类型提示的赋值
                self.add(f'{key}: {type_} = {self._repr(value)}')
            else:
                # 普通赋值
                self.add(f'{key} = {self._repr(value)}')
        # 处理 **kwargs 形式的赋值
        for key, value in kwargs.items():
            self.Value(key, value)

    # 添加注释行
    def Comment(self, text):
        for line in text.strip().split('\n'):
            line = line.strip()
            self.add(line, comment=True)

    # 添加代码自动生成提示注释
    def CommentAutoGenerage(self, file):
        """
        Args:
            file: dev_tools.button_extract (示例文件路径)
        """
        # 确保前面只有一个空行
        if len(self.lines) >= 2:
            if self.lines[-2:] == ['\n', '\n']:
                self.lines.pop(-1)
        self.Comment('This file was auto-generated, do not modify it manually. To generate:')
        self.Comment(f'``` python -m {file} ```')
        self.Empty()

    # 创建列表 TabWrapper 上下文管理器
    def List(self, key=None):
        if key is not None:
            # 带变量名的列表
            return TabWrapper(self, prefix=str(key) + ' = [', suffix=']')
        else:
            # 匿名列表
            return TabWrapper(self, prefix='[', suffix=']')

    # 添加列表项
    def ListItem(self, value):
        if isinstance(value, TabWrapper):
            # 如果列表项本身是一个嵌套的代码块（如嵌套列表/字典），设置嵌套标志并添加逗号后缀
            value.set_nested(suffix=',')
            self.add(f'{self._repr(value)}')
            return value
        else:
            # 简单列表项
            self.add(f'{self._repr(value)},')

    # 创建字典 TabWrapper 上下文管理器
    def Dict(self, key=None):
        if key is not None:
            # 带变量名的字典
            return TabWrapper(self, prefix=str(key) + ' = {', suffix='}')
        else:
            # 匿名字典
            return TabWrapper(self, prefix='{', suffix='}')

    # 添加字典项
    def DictItem(self, key=None, value=None):
        if isinstance(value, TabWrapper):
            # 如果值是嵌套代码块，设置逗号后缀
            value.set_nested(suffix=',')
            if key is not None:
                self.add(f'{self._repr(key)}: {self._repr(value)}')
            return value
        else:
            # 简单字典项
            if key is not None:
                self.add(f'{self._repr(key)}: {self._repr(value)},')

    # 创建对象实例化 TabWrapper 上下文管理器
    def Object(self, object_class, key=None):
        if key is not None:
            # 带变量名的对象实例化
            return TabWrapper(self, prefix=f'{key} = {object_class}(', suffix=')')
        else:
            # 匿名对象实例化
            return TabWrapper(self, prefix=f'{object_class}(', suffix=')')

    # 添加对象属性/参数
    def ObjectAttr(self, key=None, value=None):
        if isinstance(value, TabWrapper):
            # 如果值是嵌套代码块，设置逗号后缀
            value.set_nested(suffix=',')
            if key is None:
                # 位置参数
                self.add(f'{self._repr(value)}')
            else:
                # 关键字参数
                self.add(f'{key}={self._repr(value)}')
            return value
        else:
            # 简单参数
            if key is None:
                self.add(f'{self._repr(value)},')
            else:
                self.add(f'{key}={self._repr(value)},')

    # 创建类定义 TabWrapper 上下文管理器
    def Class(self, name, inherit=None):
        if inherit is not None:
            # 带继承的类定义
            return TabWrapper(self, prefix=f'class {name}({inherit}):')
        else:
            # 不带继承的类定义
            return TabWrapper(self, prefix=f'class {name}:')

    # 创建函数定义 TabWrapper 上下文管理器
    def Def(self, name, args=''):
        return TabWrapper(self, prefix=f'def {name}({args}):')


# 实例化一个全局 CodeGenerator 对象，供外部直接调用其方法
generator = CodeGenerator()
Import = generator.Import
Value = generator.Value
Comment = generator.Comment
Dict = generator.Dict
DictItem = generator.DictItem


# MarkdownGenerator 类：用于生成对齐良好的 Markdown 表格
class MarkdownGenerator:
    def __init__(self, column: t.List[str]):
        # 存储表格的所有行，第一行是列头
        self.rows = [column]

    # 添加一行数据
    def add_row(self, row):
        # 确保所有元素都是字符串
        self.rows.append([str(ele) for ele in row])

    # 内部方法：生成一行 Markdown 表格内容
    def product_line(self, row, max_width):
        # 使用 ljust() 将每个元素左对齐到最大宽度
        row = [ele.ljust(width) for ele, width in zip(row, max_width)]
        # 用 ' | ' 连接，并添加边框
        row = ' | '.join(row)
        row = '| ' + row + ' |'
        return row

    # 生成完整的 Markdown 表格行列表
    def generate(self) -> t.List[str]:
        import numpy as np
        # 计算每行中每个元素的长度
        width = np.array([
            [len(ele) for ele in row] for row in self.rows
        ])
        # 找出每列的最大宽度
        max_width = np.max(width, axis=0)
        # 创建分隔符行（横线）
        dash = ['-' * width for width in max_width]

        # 组合生成最终的表格行
        rows = [
                   self.product_line(self.rows[0], max_width), # 列头
                   self.product_line(dash, max_width),         # 分隔线
               ] + [
                   self.product_line(row, max_width) for row in self.rows[1:] # 数据行
               ]
        return rows