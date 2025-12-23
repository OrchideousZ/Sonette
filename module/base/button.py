'''
定义了自动化脚本中用于图像识别和点击定位的原子操作类。它将图像资源、识别区域、目标颜色和点击坐标等信息封装起来，为上层的 ModuleBase 提供了基础的图像识别能力。
    Button类： 代表单个图像模板资源，能够执行模板匹配（match_template）和颜色匹配（match_color），并能根据匹配结果计算相对偏移量（_button_offset），
               从而实现对位置不固定的 UI 元素的识别和定位。
    ButtonWrapper类： 代表多资源/多语言按钮的封装，它允许为同一 UI 元素定义多个 Button 实例（如不同语言的图片），并在运行时按优先级（服务器语言、share、cn）进行尝试匹配。
    ClickButton 类： 一个简单的类，仅包含区域和点击坐标，用于表示任何可点击的 UI 元素。
    match_template 函数： 底层 OpenCV 模板匹配的封装，是所有图像识别的基础。

    调用的模块及位置：
        module.config.server：用于获取当前运行的服务器/语言设置，决定 ButtonWrapper 优先使用哪种语言的资源。
        module.base.decorator：用于缓存属性（cached_property）及其清除（del_cached_property）。
        module.base.resource.Resource：基类，用于资源管理和释放。
        module.base.utils.*：各种图像和坐标处理工具函数（如 crop、area_offset、rgb2luma 等）。
        module.exception.ScriptError：用于抛出配置错误，例如 ButtonWrapper 找不到可用的资源时。
'''

# 导入服务器配置模块，用于多语言/服务器适配
import module.config.server as server
# 导入缓存属性装饰器及其删除函数
from module.base.decorator import cached_property, del_cached_property
# 导入资源管理基类
from module.base.resource import Resource
# 导入各种图像处理和坐标计算工具函数
from module.base.utils import *
# 导入脚本错误异常
from module.exception import ScriptError


# Button 类：代表一个单一的图像模板资源，具有位置偏移和多种匹配能力
class Button(Resource):
    def __init__(self, file, area, search, color, button, posi=None):
        """
        Args:
            file: 资源文件路径（Assets 路径）
            area: 裁剪模板的区域 (x1, y1, x2, y2)
            search: 截图上进行搜索的区域，默认比 area 略大
            color: 模板的平均颜色 (R, G, B)，用于颜色匹配
            button: 模板出现时，实际需要点击的区域 (x1, y1, x2, y2)
        """
        self.file: str = file
        self.area: t.Tuple[int, int, int, int] = area
        self.search: t.Tuple[int, int, int, int] = search
        self.color: t.Tuple[int, int, int] = color
        # 实际点击区域的原始值
        self._button: t.Tuple[int, int, int, int] = button
        self.posi: t.Optional[t.Tuple[int, int]] = posi

        # 将文件路径添加到资源管理器
        self.resource_add(self.file)
        # 模板匹配后计算出的偏移量，默认为 (0, 0)
        self._button_offset: t.Tuple[int, int] = (0, 0)

    # 属性：返回应用了偏移量后的实际点击区域
    @property
    def button(self):
        return area_offset(self._button, self._button_offset)

    # 加载另一个 Button 的偏移量
    def load_offset(self, button):
        self._button_offset = button._button_offset

    # 清除偏移量，重置为 (0, 0)
    def clear_offset(self):
        self._button_offset = (0, 0)

    # 检查偏移量是否在指定范围内
    def is_offset_in(self, x=0, y=0):
        """
        Args:
            x: X轴允许的最大偏移量（正负）
            y: Y轴允许的最大偏移量（正负）

        Returns:
            bool: 如果 _button_offset 在 (-x, -y) 到 (x, y) 范围内，则为 True。
        """
        if x:
            if self._button_offset[0] < -x or self._button_offset[0] > x:
                return False
        if y:
            if self._button_offset[1] < -y or self._button_offset[1] > y:
                return False
        return True

    # 缓存属性：加载并裁剪后的模板图像（彩色）
    @cached_property
    def image(self):
        return load_image(self.file, self.area)

    # 缓存属性：加载并转换后的模板图像（亮度/灰度）
    @cached_property
    def image_luma(self):
        return rgb2luma(self.image)

    # 资源释放：清除缓存的图像数据和偏移量
    def resource_release(self):
        del_cached_property(self, 'image')
        del_cached_property(self, 'image_luma')
        self.clear_offset()

    def __str__(self):
        return self.file

    __repr__ = __str__

    def __eq__(self, other):
        return str(self) == str(other)

    def __hash__(self):
        return hash(self.file)

    def __bool__(self):
        return True

    # 颜色匹配：通过比较平均颜色来检查按钮是否出现
    def match_color(self, image, threshold=10) -> bool:
        """
        使用平均颜色检查按钮是否出现在图像上。

        Args:
            image (np.ndarray): 当前截图。
            threshold (int): 颜色相似度阈值。

        Returns:
            bool: 颜色是否相似。
        """
        # 获取截图上 area 区域的平均颜色
        color = get_color(image, self.area)
        # 比较平均颜色与模板预设颜色
        return color_similar(
            color1=color,
            color2=self.color,
            threshold=threshold
        )

    # 模板匹配（彩色）：使用 cv2.matchTemplate
    def match_template(self, image, similarity=0.85, direct_match=False) -> bool:
        """
        通过模板匹配检测资源。如果匹配成功，将设置 self._button_offset。

        Args:
            image: 当前截图。
            similarity (float): 相似度阈值。
            direct_match: True 则忽略 self.search 区域，在整个图像上搜索。

        Returns:
            bool.
        """
        if not direct_match:
            # 裁剪搜索区域
            image = crop(image, self.search, copy=False)
        # 执行模板匹配（TM_CCOEFF_NORMED 标准化相关系数匹配）
        res = cv2.matchTemplate(self.image, image, cv2.TM_CCOEFF_NORMED)
        # 寻找最大最小值及其位置
        _, sim, _, point = cv2.minMaxLoc(res)

        # 计算并设置模板的偏移量：匹配到的位置 + 搜索区域的左上角 - 模板的左上角
        self._button_offset = np.array(point) + self.search[:2] - self.area[:2]
        return sim > similarity

    # 模板匹配（亮度/灰度）
    def match_template_luma(self, image, similarity=0.85, direct_match=False) -> bool:
        """
        通过模板匹配检测资源（使用灰度图像，忽略颜色差异）。
        """
        if not direct_match:
            image = crop(image, self.search, copy=False)
        # 将截图转换为亮度图
        image = rgb2luma(image)
        # 对亮度图执行模板匹配
        res = cv2.matchTemplate(self.image_luma, image, cv2.TM_CCOEFF_NORMED)
        _, sim, _, point = cv2.minMaxLoc(res)

        self._button_offset = np.array(point) + self.search[:2] - self.area[:2]
        return sim > similarity

    # 多结果模板匹配：返回所有匹配相似度高于阈值的位置
    def match_multi_template(self, image, similarity=0.85, direct_match=False):
        """
        通过模板匹配检测资源，返回多个结果位置。
        """
        if not direct_match:
            image = crop(image, self.search, copy=False)
        # 执行模板匹配
        res = cv2.matchTemplate(self.image, image, cv2.TM_CCOEFF_NORMED)
        # 阈值化：只保留相似度高于 similarity 的位置
        res = cv2.inRange(res, similarity, 1.)
        try:
            # 找到非零点（即匹配成功的位置）
            points = np.array(cv2.findNonZero(res))[:, 0, :]
            # 加上搜索区域的偏移量，转换为绝对坐标
            points += self.search[:2]
            return points.tolist()
        except IndexError:
            # 没有匹配结果
            return []

    # 模板匹配（亮度）和颜色匹配组合
    def match_template_color(self, image, similarity=0.85, threshold=30, direct_match=False) -> bool:
        """
        先进行亮度模板匹配，成功后再进行颜色匹配验证。
        """
        # 1. 执行亮度模板匹配
        matched = self.match_template_luma(image, similarity=similarity, direct_match=direct_match)
        if not matched:
            return False

        # 2. 如果模板匹配成功，获取应用了偏移量的实际区域
        area = area_offset(self.area, offset=self._button_offset)
        # 3. 获取该区域的平均颜色
        color = get_color(image, area)
        # 4. 颜色匹配验证
        return color_similar(
            color1=color,
            color2=self.color,
            threshold=threshold
        )


# ButtonWrapper 类：用于封装多个 Button 资源，以支持多语言和灵活匹配
class ButtonWrapper(Resource):
    def __init__(self, name='MULTI_ASSETS', **kwargs):
        self.name = name
        # 包含不同语言或版本的 Button 字典
        self.data_buttons = kwargs
        # 最近一次匹配成功的 Button 实例
        self._matched_button: t.Optional[Button] = None
        # 添加资源引用
        self.resource_add(f'{name}:{next(self.iter_buttons(), None)}')

    # 资源释放
    def resource_release(self):
        del_cached_property(self, 'buttons')
        self._matched_button = None

    def __str__(self):
        return self.name

    __repr__ = __str__

    def __eq__(self, other):
        return str(self) == str(other)

    def __hash__(self):
        return hash(self.name)

    def __bool__(self):
        return True

    # 迭代器：遍历所有 Button 资源
    def iter_buttons(self) -> t.Iterator[Button]:
        for _, assets in self.data_buttons.items():
            if isinstance(assets, Button):
                yield assets
            elif isinstance(assets, list):
                for asset in assets:
                    yield asset

    # 缓存属性：根据服务器语言优先级获取可用的 Button 列表
    @cached_property
    def buttons(self) -> t.List[Button]:
        # 匹配优先级：当前语言 -> share (共享) -> cn (默认/中文)
        for trial in [server.lang, 'share', 'cn']:
            try:
                assets = self.data_buttons[trial]
                if isinstance(assets, Button):
                    return [assets]
                elif isinstance(assets, list):
                    return assets
            except KeyError:
                pass

        # 如果所有尝试都失败，则抛出错误
        raise ScriptError(f'ButtonWrapper({self}) on server {server.lang} has no fallback button')

    # 多按钮颜色匹配
    def match_color(self, image, threshold=10) -> bool:
        for assets in self.buttons:
            if assets.match_color(image, threshold=threshold):
                self._matched_button = assets
                return True
        return False

    # 多按钮模板匹配（彩色）
    def match_template(self, image, similarity=0.85, direct_match=False) -> bool:
        for assets in self.buttons:
            if assets.match_template(image, similarity=similarity, direct_match=direct_match):
                self._matched_button = assets
                return True
        return False

    # 多按钮模板匹配（亮度）
    def match_template_luma(self, image, similarity=0.85, direct_match=False) -> bool:
        for assets in self.buttons:
            if assets.match_template_luma(image, similarity=similarity, direct_match=direct_match):
                self._matched_button = assets
                return True
        return False

    # 多结果多模板匹配：返回多个 ClickButton 结果
    def match_multi_template(self, image, similarity=0.85, threshold=5, direct_match=False):
        """
        通过模板匹配检测资源，返回多个 ClickButton 结果。
        """
        ps = []
        # 遍历所有可用资源并收集所有匹配点
        for assets in self.buttons:
            ps += assets.match_multi_template(image, similarity=similarity, direct_match=direct_match)
        if not ps:
            return []

        # 将匹配点进行分组（消除相近的重复点）
        from module.base.utils.points import Points
        ps = Points(ps).group(threshold=threshold)

        # 计算每个分组的区域和点击区域
        area_list = [area_offset(self.area, p - self.area[:2]) for p in ps]
        button_list = [area_offset(self.button, p - self.area[:2]) for p in ps]

        # 返回 ClickButton 列表
        return [
            ClickButton(area=info[0], button=info[1], name=f'{self.name}_result{i}')
            for i, info in enumerate(zip(area_list, button_list))
        ]

    # 多按钮模板匹配（亮度）和颜色匹配组合
    def match_template_color(self, image, similarity=0.85, threshold=30, direct_match=False) -> bool:
        for assets in self.buttons:
            if assets.match_template_color(
                    image, similarity=similarity, threshold=threshold, direct_match=direct_match):
                self._matched_button = assets
                return True
        return False

    # 属性：返回当前匹配到的 Button 实例，如果没有则返回列表中的第一个
    @property
    def matched_button(self) -> Button:
        if self._matched_button is None:
            # 如果没有匹配，则默认使用第一个可用的 Button
            return self.buttons[0]
        else:
            return self._matched_button

    # 以下属性均代理到 matched_button
    @property
    def area(self) -> tuple[int, int, int, int]:
        return self.matched_button.area

    @property
    def search(self) -> tuple[int, int, int, int]:
        return self.matched_button.search

    @property
    def color(self) -> tuple[int, int, int]:
        return self.matched_button.color

    @property
    def button(self) -> tuple[int, int, int, int]:
        return self.matched_button.button

    @property
    def button_offset(self) -> tuple[int, int]:
        return self.matched_button._button_offset

    @property
    def width(self) -> int:
        return area_size(self.area)[0]

    @property
    def height(self) -> int:
        return area_size(self.area)[1]

    # 加载偏移量到所有内部 Button 实例
    def load_offset(self, button):
        """
        从另一个 Button 或 ButtonWrapper 加载偏移量到内部所有 Button。
        """
        if isinstance(button, ButtonWrapper):
            button = button.matched_button
        for b in self.iter_buttons():
            b.load_offset(button)

    # 清除所有内部 Button 实例的偏移量
    def clear_offset(self):
        for b in self.iter_buttons():
            b.clear_offset()

    # 检查偏移量是否在指定范围内（代理给 matched_button）
    def is_offset_in(self, x=0, y=0):
        """
        检查 _button_offset 是否在 (-x, -y, x, y) 范围内。
        """
        return self.matched_button.is_offset_in(x=x, y=y)

    # 设置搜索区域（不可逆操作）
    def load_search(self, area):
        """
        设置 `search` 属性。
        注意：此方法是不可逆的。
        """
        for b in self.iter_buttons():
            b.search = area

    # 设置搜索区域的偏移量（兼容旧版 Alas 的 offset 机制）
    def set_search_offset(self, offset):
        """
        设置搜索区域的偏移量，会永久修改内部 Button 的 search 属性。

        Args:
            offset (tuple): (x, y) 或 (left, up, right, bottom)
        """
        if len(offset) == 2:
            left, up, right, bottom = -offset[0], -offset[1], offset[0], offset[1]
        else:
            left, up, right, bottom = offset
        for b in self.iter_buttons():
            upper_left_x, upper_left_y, bottom_right_x, bottom_right_y = b.area
            # 重新计算并设置搜索区域
            b.search = (
                upper_left_x + left,
                upper_left_y + up,
                bottom_right_x + right,
                bottom_right_y + bottom,
            )


# ClickButton 类：一个只包含点击区域的简单类
class ClickButton:
    def __init__(self, area, button=None, name='CLICK_BUTTON'):
        # 识别区域
        self.area = area
        # 实际点击区域（默认为识别区域）
        if button is None:
            self.button = area
        else:
            self.button = button
        self.name = name

    def __str__(self):
        return self.name

    __repr__ = __str__

    def __eq__(self, other):
        return str(self) == str(other)

    def __hash__(self):
        return hash(self.name)

    def __bool__(self):
        return True


# 独立的模板匹配函数（通用底层实现）
def match_template(image, template, similarity=0.85):
    """
    通用模板匹配函数。

    Args:
        image (np.ndarray): 截图。
        template (np.ndarray): 模板图像。
        similarity (float): 相似度阈值。

    Returns:
        bool: 相似度是否高于阈值。
    """
    # 执行模板匹配
    res = cv2.matchTemplate(image, template, cv2.TM_CCOEFF_NORMED)
    # 获取最大相似度和位置
    _, sim, _, point = cv2.minMaxLoc(res)
    return sim > similarity