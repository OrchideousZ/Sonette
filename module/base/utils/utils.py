'''
通用工具函数库。包含了随机数生成、图像加载/保存/裁剪/缩放、颜色处理、坐标转换等大量基础函数。
'''

import random
import re

import cv2
import numpy as np
from PIL import Image

REGEX_NODE = re.compile(r'(-?[A-Za-z]+)(-?\d+)')


def random_normal_distribution_int(a, b, n=3):
    """
    生成区间内的正态分布整数。
    通过多次随机取平均值来模拟正态分布。

    Args:
        a (int): 最小值。
        b (int): 最大值。
        n (int): 模拟次数。

    Returns:
        int
    """
    a = round(a)
    b = round(b)
    if a < b:
        total = 0
        for _ in range(n):
            total += random.randint(a, b)
        return round(total / n)
    else:
        return b


def random_rectangle_point(area, n=3):
    """
    在矩形区域内随机选择一个点。

    Args:
        area: (x1, y1, x2, y2).
        n (int): 模拟次数。

    Returns:
        tuple(int): (x, y)
    """
    x = random_normal_distribution_int(area[0], area[2], n=n)
    y = random_normal_distribution_int(area[1], area[3], n=n)
    return x, y


def random_rectangle_vector(vector, box, random_range=(0, 0, 0, 0), padding=15):
    """
    在盒子内随机放置一个向量（用于滑动操作）。

    Args:
        vector: (x, y) 滑动向量。
        box: (x1, y1, x2, y2) 限制区域。
        random_range (tuple): 给向量添加随机偏移。
        padding (int): 边距。

    Returns:
        tuple(int), tuple(int): 起点, 终点。
    """
    vector = np.array(vector) + random_rectangle_point(random_range)
    vector = np.round(vector).astype(int)
    half_vector = np.round(vector / 2).astype(int)
    box = np.array(box) + np.append(np.abs(half_vector) + padding, -np.abs(half_vector) - padding)
    center = random_rectangle_point(box)
    start_point = center - half_vector
    end_point = start_point + vector
    return tuple(start_point), tuple(end_point)


def random_rectangle_vector_opted(
        vector, box, random_range=(0, 0, 0, 0), padding=15, whitelist_area=None, blacklist_area=None):
    """
    优化版的随机向量放置。
    防止滑动操作被误判为点击（例如模拟器卡顿时）。
    支持白名单和黑名单区域过滤。

    Args:
        vector: (x, y)
        box: 限制区域。
        whitelist_area: 安全点击区域列表。
        blacklist_area: 禁止点击区域列表。

    Returns:
        tuple(int), tuple(int): 起点, 终点。
    """
    vector = np.array(vector) + random_rectangle_point(random_range)
    vector = np.round(vector).astype(int)
    half_vector = np.round(vector / 2).astype(int)
    box_pad = np.array(box) + np.append(np.abs(half_vector) + padding, -np.abs(half_vector) - padding)
    box_pad = area_offset(box_pad, half_vector)
    segment = int(np.linalg.norm(vector) // 70) + 1

    def in_blacklist(end):
        if not blacklist_area:
            return False
        for x in range(segment + 1):
            point = - vector * x / segment + end
            for area in blacklist_area:
                if point_in_area(point, area, threshold=0):
                    return True
        return False

    # 优先尝试白名单区域
    if whitelist_area:
        for area in whitelist_area:
            area = area_limit(area, box_pad)
            if all([x > 0 for x in area_size(area)]):
                end_point = random_rectangle_point(area)
                for _ in range(10):
                    if in_blacklist(end_point):
                        continue
                    return point_limit(end_point - vector, box), point_limit(end_point, box)

    # 尝试随机区域
    for _ in range(100):
        end_point = random_rectangle_point(box_pad)
        if in_blacklist(end_point):
            continue
        return point_limit(end_point - vector, box), point_limit(end_point, box)

    # 失败保底
    end_point = random_rectangle_point(box_pad)
    return point_limit(end_point - vector, box), point_limit(end_point, box)


def random_line_segments(p1, p2, n, random_range=(0, 0, 0, 0)):
    """
    将一条线段切割成多段，并添加随机偏移。
    用于模拟更真实的手指滑动轨迹。

    Args:
        p1: 起点 (x, y).
        p2: 终点 (x, y).
        n: 切割段数。
        random_range: 随机偏移范围。

    Returns:
        list[tuple]: 点列表。
    """
    return [tuple((((n - index) * p1 + index * p2) / n).astype(int) + random_rectangle_point(random_range))
            for index in range(0, n + 1)]


def ensure_time(second, n=3, precision=3):
    """
    确保时间是一个浮点数，支持随机范围字符串解析。

    Args:
        second (int, float, tuple, str): 时间，如 10, (10, 30), '10, 30'。
        n (int): 随机模拟次数。
        precision (int): 小数精度。

    Returns:
        float: 时间秒数。
    """
    if isinstance(second, tuple):
        multiply = 10 ** precision
        result = random_normal_distribution_int(second[0] * multiply, second[1] * multiply, n) / multiply
        return round(result, precision)
    elif isinstance(second, str):
        if ',' in second:
            lower, upper = second.replace(' ', '').split(',')
            lower, upper = int(lower), int(upper)
            return ensure_time((lower, upper), n=n, precision=precision)
        if '-' in second:
            lower, upper = second.replace(' ', '').split('-')
            lower, upper = int(lower), int(upper)
            return ensure_time((lower, upper), n=n, precision=precision)
        else:
            return int(second)
    else:
        return second


def ensure_int(*args):
    """
    将所有参数转换为 int。
    支持嵌套列表。
    """

    def to_int(item):
        try:
            return int(item)
        except TypeError:
            result = [to_int(i) for i in item]
            if len(result) == 1:
                result = result[0]
            return result

    return to_int(args)


def area_offset(area, offset):
    """
    移动区域。

    Args:
        area: (x1, y1, x2, y2).
        offset: (x, y).

    Returns:
        tuple: 移动后的区域。
    """
    upper_left_x, upper_left_y, bottom_right_x, bottom_right_y = area
    x, y = offset
    return upper_left_x + x, upper_left_y + y, bottom_right_x + x, bottom_right_y + y


def area_pad(area, pad=10):
    """
    缩放区域（内缩或外扩）。

    Args:
        area: (x1, y1, x2, y2).
        pad (int): 正数为内缩，负数为外扩。

    Returns:
        tuple: 缩放后的区域。
    """
    upper_left_x, upper_left_y, bottom_right_x, bottom_right_y = area
    return upper_left_x + pad, upper_left_y + pad, bottom_right_x - pad, bottom_right_y - pad


def limit_in(x, lower, upper):
    """
    限制 x 在 [lower, upper] 范围内。
    """
    return max(min(x, upper), lower)


def area_limit(area1, area2):
    """
    限制 area1 在 area2 范围内。
    """
    x_lower, y_lower, x_upper, y_upper = area2
    return (
        limit_in(area1[0], x_lower, x_upper),
        limit_in(area1[1], y_lower, y_upper),
        limit_in(area1[2], x_lower, x_upper),
        limit_in(area1[3], y_lower, y_upper),
    )


def area_size(area):
    """
    计算区域大小。

    Returns:
        tuple: (width, height).
    """
    return (
        max(area[2] - area[0], 0),
        max(area[3] - area[1], 0)
    )


def area_center(area):
    """
    计算区域中心点。

    Returns:
        tuple: (x, y).
    """
    x1, y1, x2, y2 = area
    return (x1 + x2) / 2, (y1 + y2) / 2


def point_limit(point, area):
    """
    限制点在区域内。
    """
    return (
        limit_in(point[0], area[0], area[2]),
        limit_in(point[1], area[1], area[3])
    )


def point_in_area(point, area, threshold=5):
    """
    判断点是否在区域内（带阈值）。
    """
    return area[0] - threshold < point[0] < area[2] + threshold and area[1] - threshold < point[1] < area[3] + threshold


def area_in_area(area1, area2, threshold=5):
    """
    判断 area1 是否完全在 area2 内。
    """
    return area2[0] - threshold <= area1[0] \
        and area2[1] - threshold <= area1[1] \
        and area1[2] <= area2[2] + threshold \
        and area1[3] <= area2[3] + threshold


def area_cross_area(area1, area2, threshold=5):
    """
    判断两个区域是否相交。
    """
    # https://www.yiiven.cn/rect-is-intersection.html
    xa1, ya1, xa2, ya2 = area1
    xb1, yb1, xb2, yb2 = area2
    return abs(xb2 + xb1 - xa2 - xa1) <= xa2 - xa1 + xb2 - xb1 + threshold * 2 \
        and abs(yb2 + yb1 - ya2 - ya1) <= ya2 - ya1 + yb2 - yb1 + threshold * 2


def float2str(n, decimal=3):
    """
    格式化浮点数。
    """
    return str(round(n, decimal)).ljust(decimal + 2, "0")


def point2str(x, y, length=4):
    """
    格式化坐标点字符串。
    """
    return '(%s, %s)' % (str(int(x)).rjust(length), str(int(y)).rjust(length))


def col2name(col):
    """
    将 0 索引的列号转换为 Excel 风格的列名 (A, B, ... AA, AB)。
    """
    col_neg = col < 0
    if col_neg:
        col_num = -col
    else:
        col_num = col + 1  # Change to 1-index.
    col_str = ''

    while col_num:
        remainder = col_num % 26
        if remainder == 0:
            remainder = 26
        col_letter = chr(remainder + 64)
        col_str = col_letter + col_str
        col_num = int((col_num - 1) / 26)

    if col_neg:
        return '-' + col_str
    else:
        return col_str


def name2col(col_str):
    """
    将 Excel 风格的列名转换为 0 索引的列号。
    """
    expn = 0
    col = 0
    col_neg = col_str.startswith('-')
    col_str = col_str.strip('-').upper()

    for char in reversed(col_str):
        col += (ord(char) - 64) * (26 ** expn)
        expn += 1

    if col_neg:
        return -col
    else:
        return col - 1


def node2location(node):
    """
    将节点字符串（如 'E3'）转换为坐标 (4, 2)。
    """
    res = REGEX_NODE.search(node)
    if res:
        x, y = res.group(1), res.group(2)
        y = int(y)
        if y > 0:
            y -= 1
        return name2col(x), y
    else:
        return ord(node[0]) % 32 - 1, int(node[1:]) - 1


def location2node(location):
    """
    将坐标转换为节点字符串。
    """
    x, y = location
    if y >= 0:
        y += 1
    return col2name(x) + str(y)


def xywh2xyxy(area):
    """
    转换 (x, y, w, h) -> (x1, y1, x2, y2)
    """
    x, y, w, h = area
    return x, y, x + w, y + h


def xyxy2xywh(area):
    """
    转换 (x1, y1, x2, y2) -> (x, y, w, h)
    """
    x1, y1, x2, y2 = area
    return min(x1, x2), min(y1, y2), abs(x2 - x1), abs(y2 - y1)


def load_image(file, area=None):
    """
    加载图片，移除 Alpha 通道。
    使用 PIL 读取，兼容性好。

    Args:
        file (str): 文件路径。
        area (tuple): 裁剪区域。

    Returns:
        np.ndarray: 图像数组。
    """
    with Image.open(file) as f:
        if area is not None:
            f = f.crop(area)

        image = np.array(f)

    channel = image_channel(image)
    if channel == 4:
        image = cv2.cvtColor(image, cv2.COLOR_RGBA2RGB)

    return image


def save_image(image, file):
    """
    保存图片。
    """
    Image.fromarray(image).save(file)


def copy_image(src):
    """
    复制图像，比 np.copy() 稍快。
    """
    dst = np.empty_like(src)
    cv2.copyTo(src, None, dst)
    return dst


def crop(image, area, copy=True):
    """
    裁剪图像。
    如果裁剪区域超出图像范围，会自动填充黑色背景（类似 Pillow 的行为）。

    Args:
        image (np.ndarray):
        area: (x1, y1, x2, y2)
        copy (bool): 是否返回副本。

    Returns:
        np.ndarray:
    """
    x1, y1, x2, y2 = area
    x1 = round(x1)
    y1 = round(y1)
    x2 = round(x2)
    y2 = round(y2)
    shape = image.shape
    h = shape[0]
    w = shape[1]

    # 计算溢出边界
    overflow = False
    if y1 >= 0:
        top = 0
        if y1 >= h:
            overflow = True
    else:
        top = -y1
    if y2 > h:
        bottom = y2 - h
    else:
        bottom = 0
        if y2 <= 0:
            overflow = True
    if x1 >= 0:
        left = 0
        if x1 >= w:
            overflow = True
    else:
        left = -x1
    if x2 > w:
        right = x2 - w
    else:
        right = 0
        if x2 <= 0:
            overflow = True

    # 如果完全溢出，返回全黑图像
    if overflow:
        if len(shape) == 2:
            size = (y2 - y1, x2 - x1)
        else:
            size = (y2 - y1, x2 - x1, shape[2])
        return np.zeros(size, dtype=image.dtype)

    # 限制裁剪区域在图像内
    if x1 < 0: x1 = 0
    if y1 < 0: y1 = 0
    if x2 < 0: x2 = 0
    if y2 < 0: y2 = 0

    image = image[y1:y2, x1:x2]

    # 如果有溢出，填充黑色边界
    if top or bottom or left or right:
        if len(shape) == 2:
            value = 0
        else:
            value = tuple(0 for _ in range(image.shape[2]))
        return cv2.copyMakeBorder(image, top, bottom, left, right, borderType=cv2.BORDER_CONSTANT, value=value)
    elif copy:
        return copy_image(image)
    else:
        return image


def resize(image, size):
    """
    调整图像大小，使用最近邻插值（类似 Pillow 默认行为）。
    """
    return cv2.resize(image, size, interpolation=cv2.INTER_NEAREST)


def image_channel(image):
    """
    获取图像通道数。
    """
    return image.shape[2] if len(image.shape) == 3 else 0


def image_size(image):
    """
    获取图像尺寸 (width, height)。
    """
    shape = image.shape
    return shape[1], shape[0]


def image_paste(image, background, origin):
    """
    将图像粘贴到背景上（原地修改）。
    """
    x, y = origin
    w, h = image_size(image)
    background[y:y + h, x:x + w] = image


def rgb2gray(image):
    """
    RGB 转灰度。
    使用 (MAX(r, g, b) + MIN(r, g, b)) / 2 算法。
    """
    r, g, b = cv2.split(image)
    maximum = cv2.max(r, g)
    cv2.min(r, g, dst=r)
    cv2.max(maximum, b, dst=maximum)
    cv2.min(r, b, dst=r)
    cv2.convertScaleAbs(maximum, alpha=0.5, dst=maximum)
    cv2.convertScaleAbs(r, alpha=0.5, dst=r)
    cv2.add(maximum, r, dst=maximum)
    return maximum


def rgb2hsv(image):
    """
    RGB 转 HSV。
    """
    image = cv2.cvtColor(image, cv2.COLOR_RGB2HSV).astype(float)
    cv2.multiply(image, (360 / 180, 100 / 255, 100 / 255, 0), dst=image)
    return image


def rgb2yuv(image):
    """
    RGB 转 YUV。
    """
    image = cv2.cvtColor(image, cv2.COLOR_RGB2YUV)
    return image


def rgb2luma(image):
    """
    RGB 转 YUV 中的 Y 通道（亮度）。
    """
    image = cv2.cvtColor(image, cv2.COLOR_RGB2YUV)
    luma, _, _ = cv2.split(image)
    return luma


def get_color(image, area):
    """
    获取指定区域的平均颜色。
    """
    temp = crop(image, area, copy=False)
    color = cv2.mean(temp)
    return color[:3]


class ImageNotSupported(Exception):
    pass


def get_bbox(image, threshold=0):
    """
    获取图像内容的边界框（类似 Pillow 的 getbbox）。
    大于 threshold 的像素被视为内容。

    Args:
        image (np.ndarray):
        threshold (int):

    Returns:
        tuple[int, int, int, int]: (min_x, min_y, max_x, max_y)
    """
    channel = image_channel(image)
    if channel == 3:
        mask = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        cv2.threshold(mask, threshold, 255, cv2.THRESH_BINARY, dst=mask)
    elif channel == 0:
        _, mask = cv2.threshold(image, threshold, 255, cv2.THRESH_BINARY)
    elif channel == 4:
        mask = cv2.cvtColor(image, cv2.COLOR_RGBA2GRAY)
        cv2.threshold(mask, threshold, 255, cv2.THRESH_BINARY, dst=mask)
    else:
        raise ImageNotSupported(f'shape={image.shape}')

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    min_y, min_x = mask.shape
    max_x = 0
    max_y = 0
    if not contours:
        raise ImageNotSupported(f'Cannot get bbox from a pure black image')
    for contour in contours:
        x1, y1, x2, y2 = cv2.boundingRect(contour)
        x2 += x1
        y2 += y1
        if x1 < min_x: min_x = x1
        if y1 < min_y: min_y = y1
        if x2 > max_x: max_x = x2
        if y2 > max_y: max_y = y2
    if min_x < max_x and min_y < max_y:
        return min_x, min_y, max_x, max_y
    else:
        raise ImageNotSupported(f'Empty bbox {(min_x, min_y, max_x, max_y)}')


def get_bbox_reversed(image, threshold=255):
    """
    获取图像内容的边界框（反向）。
    小于 threshold 的像素被视为内容。
    """
    channel = image_channel(image)
    if channel == 3:
        mask = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        cv2.threshold(mask, 0, threshold, cv2.THRESH_BINARY, dst=mask)
    elif channel == 0:
        mask = cv2.threshold(image, 0, threshold, cv2.THRESH_BINARY)
    elif channel == 4:
        mask = cv2.cvtColor(image, cv2.COLOR_RGBA2GRAY)
        cv2.threshold(mask, 0, threshold, cv2.THRESH_BINARY, dst=mask)
    else:
        raise ImageNotSupported(f'shape={image.shape}')

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    min_y, min_x = mask.shape
    max_x = 0
    max_y = 0
    if not contours:
        raise ImageNotSupported(f'Cannot get bbox from a pure black image')
    for contour in contours:
        x1, y1, x2, y2 = cv2.boundingRect(contour)
        x2 += x1
        y2 += y1
        if x1 < min_x: min_x = x1
        if y1 < min_y: min_y = y1
        if x2 > max_x: max_x = x2
        if y2 > max_y: max_y = y2
    if min_x < max_x and min_y < max_y:
        return min_x, min_y, max_x, max_y
    else:
        raise ImageNotSupported(f'Empty bbox {(min_x, min_y, max_x, max_y)}')


def color_similarity(color1, color2):
    """
    计算两个颜色的差异值。
    """
    diff_r = color1[0] - color2[0]
    diff_g = color1[1] - color2[1]
    diff_b = color1[2] - color2[2]

    max_positive = 0
    max_negative = 0
    if diff_r > max_positive:
        max_positive = diff_r
    elif diff_r < max_negative:
        max_negative = diff_r
    if diff_g > max_positive:
        max_positive = diff_g
    elif diff_g < max_negative:
        max_negative = diff_g
    if diff_b > max_positive:
        max_positive = diff_b
    elif diff_b < max_negative:
        max_negative = diff_b

    diff = max_positive - max_negative
    return diff


def color_similar(color1, color2, threshold=10):
    """
    判断两个颜色是否相似。
    """
    diff = color_similarity(color1, color2)
    return diff <= threshold


def color_similar_1d(image, color, threshold=10):
    """
    判断 1D 图像数组中的颜色是否相似。
    """
    diff = image.astype(int) - color
    diff = np.max(np.maximum(diff, 0), axis=1) - np.min(np.minimum(diff, 0), axis=1)
    return diff <= threshold


def color_similarity_2d(image, color):
    """
    计算 2D 图像与目标颜色的相似度图。
    返回 uint8 数组，值越大越相似（255 表示完全相同）。
    """
    diff = cv2.subtract(image, (*color, 0))
    r, g, b = cv2.split(diff)
    cv2.max(r, g, dst=r)
    cv2.max(r, b, dst=r)
    positive = r
    cv2.subtract((*color, 0), image, dst=diff)
    r, g, b = cv2.split(diff)
    cv2.max(r, g, dst=r)
    cv2.max(r, b, dst=r)
    negative = r
    cv2.add(positive, negative, dst=positive)
    cv2.subtract(255, positive, dst=positive)
    return positive


def extract_letters(image, letter=(255, 255, 255), threshold=128):
    """
    提取图像中的文字（指定颜色）。
    """
    diff = cv2.subtract(image, (*letter, 0))
    r, g, b = cv2.split(diff)
    cv2.max(r, g, dst=r)
    cv2.max(r, b, dst=r)
    positive = r
    cv2.subtract((*letter, 0), image, dst=diff)
    r, g, b = cv2.split(diff)
    cv2.max(r, g, dst=r)
    cv2.max(r, b, dst=r)
    negative = r
    cv2.add(positive, negative, dst=positive)
    if threshold != 255:
        cv2.convertScaleAbs(positive, alpha=255.0 / threshold, dst=positive)
    return positive


def extract_white_letters(image, threshold=128):
    """
    提取白色文字。
    """
    r, g, b = cv2.split(cv2.subtract((255, 255, 255, 0), image))
    maximum = cv2.max(r, g)
    cv2.min(r, g, dst=r)
    cv2.max(maximum, b, dst=maximum)
    cv2.min(r, b, dst=r)

    cv2.convertScaleAbs(maximum, alpha=0.5, dst=maximum)
    cv2.convertScaleAbs(r, alpha=0.5, dst=r)
    cv2.subtract(maximum, r, dst=r)
    cv2.add(maximum, r, dst=maximum)
    if threshold != 255:
        cv2.convertScaleAbs(maximum, alpha=255.0 / threshold, dst=maximum)
    return maximum


def color_mapping(image, max_multiply=2):
    """
    将颜色映射到 0-255 范围，增强对比度。
    """
    image = image.astype(float)
    low, high = np.min(image), np.max(image)
    multiply = min(255 / (high - low), max_multiply)
    add = (255 - multiply * (low + high)) / 2
    cv2.multiply(image, multiply, dst=image)
    cv2.add(image, add, dst=image)
    image[image > 255] = 255
    image[image < 0] = 0
    return image.astype(np.uint8)


def image_left_strip(image, threshold, length):
    """
    去除图像左侧的干扰（如标签）。
    """
    brightness = np.mean(image, axis=0)
    match = np.where(brightness < threshold)[0]

    if len(match):
        left = match[0] + length
        total = image.shape[1]
        if left < total:
            image = image[:, left:]
    return image


def red_overlay_transparency(color1, color2, red=247):
    """
    计算红色覆盖层的透明度。
    """
    return (color2[0] - color1[0]) / (red - color1[0])


def color_bar_percentage(image, area, prev_color, reverse=False, starter=0, threshold=30):
    """
    计算颜色条的百分比（如血条、进度条）。

    Args:
        image:
        area:
        prev_color:
        reverse: 是否反向（从右向左）。
        starter:
        threshold:

    Returns:
        float: 0-1.
    """
    image = crop(image, area, copy=False)
    image = image[:, ::-1, :] if reverse else image
    length = image.shape[1]
    prev_index = starter

    for _ in range(1280):
        bar = color_similarity_2d(image, color=prev_color)
        index = np.where(np.any(bar > 255 - threshold, axis=0))[0]
        if not index.size:
            return prev_index / length
        else:
            index = index[-1]
        if index <= prev_index:
            return index / length
        prev_index = index

        prev_row = bar[:, prev_index] > 255 - threshold
        if not prev_row.size:
            return prev_index / length
        # Look back 5px to get average color
        left = max(prev_index - 5, 0)
        mask = np.where(bar[:, left:prev_index + 1] > 255 - threshold)
        prev_color = np.mean(image[:, left:prev_index + 1][mask], axis=0)

    return 0.