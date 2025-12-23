'''
定义了 ModuleBase 类，它是所有具体任务类（如 Dungeon、Login 等）的基类，位于 module/base/base.py。它提供了任务执行环境所需的核心属性、基础工具和原子操作。
    作用：
        环境初始化： 负责实例化和管理配置对象（config）和设备对象（device），确保每个任务都能访问到当前配置和设备控制接口。
        核心工具集成： 集成了图像匹配（match_template、match_color）、元素层级结构查找（xpath）等自动化脚本最底层的感知和操作方法。
        状态循环： 提供了三种生成器（loop、loop_hierarchy、loop_screenshot_hierarchy），用于简化任务中不断截图/获取层级结构并检查状态的循环模式。
        防卡死机制： 通过 interval_timer 和 device.stuck_record_add 机制，辅助实现防卡死和冷却时间控制。

    调用的模块及位置：
        配置和设备：
            module.config.config.AzurLaneConfig：配置对象。
            module.device.device.Device：设备控制对象。
            module.device.method.utils.HierarchyButton：用于表示 UI 层级结构中的元素。

        图像识别和工具：
            module.base.button.*：包括 Button、ClickButton、match_template 等图像处理原子操作。
            module.base.timer.Timer：用于计时和控制任务间隔。
            module.base.utils.*：各种图像/数据处理工具函数（如 crop、color_similarity_2d）。

        其他： module.logger（日志）、module.webui.setting.cached_class_property（缓存属性装饰器）。
'''

# 导入服务器配置模块
import module.config.server as server_
# 导入基础按钮类、按钮封装类、点击按钮类以及模板匹配函数
from module.base.button import Button, ButtonWrapper, ClickButton, match_template
# 导入计时器类
from module.base.timer import Timer
# 导入各种基础工具函数（如图像处理、数学计算等）
from module.base.utils import *
# 导入基础配置类
from module.config.config import AzurLaneConfig
# 导入设备控制类
from module.device.device import Device
# 导入层级结构按钮类（基于 XPath 或其他层级信息）
from module.device.method.utils import HierarchyButton
# 导入日志模块
from module.logger import logger
# 导入缓存类属性装饰器
from module.webui.setting import cached_class_property


# 定义 ModuleBase 类，所有任务类的基类
class ModuleBase:
    # 类型标注：配置对象
    config: AzurLaneConfig
    # 类型标注：设备对象
    device: Device

    # 构造函数
    def __init__(self, config, device=None, task=None):
        """
        Args:
            config (AzurLaneConfig, str): 配置对象或配置文件名。
            device (Device, str):
                复用设备对象或设备序列号。
                如果为 None，则创建新的 Device 对象。
                如果为 str (serial)，则创建新的 Device 对象并使用指定的序列号。
            task (str):
                用于开发目的绑定特定任务的配置。自动任务调度时通常为 None。
        """
        # --- 配置对象初始化 ---
        if isinstance(config, AzurLaneConfig):
            self.config = config
            if task is not None:
                # 如果指定了 task，则初始化该 task 的配置
                self.config.init_task(task)
        elif isinstance(config, str):
            # 如果传入的是字符串，则认为是配置文件名，创建新的配置对象
            self.config = AzurLaneConfig(config, task=task)
        else:
            logger.warning('Alas ModuleBase received an unknown config, assume it is AzurLaneConfig')
            self.config = config

        # --- 设备对象初始化 ---
        if isinstance(device, Device):
            # 复用已有的 Device 对象
            self.device = device
        elif device is None:
            # 创建新的 Device 对象
            self.device = Device(config=self.config)
        elif isinstance(device, str):
            # 传入序列号，覆盖配置中的序列号并创建 Device
            self.config.override(Emulator_Serial=device)
            self.device = Device(config=self.config)
        else:
            logger.warning('Alas ModuleBase received an unknown device, assume it is Device')
            self.device = device

        # 间隔计时器字典，用于控制重复操作的冷却时间
        self.interval_timer = {}

    # 使用 @cached_class_property 装饰器，创建并缓存一个线程池供所有实例共享
    @cached_class_property
    def worker(self):
        """
        一个线程池，用于在后台运行任务（例如不影响主流程的数据更新或日志记录）。
        """
        logger.hr('Creating worker')
        from concurrent.futures import ThreadPoolExecutor
        # 创建一个单线程的线程池
        pool = ThreadPoolExecutor(1)
        return pool

    # 状态循环生成器：基于截图
    def loop(self, skip_first=True):
        """
        简化状态循环的语法糖，不断截图。

        Args:
            skip_first (bool): 通常为 True，以复用进入循环前的第一次截图。

        Yields:
            np.ndarray: 当前的截图图像。
        """
        while 1:
            if skip_first:
                skip_first = False
            else:
                # 进行截图
                self.device.screenshot()
            yield self.device.image

    # 状态循环生成器：基于 UI 层级结构（Hierarchy）
    def loop_hierarchy(self, skip_first=True):
        """
        简化基于 UI 层级结构检查的状态循环。

        Args:
            skip_first (bool): 通常为 True，以复用进入循环前的第一次层级结构。

        Yields:
            etree._Element: 当前的 UI 层级结构 XML 根元素。
        """
        while 1:
            if skip_first:
                skip_first = False
            else:
                # 导出 UI 层级结构
                self.device.dump_hierarchy()
            yield self.device.hierarchy

    # 状态循环生成器：截图和层级结构同时获取
    def loop_screenshot_hierarchy(self, skip_first=True):
        """
        同时获取截图和 UI 层级结构的状态循环。

        Args:
            skip_first (bool): 通常为 True，以复用进入循环前的第一次数据。

        Yields:
            tuple[np.ndarray, etree._Element]: 截图和层级结构。
        """
        while 1:
            if skip_first:
                skip_first = False
            else:
                self.device.screenshot()
                self.device.dump_hierarchy()
            yield self.device.image, self.device.hierarchy

    # 模板匹配（彩色图）
    def match_template(self, button, interval=0, similarity=0.85):
        """
        在当前截图上匹配模板。

        Args:
            button (ButtonWrapper): 模板按钮对象。
            interval (int, float): 两次操作之间的最小间隔时间（秒），用于防卡死。
            similarity (int, float): 相似度阈值（0到1）。

        Returns:
            bool: 是否匹配成功。
        """
        # 将按钮添加到卡死记录，用于检测重复操作
        self.device.stuck_record_add(button)

        # 检查是否达到间隔时间
        if interval and not self.interval_is_reached(button, interval=interval):
            return False

        # 执行模板匹配
        appear = button.match_template(self.device.image, similarity=similarity)

        # 如果匹配成功且设置了间隔，则重置计时器
        if appear and interval:
            self.interval_reset(button, interval=interval)

        return appear

    # 模板匹配（灰度图/亮度）
    def match_template_luma(self, button, interval=0, similarity=0.85):
        """
        在当前截图上匹配模板（基于亮度/灰度）。
        """
        self.device.stuck_record_add(button)

        if interval and not self.interval_is_reached(button, interval=interval):
            return False

        # 执行亮度模板匹配
        appear = button.match_template_luma(self.device.image, similarity=similarity)

        if appear and interval:
            self.interval_reset(button, interval=interval)

        return appear

    # 颜色匹配
    def match_color(self, button, interval=0, threshold=10):
        """
        匹配指定区域内的颜色是否符合预期。

        Args:
            button (ButtonWrapper): 颜色按钮对象，包含匹配区域和目标颜色。
            interval (int, float): 间隔时间。
            threshold (int): 颜色相似度阈值（0到255，越小越相似）。

        Returns:
            bool: 是否匹配成功。
        """
        self.device.stuck_record_add(button)

        if interval and not self.interval_is_reached(button, interval=interval):
            return False

        # 执行颜色匹配
        appear = button.match_color(self.device.image, threshold=threshold)

        if appear and interval:
            self.interval_reset(button, interval=interval)

        return appear

    # 模板和颜色综合匹配
    def match_template_color(self, button, interval=0, similarity=0.85, threshold=30):
        """
        综合模板匹配和颜色匹配。
        """
        self.device.stuck_record_add(button)

        if interval and not self.interval_is_reached(button, interval=interval):
            return False

        # 执行模板和颜色综合匹配
        appear = button.match_template_color(self.device.image, similarity=similarity, threshold=threshold)

        if appear and interval:
            self.interval_reset(button, interval=interval)

        return appear

    # 将 XPath 字符串转换为 HierarchyButton 对象
    def xpath(self, xpath) -> HierarchyButton:
        # 如果是字符串，则基于当前的 UI 层级结构创建 HierarchyButton
        if isinstance(xpath, str):
            return HierarchyButton(self.device.hierarchy, xpath)
        # 否则直接返回对象
        else:
            return xpath

    # 检查 XPath 元素是否出现
    def xpath_appear(self, xpath: str, interval=0):
        # 转换为 HierarchyButton
        button = self.xpath(xpath)

        self.device.stuck_record_add(button)

        if interval and not self.interval_is_reached(button, interval=interval):
            return False

        # 检查 HierarchyButton 对象是否包含匹配的元素（布尔转换）
        appear = bool(button)

        if appear and interval:
            self.interval_reset(button, interval=interval)

        return appear

    # 统一的“出现”检查接口（封装了图像和 XPath 检查）
    def appear(self, button, interval=0, similarity=0.85):
        """
        检查指定按钮（图像模板、颜色或 XPath 元素）是否出现。

        Args:
            button (Button, ButtonWrapper, HierarchyButton, str): 待检查的对象。
            interval (int, float): 间隔时间。

        Returns:
            bool: 是否出现。
        """
        if isinstance(button, (HierarchyButton, str)):
            # 如果是 HierarchyButton 或 XPath 字符串，则使用层级结构检查
            return self.xpath_appear(button, interval=interval)
        else:
            # 否则使用图像模板匹配
            return self.match_template(button, interval=interval, similarity=similarity)

    # 检查是否出现，如果出现则点击
    def appear_then_click(self, button, interval=5, similarity=0.85):
        # 确保 button 是可点击的对象（如果是 XPath 字符串，转换为 HierarchyButton）
        button = self.xpath(button)
        # 检查是否出现（使用间隔控制）
        appear = self.appear(button, interval=interval, similarity=similarity)
        if appear:
            # 如果出现，则点击该按钮
            self.device.click(button)
        return appear

    # 等待一个区域内的图像内容稳定不变
    def wait_until_stable(self, button, timer=Timer(0.3, count=1), timeout=Timer(5, count=10)):
        """
        等待直到指定区域内的图像内容稳定。
        """
        logger.info(f'Wait until stable: {button}')
        # 获取初始图像
        prev_image = self.image_crop(button)
        timer.reset()
        timeout.reset()
        while 1:
            self.device.screenshot()

            if timeout.reached():
                logger.warning(f'wait_until_stable({button}) timeout')
                break

            # 获取当前图像
            image = self.image_crop(button)
            # 比较当前图像和前一张图像的相似度
            if match_template(image, prev_image):
                # 如果相似，且稳定计时器达到阈值，则认为稳定
                if timer.reached():
                    logger.info(f'{button} stabled')
                    break
            else:
                # 如果不相似，更新前一张图像，重置稳定计时器
                prev_image = image
                timer.reset()

    # 从当前截图裁剪指定区域的图像
    def image_crop(self, button, copy=True):
        """
        从当前截图（self.device.image）中提取指定区域。

        Args:
            button(Button, tuple): 按钮实例或区域 (x1, y1, x2, y2) 元组。
            copy: 是否复制图像数据。
        """
        if isinstance(button, Button):
            return crop(self.device.image, button.area, copy=copy)
        elif isinstance(button, ButtonWrapper):
            return crop(self.device.image, button.area, copy=copy)
        elif hasattr(button, 'area'):
            return crop(self.device.image, button.area, copy=copy)
        else:
            return crop(self.device.image, button, copy=copy)

    # 统计指定区域内某种颜色的像素数量
    def image_color_count(self, button, color, threshold=221, count=50):
        """
        检查指定区域内目标颜色的像素数量是否超过阈值。

        Args:
            button (Button, tuple): 按钮实例或区域。
            color (tuple): 目标 RGB 颜色。
            threshold: 颜色相似度阈值。
            count (int): 像素数量阈值。

        Returns:
            bool: 像素数量是否超过 count。
        """
        # 如果传入的是 numpy 数组，则直接使用
        if isinstance(button, np.ndarray):
            image = button
        else:
            # 否则裁剪图像
            image = self.image_crop(button, copy=False)
        # 计算图像中与目标颜色相似的像素的掩码
        mask = color_similarity_2d(image, color=color)
        # 将相似度高于 threshold 的像素置为 255
        cv2.inRange(mask, threshold, 255, dst=mask)
        # 统计非零像素的数量
        sum_ = cv2.countNonZero(mask)
        return sum_ > count

    # 根据颜色在区域内寻找最匹配的点并生成一个 ClickButton
    def image_color_button(self, area, color, color_threshold=250, encourage=5, name='COLOR_BUTTON'):
        """
        在指定区域内寻找纯色区域的中心点，并创建一个 ClickButton。

        Args:
            area (tuple[int]): 搜索区域。
            color (tuple[int]): 目标颜色。
            color_threshold (int): 颜色相似度阈值。
            encourage (int): 生成的按钮的半径。
            name (str): 按钮名称。

        Returns:
            Button: 匹配到的 ClickButton，如果没有匹配则为 None。
        """
        # 计算裁剪区域内每个像素与目标颜色的相似度
        image = color_similarity_2d(self.image_crop(area, copy=False), color=color)
        # 找到相似度高于阈值的像素点坐标
        points = np.array(np.where(image > color_threshold)).T[:, ::-1]
        # 如果匹配到的像素点数量不足
        if points.shape[0] < encourage ** 2:
            return None

        # 拟合这些点以找到一个代表性的中心点
        point = fit_points(points, mod=image_size(image), encourage=encourage)
        # 将相对坐标转换为绝对坐标
        point = ensure_int(point + area[:2])
        # 根据中心点和鼓励半径计算按钮区域
        button_area = area_offset((-encourage, -encourage, encourage, encourage), offset=point)
        # 返回 ClickButton 对象
        return ClickButton(area=button_area, name=name)

    # 获取或创建用于特定按钮的间隔计时器
    def get_interval_timer(self, button, interval=5, renew=False) -> Timer:
        # 根据按钮类型获取唯一名称
        if hasattr(button, 'name'):
            name = button.name
        elif callable(button):
            name = button.__name__
        else:
            name = str(button)

        try:
            timer = self.interval_timer[name]
            # 如果 renew 且间隔时间变化，则创建新的计时器
            if renew and timer.limit != interval:
                timer = Timer(interval)
                self.interval_timer[name] = timer
            return timer
        except KeyError:
            # 第一次访问，创建新的计时器
            timer = Timer(interval)
            self.interval_timer[name] = timer
            return timer

    # 重置间隔计时器（即开始计时冷却）
    def interval_reset(self, button, interval=5):
        if isinstance(button, (list, tuple)):
            for b in button:
                self.interval_reset(b, interval)
            return

        if button is not None:
            self.get_interval_timer(button, interval=interval).reset()

    # 清除间隔计时器（即允许立即操作）
    def interval_clear(self, button, interval=5):
        if isinstance(button, (list, tuple)):
            for b in button:
                self.interval_clear(b, interval)
            return

        if button is not None:
            self.get_interval_timer(button, interval=interval).clear()

    # 检查间隔时间是否已到
    def interval_is_reached(self, button, interval=5):
        # 检查计时器是否达到限制（并自动更新计时器 limit）
        return self.get_interval_timer(button, interval=interval, renew=True).reached()

    _image_file = ''

    # 用于开发测试：从文件加载图像到 device.image
    @property
    def image_file(self):
        return self._image_file

    @image_file.setter
    def image_file(self, value):
        """
        为开发目的，从本地文件系统加载图像，并将其设置为 self.device.image，用于测试，无需从模拟器截图。
        """
        if isinstance(value, Image.Image):
            value = np.array(value)
        elif isinstance(value, str):
            value = load_image(value)

        self.device.image = value

    # 用于开发测试：设置语言配置
    def set_lang(self, lang):
        """
        为开发目的，更改语言配置，并全局生效（包括资源和服务器特定方法）。
        """
        server_.set_lang(lang)
        logger.attr('Lang', self.config.LANG)

    # 添加追踪截图（用于错误日志）
    def screenshot_tracking_add(self):
        """
        添加一个追踪图像，图像将被保存到错误日志。
        """
        # 如果配置不允许保存错误日志，则返回
        if not self.config.Error_SaveError:
            return

        logger.info('screenshot_tracking_add')
        # 获取最新的截图数据
        data = self.device.screenshot_deque[-1]
        image = data['image']
        now = data['time']

        # 图像编码函数，在新线程中执行
        def image_encode(im, ti):
            import io
            from module.handler.sensitive_info import handle_sensitive_image

            output = io.BytesIO()
            # 处理敏感信息（例如头像、昵称）
            im = handle_sensitive_image(im)
            # 将图像保存为 png 格式到内存流
            Image.fromarray(im, mode='RGB').save(output, format='png')
            output.seek(0)

            # 将时间和图像流添加到追踪队列
            self.device.screenshot_tracking.append({
                'time': ti,
                'image': output
            })

        # 在 worker 线程池中提交编码任务
        ModuleBase.worker.submit(image_encode, image, now)