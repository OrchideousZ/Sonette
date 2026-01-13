import collections
import itertools
from module.yolo.inference import yolo_agent  # 确保这里正确导入
import random

from lxml import etree

from module.device.env import IS_WINDOWS
# Patch pkg_resources before importing adbutils and uiautomator2
from module.device.pkg_resources import get_distribution

# Just avoid being removed by import optimization
_ = get_distribution

from module.base.timer import Timer
from module.device.app_control import AppControl
from module.device.control import Control
from module.device.screenshot import Screenshot
from module.exception import (
    EmulatorNotRunningError,
    GameNotRunningError,
    GameStuckError,
    GameTooManyClickError,
    RequestHumanTakeover
)
from module.logger import logger

from module.ocr.paddle_ocr import ocr_agent


def crop_image(image, box):
    """
    简单的图片裁剪函数
    Args:
        image: OpenCV 图像 (numpy array)
        box: (x1, y1, x2, y2)
    """
    x1, y1, x2, y2 = map(int, box)
    h, w = image.shape[:2]
    # 边界保护，防止报错
    x1 = max(0, x1)
    y1 = max(0, y1)
    x2 = min(w, x2)
    y2 = min(h, y2)
    return image[y1:y2, x1:x2]

class YoloTarget:
    """
    一个极简的点击目标包装类，专门骗过 ALAS 的 click 函数
    """
    def __init__(self, box, name="YoloTarget"):
        # box 格式: (x1, y1, x2, y2)
        self.button = box  # ALAS 的 random_rectangle_point 需要读取这个属性
        self.name = name   # 日志记录需要读取这个属性

    def __str__(self):
        return self.name

def show_function_call():
    """
    INFO     21:07:31.554 │ Function calls:
                       <string>   L1 <module>
                   spawn.py L116 spawn_main()
                   spawn.py L129 _main()
                 process.py L314 _bootstrap()
                 process.py L108 run()
         process_manager.py L149 run_process()
                    alas.py L285 loop()
                    alas.py  L69 run()
                     src.py  L55 rogue()
                   rogue.py  L36 run()
                   rogue.py  L18 rogue_once()
                   entry.py L335 rogue_world_enter()
                    path.py L193 rogue_path_select()
    """
    import os
    import traceback
    stack = traceback.extract_stack()
    func_list = []
    for row in stack:
        filename, line_number, function_name, _ = row
        filename = os.path.basename(filename)
        # /tasks/character/switch.py:64 character_update()
        func_list.append([filename, str(line_number), function_name])
    max_filename = max([len(row[0]) for row in func_list])
    max_linenum = max([len(row[1]) for row in func_list]) + 1

    def format_(file, line, func):
        file = file.rjust(max_filename, " ")
        line = f'L{line}'.rjust(max_linenum, " ")
        if not func.startswith('<'):
            func = f'{func}()'
        return f'{file} {line} {func}'

    func_list = [f'\n{format_(*row)}' for row in func_list]
    logger.info('Function calls:' + ''.join(func_list))


class Device(Screenshot, Control, AppControl):
    _screen_size_checked = False
    detect_record = set()
    click_record = collections.deque(maxlen=30)
    stuck_timer = Timer(60, count=60).start()

    def __init__(self, *args, **kwargs):
        for trial in range(4):
            try:
                super().__init__(*args, **kwargs)
                break
            except EmulatorNotRunningError:
                if trial >= 3:
                    logger.critical('Failed to start emulator after 3 trial')
                    raise RequestHumanTakeover
                # Try to start emulator
                if self.emulator_instance is not None:
                    self.emulator_start()
                else:
                    logger.critical(
                        f'No emulator with serial "{self.config.Emulator_Serial}" found, '
                        f'please set a correct serial'
                    )
                    raise RequestHumanTakeover

        # Auto-fill emulator info
        if IS_WINDOWS and self.config.EmulatorInfo_Emulator == 'auto':
            _ = self.emulator_instance

        self.screenshot_interval_set()
        self.method_check()

        # Auto-select the fastest screenshot method
        if not self.config.is_template_config and self.config.Emulator_ScreenshotMethod == 'auto':
            self.run_simple_screenshot_benchmark()

        # Early init
        if self.config.is_actual_task:
            if self.config.Emulator_ControlMethod == 'MaaTouch':
                self.early_maatouch_init()
            if self.config.Emulator_ControlMethod == 'minitouch':
                self.early_minitouch_init()

    def run_simple_screenshot_benchmark(self):
        """
        Perform a screenshot method benchmark, test 3 times on each method.
        The fastest one will be set into config.
        """
        logger.info('run_simple_screenshot_benchmark')
        # Check resolution first
        self.resolution_check_uiautomator2()
        # Perform benchmark
        from module.daemon.benchmark import Benchmark
        bench = Benchmark(config=self.config, device=self)
        method = bench.run_simple_screenshot_benchmark()
        # Set
        with self.config.multi_set():
            self.config.Emulator_ScreenshotMethod = method
            # if method == 'nemu_ipc':
            #     self.config.Emulator_ControlMethod = 'nemu_ipc'

    def method_check(self):
        """
        Check combinations of screenshot method and control methods
        """
        # nemu_ipc should be together
        # if self.config.Emulator_ScreenshotMethod == 'nemu_ipc' and self.config.Emulator_ControlMethod != 'nemu_ipc':
        #     logger.warning('When using nemu_ipc, both screenshot and control should use nemu_ipc')
        #     self.config.Emulator_ControlMethod = 'nemu_ipc'
        # if self.config.Emulator_ScreenshotMethod != 'nemu_ipc' and self.config.Emulator_ControlMethod == 'nemu_ipc':
        #     logger.warning('When not using nemu_ipc, both screenshot and control should not use nemu_ipc')
        #     self.config.Emulator_ControlMethod = 'minitouch'
        # Allow Hermit on VMOS only
        if self.config.Emulator_ControlMethod == 'Hermit' and not self.is_vmos:
            logger.warning('ControlMethod Hermit is allowed on VMOS only')
            self.config.Emulator_ControlMethod = 'MaaTouch'
        if self.config.Emulator_ScreenshotMethod == 'ldopengl' \
                and self.config.Emulator_ControlMethod == 'minitouch':
            logger.warning('Use MaaTouch on ldplayer')
            self.config.Emulator_ControlMethod = 'MaaTouch'

        # Fallback to auto if nemu_ipc and ldopengl are selected on non-corresponding emulators
        if self.config.Emulator_ScreenshotMethod == 'nemu_ipc':
            if not (self.is_emulator and self.is_mumu_family):
                logger.warning('ScreenshotMethod nemu_ipc is available on MuMu Player 12 only, fallback to auto')
                self.config.Emulator_ScreenshotMethod = 'auto'
        if self.config.Emulator_ScreenshotMethod == 'ldopengl':
            if not (self.is_emulator and self.is_ldplayer_bluestacks_family):
                logger.warning('ScreenshotMethod ldopengl is available on LD Player only, fallback to auto')
                self.config.Emulator_ScreenshotMethod = 'auto'
        if not IS_WINDOWS and self.config.Emulator_ScreenshotMethod in ['nemu_ipc', 'ldopengl']:
            logger.warning(f'ScreenshotMethod {self.config.Emulator_ScreenshotMethod} is available on Windows only, '
                           f'fallback to auto')
            self.config.Emulator_ScreenshotMethod = 'auto'

    def screenshot(self):
        """
        Returns:
            np.ndarray:
        """
        self.stuck_record_check()

        try:
            super().screenshot()
        except RequestHumanTakeover:
            if not self.ascreencap_available:
                logger.error('aScreenCap unavailable on current device, fallback to auto')
                self.run_simple_screenshot_benchmark()
                super().screenshot()
            else:
                raise

        return self.image

    def dump_hierarchy(self) -> etree._Element:
        self.stuck_record_check()
        return super().dump_hierarchy()

    def release_during_wait(self):
        # Scrcpy server is still sending video stream,
        # stop it during wait
        if self.config.Emulator_ScreenshotMethod == 'scrcpy':
            self._scrcpy_server_stop()
        if self.config.Emulator_ScreenshotMethod == 'nemu_ipc':
            self.nemu_ipc_release()

    def get_orientation(self):
        """
        Callbacks when orientation changed.
        """
        o = super().get_orientation()

        self.on_orientation_change_maatouch()

        return o

    def stuck_record_add(self, button):
        self.detect_record.add(str(button))

    def stuck_record_clear(self):
        self.detect_record = set()
        self.stuck_timer.reset()

    def stuck_record_check(self):
        """
        Raises:
            GameStuckError:
        """
        reached = self.stuck_timer.reached()
        if not reached:
            return False

        show_function_call()
        logger.warning('Wait too long')
        logger.warning(f'Waiting for {self.detect_record}')
        self.stuck_record_clear()

        if self.app_is_running():
            raise GameStuckError(f'Wait too long')
        else:
            raise GameNotRunningError('Game died')

    def handle_control_check(self, button):
        self.stuck_record_clear()
        self.click_record_add(button)
        self.click_record_check()

    def click_record_add(self, button):
        self.click_record.append(str(button))

    def click_record_clear(self):
        self.click_record.clear()

    def click_record_remove(self, button):
        """
        Remove a button from `click_record`

        Args:
            button (Button):

        Returns:
            int: Number of button removed
        """
        removed = 0
        for _ in range(self.click_record.maxlen):
            try:
                self.click_record.remove(str(button))
                removed += 1
            except ValueError:
                # Value not in queue
                break

        return removed

    def click_record_check(self):
        """
        Raises:
            GameTooManyClickError:
        """
        first15 = itertools.islice(self.click_record, 0, 15)
        count = collections.Counter(first15).most_common(2)
        if count[0][1] >= 12:
            # Allow more clicks in Ruan Mei event
            if 'CHOOSE_OPTION_CONFIRM' in self.click_record and 'BLESSING_CONFIRM' in self.click_record:
                count = collections.Counter(self.click_record).most_common(2)
                if count[0][0] == 'BLESSING_CONFIRM' and count[0][1] < 25:
                    return
            show_function_call()
            logger.warning(f'Too many click for a button: {count[0][0]}')
            logger.warning(f'History click: {[str(prev) for prev in self.click_record]}')
            self.click_record_clear()
            raise GameTooManyClickError(f'Too many click for a button: {count[0][0]}')
        if len(count) >= 2 and count[0][1] >= 6 and count[1][1] >= 6:
            show_function_call()
            logger.warning(f'Too many click between 2 buttons: {count[0][0]}, {count[1][0]}')
            logger.warning(f'History click: {[str(prev) for prev in self.click_record]}')
            self.click_record_clear()
            raise GameTooManyClickError(f'Too many click between 2 buttons: {count[0][0]}, {count[1][0]}')

    def disable_stuck_detection(self):
        """
        Disable stuck detection and its handler. Usually uses in semi auto and debugging.
        """
        logger.info('Disable stuck detection')

        def empty_function(*arg, **kwargs):
            return False

        self.click_record_check = empty_function
        self.stuck_record_check = empty_function

    def app_start(self):
        super().app_start()
        self.stuck_record_clear()
        self.click_record_clear()

    def app_stop(self):
        super().app_stop()
        self.stuck_record_clear()
        self.click_record_clear()

    # =========================================================================
    # YOLO AI Extensions
    # =========================================================================

    def yolo_find(self, target_label, conf=0.5):
        """
        使用 YOLO 寻找单个目标
        Returns:
            dict: {'label':..., 'box': [x1, y1, x2, y2], 'center': (cx, cy)} 或 None
        """
        screenshot = self.screenshot()
        detections = yolo_agent.predict(screenshot, conf_thres=conf)

        for item in detections:
            if item['label'] == target_label:
                return item
        return None

    def yolo_find_all(self, target_label, conf=0.5):
        """
        使用 YOLO 寻找所有匹配的目标
        """
        screenshot = self.screenshot()
        detections = yolo_agent.predict(screenshot, conf_thres=conf)
        return [d for d in detections if d['label'] == target_label]

    def yolo_click(self, target_label, conf=0.5, sleep_time=1):
        """
        使用 YOLO 查找并点击目标 (使用轻量化 YoloTarget 绕过 ALAS Button 限制)
        """
        item = self.yolo_find(target_label, conf)

        if item:
            # 1. 提取坐标框并转为元组
            box = tuple(item['box'])

            # 2. 实例化我们自己写的轻量级对象 (鸭子类型)
            # 这不需要 file, search 等一堆废话参数
            target = YoloTarget(box=box, name=target_label)

            logger.info(f'YOLO Click: {target_label} @ {box} (Conf: {item["conf"]:.2f})')

            # 3. 调用 ALAS 底层的点击
            # self.click 内部只关心 target.button 属性，所以这能完美运行
            self.click(target)

            # 4. 记录防卡死
            self.handle_control_check(target)

            self.sleep(sleep_time)
            return True

        return False

    def scan_screen(self, conf=0.5):
        """
        [感知] 全屏扫描
        一次性对当前屏幕进行推理，返回所有检测到的目标。

        Returns:
            list: 检测结果列表, e.g.
                  [{'label': 'btn_start', 'conf': 0.9, 'box': [...], 'center': (x,y)},
                   {'label': 'btn_cancel', 'conf': 0.8, ...}]
            dict: 标签索引字典, 方便快速查询 e.g. {'btn_start': <item>, 'btn_cancel': <item>}
        """
        screenshot = self.screenshot()
        detections = yolo_agent.predict(screenshot, conf_thres=conf)

        # 生成一个字典方便快速查找是否存在某个标签
        # 注意：如果同屏有多个相同标签，字典只会存最后一个 (通常对于UI按钮够用了)
        detections_map = {item['label']: item for item in detections}

        return detections, detections_map

    def click_result(self, item, sleep_time=1):
        """
        [行动] 点击扫描到的结果
        Args:
            item (dict): scan_screen 返回列表中的某一项
        """
        if not item:
            return False

        label = item['label']
        box = tuple(item['box'])

        # 实例化轻量级对象 (鸭子类型)
        target = YoloTarget(box=box, name=label)

        logger.info(f'YOLO Click: {label} @ {box} (Conf: {item["conf"]:.2f})')

        self.click(target)
        self.handle_control_check(target)
        self.sleep(sleep_time)
        return True

    def ocr_yolo_box(self, yolo_item):
        """
        对 YOLO 识别到的某个目标区域进行 OCR 文字识别
        Args:
            yolo_item (dict): yolo_find 返回的结果字典
        Returns:
            str: 识别到的文本 (已去除空格)
        """
        if not yolo_item:
            return ""

        # 1. 获取当前屏幕截图
        # 注意：这里直接用 self.image，假设在调用此方法前已经执行过 screenshot()
        # 如果不确定，可以再次调用 self.screenshot()
        image = self.image

        # 2. 裁剪区域 (box: x1, y1, x2, y2)
        box = yolo_item['box']
        crop = crop_image(image, box)

        # 3. OCR 识别
        text = ocr_agent.predict(crop)

        # 4. 清理文本 (去掉空格，方便转数字)
        clean_text = text.replace(" ", "")
        logger.info(f"OCR Result [{yolo_item['label']}]: {clean_text}")

        return clean_text