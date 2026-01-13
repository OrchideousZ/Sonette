from module.base.base import ModuleBase
from module.logger import logger
from module.base.timer import Timer


class Login(ModuleBase):
    def app_restart(self):
        logger.info('Login task calling app_restart')
        self.device.app_stop()
        self.device.app_start()

    def app_start(self):
        logger.info('Login task calling app_start')
        self.device.app_start()

    def app_stop(self):
        logger.info('Login task calling app_stop')
        self.device.app_stop()

    def run(self):
        logger.info("Script Start: Waiting for Log in")
        self.app_start()

        # 超时保护
        timeout = Timer(5 * 60).start()

        # 定义需要处理的弹窗列表 (通用干扰项)
        # 只要屏幕上出现这些，就优先点掉
        POPUP_INTERRUPTS = [
            'btn_cancel',  # 通用关闭
            'btn_confirm',  # 签到确认
            'btn_skip',  # 跳过剧情
        ]

        while 1:
            if timeout.reached():
                logger.warning("Login timeout!")
                break

            # 1. [感知] 看一眼屏幕，获取所有信息
            # detection_list 是列表，detection_map 是字典 {'label': item}
            detection_list, detection_map = self.device.scan_screen(conf=0.6)

            # 调试日志：看看当前看见了啥 (可选)
            # labels = [d['label'] for d in detection_list]
            # logger.info(f"Screen Scan: {labels}")

            # 2. [决策] 状态机判断

            # --- 登录 ---
            if 'btn_start' in detection_map:
                logger.info("Action: Start Game")
                self.device.click_result(detection_map['btn_start'], sleep_time=4)
                continue  # 点完动作后，重新下一轮扫描

            # --- 看到了荒原入口，启动成功 ---
            if 'icon_wilderness' in detection_map:
                logger.info("Detected Home Screen (Wilderness Icon). Login Success.")
                break  # 退出循环

            # --- 异常处理 (通用弹窗) ---
            # 遍历干扰项列表，看看当前屏幕上有没有
            handled_popup = False
            for popup_label in POPUP_INTERRUPTS:
                if popup_label in detection_map:
                    logger.info(f"Action: Handle Popup ({popup_label})")
                    self.device.click_result(detection_map[popup_label], sleep_time=1)
                    handled_popup = True
                    break  # 一次只点一个，点完重新扫描

            if handled_popup:
                continue

            # --- 兜底逻辑 ---
            # 如果什么都没发现，可能是加载中，稍微等一下
            self.device.sleep(0.5)
