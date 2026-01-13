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
        logger.info("Script Start: Login & Harvest")

        # 启动应用
        self.app_start()

        # ===========================================
        # 阶段一：登录循环 (直到检测到主页)
        # ===========================================
        timeout = Timer(5 * 60).start()

        while 1:
            if timeout.reached():
                logger.warning("Login timeout! Game stuck.")
                break

            # 成功检测：找到主页的荒原入口
            if self.device.yolo_find("icon_wilderness"):
                logger.info("Detected Home Screen! Login Success.")
                break

            # 过程处理：点击开始
            if self.device.yolo_click("btn_start", sleep_time=5):
                logger.info("Clicked Start Game")
                continue

            # 过程处理：关闭弹窗
            if self.device.yolo_click("btn_cancel", sleep_time=1):
                logger.info("Closed popup")
                continue

            self.device.sleep(1)

        # ===========================================
        # 阶段二：荒原收菜流程
        # 逻辑：主页 -> 荒原 -> 大厅(icon_home) -> 收取(btn_resource)
        # ===========================================
        logger.info("Starting Wilderness Routine...")

        # 1. 点击进入荒原
        if self.device.yolo_click("icon_wilderness", sleep_time=4):
            logger.info("Entered Wilderness Map")

            # 2. 点击进入大厅 (icon_home)
            # 这里可能需要多试几次，防止地图没加载出来
            hall_entered = False
            for _ in range(3):
                if self.device.yolo_click("icon_home", sleep_time=2):
                    logger.info("Entered Paleohall (icon_home)")
                    hall_entered = True
                    break
                self.device.sleep(1)

            if hall_entered:
                # 3. 点击资源收取 (btn_resource)
                # 假设点这个按钮就直接收了，或者弹个窗
                if self.device.yolo_click("btn_resource", sleep_time=2):
                    logger.info("Resources Collected (btn_resource clicked)")

                    # (可选) 如果收完需要点空白处或返回，可以在这里加
                    # self.device.yolo_click("btn_cancel")
                else:
                    logger.warning("btn_resource not found (Maybe already collected?)")

                # 4. (可选) 返回主页
                # 如果你需要收完菜回到游戏主菜单，可以在这里加 btn_back 或 btn_home(注意标签名别搞混)
                # self.device.yolo_click("btn_back")

            else:
                logger.warning("Failed to find 'icon_home' in Wilderness.")

        else:
            logger.warning("Failed to find 'icon_wilderness' on Main Menu.")

        logger.info("Task Finished.")