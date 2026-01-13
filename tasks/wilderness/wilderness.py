from module.base.base import ModuleBase
from module.logger import logger
from module.base.timer import Timer
import re # 导入正则


class Wilderness(ModuleBase):
    def run(self):
        """
        荒原收菜任务主入口
        """
        logger.info("Task Start: Wilderness")

        # 1. 基础环境检查 (只检查包名，不重启)
        if not self.device.app_is_running():
            logger.error("Game is not running! Please start the game first.")
            return

        # 2. 初始化状态标志
        self.resources_claimed = False  # 是否已完成收菜
        self.bond_clicked_count = 0  # 摸头次数计数
        self.well_processed = False  # 好梦井是否处理完

        # 3. 设置超时 (防止任务卡死)
        timeout = Timer(5 * 60).start()

        # 4. 定义干扰项 (看见就点)
        POPUP_INTERRUPTS = ['btn_cancel',
                            'btn_confirm',
                            'btn_skip']

        # ================= Loop Start =================
        while 1:
            if timeout.reached():
                logger.warning("Wilderness task timeout!")
                break

            # [感知] 全屏扫描
            # detection_map: {'label': item_dict, ...}
            detection_list, detection_map = self.device.scan_screen(conf=0.6)

            # [决策] 状态机逻辑

            # --------------------------------------------------
            # 优先级 0: 任务完成，返回主页
            # --------------------------------------------------
            if self.resources_claimed:
                # 如果已经收过菜了，目标是回到主页
                if 'icon_wilderness' in detection_map:
                    logger.info("Returned to Main Menu. Task Completed.")
                    break  # 任务结束

                # 如果还在荒原或大厅，找返回键或主页键
                if 'btn_home' in detection_map:  # 这里的 btn_home 指左上角的UI主页键
                    self.device.click_result(detection_map['btn_home'])
                    continue
                if 'btn_back' in detection_map:
                    self.device.click_result(detection_map['btn_back'])
                    continue

            # --------------------------------------------------
            # 优先级 1: 处理弹窗 (断线重连)
            # --------------------------------------------------
            handled_popup = False
            for popup in POPUP_INTERRUPTS:
                if popup in detection_map:
                    logger.info(f"Action: Handle Popup ({popup})")
                    self.device.click_result(detection_map[popup])
                    handled_popup = True
                    break
            if handled_popup: continue  # 这轮处理完了一个弹窗，不执行后面的代码，直接进入下一轮重新取画面

            # --------------------------------------------------
            # 优先级 2: 收菜 (在荒原古厅内)
            # --------------------------------------------------
            if 'btn_resource' in detection_map:
                logger.info("Action: Claim Resources")
                resource_btns = [item for item in detection_list if item['label'] == 'btn_resource']

                # 全部收菜按钮点击两轮
                for btn in resource_btns:
                    self.device.click_result(btn, sleep_time=0.5)  # 快速连点
                self.device.sleep(2)
                for btn in resource_btns:
                    self.device.click_result(btn, sleep_time=0.5)  # 快速连点
                self.device.sleep(2)

                # 简单处理：点过一次就认为完成了，触发返回逻辑
                self.resources_claimed = True
                # 回荒原主界面
                self.device.yolo_click("btn_back")
                continue
                # 也可以再次扫描确认是否还有资源按钮，如果没有了，标记完成

            # --------------------------------------------------
            # 优先级 3: 摸头 (在荒原主界面)
            # --------------------------------------------------
            if 'bubble_bond' in detection_map and self.bond_clicked_count < 5:
                logger.info("Action: Click Bond Bubble")
                self.device.click_result(detection_map['bubble_bond'])
                self.bond_clicked_count += 1
                continue

            # --------------------------------------------------
            # 优先级 4: 进入好梦井 (在荒原主界面)
            # --------------------------------------------------
            # 场景：在荒原主页看到了井，且还没收过
            if 'icon_well' in detection_map and not self.well_processed:
                logger.info("Action: Enter Wishing Well")
                self.device.click_result(detection_map['icon_well'], sleep_time=2.5)
                # 点完之后下一轮循环应该就会触发“优先级2”
                continue

            # --------------------------------------------------
            # 优先级 5: 导航逻辑 (进大厅 > 进荒原)
            # --------------------------------------------------
            # A. 如果在荒原地图，看到了大厅入口 (icon_home = 荒原古厅的icon)
            if 'icon_home' in detection_map:
                if not self.resources_claimed:
                    logger.info("Action: Enter Paleohall (icon_home)")
                    self.device.click_result(detection_map['icon_home'], sleep_time=2)
                    continue

            # B. 如果在主页，看到了荒原入口
            if 'icon_wilderness' in detection_map:
                logger.info("Action: Enter Wilderness Map")
                self.device.click_result(detection_map['icon_wilderness'], sleep_time=4)
                continue

            # ======================================================
            # 好梦井 (Wishing Well) 核心逻辑
            # ======================================================

            # 场景：检测到有好梦井的“打捞”标签，说明打开了好梦井界面
            if 'well_produce' in detection_map and not self.well_processed:

                # 1. OCR 识别剩余数量
                text = self.device.ocr_yolo_box(detection_map['well_remain'])
                # text 可能是 "4/20", "12/24", "0/20" 等

                # 2. 提取斜杠前面的数字
                # 正则匹配：找到第一个数字
                match = re.search(r'(\d+)', text)

                if match:
                    remain_count = int(match.group(1))  # 比如 4
                    logger.info(f"Wishing Well Remain: {remain_count}")

                    if remain_count > 0:
                        # 3. 点击“打捞”按钮 (well_produce) 唤出次数选择
                        if 'well_produce' in detection_map:
                            self.device.click_result(detection_map['well_produce'], sleep_time=1)

                            # 根据数量选择次数
                            # 注意：点击 well_produce 后，次数按钮才会浮现，
                            # 这里做一个简单的内层查找，也可以等下一轮循环检测 well_timeX
                            # 为了连贯性，这里用 yolo_click 快速点掉

                            target_btn = None
                            if remain_count >= 4:
                                target_btn = "well_time4"
                            elif remain_count == 3:
                                target_btn = "well_time3"
                            elif remain_count == 2:
                                target_btn = "well_time2"
                            else:
                                target_btn = "well_time1"

                            logger.info(f"Selecting harvest amount: {target_btn}")

                            # 尝试点击对应的次数按钮
                            # 因为刚才点了 well_produce，界面变了，最好重新截个图或直接盲点 YOLO
                            # 建议：这里直接调用 yolo_click 实时查找
                            if self.device.yolo_click(target_btn, sleep_time=2):
                                logger.info("Harvest confirmed.")
                            else:
                                logger.warning(f"Could not find button {target_btn}")
                        else:
                            logger.warning("Found remain text but no produce button?")
                    else:
                        logger.info("No bait remaining.")
                else:
                    logger.warning(f"OCR failed to parse number from: {text}")

                # 标记为已处理，避免死循环 (或者可以依靠 remain_count == 0 来跳出)
                # 建议处理一次后就标记 True，防止OCR读错导致死循环
                self.well_processed = True

                # 关闭好梦井界面 (点返回或关闭)
                self.device.yolo_click("btn_back")
                continue

            # --------------------------------------------------
            # 什么都没识别到，进Idle
            # --------------------------------------------------
            logger.info("Idle... Waiting for UI update.")
            self.device.sleep(1)