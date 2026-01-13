# test_login.py
from Sonette import StarRailCopilot
from module.logger import logger
# 【新增】导入 Login 任务类
from tasks.login.login import Login


def test_connection_and_login():
    logger.info("========== 开始测试连接与登录 ==========")

    # 1. 初始化主程序
    src = StarRailCopilot('src')

    # 2. 强制初始化设备 (截图测试)
    logger.info("正在尝试连接 ADB...")
    src.device.screenshot()
    logger.info("ADB 连接成功，截图功能正常！")

    # 3. 实例化 Login 任务
    # 注意：我们要拿到这个 task 对象，才能调用它的 run 方法
    task = Login(src.config, device=src.device)

    # 4. 执行重启 (先杀后台再进游戏，确保环境纯净)
    logger.info("正在重启游戏...")
    task.app_restart()

    # 5. 【关键】执行业务逻辑 (Run!)
    # 这才会触发你写的 while 循环和 YOLO 识别
    logger.info("正在执行登录与荒原逻辑...")
    task.run()

    logger.info("========== 测试结束 ==========")


if __name__ == '__main__':
    test_connection_and_login()