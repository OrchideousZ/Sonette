'''
定义了 StarRailCopilot 类，它是整个自动化脚本的主程序入口（Main Entry）。
    作用：
        继承自一个基础自动化脚本类（AzurLaneAutoScript），复用了其底层框架。
        封装了一系列针对《崩坏：星穹铁道》特定功能的方法（如 dungeon、rogue、daily_quest 等）。
        每个方法都负责导入并实例化 tasks/ 目录下相应的任务类，然后执行该任务类的 run() 方法。
        通过 if __name__ == ’__main__’: 结构，实现程序的命令行或 GUI 启动和任务循环（src.loop()）。

    调用的模块及位置：
        module.alas.AzurLaneAutoScript： 基础框架类，位于 module/alas.py（虽然它在 StarRailCopilot 中被重命名或继承）。
        module.logger： 日志记录模块，用于在运行时输出信息。
        tasks.login.login.Login： 负责应用的启动、停止和重启。
        tasks.base.ui.UI： 负责导航到游戏主界面。
        tasks/dungeon/dungeon 等其他所有任务模块：负责实现具体的游戏自动化逻辑，位于 tasks/ 目录下的相应子目录中。
'''

# 从 module/alas 模块导入基础自动化脚本类 AzurLaneAutoScript
# StarRailCopilot 继承此基础类，复用其设备连接、日志、配置等底层功能
from module.alas import AzurLaneAutoScript
# 导入日志模块
from module.logger import logger


# 定义 StarRailCopilot 类，作为整个应用程序的核心调度器
class StarRailCopilot(AzurLaneAutoScript):
    # 重启游戏应用的方法
    def restart(self):
        # 延迟导入，从 tasks/login/login.py 导入 Login 类
        from tasks.login.login import Login
        # 实例化 Login 任务并执行应用重启方法
        Login(self.config, device=self.device).app_restart()

    # 启动游戏应用的方法
    def start(self):
        # 延迟导入 Login 类
        from tasks.login.login import Login
        # 实例化 Login 任务并执行应用启动方法
        Login(self.config, device=self.device).app_start()

    # 停止游戏应用的方法
    def stop(self):
        # 延迟导入 Login 类
        from tasks.login.login import Login
        # 实例化 Login 任务并执行应用停止方法
        Login(self.config, device=self.device).app_stop()

    # 导航到游戏主界面的方法
    def goto_main(self):
        # 延迟导入 Login 类
        from tasks.login.login import Login
        # 延迟导入 UI 基础导航类
        from tasks.base.ui import UI
        # 检查应用是否正在运行
        if self.device.app_is_running():
            logger.info('App is already running, goto main page')
            # 如果运行，直接使用 UI 任务导航到主界面
            UI(self.config, device=self.device).ui_goto_main()
        else:
            logger.info('App is not running, start app and goto main page')
            # 如果未运行，先启动应用
            Login(self.config, device=self.device).app_start()
            # 再使用 UI 任务导航到主界面
            UI(self.config, device=self.device).ui_goto_main()

    # 任务失败后的错误后处理方法
    def error_postprocess(self):
        # 如果配置为云游戏，在错误发生后退出应用，以节省费用
        if self.config.is_cloud_game:
            # 延迟导入 Login 类
            from tasks.login.login import Login
            # 停止应用（退出云游戏）
            Login(self.config, device=self.device).app_stop()

    # 执行普通副本（Dungeon）任务
    def dungeon(self):
        # 延迟导入 Dungeon 任务类
        from tasks.dungeon.dungeon import Dungeon
        # 实例化并运行 Dungeon 任务
        Dungeon(config=self.config, device=self.device).run()

    # 执行每周副本（Weekly Dungeon/历战余响）任务
    def weekly(self):
        # 延迟导入 WeeklyDungeon 任务类
        from tasks.dungeon.weekly import WeeklyDungeon
        # 实例化并运行 WeeklyDungeon 任务
        WeeklyDungeon(config=self.config, device=self.device).run()

    # 执行每日任务（Daily Quest）
    def daily_quest(self):
        # 延迟导入 DailyQuestUI 任务类
        from tasks.daily.daily_quest import DailyQuestUI
        # 实例化并运行 DailyQuestUI 任务
        DailyQuestUI(config=self.config, device=self.device).run()

    # 执行无名勋礼/大月卡（Battle Pass）奖励领取
    def battle_pass(self):
        # 延迟导入 BattlePassUI 任务类
        from tasks.battle_pass.battle_pass import BattlePassUI
        # 实例化并运行 BattlePassUI 任务
        BattlePassUI(config=self.config, device=self.device).run()

    # 执行派遣任务（Assignment）
    def assignment(self):
        # 延迟导入 Assignment 任务类
        from tasks.assignment.assignment import Assignment
        # 实例化并运行 Assignment 任务
        Assignment(config=self.config, device=self.device).run()

    # 执行数据更新/同步任务
    def data_update(self):
        # 延迟导入 DataUpdate 任务类
        from tasks.item.data_update import DataUpdate
        # 实例化并运行 DataUpdate 任务
        DataUpdate(config=self.config, device=self.device).run()

    # 执行免费资源领取（Freebies，如邮件/签到）任务
    def freebies(self):
        # 延迟导入 Freebies 任务类
        from tasks.freebies.freebies import Freebies
        # 实例化并运行 Freebies 任务
        Freebies(config=self.config, device=self.device).run()

    # 执行模拟宇宙（Rogue-like/Simulated Universe）任务
    def rogue(self):
        # 延迟导入 Rogue 任务类
        from tasks.rogue.rogue import Rogue
        # 实例化并运行 Rogue 任务
        Rogue(config=self.config, device=self.device).run()

    # 执行位面饰品（Ornament）副本任务
    def ornament(self):
        # 延迟导入 Ornament 任务类
        from tasks.ornament.ornament import Ornament
        # 实例化并运行 Ornament 任务
        Ornament(config=self.config, device=self.device).run()

    # 执行性能基准测试任务
    def benchmark(self):
        # 延迟导入 run_benchmark 函数
        from module.daemon.benchmark import run_benchmark
        # 执行基准测试
        run_benchmark(config=self.config)

    # 执行后台托管服务任务
    def daemon(self):
        # 延迟导入 Daemon 任务类
        from tasks.base.daemon import Daemon
        # 实例化并运行 Daemon 任务
        Daemon(config=self.config, device=self.device, task="Daemon").run()

    # 执行规划器扫描任务（Planner Scan，用于角色养成规划）
    def planner_scan(self):
        # 延迟导入 PlannerScan 任务类
        from tasks.planner.scan import PlannerScan
        # 实例化并运行 PlannerScan 任务
        PlannerScan(config=self.config, device=self.device, task="PlannerScan").run()


# 当脚本作为主程序运行时
if __name__ == '__main__':
    # 实例化 StarRailCopilot，传入 'src' 作为名称
    src = StarRailCopilot('src')
    # 启动主循环，通常负责处理命令行参数和任务调度
    src.loop()