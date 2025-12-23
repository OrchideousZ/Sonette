'''
定义了 AzurLaneAutoScript 类，它是整个自动化项目的基础框架类（Base Framework）。src.py 中的 StarRailCopilot 继承自这个类。
    作用： 提供了自动化脚本运行所需的核心基础设施和任务调度机制。它不包含任何具体的游戏操作逻辑，而是负责：
        初始化配置（config）、设备连接（device）和服务器状态检查器（checker）。
        实现主循环（loop）和任务调度逻辑（get_next_task）。
        提供统一的异常处理和错误日志记录（run 方法中的 try...except 块）。
        定义了所有具体任务（如 restart, dungeon）的抽象接口（在 StarRailCopilot 中被实现）。

    调用的模块及位置：
        module.base.decorator.del_cached_property： 用于清除缓存属性。
        module.config.config.AzurLaneConfig / TaskEnd： 配置加载和任务结束异常。
        module.config.deep： 用于处理深层配置的读取和设置（deep_get, deep_set）。
        module.exception： 各种自定义异常类（如 GameStuckError, ScriptError 等）。
        module.logger： 日志记录和错误日志保存。
        module.notify： 推送通知处理（如 OnePush）。
        module.device.device.Device： 设备控制模块，负责与模拟器/手机交互。
        module.server_checker.ServerChecker： 服务器状态检查模块。
        module.base.resource.release_resources： 资源释放函数。
'''

# 导入线程模块，用于处理事件
import threading
# 导入时间模块
import time
# 导入日期时间模块，用于调度
from datetime import datetime, timedelta

# 导入 inflection 库，用于字符串格式化，如驼峰转下划线（CamelCase to snake_case）
import inflection
# 导入 cached_property，用于惰性计算属性并缓存结果
from cached_property import cached_property

# 导入用于清除缓存属性的函数
from module.base.decorator import del_cached_property
# 导入 AzurLaneConfig（基础配置类）和 TaskEnd（任务结束异常）
from module.config.config import AzurLaneConfig, TaskEnd
# 导入用于深层读取和设置配置的函数
from module.config.deep import deep_get, deep_set
# 导入自定义异常类
from module.exception import *
# 导入日志记录器和错误日志保存函数
from module.logger import logger, save_error_log
# 导入通知处理函数（如 OnePush）
from module.notify import handle_notify


# 定义 AzurLaneAutoScript 类，作为所有自动化脚本的基类
class AzurLaneAutoScript:
    # 类变量：用于停止任务的线程事件
    stop_event: threading.Event = None

    # 构造函数
    def __init__(self, config_name='alas'):
        # 记录日志标题：开始
        logger.hr('Start', level=0)
        # 配置名称，用于加载特定的配置文件
        self.config_name = config_name
        # 跳过首次重启标志，防止程序启动时立即执行一次重启
        self.is_first_task = True
        # 任务失败记录字典
        # 键: 任务名称 (str), 值: 失败次数 (int)
        self.failure_record = {}

    # 使用 @cached_property 装饰器，确保 config 只在首次访问时创建，并被缓存
    @cached_property
    def config(self):
        try:
            # 实例化配置类，加载配置
            config = AzurLaneConfig(config_name=self.config_name)
            return config
        except RequestHumanTakeover:
            # 如果请求人工接管（配置错误等严重问题），记录并退出
            logger.critical('Request human takeover')
            exit(1)
        except Exception as e:
            # 捕获其他异常并退出
            logger.exception(e)
            exit(1)

    # 实例化设备控制模块
    @cached_property
    def device(self):
        try:
            # 延迟导入 Device 类（避免循环导入和不必要的初始化）
            from module.device.device import Device
            # 实例化设备对象，用于截图、点击等操作
            device = Device(config=self.config)
            return device
        except RequestHumanTakeover:
            logger.critical('Request human takeover')
            exit(1)
        except Exception as e:
            logger.exception(e)
            exit(1)

    # 实例化服务器状态检查器
    @cached_property
    def checker(self):
        try:
            # 延迟导入 ServerChecker 类
            from module.server_checker import ServerChecker
            # 实例化检查器，使用配置中的包名检查服务器/网络状态
            checker = ServerChecker(server=self.config.Emulator_PackageName)
            return checker
        except Exception as e:
            logger.exception(e)
            exit(1)

    # 抽象方法：重启应用，要求子类（StarRailCopilot）实现
    def restart(self):
        raise NotImplemented

    # 抽象方法：启动应用，要求子类实现
    def start(self):
        raise NotImplemented

    # 抽象方法：停止应用，要求子类实现
    def stop(self):
        raise NotImplemented

    # 抽象方法：导航到游戏主界面，要求子类实现
    def goto_main(self):
        raise NotImplemented

    # 核心执行方法：运行指定的任务命令
    def run(self, command):
        try:
            # 运行任务前先截图
            self.device.screenshot()
            # 清除截图跟踪记录
            self.device.screenshot_tracking.clear()
            # 通过字符串名称调用对应的方法（例如 self.dungeon()）
            self.__getattribute__(command)()
            # 任务成功完成
            return True
        except TaskEnd:
            # 任务主动抛出 TaskEnd 异常来结束自身，视为成功
            return True
        except GameNotRunningError as e:
            # 游戏未运行错误
            logger.warning(e)
            # 触发重启任务
            self.config.task_call('Restart')
            return False
        except (GameStuckError, GameTooManyClickError) as e:
            # 游戏卡住或点击过多错误（可能是识别失败导致重复点击）
            logger.error(e)
            self.save_error_log()
            logger.warning(f'Game stuck, {self.device.package} will be restarted in 10 seconds')
            self.config.task_call('Restart')
            self.device.sleep(10)
            return False
        except GameBugError as e:
            # 游戏客户端发生 Bug，脚本无法处理
            logger.warning(e)
            self.save_error_log()
            logger.warning('An error has occurred in Star Rail game client, Src is unable to handle')
            logger.warning(f'Restarting {self.device.package} to fix it')
            self.config.task_call('Restart')
            self.device.sleep(10)
            return False
        except GamePageUnknownError:
            # 游戏页面无法识别的严重错误
            # 检查服务器状态，确认是否为维护或网络问题
            self.checker.check_now()
            if self.checker.is_available():
                # 如果服务器可用，则确认为脚本或环境问题
                logger.critical('Game page unknown')
                self.save_error_log()
                # 发送错误通知
                handle_notify(
                    self.config.Error_OnePushConfig,
                    title=f"Src <{self.config_name}> crashed",
                    content=f"<{self.config_name}> GamePageUnknownError",
                )
                exit(1)
            else:
                # 如果服务器不可用（维护），则等待直到恢复
                self.checker.wait_until_available()
                return False
        except HandledError as e:
            # 已知且已处理的错误，不影响程序继续运行
            logger.error(e)
            return False
        except ScriptError as e:
            # 脚本内部错误（开发者错误或随机问题）
            logger.exception(e)
            self.error_postprocess()
            logger.critical('This is likely to be a mistake of developers, but sometimes just random issues')
            self.save_error_log()
            # 发送错误通知并退出
            handle_notify(
                self.config.Error_OnePushConfig,
                title=f"Src <{self.config_name}> crashed",
                content=f"<{self.config_name}> ScriptError",
            )
            exit(1)
        except RequestHumanTakeover:
            # 请求人工接管（配置或环境问题）
            logger.critical('Request human takeover')
            self.error_postprocess()
            # 发送错误通知并退出
            handle_notify(
                self.config.Error_OnePushConfig,
                title=f"Src <{self.config_name}> crashed",
                content=f"<{self.config_name}> RequestHumanTakeover",
            )
            exit(1)
        except Exception as e:
            # 捕获所有未被处理的通用异常
            logger.exception(e)
            self.error_postprocess()
            self.save_error_log()
            # 发送错误通知并退出
            handle_notify(
                self.config.Error_OnePushConfig,
                title=f"Src <{self.config_name}> crashed",
                content=f"<{self.config_name}> Exception occured",
            )
            exit(1)

    # 保存错误日志和截图
    def save_error_log(self):
        """
        Save last 60 screenshots in ./log/error/<timestamp>
        Save logs to ./log/error/<timestamp>/log.txt
        """
        save_error_log(config=self.config, device=self.device)

    # 错误发生后的后处理（子类可重写，例如 StarRailCopilot 用于退出云游戏）
    def error_postprocess(self):
        """
        Do something when error occurred
        """
        pass

    # 等待直到指定的时间点
    def wait_until(self, future):
        """
        Wait until a specific time.

        Args:
            future (datetime): 目标时间

        Returns:
            bool: True 如果等待完成，False 如果配置在等待期间被重新加载
        """
        # 增加一秒缓冲
        future = future + timedelta(seconds=1)
        # 开始监听配置文件的变化
        self.config.start_watching()
        while 1:
            # 如果当前时间超过了目标时间，则退出等待
            if datetime.now() > future:
                return True

            # 检查是否有外部停止事件（例如 GUI/Web UI 的更新指令）
            if self.stop_event is not None:
                if self.stop_event.is_set():
                    logger.info("Update event detected")
                    logger.info(f"[{self.config_name}] exited. Reason: Update")
                    exit(0)

            # 每次循环休眠 5 秒
            time.sleep(5)

            # 检查配置是否在等待期间被外部更改并需要重新加载
            if self.config.should_reload():
                return False

    # 获取下一个要执行的任务
    def get_next_task(self):
        """
        Returns:
            str: Name of the next task.
        """
        while 1:
            # 从任务队列中获取下一个任务
            task = self.config.get_next()
            # 设置当前任务，并绑定配置
            self.config.task = task
            self.config.bind(task)

            # 延迟导入资源释放函数
            from module.base.resource import release_resources
            # 如果下一个任务不是 'Alas'（基础任务），则释放不再需要的资源
            if self.config.task.command != 'Alas':
                release_resources(next_task=task.command)

            # 如果下一个任务的运行时间在未来，则进入等待
            if task.next_run > datetime.now():
                logger.info(f'Wait until {task.next_run} for task `{task.command}`')
                # 标记不是首次任务
                self.is_first_task = False
                # 获取任务队列为空时的优化处理方法
                method = self.config.Optimization_WhenTaskQueueEmpty

                # --- 根据优化方法执行不同的等待策略 ---
                if method == 'close_game':
                    logger.info('Close game during wait')
                    self.run('stop')  # 停止游戏
                    release_resources()
                    self.device.release_during_wait()
                    # 执行等待，如果返回 False 说明配置更改，需跳出循环重新获取任务
                    if not self.wait_until(task.next_run):
                        del_cached_property(self, 'config')
                        continue
                    # 等待结束后，如果任务不是 Restart，则手动插入一个 Restart 任务
                    if task.command != 'Restart':
                        self.config.task_call('Restart')
                        del_cached_property(self, 'config')
                        continue
                elif method == 'goto_main':
                    logger.info('Goto main page during wait')
                    self.run('goto_main')  # 导航到主界面
                    release_resources()
                    self.device.release_during_wait()
                    if not self.wait_until(task.next_run):
                        del_cached_property(self, 'config')
                        continue
                elif method == 'stay_there':
                    logger.info('Stay there during wait')  # 保持当前状态
                    release_resources()
                    self.device.release_during_wait()
                    if not self.wait_until(task.next_run):
                        del_cached_property(self, 'config')
                        continue
                elif method == 'close_emulator':
                    logger.info('Close emulator during wait')
                    self.run('stop')  # 停止游戏
                    release_resources()
                    self.device.release_during_wait()
                    # 关闭模拟器
                    try:
                        self.device.emulator_stop()
                        logger.info('Emulator stopped successfully')
                    except Exception as e:
                        logger.warning(f'Failed to stop emulator: {e}')
                    if not self.wait_until(task.next_run):
                        # 如果配置更改，需要清除 config 和 device 缓存
                        del_cached_property(self, 'config')
                        del_cached_property(self, 'device')
                        continue
                    if task.command == 'Restart':
                        del_cached_property(self, 'config')
                        del_cached_property(self, 'device')
                        continue
                    # 重新启动模拟器并插入重启任务
                    if task.command != 'Restart':
                        self.config.task_call('Restart')
                        del_cached_property(self, 'config')
                        del_cached_property(self, 'device')
                        continue
                else:
                    # 策略无效时，退回到 stay_there
                    logger.warning(f'Invalid Optimization_WhenTaskQueueEmpty: {method}, fallback to stay_there')
                    release_resources()
                    self.device.release_during_wait()
                    if not self.wait_until(task.next_run):
                        del_cached_property(self, 'config')
                        continue
            # 成功获取到需要立即执行的任务，跳出循环
            break

        # 任务开始前，重置是否为囤积任务的标志
        AzurLaneConfig.is_hoarding_task = False
        return task.command

    # 主循环方法：脚本的核心运行逻辑
    def loop(self):
        # 设置文件日志记录器
        logger.set_file_logger(self.config_name)
        logger.info(f'Start scheduler loop: {self.config_name}')

        while 1:
            # 检查是否有外部停止/更新事件
            if self.stop_event is not None:
                if self.stop_event.is_set():
                    logger.info("Update event detected")
                    logger.info(f"[{self.config_name}] exited.")
                    break

            # 检查游戏服务器是否可用，如果不可用则等待
            self.checker.wait_until_available()
            if self.checker.is_recovered():
                # 如果服务器恢复，清除缓存配置并插入重启任务
                del_cached_property(self, 'config')
                logger.info('Server or network is recovered. Restart game client')
                self.config.task_call('Restart')

            # 获取下一个任务名称
            task = self.get_next_task()

            # 确保设备和配置是最新的
            _ = self.device
            self.device.config = self.config

            # 如果是首次任务，且任务为 Restart，则跳过
            if self.is_first_task and task == 'Restart':
                logger.info('Skip task `Restart` at scheduler start')
                # 延迟任务执行时间
                self.config.task_delay(server_update=True)
                del_cached_property(self, 'config')
                continue

            # --- 运行任务 ---
            logger.info(f'Scheduler: Start task `{task}`')
            # 清除设备中的卡住记录和点击记录
            self.device.stuck_record_clear()
            self.device.click_record_clear()
            # 记录任务开始的标题
            logger.hr(task, level=0)
            # 将任务名称从 CamelCase 转换为 snake_case（例如: Dungeon -> dungeon），然后调用 self.dungeon()
            success = self.run(inflection.underscore(task))
            logger.info(f'Scheduler: End task `{task}`')
            # 标记首次任务已完成
            self.is_first_task = False

            # --- 失败次数检查 ---
            failed = deep_get(self.failure_record, keys=task, default=0)
            failed = 0 if success else failed + 1
            deep_set(self.failure_record, keys=task, value=failed)

            # 如果任务连续失败达到 3 次
            if failed >= 3:
                logger.critical(f"Task `{task}` failed 3 or more times.")
                logger.critical("Possible reason #1: You haven't used it correctly. "
                                "Please read the help text of the options.")
                logger.critical("Possible reason #2: There is a problem with this task. "
                                "Please contact developers or try to fix it yourself.")
                logger.critical('Request human takeover')
                # 发送通知并退出
                handle_notify(
                    self.config.Error_OnePushConfig,
                    title=f"Src <{self.config_name}> crashed",
                    content=f"<{self.config_name}> RequestHumanTakeover\nTask `{task}` failed 3 or more times.",
                )
                exit(1)

            # --- 任务后处理 ---
            if success:
                # 如果成功，清除配置缓存，以便下次加载最新配置
                del_cached_property(self, 'config')
                continue
            else:
                # 如果失败，清除配置缓存，并立即检查服务器状态
                # self.config.task_delay(success=False) # 任务延迟已注释
                del_cached_property(self, 'config')
                self.checker.check_now()
                continue


# 如果 alas.py 作为主程序运行（通常在测试时）
if __name__ == '__main__':
    # 实例化基类
    alas = AzurLaneAutoScript()
    # 启动主循环
    alas.loop()