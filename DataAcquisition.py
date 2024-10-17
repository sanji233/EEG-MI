import threading
import time
import pandas as pd
import numpy as np
import heapq
import mne
import random  # 导入 random 模块
from brainflow.board_shim import BoardShim, BrainFlowInputParams, BoardIds
from brainflow.data_filter import DataFilter
import sys
import os

# 导入 PyQt5 模块
from PyQt5.QtWidgets import QMainWindow, QApplication
from PyQt5.QtCore import QTimer, Qt, QPoint
from PyQt5.QtGui import QPainter, QPen, QBrush

class GlobalVariableController:
    """全局变量控制器，用于管理全局变量的值和变化日志。"""
    def __init__(self):
        self._global_variable = 0
        self._change_log = []
        self._lock = threading.Lock()
        self._schedule_lock = threading.Lock()
        self._schedule = []  # 使用最小堆实现的优先队列
        self._stop_event = threading.Event()
        self._thread = None

    def set_variable(self, value):
        """设置全局变量的值，并记录变化。"""
        with self._lock:
            previous_value = self._global_variable
            change_time = time.time()  # 使用 time.time() 获取时间戳（秒）

            if previous_value != value:
                self._change_log.append({
                    'Value': previous_value,
                    'Timestamp': change_time,
                    'Event': 'end'
                })

            self._global_variable = value

            self._change_log.append({
                'Value': value,
                'Timestamp': change_time,
                'Event': 'start'
            })

            # 添加调试信息
            print(f"[{change_time}] Global variable changed from {previous_value} to {value}")

    def get_variable(self):
        """获取当前全局变量的值。"""
        with self._lock:
            return self._global_variable

    def get_change_log(self):
        """返回全局变量变化的日志，以 pandas DataFrame 格式。"""
        with self._lock:
            df = pd.DataFrame(self._change_log)
        return df

    def save_change_log(self, filename):
        """将全局变量变化日志保存到 CSV 文件。"""
        df = self.get_change_log()
        df.to_csv(filename, index=False)
        print(f"Global variable change log saved to {filename}")

    def reset(self):
        """重置全局变量并清空日志。"""
        with self._lock:
            self._global_variable = 0
            self._change_log.clear()

    def add_schedule(self, value, delay):
        """添加一个计划，在指定的延迟（秒）后更改全局变量的值。"""
        execute_time = time.time() + delay  # 使用 time.time() 作为时间基准
        with self._schedule_lock:
            heapq.heappush(self._schedule, (execute_time, value))
            print(f"Added schedule: Set variable to {value} after {delay} seconds.")

    def add_schedules(self, events):
        """添加多个计划事件。

        参数：
        - events: 列表，包含多个元组 (value, delay)，表示计划的值和延迟时间（秒）。
        """
        current_time = time.time()
        with self._schedule_lock:
            for value, delay in events:
                execute_time = current_time + delay
                heapq.heappush(self._schedule, (execute_time, value))
                print(f"Added schedule: Set variable to {value} after {delay} seconds.")




    def add_random_schedules(self, num_events, interval, possible_values):
        """添加多个随机参数的计划事件，具有固定的时间间隔。
        参数：
        - num_events: 要生成的计划事件的数量。
        - interval: 事件之间的固定时间间隔（秒）。
        - possible_values: 列表，包含可能的参数值。
        """
        current_time = time.time()
        previous_value = None
        with self._schedule_lock:
            for i in range(num_events):
                # 确保新值与前一个值不同
                if previous_value is not None:
                    available_values = [v for v in possible_values if v != previous_value]
                else:
                    available_values = possible_values.copy()
                value = random.choice(available_values)
                previous_value = value

                execute_time = current_time + i * interval
                heapq.heappush(self._schedule, (execute_time, value))
                print(f"Added random schedule: Set variable to {value} at {i * interval:.2f} seconds.")


    def start_schedule(self):
        """开始执行计划中的变化。"""
        def run_schedule():
            try:
                while not self._stop_event.is_set():
                    with self._schedule_lock:
                        if not self._schedule:
                            print("No more schedules to execute. Exiting schedule thread.")
                            break
                        execute_time, value = heapq.heappop(self._schedule)

                    now = time.time()
                    wait_time = execute_time - now
                    print(f"Scheduled to set variable to {value} in {wait_time} seconds.")

                    if wait_time > 0:
                        if self._stop_event.wait(timeout=wait_time):
                            break

                    self.set_variable(value)
            except Exception as e:
                print(f"Exception in schedule execution: {e}")

        self._stop_event.clear()
        self._thread = threading.Thread(target=run_schedule)
        self._thread.start()

    def stop_schedule(self):
        """停止计划的执行。"""
        self._stop_event.set()
        if self._thread and self._thread.is_alive():
            self._thread.join()

    def is_running(self):
        """检查计划是否正在运行。"""
        return self._thread is not None and self._thread.is_alive()

class EEGDataAcquisition:
    """
    EEG 数据采集类，用于从 EEG 设备采集数据，并将全局变量的变化作为标记记录。

    参数：
    - global_var_controller: GlobalVariableController 实例，用于监控全局变量的变化。
    - serial_port: EEG 设备的串口端口。
    - duration_minutes: 采集的持续时间（以分钟为单位）。
    """
    def __init__(self, global_var_controller, serial_port, duration_minutes):
        self.global_var_controller = global_var_controller
        self.serial_port = serial_port
        self.duration_minutes = duration_minutes
        self.params = BrainFlowInputParams()
        self.params.serial_port = serial_port
        self.board_id = BoardIds.CYTON_BOARD.value
        self.board = BoardShim(self.board_id, self.params)
        self.eeg_data = None
        self.is_streaming = False
        self.data_thread = None
        self.marker_thread = None
        self.event_markers = []  # 存储事件标记 (timestamp, value)
        self.lock = threading.Lock()

    def start_acquisition(self):
        """启动 EEG 数据采集和事件标记监控。"""
        BoardShim.enable_dev_board_logger()
        self.board.prepare_session()
        self.board.start_stream(45000, '')  # 增加缓冲区大小，防止数据丢失
        print("EEGDataAcquisition: Stream started")
        self.is_streaming = True

        # 启动数据采集线程
        self.data_thread = threading.Thread(target=self._acquire_data)
        self.data_thread.start()

        # 启动全局变量监控线程
        self.marker_thread = threading.Thread(target=self._monitor_global_variable)
        self.marker_thread.start()

    def stop_acquisition(self):
        """停止 EEG 数据采集和事件标记监控。"""
        self.is_streaming = False
        if self.data_thread and self.data_thread.is_alive():
            self.data_thread.join()
        if self.marker_thread and self.marker_thread.is_alive():
            self.marker_thread.join()
        time.sleep(1)  # 等待一段时间，确保所有数据都已接收
        self.board.stop_stream()
        print("EEGDataAcquisition: Stream stopped")
        # 获取所有剩余的数据
        data = self.board.get_board_data()
        if data.size > 0:
            with self.lock:
                if self.eeg_data is not None:
                    self.eeg_data = np.hstack((self.eeg_data, data))
                else:
                    self.eeg_data = data
        self.board.release_session()
        print("EEGDataAcquisition: Session released")

    def _acquire_data(self):
        """数据采集线程，从 EEG 设备持续获取数据。"""
        start_time = time.time()
        try:
            while self.is_streaming and (time.time() - start_time) < self.duration_minutes * 60:
                data = self.board.get_current_board_data(250)  # 获取最新的 250 个数据点
                if data.size > 0:
                    with self.lock:
                        if self.eeg_data is not None:
                            self.eeg_data = np.hstack((self.eeg_data, data))
                        else:
                            self.eeg_data = data
                time.sleep(1)  # 控制数据获取频率
        except Exception as e:
            print(f"EEGDataAcquisition: Exception in data acquisition - {e}")
        finally:
            print("EEGDataAcquisition: Data acquisition finished")

    def _monitor_global_variable(self):
        """全局变量监控线程，记录变量的变化时间和值。"""
        previous_value = self.global_var_controller.get_variable()
        try:
            while self.is_streaming:
                current_value = self.global_var_controller.get_variable()
                if current_value != previous_value:
                    # 获取当前时间戳（秒）
                    timestamp = time.time()
                    with self.lock:
                        self.event_markers.append((timestamp, current_value))
                    print(f"[{timestamp}] Event marker recorded: {current_value}")
                    previous_value = current_value
                time.sleep(0.01)  # 根据需要调整检查频率
        except Exception as e:
            print(f"EEGDataAcquisition: Exception in marker monitoring - {e}")

    def save_data(self, filename):
        """保存 EEG 数据和事件标记为 MNE 的 .fif 文件，并输出未经处理的 BrainFlow 数据文件。

        参数：
        - filename: 保存的文件名。
        """
        if self.eeg_data is None or self.eeg_data.size == 0:
            print("No data to save.")
            return

        # 保存未经处理的 BrainFlow 数据为 CSV 文件
        raw_data_file = 'brainflow_raw_data.csv'
        DataFilter.write_file(self.eeg_data, raw_data_file, 'w')
        print(f"Raw BrainFlow data saved to {raw_data_file}")

        # 使用 MNE 加载数据
        eeg_channels = BoardShim.get_eeg_channels(self.board_id)
        eeg_data = self.eeg_data[eeg_channels, :]

        # 定义通道信息
        ch_types = ['eeg'] * len(eeg_channels)
        ch_names = ['C1', 'C2', 'C3', 'C4', 'C5', 'C6', 'C7', 'C8']
        sfreq = BoardShim.get_sampling_rate(self.board_id)
        info = mne.create_info(ch_names=ch_names, sfreq=sfreq, ch_types=ch_types)
        raw = mne.io.RawArray(eeg_data, info)

        # 应用滤波器（可选）
        raw.filter(0.5, 40, fir_design='firwin', skip_by_annotation='edge')

        # 添加标记（Annotations）
        annotations = mne.Annotations(onset=[], duration=[], description=[])
        if self.event_markers:
            first_timestamp = self.event_markers[0][0]
            for timestamp, value in self.event_markers:
                onset = timestamp - first_timestamp  # 计算相对时间（秒）
                annotations.append(onset=onset, duration=0, description=str(value))
            raw.set_annotations(annotations)
        else:
            print("No event markers to add.")

        # 保存数据
        raw.save(filename, overwrite=True)
        print(f"Data saved to {filename}")

    def save_raw_brainflow_data(self, filename):
        """保存未经处理的 BrainFlow 数据为 CSV 文件。

        参数：
        - filename: 保存的文件名。
        """
        if self.eeg_data is None or self.eeg_data.size == 0:
            print("No raw data to save.")
            return

        DataFilter.write_file(self.eeg_data, filename, 'w')
        print(f"Raw BrainFlow data saved to {filename}")

class MainWindow(QMainWindow):
    """PyQt5 GUI 主窗口，显示上下左右四个箭头和一个中心圆点，根据全局变量改变颜色。"""
    def __init__(self, global_var_controller):
        super().__init__()
        self.global_var_controller = global_var_controller
        self.initUI()

    def initUI(self):
        """初始化界面。"""
        self.setGeometry(300, 300, 600, 600)
        self.setWindowTitle('EEG Collection GUI')
        self.show()
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.update)
        self.timer.start(50)  # 每 50 毫秒刷新一次界面
        print("GUI timer started, refreshing every 50 ms.")

    def closeEvent(self, event):
        """在 GUI 关闭时，执行必要的清理操作。"""
        print("GUI window is closing.")
        event.accept()  # 允许窗口关闭

    def paintEvent(self, event):
        """绘制界面元素。"""
        qp = QPainter(self)
        qp.setPen(QPen(Qt.black, 2, Qt.SolidLine))

        center_x, center_y = int(self.width() // 2), int(self.height() // 2)
        dot_size = 40

        default_color = Qt.gray
        active_color = Qt.green

        # 获取当前全局变量的值
        current_value = self.global_var_controller.get_variable()
        # print(f"Current global variable value in GUI: {current_value}")

        # 根据全局变量改变中心圆点的颜色
        dot_color = active_color if current_value == 5 else default_color
        qp.setBrush(QBrush(dot_color, Qt.SolidPattern))
        qp.drawEllipse(int(center_x - dot_size / 2), int(center_y - dot_size / 2), dot_size, dot_size)

        # 根据全局变量改变箭头的颜色
        arrow_colors = [default_color] * 4
        if current_value in {1, 2, 3, 4}:
            arrow_colors[current_value - 1] = active_color

        self.draw_arrow(qp, center_x, 100, arrow_colors[0], 'up')  # 上箭头
        self.draw_arrow(qp, center_x, int(self.height()) - 100, arrow_colors[1], 'down')  # 下箭头
        self.draw_arrow(qp, 100, center_y, arrow_colors[2], 'left')  # 左箭头
        self.draw_arrow(qp, int(self.width()) - 100, center_y, arrow_colors[3], 'right')  # 右箭头

    def draw_arrow(self, qp, x, y, color, direction):
        """绘制箭头。"""
        arrow_size = 40
        qp.setBrush(QBrush(color, Qt.SolidPattern))
        if direction == 'up':
            qp.drawPolygon(
                QPoint(int(x), int(y - arrow_size)),
                QPoint(int(x - arrow_size // 2), int(y)),
                QPoint(int(x + arrow_size // 2), int(y))
            )
        elif direction == 'down':
            qp.drawPolygon(
                QPoint(int(x), int(y + arrow_size)),
                QPoint(int(x - arrow_size // 2), int(y)),
                QPoint(int(x + arrow_size // 2), int(y))
            )
        elif direction == 'left':
            qp.drawPolygon(
                QPoint(int(x - arrow_size), int(y)),
                QPoint(int(x), int(y - arrow_size // 2)),
                QPoint(int(x), int(y + arrow_size // 2))
            )
        elif direction == 'right':
            qp.drawPolygon(
                QPoint(int(x + arrow_size), int(y)),
                QPoint(int(x), int(y - arrow_size // 2)),
                QPoint(int(x), int(y + arrow_size // 2))
            )
def run_gui(global_var_controller):
    """运行 PyQt5 GUI。"""
    print("Initializing QApplication...")
    app = QApplication(sys.argv)
    ex = MainWindow(global_var_controller)
    print("Entering main event loop...")
    app.exec_()
    print("GUI closed.")

if __name__ == "__main__":
    # 初始化全局变量控制器
    global_var_controller = GlobalVariableController()

    # 定义参数
    num_events = 80  # 生成 10 个计划事件
    interval = 3     # 事件之间的固定时间间隔为 3 秒
    possible_values = [1, 2, 3, 4, 5]  # 可能的参数值

    # 添加随机计划事件
    global_var_controller.add_random_schedules(num_events, interval, possible_values)

    # 启动计划
    global_var_controller.start_schedule()

    # 初始化 EEG 数据采集（如果需要）
    serial_port = 'COM5'  # 请根据实际情况调整
    duration_minutes = (num_events * interval) / 60.0 + 0.3  # 根据计划事件的总时长，增加 0.5 分钟的缓冲时间
    eeg_acquisition = EEGDataAcquisition(global_var_controller, serial_port, duration_minutes)
    eeg_acquisition.start_acquisition()

    # 启动 GUI，在主线程中运行
    print("Starting GUI...")
    run_gui(global_var_controller)
    print("GUI has been closed.")

    # 停止计划
    global_var_controller.stop_schedule()

    # 停止数据采集并保存数据
    eeg_acquisition.stop_acquisition()
    eeg_acquisition.save_data('raw.fif')  # 保存 MNE 格式的数据
    eeg_acquisition.save_raw_brainflow_data('brainflow_raw_data.csv')  # 保存未经处理的 BrainFlow 数据

    # 获取并保存全局变量的变化日志
    change_log = global_var_controller.get_change_log()
    print("Global Variable Change Log:")
    print(change_log)
    global_var_controller.save_change_log('global_variable_change_log.csv')
    print("Program finished.")
