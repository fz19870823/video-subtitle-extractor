# -*- coding: utf-8 -*-
"""
@Author  : Fang Yao
@Time    : 2021/3/24 9:28 上午
@FileName: main.py
@desc: 主程序入口文件
"""
import os
import re
import random
import shutil
import traceback
from collections import Counter, namedtuple
import unicodedata
from threading import Thread
from pathlib import Path
import cv2
from Levenshtein import ratio
from PIL import Image
from numpy import average, dot, linalg
import numpy as np
from tqdm import tqdm
import sys

sys.path.insert(0, os.path.dirname(__file__))
import subprocess
from backend.config import *
from backend.tools.hardware_accelerator import HardwareAccelerator
from backend.tools import reformat
from backend.tools.ocr import OcrRecogniser, get_coordinates
from backend.tools import subtitle_ocr
from backend.tools.paddle_model_config import PaddleModelConfig
from backend.tools.process_manager import ProcessManager
from backend.tools.language_filter import is_target_language
from backend.tools.subtitle_detect import SubtitleDetect
from backend.bean.subtitle_area import SubtitleArea
import threading
import platform
import multiprocessing
import time
import pysrt

def _max_batch_size():
    """OCR 帧批量大小：VSE_MAX_BATCH 环境变量优先（做 A/B 对比用），否则读配置。"""
    env = os.environ.get('VSE_MAX_BATCH')
    if env:
        try:
            return max(1, int(env))
        except ValueError:
            pass
    try:
        return max(1, int(config.maxBatchSize.value))
    except Exception:
        return 1


class SubtitleExtractor:
    """
    视频字幕提取类
    """

    def __init__(self, vd_path):
        # 线程锁
        self.lock = threading.RLock()
        # 用户指定的字幕区域位置
        self.sub_area = None
        self.hardware_accelerator = HardwareAccelerator.instance()
        # 是否使用硬件加速
        self.hardware_accelerator.set_enabled(config.hardwareAcceleration.value)
        self.model_config = PaddleModelConfig(self.hardware_accelerator)
        # 创建字幕检测对象
        self.sub_detector = SubtitleDetect()
        # 视频路径
        self.video_path = vd_path
        self.video_cap = cv2.VideoCapture(vd_path)
        # 通过视频路径获取视频名称
        self.vd_name = Path(self.video_path).stem
        # 临时存储文件夹
        self.temp_output_dir = os.path.join(os.path.dirname(BASE_DIR), 'output', str(self.vd_name))
        # 视频帧总数
        self.frame_count = self.video_cap.get(cv2.CAP_PROP_FRAME_COUNT)
        # 视频帧率
        self.fps = self.video_cap.get(cv2.CAP_PROP_FPS)
        # 视频尺寸
        self.frame_height = int(self.video_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.frame_width = int(self.video_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        # 提取的视频帧储存目录
        self.frame_output_dir = os.path.join(self.temp_output_dir, 'frames')
        # 提取的字幕文件存储目录
        self.subtitle_output_dir = os.path.join(self.temp_output_dir, 'subtitle')
        # 定义是否使用vsf提取字幕帧
        self.use_vsf = False
        # 定义vsf的字幕输出路径
        self.vsf_subtitle = os.path.join(self.subtitle_output_dir, 'raw_vsf.srt')
        # 提取的原始字幕文本存储路径
        self.raw_subtitle_path = os.path.join(self.subtitle_output_dir, 'raw.txt')
        # 定义输出字幕文件路径
        self.subtitle_output_path = os.path.splitext(self.video_path)[0] + '.srt'
        # 自定义ocr对象
        self.ocr = None
        # 总处理进度（帧提取100 + OCR100 + 后处理100 = 300）
        self.progress_total = 300
        # 视频帧提取进度
        self.progress_frame_extract = 0
        # OCR识别进度
        self.progress_ocr = 0
        # 后处理进度
        self.progress_post = 0
        # 是否完成
        self.isFinished = False
        # 字幕OCR任务队列
        self.subtitle_ocr_task_queue = None
        # 字幕OCR进度队列
        self.subtitle_ocr_progress_queue = None
        # vsf运行状态
        self.vsf_running = False
        # 进度监听器列表
        self.progress_listeners = []

    def run(self):
        """
        运行整个提取视频的步骤
        """
        # 记录开始运行的时间
        start_time = time.time()
        self.lock.acquire()
        # 重置进度条
        self.update_progress(ocr=0, frame_extract=0, post=0)
        self.append_output('-----------------------------')
        # 打印识别语言与识别模式
        self.append_output(f"  {tr['Main']['RecSubLang']}：{config.language.value}  |  {tr['Main']['RecMode']}：{config.mode.value}")
        # 如果使用GPU加速，则打印GPU加速提示
        if self.hardware_accelerator.has_accelerator():
            self.append_output(f"  {tr['Main']['AcceleratorON'].format(self.hardware_accelerator.accelerator_name)}")

        # 打印视频帧数与帧率
        self.append_output(f"  {tr['Main']['FrameCount']}：{self.frame_count}"
              f"  |  {tr['Main']['FrameRate']}：{self.fps}")
        # 打印加载模型信息
        det_desc, rec_desc = [part.strip() for part in self.model_config.describe().split('|', 1)]
        self.append_output(f"  DET: {det_desc}  |  REC: {rec_desc}")
        # 当前语言没有 v6 专用模型时给出提示
        if self.model_config.LANG_WARNING:
            self.append_output(f"  {self.model_config.LANG_WARNING}")
        self.append_output('-----------------------------')
        # 打印视频帧提取开始提示
        self.append_output(tr['Main']['StartProcessFrame'])
        # 删除缓存
        self.__delete_frame_cache()
        # 若目录不存在，则创建文件夹
        if not os.path.exists(self.frame_output_dir):
            os.makedirs(self.frame_output_dir)
        if not os.path.exists(self.subtitle_output_dir):
            os.makedirs(self.subtitle_output_dir)
        self.capture_frame_with_subtitle_area()
        # 创建一个字幕OCR识别进程
        subtitle_ocr_process = self.start_subtitle_ocr_async()
        if self.sub_area is not None:
            if platform.system() in ['Windows', 'Linux', 'Darwin']:
                # 有 GPU 时"精准"和"自动"都走 det（GPU 检测）：在这类机器上 det 比
                # VideoSubFinder 又快又准，而 VSF 会把 CPU 核心吃满、GPU 全程闲置。
                # 4K 2 分钟样片实测：det 71.9s / 30 条 / CPU 1~2 核 / GPU 忙；
                #                    VSF 124.1s / 19 条 / VSF 单进程吃 8~10.5 核 / GPU 1~4%。
                # 无 GPU 时 auto 仍退回 VSF —— CPU 上逐帧 det 太慢（约 13s/帧）。
                # 显式选"快速"的仍然走 VSF，保留原来的行为。
                if self.hardware_accelerator.has_accelerator() and config.mode.value in ('accurate', 'auto'):
                    self.extract_frame_by_det()
                else:
                    self.extract_frame_by_vsf()
            else:
                self.extract_frame_by_fps()
        else:
            self.extract_frame_by_fps()

        # 往字幕OCR任务队列中，添加OCR识别任务结束标志
        # 任务格式为：(total_frame_count总帧数, current_frame_no当前帧, dt_box检测框, rec_res识别结果, 当前帧时间， subtitle_area字幕区域)
        self.subtitle_ocr_task_queue.put((self.frame_count, -1, None, None, None, None))
        # 等待子线程完成（带超时：OCR 线程若异常退出，无超时会永久挂住主进程）
        subtitle_ocr_process.join(timeout=1800)
        if subtitle_ocr_process.is_alive():
            raise RuntimeError('OCR 识别进程 30 分钟内未结束，请检查 loss 日志或重试')
        # 打印完成提示
        self.append_output(tr['Main']['FinishProcessFrame'])
        self.append_output(tr['Main']['FinishFindSub'])

        if self.sub_area is None:
            self.append_output(tr['Main']['StartDetectWaterMark'])
            # 询问用户视频是否有水印区域
            user_input = input(tr['Main']['checkWaterMark']).strip()
            if user_input == 'y':
                self.filter_watermark()
                self.append_output(tr['Main']['FinishDetectWaterMark'])
            else:
                self.append_output('-----------------------------')

        if self.sub_area is None:
            self.append_output(tr['Main']['StartDeleteNonSub'])
            self.filter_scene_text()
            self.append_output(tr['Main']['FinishDeleteNonSub'])

        self.update_progress(post=20)

        # 打印开始字幕生成提示
        self.append_output(tr['Main']['StartGenerateSub'])
        # 判断是否使用了vsf提取字幕
        if self.use_vsf:
            # 如果使用了vsf提取字幕，则使用vsf的字幕生成方法
            self.generate_subtitle_file_vsf()
        else:
            # 如果未使用vsf提取字幕，则使用常规字幕生成方法
            self.generate_subtitle_file()

        self.update_progress(post=90)

        if config.wordSegmentation.value:
            reformat.execute(self.subtitle_output_path, config.language.value)
        self.append_output(tr['Main']['FinishGenerateSub'], f"{round(time.time() - start_time, 2)}s")
        self.append_output('-----------------------------')
        # isFinished 必须在最后一次 update_progress 之前置位：GUI 是靠进度回调里的
        # isFinished 判定"流程真的走完了"的，先发进度再置位会让 GUI 永远收不到完成信号。
        self.isFinished = True
        self.update_progress(ocr=100, frame_extract=100, post=100)
        # 删除缓存文件
        self.empty_cache()
        self.lock.release()
        if config.generateTxt.value:
            self.srt2txt(self.subtitle_output_path)

    def capture_frame_with_subtitle_area(self):
        """
        截取视频的一帧，并在上面绘制字幕区域，保存到temp_output_dir/sub_area.jpg
        """
        # 确保输出目录存在
        if not os.path.exists(self.temp_output_dir):
            os.makedirs(self.temp_output_dir)
            
        # 确保视频已打开
        if not self.video_cap.isOpened():
            self.video_cap = cv2.VideoCapture(self.video_path)
            
        # 将视频指针设置到第一帧
        # self.video_cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        
        # 读取第一帧
        ret, frame = self.video_cap.read()
        
        if ret:
            # 如果有字幕区域，绘制矩形
            sub_area = self.sub_area
            if sub_area is not None:
                # 绘制绿色矩形框
                cv2.rectangle(frame, (sub_area.xmin, sub_area.ymin), (sub_area.xmax, sub_area.ymax), (0, 255, 0), 2)
                # 添加文字标注
                cv2.putText(frame, "Subtitle Area", (sub_area.xmin, sub_area.ymin - 10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
            
            # 保存图像
            output_path = os.path.join(self.temp_output_dir, 'sub_area.jpg')
            cv2.imwrite(output_path, frame)
            
            # 重置视频指针到第一帧
            self.video_cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    def extract_frame_by_fps(self):
        """
        根据帧率，定时提取视频帧，容易丢字幕，但速度快，将提取到的视频帧加入ocr识别任务队列
        """
        # 计算采样间隔（每隔多少帧取一帧）
        frame_interval = int(self.fps // config.extractFrequency.value)
        if frame_interval < 1:
            frame_interval = 1
        # 当前视频帧的帧号
        current_frame_no = 0
        total_frames = int(self.frame_count)
        while current_frame_no < total_frames:
            # 直接跳到目标帧
            self.video_cap.set(cv2.CAP_PROP_POS_FRAMES, current_frame_no)
            ret, frame = self.video_cap.read()
            if not ret:
                break
            current_frame_no += 1
            # subtitle_ocr_task_queue: (total_frame_count总帧数, current_frame_no当前帧, dt_box检测框, rec_res识别结果, 当前帧时间，subtitle_area字幕区域)
            task = (self.frame_count, current_frame_no, None, None, None, config.subtitleArea.value)
            self.subtitle_ocr_task_queue.put(task)
            # 更新进度条
            self.update_progress(frame_extract=(current_frame_no / self.frame_count) * 100)
            # 跳到下一个采样帧
            current_frame_no = current_frame_no + frame_interval - 1

        self.video_cap.release()

    def extract_frame_by_det(self):
        """
        通过检测字幕区域位置提取字幕帧

        三层优化（输出与逐帧版一致）：
        1. 字幕条管道解码：只让 ffmpeg 解码并裁出字幕选区（外扩 16px），
           不再整帧解码 + YUV→BGR 转换（4K 每帧 15.2ms → 5.5ms）。
           像素与 OpenCV 逐字节一致（色彩矩阵固定 BT.601，见 DECODE_COLOR）。
        2. 签名跳帧：字幕条降采样签名与上一帧相同（差异像素 < 0.3%）时，
           复用上一帧的 det/rec 结果，跳过两次模型调用（动画字幕每条持续数十帧）。
        3. rec-only 识别：复用 det 已算出的框只跑识别，跳过重复检测。
        """
        # 当前视频帧的帧号
        current_frame_no = 0
        frame_lru_list = []
        frame_lru_list_max_size = 2
        ocr_args_list = []
        compare_ocr_result_cache = {}
        tbar = tqdm(total=int(self.frame_count), unit='f', position=0, file=sys.__stdout__)
        first_flag = True
        is_finding_start_frame_no = False
        is_finding_end_frame_no = False
        start_frame_no = 0
        start_end_frame_no = []
        start_frame = None
        if self.ocr is None:
            self.ocr = OcrRecogniser()
            self.ocr.hardware_accelerator = self.hardware_accelerator

        sig_region = None
        sig_prev = None     # 上一帧签名
        # 最近一次真实 det 的结果: (frame_no, dt_boxes, coordinate_list)
        sig_src = None
        # frame_no -> compare 缓存条目：复用帧需要引用源帧条目，且要跨过 compare 缓存的 10 帧淘汰窗口
        entry_by_frame = {}
        # 调试：VSE_DEBUG_STATE=1 时把状态机事件写入 state_events.log（不干扰 tqdm 输出）
        debug_state = os.environ.get('VSE_DEBUG_STATE', '') == '1'
        dbg_fh = open(os.path.join(os.getcwd(), 'state_events.log'), 'a', encoding='utf-8') if debug_state else None
        det_calls = 0
        reuse_frames = 0

        def dbg(*args):
            if dbg_fh is not None and not dbg_fh.closed:
                print(*args, file=dbg_fh, flush=True)

        # frame 可能是整帧（origin=(0,0)）或字幕条局部图（origin 为它在整帧中的左上角），
        # 下游一律使用整帧坐标系，靠 origin 换算。
        for current_frame_no, frame, origin in self._iter_frames():
            tbar.update(1)

            # ---- 签名判定：字幕条与上一帧内容相同则复用上一次真实 det/rec 的结果 ----
            if sig_region is None:
                sig_region = self._signature_region((self.frame_height, self.frame_width))
            sig = self._strip_signature(frame, sig_region, origin) if self.SIGNATURE_MAX_FRAC > 0 else None
            frac = None
            if sig is not None and sig_prev is not None and sig.shape == sig_prev.shape:
                d = np.abs(sig.astype(np.int16) - sig_prev.astype(np.int16))
                frac = float((d > self.SIGNATURE_DIFF_GRAY).mean())
            reused = frac is not None and frac < self.SIGNATURE_MAX_FRAC and sig_src is not None
            entry = None
            if reused:
                src_no, dt_boxes, coordinate_list = sig_src
                if len(dt_boxes):
                    entry = entry_by_frame.get(src_no)
                    if entry is None:
                        # 源帧条目缺失（理论上不会发生）时退回真实检测，保证正确性
                        reused = False
            if reused:
                reuse_frames += 1
                dbg(f'[FRAME] f={current_frame_no} REUSE src={src_no} frac={frac:.5f} boxes={len(dt_boxes)}')
            else:
                dt_boxes, elapse = self._detect_subtitle_in_area(frame, origin)
                det_calls += 1
                coordinate_list = get_coordinates(dt_boxes.tolist()) if len(dt_boxes) else []
                sig_src = (current_frame_no, dt_boxes, coordinate_list)
                dbg(f'[FRAME] f={current_frame_no} DET frac={None if frac is None else round(frac, 5)} '
                    f'boxes={len(dt_boxes)}')
                if len(dt_boxes):
                    dt_box, rec_res = self._ocr_in_area(frame, dt_boxes, coordinate_list, origin)
                    entry = {'text': "".join(self.__get_area_text((dt_box, rec_res))),
                             'dt_box': dt_box, 'rec_res': rec_res}
            sig_prev = sig

            # 登记缓存：复用帧与源帧共享同一份条目（下游只读，共享安全）
            if len(dt_boxes):
                entry_by_frame[current_frame_no] = entry
                if current_frame_no not in compare_ocr_result_cache:
                    compare_ocr_result_cache[current_frame_no] = entry
                dbg(f'[REG] f={current_frame_no} src={sig_src[0]} reused={reused} text={entry["text"]!r}')

            # ---- 状态机（与原始逐帧实现一致）----
            has_subtitle = False
            sub_area = self.sub_area
            if sub_area is not None:
                for coordinate in coordinate_list:
                    xmin, xmax, ymin, ymax = coordinate
                    if (sub_area.xmin <= xmin and xmax <= sub_area.xmax
                            and sub_area.ymin <= ymin
                            and ymax <= sub_area.ymax):
                        has_subtitle = True
                        break
            else:
                has_subtitle = len(dt_boxes) > 0
            # 选区内只有噪声框（logo、省略号）的帧不算字幕，否则会产出 'T' / '?' / 'oo'
            # 这类 1 帧垃圾条目
            if has_subtitle and entry is not None and self.NOISE_FILTER \
                    and self._is_noise_text(entry['text']):
                has_subtitle = False
            # 检测到包含字幕帧的起始帧号与结束帧号
            if has_subtitle:
                # 检测到字幕时，如果列表为空，则为字幕头
                if first_flag:
                    is_finding_start_frame_no = True
                    first_flag = False
                # 判断是字幕头还是尾
                if is_finding_start_frame_no:
                    start_frame_no = current_frame_no
                    dbg(f'[START] f={current_frame_no} reused={reused} '
                        f'text={entry["text"]!r}')
                    frame_lru_list.append((frame, current_frame_no))
                    ocr_args_list.append((self.frame_count, current_frame_no))
                    # 缓存头帧
                    start_frame = frame
                    # 开始找尾
                    is_finding_start_frame_no = False
                    is_finding_end_frame_no = True
                # 判断是否为最后一帧
                if is_finding_end_frame_no and current_frame_no == self.frame_count:
                    is_finding_end_frame_no = False
                    is_finding_start_frame_no = False
                    end_frame_no = current_frame_no
                    frame_lru_list.append((frame, current_frame_no))
                    ocr_args_list.append((self.frame_count, current_frame_no))
                    start_end_frame_no.append((start_frame_no, end_frame_no))
                # 如果在找结束帧的时候
                if is_finding_end_frame_no:
                    # 判断该帧与头帧ocr内容是否一致,若不一致则找到尾，尾巴为前一帧
                    if not self._compare_ocr_result(compare_ocr_result_cache, None, start_frame_no, frame, current_frame_no,
                                                    img2_polys=dt_boxes, img2_coordinates=coordinate_list,
                                                    img2_origin=origin):
                        is_finding_end_frame_no = False
                        is_finding_start_frame_no = True
                        end_frame_no = current_frame_no - 1
                        dbg(f'[END-DIFF] f={current_frame_no} end={end_frame_no} start={start_frame_no} reused={reused}')
                        frame_lru_list.append((start_frame, end_frame_no))
                        ocr_args_list.append((self.frame_count, end_frame_no))
                        start_end_frame_no.append((start_frame_no, end_frame_no))

            else:
                # 如果检测到字幕头后有没有字幕，则找到结尾，尾巴为前一帧
                if is_finding_end_frame_no:
                    end_frame_no = current_frame_no - 1
                    dbg(f'[END-NOSUB] f={current_frame_no} end={end_frame_no} start={start_frame_no} reused={reused}')
                    is_finding_end_frame_no = False
                    is_finding_start_frame_no = True
                    frame_lru_list.append((start_frame, end_frame_no))
                    ocr_args_list.append((self.frame_count, end_frame_no))
                    start_end_frame_no.append((start_frame_no, end_frame_no))

            while len(frame_lru_list) > frame_lru_list_max_size:
                frame_lru_list.pop(0)

            # if len(start_end_frame_no) > 0:
                # self.append_output(start_end_frame_no)

            while len(ocr_args_list) > 1:
                total_frame_count, ocr_info_frame_no = ocr_args_list.pop(0)
                # 必须按入队的帧号取缓存，否则会把当前帧的文本贴到别的帧号上（文本与时间轴错配）
                predict_result = self._take_flush_result(
                    compare_ocr_result_cache, entry_by_frame, ocr_info_frame_no, dbg)
                if predict_result is not None:
                    dt_box, rec_res = predict_result['dt_box'], predict_result['rec_res']
                else:
                    dt_box, rec_res = None, None
                # subtitle_ocr_task_queue: (total_frame_count总帧数, current_frame_no当前帧, dt_box检测框, rec_res识别结果, 当前帧时间， subtitle_area字幕区域)
                task = (total_frame_count, ocr_info_frame_no, dt_box, rec_res, None, config.subtitleArea.value)
                # 添加任务
                self.subtitle_ocr_task_queue.put(task)
                self.update_progress(frame_extract=(current_frame_no / self.frame_count) * 100)

        dbg(f'[STAT] frames={current_frame_no} det_calls={det_calls} reused_frames={reuse_frames} '
            f'skip_ratio={reuse_frames / max(current_frame_no, 1):.1%}')
        if dbg_fh is not None:
            dbg_fh.close()
        while len(ocr_args_list) > 0:
            total_frame_count, ocr_info_frame_no = ocr_args_list.pop(0)
            predict_result = self._take_flush_result(
                compare_ocr_result_cache, entry_by_frame, ocr_info_frame_no, dbg)
            if predict_result is not None:
                dt_box, rec_res = predict_result['dt_box'], predict_result['rec_res']
            else:
                dt_box, rec_res = None, None
            task = (total_frame_count, ocr_info_frame_no, dt_box, rec_res, None, config.subtitleArea.value)
            # 添加任务
            self.subtitle_ocr_task_queue.put(task)
        self.video_cap.release()

    @staticmethod
    def vsf_executable_path():
        """
        返回 VideoSubFinder CLI 所在路径
        Windows/Linux 使用 6.10.x 版 CLI（eritpchy/videosubfinder-cli），macOS 暂沿用 5.60 版
        """
        system = platform.system()
        if system == 'Windows':
            return os.path.join(BASE_DIR, 'subfinder', 'windows', 'VideoSubFinderCli.exe')
        if system == 'Darwin':
            path_vsf = os.path.join(BASE_DIR, 'subfinder', 'macos', 'VideoSubFinderCli')
        else:
            path_vsf = os.path.join(BASE_DIR, 'subfinder', 'linux', 'VideoSubFinderCli.run')
        os.chmod(path_vsf, 0o775)
        return path_vsf

    @staticmethod
    def parse_vsf_time(name):
        """
        解析 VideoSubFinder 输出的字幕时间
        name 形如：0_00_02_300__0_00_03_566_<hash>
        返回起始时间（毫秒），解析失败返回 None
        """
        match = re.match(r'^(\d+)_(\d+)_(\d+)_(\d+)__', name.strip())
        if not match:
            return None
        h, m, s, ms = (int(value) for value in match.groups())
        return ms + s * 1000 + m * 60 * 1000 + h * 3600 * 1000

    def build_vsf_command(self, path_vsf):
        """
        构造 VideoSubFinder CLI 参数
        6.x 的改动：-nocrthr 已移除、-nthr 限制为 2-64、CPU 版传 --use_cuda 会被直接拒绝
        """
        # top_end：图像上半部分所占百分比，取值【0-1】
        top_end = 1 - self.sub_area.ymin / self.frame_height
        # bottom_end：图像下半部分所占百分比，取值【0-1】
        bottom_end = 1 - self.sub_area.ymax / self.frame_height
        # left_end：图像左半部分所占百分比，取值【0-1】
        left_end = self.sub_area.xmin / self.frame_width
        # right_end：图像右半部分所占百分比，取值【0-1】
        right_end = self.sub_area.xmax / self.frame_width
        if (not self.hardware_accelerator.has_cuda()) and len(self.hardware_accelerator.onnx_providers) > 0:
            cpu_count = multiprocessing.cpu_count()
        else:
            # 留2核心来给其他任务使用
            cpu_count = max(multiprocessing.cpu_count() - 2, 1)
        if config.videoSubFinderCpuCores.value > 0:
            cpu_count = config.videoSubFinderCpuCores.value
        # CLI 只接受 2-64 的线程数
        cpu_count = max(2, min(64, int(cpu_count)))

        cmd = [
            path_vsf,
            '-c', '-r',
            # 只需要时间轴，不落盘图像（OCR 直接从视频取帧）
            '-dsi',
            '-i', self.video_path,
            '-o', self.temp_output_dir,
            '-ces', self.vsf_subtitle,
            '-te', f'{top_end:.6f}', '-be', f'{bottom_end:.6f}',
            '-le', f'{left_end:.6f}', '-re', f'{right_end:.6f}',
            '-nthr', str(cpu_count),
            f'--open_video_{config.videoSubFinderDecoder.value.value.lower()}',
        ]
        if self.vsf_cuda_enabled():
            cmd.append('--use_cuda')
        return cmd

    def vsf_cuda_enabled(self):
        """
        是否给 VideoSubFinder 传 --use_cuda
        随包携带的是 CPU 静态版本，传 --use_cuda 会被拒绝，因此默认关闭；
        替换为 CUDA 版后设置环境变量 VSE_VSF_USE_CUDA=1 即可启用。
        macOS 沿用旧版行为（有加速器就开启）。
        """
        if os.environ.get('VSE_VSF_USE_CUDA', '').strip().lower() in ('1', 'true', 'yes', 'on'):
            return True
        return platform.system() == 'Darwin' and self.hardware_accelerator.has_accelerator()

    def extract_frame_by_vsf(self):
        """
       通过调用videoSubFinder获取字幕帧
       """
        self.use_vsf = True
        if self.video_cap:
            self.video_cap.release()
            self.video_cap = None

        duration_ms = (self.frame_count / self.fps) * 1000 if self.fps else 0
        # 初始值用 -1，避免 0:00:00.000 处的字幕被判为“未前进”而丢弃
        state = {'last_total_ms': -1}

        def post_ocr_task(total_ms):
            """按时间轴投放 OCR 任务"""
            if total_ms <= state['last_total_ms']:
                return
            state['last_total_ms'] = total_ms
            frame_no = int(total_ms / self.fps) if self.fps else 0
            task = (self.frame_count, frame_no, None, None, total_ms, config.subtitleArea.value)
            self.subtitle_ocr_task_queue.put(task)
            if duration_ms > 0:
                self.update_progress(frame_extract=min(total_ms / duration_ms * 100, 100))

        def vsf_output(out):
            for raw_line in iter(out.readline, b''):
                line = raw_line.decode('utf-8', 'replace').strip()
                if not line:
                    continue
                if line.startswith('Frame: '):
                    total_ms = self.parse_vsf_time(line[len('Frame: '):])
                    if total_ms is not None:
                        post_ocr_task(total_ms)
                elif line.startswith('Progress:') or line == 'FINISH':
                    # 进度信息与结束标记，不输出到界面
                    continue
                else:
                    self.append_output(line)
            out.close()

        path_vsf = self.vsf_executable_path()
        cmd = self.build_vsf_command(path_vsf)
        self.append_output(f"VSF: {subprocess.list2cmdline(cmd)}")
        self.vsf_running = True
        try:
            # 已知BUG: test_chinese_cht.flv在net drive上会导致无法停止, 但在本地不会, 可能是vsf的原因
            popen_kwargs = dict(
                stdout=subprocess.PIPE, stderr=subprocess.PIPE, bufsize=1,
                close_fds='posix' in sys.builtin_module_names, shell=False,
            )
            if platform.system() == 'Windows':
                popen_kwargs['creationflags'] = subprocess.CREATE_NEW_PROCESS_GROUP
            else:
                popen_kwargs['start_new_session'] = True
            p = subprocess.Popen(cmd, **popen_kwargs)
            ProcessManager.instance().add_process(p)
            self.manage_process(p.pid)
            # 同时消费 stdout/stderr，避免管道写满导致子进程阻塞
            Thread(target=vsf_output, daemon=True, args=(p.stderr,)).start()
            Thread(target=vsf_output, daemon=True, args=(p.stdout,)).start()
            p.wait()
            if p.returncode != 0:
                self.append_output(f"VideoSubFinder 退出码: {p.returncode}（请检查视频路径、解码方式与 settings/general.cfg）")
        finally:
            self.vsf_running = False
    def filter_watermark(self):
        """
        去除原始字幕文本中的水印区域的文本
        """
        # 获取潜在水印区域
        watermark_areas = self._detect_watermark_area()

        # 随机选择一帧, 将所水印区域标记出来，用户看图判断是否是水印区域
        cap = cv2.VideoCapture(self.video_path)
        ret, sample_frame = False, None
        for i in range(10):
            frame_no = random.randint(int(self.frame_count * 0.1), int(self.frame_count * 0.9))
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_no)
            ret, sample_frame = cap.read()
            if ret:
                break
        cap.release()

        if not ret:
            self.append_output("Error in filter_watermark: reading frame from video")
            return

        # 给潜在的水印区域编号
        area_num = ['E', 'D', 'C', 'B', 'A']

        for watermark_area in watermark_areas:
            ymin = min(watermark_area[0][2], watermark_area[0][3])
            ymax = max(watermark_area[0][3], watermark_area[0][2])
            xmin = min(watermark_area[0][0], watermark_area[0][1])
            xmax = max(watermark_area[0][1], watermark_area[0][0])
            cover = sample_frame[ymin:ymax, xmin:xmax]
            cover = cv2.blur(cover, (10, 10))
            cv2.rectangle(cover, pt1=(0, cover.shape[0]), pt2=(cover.shape[1], 0), color=(0, 0, 255), thickness=3)
            sample_frame[ymin:ymax, xmin:xmax] = cover
            position = ((xmin + xmax) // 2, ymax)

            cv2.putText(sample_frame, text=area_num.pop(), org=position, fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                        fontScale=1, color=(255, 0, 0), thickness=2, lineType=cv2.LINE_AA)

        sample_frame_file_path = os.path.join(os.path.dirname(self.frame_output_dir), 'watermark_area.jpg')
        cv2.imwrite(sample_frame_file_path, sample_frame)
        self.append_output(f"{tr['Main']['WatchPicture']}: {sample_frame_file_path}")

        area_num = ['E', 'D', 'C', 'B', 'A']
        for watermark_area in watermark_areas:
            user_input = input(f"{area_num.pop()}{str(watermark_area)} "
                               f"{tr['Main']['QuestionDelete']}").strip()
            if user_input == 'y' or user_input == '\n':
                with open(self.raw_subtitle_path, mode='r+', encoding='utf-8') as f:
                    content = f.readlines()
                    f.seek(0)
                    for i in content:
                        if i.find(str(watermark_area[0])) == -1:
                            f.write(i)
                    f.truncate()
                self.append_output(tr['Main']['FinishDelete'])
        self.append_output(tr['Main']['FinishWaterMarkFilter'])
        # 删除缓存
        if os.path.exists(sample_frame_file_path):
            os.remove(sample_frame_file_path)

    def filter_scene_text(self):
        """
        将场景里提取的文字过滤，仅保留字幕区域
        """
        # 获取潜在字幕区域
        subtitle_area = self._detect_subtitle_area()[0][0]

        # 随机选择一帧，将所水印区域标记出来，用户看图判断是否是水印区域
        cap = cv2.VideoCapture(self.video_path)
        ret, sample_frame = False, None
        for i in range(10):
            frame_no = random.randint(int(self.frame_count * 0.1), int(self.frame_count * 0.9))
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_no)
            ret, sample_frame = cap.read()
            if ret:
                break
        cap.release()

        if not ret:
            self.append_output("Error in filter_scene_text: reading frame from video")
            return

        # 为了防止有双行字幕，根据容忍度，将字幕区域y范围加高
        ymin = abs(subtitle_area[0] - config.subtitleAreaDeviationPixel.value)
        ymax = subtitle_area[1] + config.subtitleAreaDeviationPixel.value
        # 画出字幕框的区域
        cv2.rectangle(sample_frame, pt1=(0, ymin), pt2=(sample_frame.shape[1], ymax), color=(0, 0, 255), thickness=3)
        sample_frame_file_path = os.path.join(os.path.dirname(self.frame_output_dir), 'subtitle_area.jpg')
        cv2.imwrite(sample_frame_file_path, sample_frame)
        self.append_output(f"{tr['Main']['CheckSubArea']} {sample_frame_file_path}")

        user_input = input(f"{(ymin, ymax)} {tr['Main']['DeleteNoSubArea']}").strip()
        if user_input == 'y' or user_input == '\n':
            with open(self.raw_subtitle_path, mode='r+', encoding='utf-8') as f:
                content = f.readlines()
                f.seek(0)
                for i in content:
                    i_ymin = int(i.split('\t')[1].split('(')[1].split(')')[0].split(', ')[2])
                    i_ymax = int(i.split('\t')[1].split('(')[1].split(')')[0].split(', ')[3])
                    if ymin <= i_ymin and i_ymax <= ymax:
                        f.write(i)
                f.truncate()
            self.append_output(tr['Main']['FinishDeleteNoSubArea'])
        # 删除缓存
        if os.path.exists(sample_frame_file_path):
            os.remove(sample_frame_file_path)

    def generate_subtitle_file(self):
        """
        生成srt格式的字幕文件
        """
        if not self.use_vsf:
            subtitle_content = self._remove_duplicate_subtitle()
            # 保存持续时间不足1秒的字幕行，用于后续处理
            post_process_subtitle = []
            with open(self.subtitle_output_path, mode='w', encoding='utf-8') as f:
                for index, content in enumerate(subtitle_content):
                    line_code = index + 1
                    start_frame = int(content[0])
                    origin_end_frame = int(content[1])
                    frame_start = self._frame_to_timecode(start_frame)
                    # 比较起始帧号与结束帧号， 如果字幕持续时间不足1秒，则将显示时间设为1s
                    if abs(origin_end_frame - start_frame) < self.fps:
                        end_frame = start_frame + self.fps
                        # 补足到 1s 后不能越过下一条字幕的起始帧，否则两条字幕在时间轴上重叠
                        if index + 1 < len(subtitle_content):
                            next_start_frame = int(subtitle_content[index + 1][0])
                            end_frame = min(end_frame, next_start_frame - 1)
                        # 与下一条几乎紧邻时可能补不出时长，此时保留原始结束帧（重叠源于原始数据本身）
                        end_frame = max(end_frame, origin_end_frame)
                        frame_end = self._frame_to_timecode(end_frame)
                        post_process_subtitle.append(line_code)
                    else:
                        frame_end = self._frame_to_timecode(origin_end_frame)
                    frame_content = content[2]
                    subtitle_line = f'{line_code}\n{frame_start} --> {frame_end}\n{frame_content}\n'
                    f.write(subtitle_line)
            self.append_output(tr['Main']['SubLocation'].format(self.subtitle_output_path))
            # 返回持续时间低于1s的字幕行
            return post_process_subtitle

    def generate_subtitle_file_vsf(self):
        if not self.use_vsf:
            return
        subs = pysrt.open(self.vsf_subtitle)
        sub_no_map = {}
        for sub in subs:
            sub.start.no = self._timestamp_to_frameno(sub.start.ordinal)
            sub_no_map[sub.start.no] = sub

        subtitle_content = self._remove_duplicate_subtitle()
        subtitle_content_start_map = {int(a[0]): a for a in subtitle_content}
        final_subtitles = []
        for sub in subs:
            found = sub.start.no in subtitle_content_start_map
            if found:
                subtitle_content_line = subtitle_content_start_map[sub.start.no]
                sub.text = subtitle_content_line[2]
                end_no = int(subtitle_content_line[1])
                sub.end = sub_no_map[end_no].end if end_no in sub_no_map else sub.end
                sub.index = len(final_subtitles) + 1
                final_subtitles.append(sub)

            if not found and not config.deleteEmptyTimeStamp.value:
                # 保留时间轴
                sub.text = ""
                sub.index = len(final_subtitles) + 1
                final_subtitles.append(sub)
                continue

        pysrt.SubRipFile(final_subtitles).save(self.subtitle_output_path, encoding='utf-8')
        self.append_output(tr['Main']['SubLocation'].format(self.subtitle_output_path))

    def _detect_watermark_area(self):
        """
        根据识别出来的raw txt文件中的坐标点信息，查找水印区域
        假定：水印区域（台标）的坐标在水平和垂直方向都是固定的，也就是具有(xmin, xmax, ymin, ymax)相对固定
        根据坐标点信息，进行统计，将一直具有固定坐标的文本区域选出
        :return 返回最有可能的水印区域
        """
        with open(self.raw_subtitle_path, mode='r', encoding='utf-8') as f:
            lines = f.readlines()
        # 一次性解析所有行
        parsed = []
        for line in lines:
            parts = line.split('\t')
            if len(parts) < 3:
                continue
            frame_no = parts[0]
            text_position = parts[1].split('(')[1].split(')')[0].split(', ')
            content = parts[2]
            coordinate = (int(text_position[0]), int(text_position[1]),
                          int(text_position[2]), int(text_position[3]))
            parsed.append((frame_no, coordinate, content))

        frame_no_list = [p[0] for p in parsed]
        coordinates_list = [p[1] for p in parsed]
        content_list = [p[2] for p in parsed]
        # 将坐标列表的相似值统一
        coordinates_list = self._unite_coordinates(coordinates_list)

        # 将原txt文件的坐标更新为归一后的坐标
        with open(self.raw_subtitle_path, mode='w', encoding='utf-8') as f:
            for frame_no, coordinate, content in zip(frame_no_list, coordinates_list, content_list):
                f.write(f'{frame_no}\t{coordinate}\t{content}')

        count = Counter(coordinates_list).most_common()
        num = min(config.waterarkAreaNum.value, len(count))
        return count[:num]

    def _detect_subtitle_area(self):
        """
        读取过滤水印区域后的raw txt文件，根据坐标信息，查找字幕区域
        假定：字幕区域在y轴上有一个相对固定的坐标范围，相对于场景文本，这个范围出现频率更高
        :return 返回字幕的区域位置
        """
        with open(self.raw_subtitle_path, mode='r', encoding='utf-8') as f:
            lines = f.readlines()
        y_coordinates_list = []
        for line in lines:
            parts = line.split('\t')
            if len(parts) < 2:
                continue
            text_position = parts[1].split('(')[1].split(')')[0].split(', ')
            y_coordinates_list.append((int(text_position[2]), int(text_position[3])))
        return Counter(y_coordinates_list).most_common(1)

    def _frame_to_timecode(self, frame_no):
        """
        将视频帧转换成时间（纯计算，不打开视频文件）
        :param frame_no: 视频的帧号，i.e. 第几帧视频帧
        :returns: SMPTE格式时间戳 as string, 如'01:02:12,032'
        """
        total_ms = frame_no / self.fps * 1000
        total_ms_int = int(total_ms)
        milliseconds = total_ms_int % 1000
        total_seconds = total_ms_int // 1000
        seconds = total_seconds % 60
        minutes = (total_seconds // 60) % 60
        hours = total_seconds // 3600
        return "%02d:%02d:%02d,%03d" % (hours, minutes, seconds, milliseconds)

    def _timestamp_to_frameno(self, time_ms):
        return int(time_ms / self.fps)

    def _frameno_to_milliseconds(self, frame_no):
        return float(int(frame_no / self.fps * 1000))

    def _remove_duplicate_subtitle(self):
        """
        读取原始的raw txt，去除重复行，返回去除了重复后的字幕列表
        """
        self._concat_content_with_same_frameno()
        with open(self.raw_subtitle_path, mode='r', encoding='utf-8') as r:
            lines = r.readlines()
        RawInfo = namedtuple('RawInfo', 'no content')
        content_list = []
        lang_dropped = 0
        for line in lines:
            parts = line.split('\t')
            if len(parts) < 3:
                continue
            frame_no = parts[0]
            content = parts[2]
            # 兜底：子进程自行识别的行（框越界导致主进程没给结果）没经过 _filter_noise_boxes，
            # 这里丢掉纯噪声行，避免 'T' / 'oo' / logo 文本变成字幕条目
            if self.NOISE_FILTER and self._is_noise_text(content):
                continue
            # 同一层兜底：子进程自识别的行也要过语言过滤，否则 '¥750'、'J.C.STAFF'
            # 这类非目标语言内容会绕过 _filter_noise_boxes 直接进 raw
            if self.LANG_FILTER and self._is_wrong_language(content):
                lang_dropped += 1
                continue
            content_list.append(RawInfo(frame_no, content))
        if lang_dropped:
            self.append_output(f'强制单一语言：丢弃 {lang_dropped} 行非「{config.language.value}」识别结果')
        # 去重后的字幕列表
        unique_subtitle_list = []
        idx_i = 0
        content_list_len = len(content_list)
        # 循环遍历每行字幕，记录开始时间与结束时间
        last_reported_idx = -1
        report_interval = max(1, content_list_len // 100)
        while idx_i < content_list_len:
            i = content_list[idx_i]
            start_frame = i.no
            idx_j = idx_i
            while idx_j < content_list_len:
                # 定期更新去重进度（post: 20→85），基于 idx_j 推进
                if idx_j - last_reported_idx >= report_interval:
                    last_reported_idx = idx_j
                    self.update_progress(post=20 + int(65 * idx_j / content_list_len))
                # 计算当前行与下一行的Levenshtein距离
                # 判决idx_j的下一帧是否与idx_i不同，若不同（或者是最后一帧）则找到结束帧
                if idx_j + 1 == content_list_len or ratio(i.content.replace(' ', ''), content_list[idx_j + 1].content.replace(' ', '')) < (config.thresholdTextSimilarity.value / 100.0):
                    # 若找到终点帧,定义字幕结束帧帧号
                    end_frame = content_list[idx_j].no
                    if not self.use_vsf:
                        if end_frame == start_frame and idx_j + 1 < content_list_len:
                            # 针对只有一帧的情况，以下一帧的开始时间为准(除非是最后一帧)
                            end_frame = content_list[idx_j + 1][0]
                    # 寻找最长字幕
                    similar_list = content_list[idx_i:idx_j + 1]
                    similar_content_strip_list = [item.content.replace(' ', '') for item in similar_list]
                    index, _ = max(enumerate(similar_content_strip_list), key=lambda x: len(x[1]))

                    # 添加进列表
                    unique_subtitle_list.append((start_frame, end_frame, similar_list[index].content))
                    idx_i = idx_j + 1
                    break
                else:
                    idx_j += 1
                    continue
        return unique_subtitle_list

    def _concat_content_with_same_frameno(self):
        """
        将raw txt文本中具有相同帧号的字幕行合并
        """
        with open(self.raw_subtitle_path, mode='r', encoding='utf-8') as r:
            lines = r.readlines()

        # 用dict按frame_no分组，保留原始顺序
        from collections import OrderedDict
        grouped = OrderedDict()
        for line in lines:
            parts = line.split('\t', 2)
            if len(parts) < 3:
                continue
            frame_no, coordinate, content = parts
            if frame_no not in grouped:
                grouped[frame_no] = (coordinate, [])
            grouped[frame_no][1].append(content)

        with open(self.raw_subtitle_path, mode='w', encoding='utf-8') as f:
            for frame_no, (coordinate, contents) in grouped.items():
                merged = ' '.join(contents).replace('\n', ' ')
                if self.LANG_FILTER and self._is_wrong_language(merged):
                    # 整帧都不是目标语言 → 该帧不写回 raw，下游自然不会产出条目。
                    # 判定放在合并之后：同一帧里 'Level 5' 和中文台词是两个框，
                    # 拼起来是一行有效字幕，不能逐框丢。
                    continue
                if self.NOISE_FILTER:
                    # 同一帧有多个框时上面是按 ' '.join 拼的，如果两个框识别出同一句，
                    # 拼完就是 'A A'。主进程的 _filter_noise_boxes 只处理单帧内的框，
                    # 这里再折一次，覆盖子进程自行识别（框越界、缓存未命中）写入的行。
                    merged = self._collapse_repeat(merged)
                if not merged.endswith('\n'):
                    merged += '\n'
                merged = unicodedata.normalize('NFKC', merged)
                f.write(f'{frame_no}\t{coordinate}\t{merged}')

    def _unite_coordinates(self, coordinates_list):
        """
        给定一个坐标列表，将这个列表中相似的坐标统一为一个值
        e.g. 由于检测框检测的结果不是一致的，相同位置文字的坐标可能一次检测为(255,123,456,789)，另一次检测为(253,122,456,799)
        因此要对相似的坐标进行值的统一
        :param coordinates_list 包含坐标点的列表
        :return: 返回一个统一值后的坐标列表
        """
        if not coordinates_list:
            return coordinates_list
        # 按xmin排序后用滑动窗口，相似坐标的xmin差值在容忍范围内
        n = len(coordinates_list)
        indexed = sorted(enumerate(coordinates_list), key=lambda x: x[1][0])
        # parent数组用于并查集
        parent = list(range(n))
        def find(i):
            while parent[i] != i:
                parent[i] = parent[parent[i]]
                i = parent[i]
            return i
        def union(i, j):
            ri, rj = find(i), find(j)
            if ri != rj:
                # 保留较小索引的坐标作为代表
                if ri > rj:
                    ri, rj = rj, ri
                parent[rj] = ri
        # 滑动窗口：xmin已排序，只要xmin差值超过容忍度就移动左边界
        left = 0
        for right in range(n):
            while left < right and indexed[right][1][0] - indexed[left][1][0] >= config.tolerantPixelX:
                left += 1
            # 检查窗口内所有元素与当前元素是否相似
            for k in range(left, right):
                if self.__is_coordinate_similar(indexed[k][1], indexed[right][1]):
                    union(indexed[k][0], indexed[right][0])
        # 将所有坐标统一为其组代表的坐标
        result = list(coordinates_list)
        for i in range(n):
            result[i] = coordinates_list[find(i)]
        return result

    def _compute_image_similarity(self, image1, image2):
        """
        计算两张图片的余弦相似度
        """
        image1 = self.__get_thum(image1)
        image2 = self.__get_thum(image2)
        images = [image1, image2]
        vectors = []
        norms = []
        for image in images:
            vector = []
            for pixel_tuple in image.getdata():
                vector.append(average(pixel_tuple))
            vectors.append(vector)
            # linalg=linear（线性）+algebra（代数），norm则表示范数
            # 求图片的范数
            norms.append(linalg.norm(vector, 2))
        a, b = vectors
        a_norm, b_norm = norms
        # dot返回的是点积，对二维数组（矩阵）进行计算
        res = dot(a / a_norm, b / b_norm)
        return res

    def __get_area_text(self, ocr_result):
        """
        获取字幕区域内的文本内容
        """
        box, text = ocr_result
        coordinates = get_coordinates(box)
        area_text = []
        for content, coordinate in zip(text, coordinates):
            sub_area = self.sub_area
            if sub_area is not None:
                xmin = coordinate[0]
                xmax = coordinate[1]
                ymin = coordinate[2]
                ymax = coordinate[3]
                if sub_area.xmin <= xmin and xmax <= sub_area.xmax and sub_area.ymin <= ymin and ymax <= sub_area.ymax:
                    area_text.append(content[0])
        if self.NOISE_FILTER and len(area_text) > 1:
            kept = [t for t in area_text if not self._is_noise_text(t)]
            # 全是噪声时原样返回：是否算作字幕由调用方的 _is_noise_text 统一判定
            if kept:
                area_text = kept
        # 强制单一语言：非目标语言的行不算字幕。整帧只剩外文招牌时 area_text 变空，
        # 上游 has_subtitle 会因此判为无字幕，不会为这一帧产出条目。
        # 按"整行"判定而不是逐框：det 常把一句拆成中文框 + 英文框
        # （'Level 5' + '7名超能力者的第三位'），逐框过滤会把 'Level 5' 单独删掉。
        if self.LANG_FILTER and area_text and self._is_wrong_language(''.join(area_text)):
            return []
        return area_text

    @staticmethod
    def _is_noise_text(text):
        """
        判断一条识别结果是不是噪声（省略号被识成 'oo'、logo 被认成 'T' / '?'）。
        含中日文或数字 → 一定是有效内容；纯拉丁/符号再按长度和重复度收窄，
        避免误伤 'OK' 这类真实短字幕。

        例外：单个字符用空格隔开重复（'1  1' / '1  1  1  1'）是省略号「……」被
        识别成数字的典型形态，属于噪声。语言过滤放开"纯数字行"之后这类会漏进来，
        所以在数字早退之前先拦一道——不动 '1' / '10' / '750' 这些真数字。
        """
        t = text.replace(' ', '').strip()
        if not t:
            return True
        stripped = text.strip()
        if len(t) > 1 and len(t) <= 4 and len(set(t)) == 1 and len(stripped) > len(t):
            return True                      # '1  1' / '1  1  1  1' / 'oo oo'
        for ch in t:
            if ch.isdigit() or '\u3040' <= ch <= '\u30ff' or '\u4e00' <= ch <= '\u9fff':
                return False
        if len(t) == 1:
            return True                      # 'T' / '?' / 'M'
        if len(t) <= 4 and len(set(t)) == 1:
            return True                      # 'oo' / 'CC' / '....'
        if not any(ch.isalnum() for ch in t):
            return True                      # '...' / '—' / '·'
        return False

    @classmethod
    def _is_wrong_language(cls, text):
        """
        强制单一语言：判断这行是不是"非所选语言"的识别结果。
        未开启 config.forceSingleLanguage（或语言未收录）时恒为 False，不影响原有行为。
        """
        if not cls.LANG_FILTER:
            return False
        try:
            if not config.forceSingleLanguage.value:
                return False
            lang = config.language.value
        except Exception:
            return False
        return not is_target_language(text, lang)

    @staticmethod
    def _take_flush_result(compare_cache, entry_by_frame, frame_no, dbg=None):
        """
        取出某个帧号的识别结果，用于写给 OCR 子进程。
        compare 缓存只保留最近 10 帧，而条目的**结束帧**往往在 flush 时已被淘汰，此时退回
        持久的 entry_by_frame：否则子进程会重新 seek+解码+整帧 OCR，既慢，又会因为整帧 OCR
        的 sub_area 越界判定与主进程不同而丢掉该结束帧 —— raw 少了结束帧行，
        _remove_duplicate_subtitle 的“只有一帧的条目”兜底就会把结束时间拉长到下一帧起始
        （实测 S01E01 20:05 的「黑子」被从 1s 拉到 4.4s）。
        """
        res = compare_cache.get(frame_no)
        if res is None:
            res = entry_by_frame.get(frame_no)
        # 主进程一个框都没给出（框整体越出字幕选区）时必须退回 None，让子进程自己识别这一帧。
        # 直接传空框会让这一帧被静默跳过：实测 S01E01 17:34 的「明明还是白天」和 OP 2:59
        # 整段因此消失。
        if res is not None and not res.get('dt_box'):
            res = None
        if dbg is not None:
            dbg(f'[FLUSH] f={frame_no} cache_hit={res is not None}')
        return res

    # ---- 签名跳帧参数 ----
    # 字幕条降采样签名中，灰度差超过 SIGNATURE_DIFF_GRAY 的像素占比低于 SIGNATURE_MAX_FRAC
    # 时，判定本帧与上一帧“内容相同”，直接复用上一次真实 det/rec 的结果，跳过两次模型调用。
    # 实测（4K 超炮 S01E01 2min）：阈值 0.3% 跳帧率 64.8%，耗时 152s→102s；
    # 收紧到 0.005% 跳帧率仍有 60%（被跳过的绝大多数是真正静止的画面）。
    # 注意：跳帧会改变模型调用序列，GPU 在淡入淡出过渡帧上可能给出略有差异的框/文本，
    # 实测只影响过渡帧产生的 1 帧垃圾条目，所有真实字幕行逐字节一致。
    # 可用环境变量 VSE_SIG_MAX_FRAC 覆盖（设为 0 可完全关闭跳帧）。
    SIGNATURE_DIFF_GRAY = 16
    SIGNATURE_MAX_FRAC = float(os.environ.get('VSE_SIG_MAX_FRAC', '0.003'))

    # ---- 同行框合并 / 识别噪声过滤 ----
    # det 会把一行字幕拆成 1~2 个框，且框数会在 1↔2 之间逐帧跳变，第二个框常常只是省略号
    # 或标点，被 rec 成 'oo' / '...' 之类。后果有两个（S01E01 整集实测）：
    #   ① 拼接文本在 '就算是我' ↔ '就算是我oo' 之间反复切换，Levenshtein ratio 恰好 =0.8
    #      卡在 thresholdTextSimilarity 的边界上，状态机每次跳变都断开条目 → 一条字幕被切成
    #      数条 <500ms 碎片（正片占 16%，OP 占 54%）；
    #   ② 拆开的两个框各自识别出同一行内容时，拼出重复文本
    #      （'振り返ることしか  振り返ることしか'）。
    # 修法：① 同行水平相邻的框先合并成一个框再识别（根除拆框与重复拼接）；
    #       ② 识别结果里的纯符号/极短拉丁噪声框丢弃（消除 'oo' / 'T' / '?' 造成的文本抖动）。
    # 环境变量可分别关闭，便于回归对比：VSE_MERGE_LINE=0 / VSE_NOISE_FILTER=0
    MERGE_SAME_LINE = os.environ.get('VSE_MERGE_LINE', '1') == '1'
    NOISE_FILTER = os.environ.get('VSE_NOISE_FILTER', '1') == '1'
    # 强制单一语言的总开关（真正是否生效还要看 config.forceSingleLanguage）。
    # 环境变量 VSE_LANG_FILTER=0 可临时关掉，便于回归对比。
    LANG_FILTER = os.environ.get('VSE_LANG_FILTER', '1') == '1'

    # ---- 管道解码参数 ----
    # 精确模式只需要字幕条上的像素，整帧 4K 每帧 24MB 的 YUV→BGR 转换和 Mat 拷贝是纯浪费。
    # 改成 ffmpeg 顺序解码 + crop 到字幕条后走管道，实测 4K 每帧 15.2ms → 5.5ms（软解）、
    # 4.6ms（NVDEC，且 CPU 占用降到约 1/4，给 det/rec 让出核心）。
    # VSE_PIPE_DECODE=0 可退回原来的 cv2 整帧读取；VSE_DECODE_HWACCEL=0 只关 GPU 解码。
    PIPE_DECODE = os.environ.get('VSE_PIPE_DECODE', '1') == '1'
    DECODE_HWACCEL = os.environ.get('VSE_DECODE_HWACCEL', '0') == '1'
    # YUV→RGB 色彩矩阵：OpenCV 的 sws 上下文不读流标签，固定按 BT.601 转换；
    # ffmpeg CLI 默认按流标签（本片源标 bt709），两者通道值差 B+4.6 / R-5.5，
    # 会让 det 在淡入淡出帧上给不同结果。显式指定 bt601 后逐像素完全一致。
    DECODE_COLOR = os.environ.get('VSE_DECODE_COLOR', 'bt601')

    def _strip_box(self, frame_shape):
        """
        字幕条解码范围（整帧坐标），与 _signature_region / _detect_subtitle_in_area 的留白一致。
        返回 (x0, y0, x1, y1)。
        """
        h, w = frame_shape[:2]
        sub_area = self.sub_area
        if sub_area is None:
            return (0, 0, w, h)
        pad = 16
        x0 = max(0, int(sub_area.xmin) - pad)
        x1 = min(w, int(sub_area.xmax) + pad)
        y0 = max(0, int(sub_area.ymin) - pad)
        y1 = min(h, int(sub_area.ymax) + pad)
        if x1 <= x0 or y1 <= y0:
            return (0, 0, w, h)
        return (x0, y0, x1, y1)

    @staticmethod
    def _iter_cap_frames(video_cap):
        """cv2 整帧顺序读（管道解码不可用时的回退路径），产出 (frame_no, frame, origin)"""
        frame_no = 0
        while video_cap.isOpened():
            ret, frame = video_cap.read()
            if not ret:
                break
            frame_no += 1
            yield frame_no, frame, (0, 0)

    def _spawn_strip_decoder(self, x, y, w, h, use_hwaccel):
        """启动一个只输出字幕条 rawvideo 的 ffmpeg 进程；失败返回 None"""
        ffmpeg = None
        try:
            import imageio_ffmpeg
            ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
        except Exception:
            ffmpeg = shutil.which('ffmpeg')
        if not ffmpeg or not os.path.exists(ffmpeg):
            ffmpeg = shutil.which('ffmpeg')
        if not ffmpeg:
            return None
        cmd = [ffmpeg, '-hide_banner', '-nostats', '-loglevel', 'error']
        if use_hwaccel:
            cmd += ['-hwaccel', 'cuda']
        cmd += ['-i', self.video_path,
                '-vf', f'crop={w}:{h}:{x}:{y},scale=in_color_matrix={self.DECODE_COLOR}:in_range=auto',
                # 必须 1:1 输出：默认的 cfr 模式会复制帧补齐时间戳，
                # 实测 4K HEVC 会多出 7 帧（开头 3 帧），导致整条时间轴偏移
                '-fps_mode', 'passthrough',
                '-an', '-pix_fmt', 'bgr24', '-f', 'rawvideo', '-']
        try:
            return subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
                                    bufsize=w * h * 3)
        except Exception as e:
            print(f'[decode] ffmpeg 启动失败: {e}', flush=True)
            return None

    def _iter_strip_frames(self):
        """
        顺序解码视频，只输出字幕条区域。产出 (frame_no, strip, (x0, y0))：
        strip 是字幕条局部图像，origin 是它在整帧中的左上角（下游坐标仍用整帧坐标系）。
        优先尝试 NVDEC，失败自动回退软解；两者都不可用时回退 cv2 整帧读取。
        """
        h, w = self.frame_height, self.frame_width
        bx0, by0, bx1, by1 = self._strip_box((h, w))
        # yuv420 的 crop 要求 x/y/w/h 为偶数：先向外扩到偶数，Python 侧再裁回精确范围
        dx0, dy0 = bx0 - (bx0 & 1), by0 - (by0 & 1)
        dx1, dy1 = min(w, bx1 + (bx1 & 1)), min(h, by1 + (by1 & 1))
        cw, ch = dx1 - dx0, dy1 - dy0
        if cw <= 0 or ch <= 0:
            yield from self._iter_cap_frames(self.video_cap)
            return
        frame_bytes = cw * ch * 3
        # 精确裁回目标范围的切片（偶数对齐可能多出 1px）
        sx0, sy0, sx1, sy1 = bx0 - dx0, by0 - dy0, bx1 - dx0, by1 - dy0
        modes = [True, False] if self.DECODE_HWACCEL else [False]
        for use_hwaccel in modes:
            proc = self._spawn_strip_decoder(dx0, dy0, cw, ch, use_hwaccel)
            if proc is None:
                continue
            frame_no = 0
            try:
                while True:
                    buf = proc.stdout.read(frame_bytes)
                    if not buf or len(buf) < frame_bytes:
                        break
                    strip = np.frombuffer(buf, np.uint8).reshape(ch, cw, 3)
                    frame_no += 1
                    yield frame_no, strip[sy0:sy1, sx0:sx1], (bx0, by0)
            finally:
                try:
                    proc.stdout.close()
                except Exception:
                    pass
                try:
                    proc.kill()
                except Exception:
                    pass
                try:
                    proc.wait(timeout=5)
                except Exception:
                    pass
            if frame_no > 0:
                if abs(frame_no - int(self.frame_count)) > 1:
                    # 管道提前结束（解码错误等）：帧号会与视频总帧数错位，明确告警而不是静默截断
                    print(f'[decode] 警告：管道输出 {frame_no} 帧，与视频总帧数 '
                          f'{int(self.frame_count)} 不符，时间轴可能偏移', flush=True)
                return
            # 这个模式一帧都没读出来（例如无 NVIDIA 驱动）→ 试下一个模式
        # 全部失败 → 回退整帧读取，保证功能不受影响
        yield from self._iter_cap_frames(self.video_cap)

    def _iter_frames(self):
        """统一取帧入口：管道解码（字幕条）或 cv2 整帧，产出 (frame_no, frame, origin)"""
        if self.PIPE_DECODE and self.sub_area is not None:
            yield from self._iter_strip_frames()
        else:
            yield from self._iter_cap_frames(self.video_cap)

    def _signature_region(self, frame_shape):
        """
        签名采样区域：与 _detect_subtitle_in_area 的裁剪范围一致（sub_area 外扩 16px），
        sub_area 为 None 时退化为整帧。返回 (y0, y1, x0, x1, step)。
        """
        h, w = frame_shape[:2]
        sub_area = self.sub_area
        if sub_area is None:
            return (0, h, 0, w, 16)
        pad = 16
        y0 = max(0, int(sub_area.ymin) - pad)
        y1 = min(h, int(sub_area.ymax) + pad)
        x0 = max(0, int(sub_area.xmin) - pad)
        x1 = min(w, int(sub_area.xmax) + pad)
        if x1 <= x0 or y1 <= y0:
            return (0, h, 0, w, 16)
        return (y0, y1, x0, x1, 8)

    @staticmethod
    def _strip_signature(frame, region, origin=(0, 0)):
        """
        字幕条签名：单通道、步长采样的小矩阵（4K 字幕条约 57x432，拷贝开销 ~0.05ms）。
        视频解码对相同内容块输出相同像素，直接数值比较即可。
        :param origin: frame 左上角在整帧坐标系中的位置，region 是整帧坐标下的采样区域
        """
        y0, y1, x0, x1, step = region
        ox, oy = origin
        ry0, ry1 = max(0, y0 - oy), min(frame.shape[0], y1 - oy)
        rx0, rx1 = max(0, x0 - ox), min(frame.shape[1], x1 - ox)
        if ry1 <= ry0 or rx1 <= rx0:
            return None
        return np.ascontiguousarray(frame[ry0:ry1:step, rx0:rx1:step, 0])

    def _detect_subtitle_in_area(self, frame, origin=(0, 0)):
        """
        只在字幕选区（外加少量留白）上跑文本检测，并把坐标换算回整帧坐标系。
        整帧检测实测 ~33ms，裁到字幕条后 ~16ms，且选区外的文字本来也不参与判定。
        :param origin: frame 左上角在整帧坐标系中的位置；传字幕条时用它换算裁剪范围，
                       只要字幕条覆盖了选区留白范围，检测输入与整帧路径完全一致
        """
        sub_area = self.sub_area
        if sub_area is None:
            return self.sub_detector.detect_subtitle(frame)
        h, w = frame.shape[:2]
        ox, oy = origin
        pad = 16
        # 整帧坐标下的目标范围 → 换算到 frame 局部坐标
        x0 = max(0, int(sub_area.xmin) - pad - ox)
        x1 = min(w, int(sub_area.xmax) + pad - ox)
        y0 = max(0, int(sub_area.ymin) - pad - oy)
        y1 = min(h, int(sub_area.ymax) + pad - oy)
        if x1 <= x0 or y1 <= y0:
            return self.sub_detector.detect_subtitle(frame)
        crop = frame[y0:y1, x0:x1]
        if crop.size == 0:
            return self.sub_detector.detect_subtitle(frame)
        dt_boxes, elapse = self.sub_detector.detect_subtitle(crop)
        if len(dt_boxes) == 0:
            return dt_boxes, elapse
        # 坐标从裁剪图坐标系平移回整帧坐标系
        dt_boxes = dt_boxes.copy()
        dt_boxes[:, :, 0] += x0 + ox
        dt_boxes[:, :, 1] += y0 + oy
        # 合并高度重叠的框：det 偶尔把一行字幕拆成两个重叠框，不合并会拼出重复文本
        dt_boxes = self._dedup_polys(dt_boxes)
        # 再合并同一行内水平相邻的框：拆框不重叠时 _dedup_polys 不生效，但会让文本在
        # 两段/整行之间来回跳变，把一条字幕切碎
        if self.MERGE_SAME_LINE:
            dt_boxes = self._merge_same_line_polys(
                dt_boxes, valid_rect=(sub_area.xmin, sub_area.ymin, sub_area.xmax, sub_area.ymax))
        return dt_boxes, elapse

    @staticmethod
    def _poly_rect(poly):
        """
        多边形的 AABB，返回 [xmin, ymin, xmax, ymax]（float）。
        必须与 get_coordinates() 用同一套定义：det 输出的框常带轻微倾斜，
        取“所有点的 min/max”会比对角点定义宽出十几像素，合并后可能越过字幕选区边界，
        导致本来能识别的帧被 _ocr_in_area 判为越界而整帧丢弃。
        """
        (x1, y1) = poly[0]
        (x2, y2) = poly[1]
        (x3, y3) = poly[2]
        (x4, y4) = poly[3]
        xmin = float(max(x1, x4))
        xmax = float(min(x2, x3))
        ymin = float(max(y1, y2))
        ymax = float(min(y3, y4))
        # 极端情况下退化框（xmax<=xmin）退回全点范围，避免出现负宽高
        if xmax <= xmin or ymax <= ymin:
            xs = [float(p[0]) for p in poly]
            ys = [float(p[1]) for p in poly]
            return [min(xs), min(ys), max(xs), max(ys)]
        return [xmin, ymin, xmax, ymax]

    @classmethod
    def _merge_same_line_polys(cls, dt_boxes, v_overlap_thresh=0.6, gap_ratio=0.8, valid_rect=None):
        """
        合并同一行内水平相邻的检测框（det 把一句话拆成多个框时使用）。
        两个框同时满足以下条件才合并：
          1. 垂直重叠高度 / 较矮框高度 > v_overlap_thresh → 判定在同一行
          2. 水平间隙 < min(两框高度) * gap_ratio → 判定是被拆开的同一句话
             （双语字幕的上下一行垂直不重叠，走不到这里，不会被误合并）
          3. 两个框都在字幕选区内（valid_rect）。越界框一旦被并进来，合并后的外接矩形
             会越过选区边界，整帧会被 _ocr_in_area 判为越界丢弃 —— 连带丢掉本来能用的
             那一半文本（实测预告段因此丢了「一緒に」）。
        合并结果取外接矩形并写成 4 点多边形，按 xmin 输出。
        """
        if dt_boxes is None or len(dt_boxes) < 2:
            return dt_boxes
        rects = [cls._poly_rect(poly) for poly in dt_boxes]
        if valid_rect is not None:
            vx0, vy0, vx1, vy1 = valid_rect
            inside = [r[0] >= vx0 and r[2] <= vx1 and r[1] >= vy0 and r[3] <= vy1 for r in rects]
        else:
            inside = [True] * len(rects)
        order = sorted(range(len(rects)), key=lambda i: rects[i][0])
        used = [False] * len(rects)
        merged = []
        for i in order:
            if used[i]:
                continue
            used[i] = True
            ax0, ay0, ax1, ay1 = rects[i]
            can_merge = inside[i]
            # 合并后外接矩形变大，可能又能与更远的框相连，迭代到不再变化
            grown = True
            while grown:
                grown = False
                for j in order:
                    if used[j] or not can_merge or not inside[j]:
                        continue
                    bx0, by0, bx1, by1 = rects[j]
                    ah, bh = ay1 - ay0, by1 - by0
                    if ah <= 0 or bh <= 0:
                        continue
                    inter_h = min(ay1, by1) - max(ay0, by0)
                    if inter_h <= 0 or inter_h / min(ah, bh) <= v_overlap_thresh:
                        continue
                    # 相交时为负，表示重叠
                    gap = max(bx0 - ax1, ax0 - bx1)
                    if gap > min(ah, bh) * gap_ratio:
                        continue
                    ax0, ay0 = min(ax0, bx0), min(ay0, by0)
                    ax1, ay1 = max(ax1, bx1), max(ay1, by1)
                    used[j] = True
                    grown = True
            merged.append([[ax0, ay0], [ax1, ay0], [ax1, ay1], [ax0, ay1]])
        if len(merged) == len(rects):
            return dt_boxes  # 无可合并项，原样返回，保持下游行为完全不变
        return np.asarray(merged, dtype=dt_boxes.dtype)

    @classmethod
    def _dedup_polys(cls, dt_boxes, overlap_thresh=0.6):
        """
        合并互相高度重叠的检测框（按交叠面积 / 较小框面积判定），
        合并结果取两者的外接矩形，保持 (N, 4, 2) 的多边形格式。
        """
        if dt_boxes is None or len(dt_boxes) < 2:
            return dt_boxes
        rects = [cls._poly_rect(poly) for poly in dt_boxes]
        used = [False] * len(rects)
        merged = []
        for i in range(len(rects)):
            if used[i]:
                continue
            ax0, ay0, ax1, ay1 = rects[i]
            area_a = max(ax1 - ax0, 1) * max(ay1 - ay0, 1)
            used[i] = True
            for j in range(i + 1, len(rects)):
                if used[j]:
                    continue
                bx0, by0, bx1, by1 = rects[j]
                inter_w = min(ax1, bx1) - max(ax0, bx0)
                inter_h = min(ay1, by1) - max(ay0, by0)
                if inter_w <= 0 or inter_h <= 0:
                    continue
                area_b = max(bx1 - bx0, 1) * max(by1 - by0, 1)
                if inter_w * inter_h / min(area_a, area_b) > overlap_thresh:
                    ax0, ay0 = min(ax0, bx0), min(ay0, by0)
                    ax1, ay1 = max(ax1, bx1), max(ay1, by1)
                    area_a = max(ax1 - ax0, 1) * max(ay1 - ay0, 1)
                    used[j] = True
            merged.append([[ax0, ay0], [ax1, ay0], [ax1, ay1], [ax0, ay1]])
        return np.asarray(merged, dtype=dt_boxes.dtype)

    def _ocr_in_area(self, frame, dt_boxes, coordinate_list, origin=(0, 0)):
        """
        复用本帧已算出的检测框，只对完全落在字幕选区内的框做识别（跳过重复的 det）。
        返参格式与 OcrRecogniser.predict() 一致。
        :param origin: frame 左上角在整帧坐标系中的位置；dt_boxes/coordinate_list 始终是整帧坐标
        """
        if dt_boxes is None or len(dt_boxes) == 0:
            return [], []
        sub_area = self.sub_area
        if sub_area is None:
            polys = list(dt_boxes)
        else:
            polys = [poly for poly, c in zip(dt_boxes, coordinate_list)
                     if (sub_area.xmin <= c[0] and c[1] <= sub_area.xmax
                         and sub_area.ymin <= c[2] and c[3] <= sub_area.ymax)]
        if not polys:
            return [], []
        dt_box, rec_res = self.ocr.recognise_polys(frame, polys, origin=origin)
        if self.NOISE_FILTER:
            # 必须在 rec 结果这一层过滤：raw.txt 是子进程按 rec_res 直接写的，只在拼接文本层
            # 过滤会让 'oo' / 'T' 这类噪声照样进 raw，最后仍变成 srt 条目。
            dt_box, rec_res = self._filter_noise_boxes(dt_box, rec_res)
        return dt_box, rec_res

    @classmethod
    def _filter_noise_boxes(cls, dt_box, rec_res):
        """
        过滤识别结果里的两类无效框：
          1. 噪声框：纯符号 / 极短拉丁字符（省略号被识成 'oo'、logo 被认成 'T' / '?'）
          2. 同行近重复框：det 把一句拆成两个框，两个框各自识别出同一行内容，
             拼接后就是 '振り返ることしか  振り返ることしか'
        返回过滤后的 (dt_box, rec_res)，两者保持一一对应。
        """
        keep = []
        for box, (text, score) in zip(dt_box, rec_res):
            # 先折叠“整句被印了两遍”的识别结果：det 给出一个宽框时，rec 有时会把同一句
            # 输出两次（'イジョウブ!迷わないで  イジョウブ!迷わないで'），去重只看框间文本
            # 抓不到这种情况
            text = cls._collapse_repeat(text)
            if cls._is_noise_text(text):
                continue
            norm = text.replace(' ', '')
            duplicated = False
            for kbox, ktext, _ in keep:
                if not cls._same_line(kbox, box):
                    continue
                knorm = ktext.replace(' ', '')
                # 只有“几乎逐字相同且长度相当”才算重复框。阈值必须收紧：
                # '泣いていた' 与 '一緒に泣いていた' 的 ratio 就有 0.857，
                # 用 0.85 会把半句当重复删掉（实测预告段丢了「一緒に」）。
                if len(norm) < len(knorm) * 0.7 or len(knorm) < len(norm) * 0.7:
                    continue
                if ratio(knorm, norm) > 0.95:
                    duplicated = True
                    break
            if not duplicated:
                keep.append((box, text, score))
        # 强制单一语言：按"整帧一行"判断。整帧文本都不是目标语言时才清空，
        # 而不是逐框删——det 常把一句拆成中文框 + 英文框（'Level 5' + '7名超能力者的第三位'），
        # 逐框过滤会把 'Level 5' 这种句中成分一起删掉。
        if keep and cls.LANG_FILTER and cls._is_wrong_language(''.join(t for _, t, _ in keep)):
            return [], []
        return [b for b, _, _ in keep], [(t, s) for _, t, s in keep]

    @staticmethod
    def _collapse_repeat(text):
        """
        把“同一句被印了两遍”的识别结果折叠成一份：'A  A' / 'AA' -> 'A'。
        片源中 ED、OP 常有重复排版，det 也可能给出包含两遍文字的宽框。
        """
        t = text.strip()
        if len(t) < 4:
            return text
        # 有分隔符：按分隔符切段，各段完全相同时只留一段；
        # 或前一半段与后一半段完全相同（两框各识别整句、拼接后中间只有一个空格时的情况：
        # '一緒に 泣いていた 一緒に 泣いていた'）
        for sep in ('  ', ' '):
            parts = [p for p in t.split(sep) if p.strip()]
            if len(parts) < 2 or len(parts[0]) < 2:
                continue
            if len(set(parts)) == 1:
                return parts[0]
            half = len(parts) // 2
            if len(parts) % 2 == 0 and parts[:half] == parts[half:]:
                return sep.join(parts[:half])
        # 无分隔符：整串是某个子串的整数倍重复
        n = len(t)
        for k in range(2, n // 2 + 1):
            if n % k == 0 and t == t[:n // k] * k:
                return t[:n // k]
        return text

    @staticmethod
    def _same_line(box1, box2, v_overlap_thresh=0.5):
        """两个框是否在同一行（垂直重叠比例超过阈值）"""
        y1 = [p[1] for p in box1]
        y2 = [p[1] for p in box2]
        h1 = max(y1) - min(y1)
        h2 = max(y2) - min(y2)
        if h1 <= 0 or h2 <= 0:
            return False
        inter = min(max(y1), max(y2)) - max(min(y1), min(y2))
        return inter > 0 and inter / min(h1, h2) > v_overlap_thresh

    def _compare_ocr_result(self, result_cache, img1, img1_no, img2, img2_no,
                            img2_polys=None, img2_coordinates=None, img2_origin=(0, 0)):
        """
        比较两张图片预测出的字幕区域文本是否相同
        img2_polys / img2_coordinates 可传入该帧已算出的检测结果，避免重复检测
        """
        if self.ocr is None:
            self.ocr = OcrRecogniser()
            self.ocr.hardware_accelerator = self.hardware_accelerator
        if img1_no in result_cache:
            area_text1 = result_cache[img1_no]['text']
        else:
            dt_box, rec_res = self.ocr.predict(img1)
            area_text1 = "".join(self.__get_area_text((dt_box, rec_res)))
            result_cache[img1_no] = {'text': area_text1, 'dt_box': dt_box, 'rec_res': rec_res}

        if img2_no in result_cache:
            area_text2 = result_cache[img2_no]['text']
        else:
            if img2_polys is not None and img2_coordinates is not None:
                dt_box, rec_res = self._ocr_in_area(img2, img2_polys, img2_coordinates, origin=img2_origin)
            else:
                dt_box, rec_res = self.ocr.predict(img2)
            area_text2 = "".join(self.__get_area_text((dt_box, rec_res)))
            result_cache[img2_no] = {'text': area_text2, 'dt_box': dt_box, 'rec_res': rec_res}
        delete_no_list = []
        for no in result_cache:
            if no < min(img1_no, img2_no) - 10:
                delete_no_list.append(no)
        for no in delete_no_list:
            del result_cache[no]
        if ratio(area_text1, area_text2) > config.thresholdTextSimilarity.value / 100.0:
            return True
        else:
            return False

    @staticmethod
    def __is_coordinate_similar(coordinate1, coordinate2):
        """
        计算两个坐标是否相似，如果两个坐标点的xmin,xmax,ymin,ymax的差值都在像素点容忍度内
        则认为这两个坐标点相似
        """
        return abs(coordinate1[0] - coordinate2[0]) < config.tolerantPixelX and \
            abs(coordinate1[1] - coordinate2[1]) < config.tolerantPixelX and \
            abs(coordinate1[2] - coordinate2[2]) < config.tolerantPixelY and \
            abs(coordinate1[3] - coordinate2[3]) < config.tolerantPixelY

    @staticmethod
    def __get_thum(image, size=(64, 64), greyscale=False):
        """
        对图片进行统一化处理
        """
        # 利用image对图像大小重新设置, Image.ANTIALIAS为高质量的
        image = image.resize(size, Image.ANTIALIAS)
        if greyscale:
            # 将图片转换为L模式，其为灰度图，其每个像素用8个bit表示
            image = image.convert('L')
        return image

    def __delete_frame_cache(self):
        if os.path.exists(self.temp_output_dir):
            shutil.rmtree(self.temp_output_dir, True)

    def empty_cache(self):
        """
        删除字幕提取过程中所有生产的缓存文件
        """
        if not config.debugNoDeleteCache.value:
            if os.path.exists(self.temp_output_dir):
                shutil.rmtree(self.temp_output_dir, True)

    def update_progress(self, ocr=None, frame_extract=None, post=None):
        """
        更新进度条
        :param ocr ocr进度
        :param frame_extract 视频帧提取进度
        :param post 后处理进度
        """
        if ocr is not None:
            self.progress_ocr = max(0, min(100, ocr))  # Clamp value between 0 and 100
        if frame_extract is not None:
            self.progress_frame_extract = max(0, min(100, frame_extract))
        if post is not None:
            self.progress_post = max(0, min(100, post))
        # 通知所有监听器
        self.notify_progress_listeners()

    def start_subtitle_ocr_async(self):
        def get_ocr_progress():
            """
            获取ocr识别进度
            生产者在所有帧入队后发送 (-2, total_tasks) 告知总帧数
            消费者每处理一帧发送 (frame_no, processed_count)
            消费者处理完全部帧后发送 (-1, total_tasks)
            """
            notify = True
            total_tasks = None
            while True:
                value = self.subtitle_ocr_progress_queue.get(block=True)
                if notify:
                    self.append_output(tr['Main']['StartFindSub'])
                    notify = False
                # 生产者告知总帧数 value = (-2, total_tasks)
                if isinstance(value, tuple) and len(value) == 2 and value[0] == -2:
                    total_tasks = value[1]
                    continue
                # 终止条件
                if value == -1:
                    self.update_progress(ocr=100)
                    return
                if isinstance(value, tuple) and value[0] == -1:
                    self.update_progress(ocr=100)
                    return
                # value = (frame_no, processed_count)
                if isinstance(value, tuple) and len(value) == 2:
                    _, processed_count = value
                    if total_tasks and total_tasks > 0:
                        ocr_pct = min(99, processed_count / total_tasks * 100)
                    else:
                        ocr_pct = min(99, processed_count)
                    self.update_progress(ocr=ocr_pct)
        options = {
            'REC_CHAR_TYPE': config.language.value,
            'DROP_SCORE': config.dropScore.value / 100.0,
            'SUB_AREA_DEVIATION_RATE': config.subtitleAreaDeviationRate.value / 100.0,
            'DEBUG_OCR_LOSS': config.debugOcrLoss.value,
            'HARDWARD_ACCELERATOR': self.hardware_accelerator,
            # 一次送进 OCR 的帧数（快速模式/无预检测结果时才生效）
            'MAX_BATCH_SIZE': _max_batch_size(),
        }
        process, task_queue, progress_queue = subtitle_ocr.async_start(self.video_path, self.raw_subtitle_path, self.sub_area, options)
        ProcessManager.instance().add_process(process)
        self.manage_process(process.pid)
        self.subtitle_ocr_task_queue = task_queue
        self.subtitle_ocr_progress_queue = progress_queue
        # 开启线程负责更新OCR进度
        Thread(target=get_ocr_progress, daemon=True).start()
        return process

    def srt2txt(self, srt_file):
        subs = pysrt.open(srt_file, encoding='utf-8')
        output_path = os.path.join(os.path.dirname(srt_file), Path(srt_file).stem + '.txt')
        self.append_output(output_path)
        with open(output_path, 'w', encoding='utf-8') as f:
            for sub in subs:
                f.write(f'{sub.text}\n')

    def append_output(self, *args):
        """输出信息到控制台
        Args:
            *args: 要输出的内容，多个参数将用空格连接
        """
        print(*args)

    def add_progress_listener(self, listener):
        """
        添加进度监听器
        
        Args:
            listener: 一个回调函数，接收参数 (progress_ocr, progress_frame_extract, progress_total, isFinished)
        """
        if listener not in self.progress_listeners:
            self.progress_listeners.append(listener)
    
    def remove_progress_listener(self, listener):
        """
        移除进度监听器
        
        Args:
            listener: 要移除的监听器函数
        """
        if listener in self.progress_listeners:
            self.progress_listeners.remove(listener)
            
    def notify_progress_listeners(self):
        """
        通知所有进度监听器当前进度
        """
        for listener in self.progress_listeners:
            try:
                listener(self.progress_ocr, self.progress_frame_extract, self.progress_total, self.isFinished, self.progress_post)
            except Exception as e:
                traceback.print_exc()

    @staticmethod
    def manage_process(pid):
        # GUI 会把这个方法替换成远程调用版本；CLI 下为占位实现。
        # 缺 @staticmethod 时 self.manage_process(pid) 会把 self 当作 pid 传进来，
        # 直接 TypeError，导致 CLI 启动 OCR 子进程后立刻崩溃。
        pass

if __name__ == '__main__':
    multiprocessing.set_start_method("spawn")
    # 提示用户输入视频路径
    video_path = input(f"{tr['Main']['InputVideo']}").strip()
    # 提示用户输入字幕区域
    try:
        y_min, y_max, x_min, x_max = map(int, input(
            f"{tr['Main']['ChooseSubArea']} (ymin ymax xmin xmax)：").split())
        subtitle_area = SubtitleArea(y_min, y_max, x_min, x_max)
    except ValueError as e:
        subtitle_area = None
    # 新建字幕提取对象
    se = SubtitleExtractor(video_path)
    se.sub_area = subtitle_area
    # 开始提取字幕
    se.run()
