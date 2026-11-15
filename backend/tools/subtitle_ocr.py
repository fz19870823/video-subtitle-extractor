import os
import re
import time
import traceback
from multiprocessing import Queue, Process
import cv2
from PIL import ImageFont, ImageDraw, Image
from tqdm import tqdm
from backend.tools.ocr import OcrRecogniser, get_coordinates
from backend.tools.constant import SubtitleArea
from backend.bean.subtitle_area import SubtitleArea as SubtitleAreaRect
from backend.tools import constant
from threading import Thread
import queue
from types import SimpleNamespace
import shutil
import numpy as np
from collections import namedtuple
from backend.config import tr


def extract_subtitles(data, text_recogniser, img, raw_subtitles,
                      sub_area, options, dt_box_arg, rec_res_arg, ocr_loss_debug_path,
                      sub_area_local=None):
    """
    提取视频帧中的字幕信息
    """
    # 从参数中获取检测框与检测结果
    dt_box = dt_box_arg
    rec_res = rec_res_arg
    # 上游（精确模式）给出结果时，框是整帧坐标，越界判定用整帧选区；
    # 没给则下面在裁剪图上自己检测，坐标是局部坐标，要用平移后的选区，否则整片都会被判越界。
    # 如果没有检测结果，则获取检测结果
    if dt_box is None or rec_res is None:
        dt_box, rec_res = text_recogniser.predict(img)
        check_area = sub_area_local if sub_area_local is not None else sub_area
    else:
        check_area = sub_area
    # rec_res格式为： ("hello", 0.997)
    # 获取文本坐标
    coordinates = get_coordinates(dt_box)
    # 将结果写入txt文本中
    if options.REC_CHAR_TYPE == 'en':
        # 如果识别语言为英文，则去除中文
        text_res = [(re.sub('[\u4e00-\u9fa5]', '', res[0]), res[1]) for res in rec_res]
    else:
        text_res = [(res[0], res[1]) for res in rec_res]
    line = ''
    loss_list = []
    for content, coordinate in zip(text_res, coordinates):
        text = content[0]
        prob = content[1]
        if check_area is not None:
            selected = False
            # 初始化超界偏差为0
            overflow_area_rate = 0
            # 使用AABB矩形重叠判断（比Shapely Polygon快得多）
            c_xmin, c_xmax, c_ymin, c_ymax = coordinate
            # 计算交集矩形
            inter_xmin = max(check_area.xmin, c_xmin)
            inter_ymin = max(check_area.ymin, c_ymin)
            inter_xmax = min(check_area.xmax, c_xmax)
            inter_ymax = min(check_area.ymax, c_ymax)
            has_intersection = inter_xmin < inter_xmax and inter_ymin < inter_ymax
            drop_reason = ''
            # 如果有交集
            if has_intersection:
                sub_area_w = check_area.xmax - check_area.xmin
                sub_area_h = check_area.ymax - check_area.ymin
                sub_area_size = sub_area_w * sub_area_h
                inter_area = (inter_xmax - inter_xmin) * (inter_ymax - inter_ymin)
                coord_area = (c_xmax - c_xmin) * (c_ymax - c_ymin)
                # 计算越界允许偏差
                overflow_area_rate = ((sub_area_size + coord_area - inter_area) / sub_area_size) - 1
                # 如果越界比例低于设定阈值且该行文本识别的置信度高于设定阈值
                not_overflow = overflow_area_rate <= options.SUB_AREA_DEVIATION_RATE
                confident = prob > options.DROP_SCORE
                if not_overflow and confident:
                    # 保留该帧
                    selected = True
                    line += f'{str(data["i"]).zfill(8)}\t{coordinate}\t{text}\n'
                    raw_subtitles.append(f'{str(data["i"]).zfill(8)}\t{coordinate}\t{text}\n')
                else:
                    if not not_overflow:
                        drop_reason = tr['Main']['OcrDropOutOfBoxRate'].format(int(options.SUB_AREA_DEVIATION_RATE * 100), int(overflow_area_rate * 100))
                    elif not confident:
                        drop_reason = tr['Main']['OcrDropConfidentLow'].format(int(options.DROP_SCORE * 100))
            else:
                drop_reason = tr['Main']['OcrDropNoIntercetion']
            if drop_reason:
                tqdm.write(tr['Main']['OcrResultWithDropReason'].format(text, round(prob * 100,1), drop_reason))
            else:
                tqdm.write(tr['Main']['OcrResult'].format(text, round(prob * 100,1)))
            # 保存丢掉的识别结果
            loss_info = namedtuple('loss_info', 'text prob overflow_area_rate coordinate selected')
            loss_list.append(loss_info(text, prob, overflow_area_rate, coordinate, selected))
        else:
            raw_subtitles.append(f'{str(data["i"]).zfill(8)}\t{coordinate}\t{text}\n')
    # 输出调试信息
    dump_debug_info(options, line, img, loss_list, ocr_loss_debug_path, sub_area, data)


def dump_debug_info(options, line, img, loss_list, ocr_loss_debug_path, sub_area, data):
    # 精确模式下上游已给出识别结果，这里不再解码视频帧（img 为 None），无法画调试图
    if img is None:
        return
    loss = False
    if options.DEBUG_OCR_LOSS and options.REC_CHAR_TYPE in ('ch', 'japan ', 'korea', 'ch_tra'):
        loss = len(line) > 0 and re.search(r'[\u4e00-\u9fa5\u3400-\u4db5\u3130-\u318F\uAC00-\uD7A3\u0800-\u4e00]', line) is None
    if loss:
        if not os.path.exists(ocr_loss_debug_path):
            os.makedirs(ocr_loss_debug_path, mode=0o777, exist_ok=True)
        img = cv2.rectangle(img, (sub_area.xmin, sub_area.ymin), (sub_area.xmax, sub_area.ymax), constant.BGR_COLOR_BLUE, 2)
        for loss_info in loss_list:
            coordinate = loss_info.coordinate
            color = constant.BGR_COLOR_GREEN if loss_info.selected else constant.BGR_COLOR_RED
            text = f"[{loss_info.text}] prob:{loss_info.prob:.4f} or:{loss_info.overflow_area_rate:.2f}"
            img = paint_chinese_opencv(img, text, pos=(coordinate[0], coordinate[2] - 30), color=color)
            img = cv2.rectangle(img, (coordinate[0], coordinate[2]), (coordinate[1], coordinate[3]), color, 2)
        cv2.imwrite(os.path.join(os.path.abspath(ocr_loss_debug_path), f'{str(data["i"]).zfill(8)}.png'), img)


FONT_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'NotoSansCJK-Bold.otf')
FONT = ImageFont.truetype(FONT_PATH, 20)


def paint_chinese_opencv(im, chinese, pos, color):
    img_pil = Image.fromarray(im)
    fill_color = color  # (color[2], color[1], color[0])
    position = pos
    draw = ImageDraw.Draw(img_pil)
    draw.text(position, chinese, font=FONT, fill=fill_color)
    img = np.array(img_pil)
    return img


def ocr_task_consumer(ocr_queue, raw_subtitle_path, sub_area, video_path, options, progress_queue,
                       sub_area_local=None):
    """
    消费者： 消费ocr_queue，将ocr队列中的数据取出，进行ocr识别，写入字幕文件中
    :param ocr_queue (current_frame_no当前帧帧号, frame 视频帧, dt_box检测框, rec_res识别结果)
    :param raw_subtitle_path
    :param sub_area
    :param video_path
    :param options
    """
    data = {'i': 1}
    # 文本识别对象延迟创建：精确模式下上游已给出 dt_box/rec_res，这里只是整理文本，
    # 不需要模型。提前创建会让子进程/线程白白加载一套 paddle（打包版同进程运行时会
    # 和主进程的 predictor 抢 CUDA，实测直接死锁），也让启动快很多。
    text_recogniser = None
    # 一次识别的帧数。攒批是"帧级"批量：把 N 张同尺寸的字幕条一次性送进 det+rec，
    # 减少 Python 侧调用与显存同步次数。N 取自配置 MaxBatchSize。
    batch_size = int(getattr(options, 'MAX_BATCH_SIZE', 1) or 1)
    batch_size = max(1, min(batch_size, 32))
    # 丢失字幕的存储路径
    ocr_loss_debug_path = os.path.join(os.path.abspath(os.path.splitext(video_path)[0]), 'loss')
    # 删除之前的缓存垃圾
    if os.path.exists(ocr_loss_debug_path):
        shutil.rmtree(ocr_loss_debug_path, True)

    raw_subtitles = []
    processed_count = 0

    def process_items(items):
        """处理一批帧：需要识别的合并成一次 predict，再把结果按原顺序写回。"""
        nonlocal text_recogniser, processed_count
        need_idx = [i for i, it in enumerate(items) if it[2] is None or it[3] is None]
        result_by_idx = {}
        if need_idx:
            if text_recogniser is None:
                text_recogniser = OcrRecogniser(batch_size=batch_size)
                text_recogniser.hardware_accelerator = options.HARDWARD_ACCELERATOR
            frames = [items[i][1] for i in need_idx]
            results = text_recogniser.predict_batch(frames)
            result_by_idx = dict(zip(need_idx, results))
        for i, (frame_no, frame, dt_box, rec_res) in enumerate(items):
            if i in result_by_idx:
                dt_box, rec_res = result_by_idx[i]
            data['i'] = frame_no
            extract_subtitles(data, text_recogniser, frame, raw_subtitles, sub_area, options, dt_box,
                              rec_res, ocr_loss_debug_path, sub_area_local)
            processed_count += 1
            progress_queue.put((frame_no, processed_count))

    try:
        while True:
            try:
                first = ocr_queue.get(block=True)
                if first[0] == -1:
                    # frame 是生产者统计的总帧数
                    total_tasks = first[1] if first[1] is not None else processed_count
                    progress_queue.put((-1, total_tasks))
                    return
                items = [first]
                end_item = None
                if batch_size > 1:
                    # 攒满一批再识别；队列暂时空时最多等 50ms，避免尾部几帧拖住收尾。
                    deadline = time.monotonic() + 0.05
                    while len(items) < batch_size:
                        try:
                            item = ocr_queue.get_nowait()
                        except queue.Empty:
                            if time.monotonic() >= deadline:
                                break
                            time.sleep(0.002)
                            continue
                        if item[0] == -1:
                            end_item = item
                            break
                        items.append(item)
                process_items(items)
                if end_item is not None:
                    total_tasks = end_item[1] if end_item[1] is not None else processed_count
                    progress_queue.put((-1, total_tasks))
                    return
            except Exception as e:
                # 这里不能只 print(e)：打包版 stdout 是 GBK 时 print 自身就会抛
                # UnicodeEncodeError，把整个消费线程打死而生产者继续入队 →
                # 队列撑满，主进程 join 永久阻塞。打完整栈便于定位。
                traceback.print_exc()
                progress_queue.put(-1)
                break
    finally:
        with open(raw_subtitle_path, mode='w+', encoding='utf-8') as raw_subtitle_file:
            for line in raw_subtitles:
                raw_subtitle_file.write(line)


def ocr_task_producer(ocr_queue, task_queue, progress_queue, video_path, raw_subtitle_path,
                      crop_rect=None):
    """
    生产者：负责生产用于OCR识别的数据，将需要进行ocr识别的数据加入ocr_queue中
    :param ocr_queue (current_frame_no当前帧帧号, frame 视频帧, dt_box检测框, rec_res识别结果)
    :param task_queue (total_frame_count总帧数, current_frame_no当前帧帧号, dt_box检测框, rec_res识别结果, subtitle_area字幕区域)
    :param progress_queue
    :param video_path
    :param raw_subtitle_path
    """
    cap = cv2.VideoCapture(video_path)
    tbar = None
    frame_count = 0

    def _put(item):
        """带超时入队：消费者若已退出，这里不能永久阻塞（否则主进程 join 挂死）"""
        try:
            ocr_queue.put(item, timeout=60)
            return True
        except queue.Full:
            print('[subtitle_ocr] 消费线程疑似已退出，停止入队', flush=True)
            return False

    while True:
        try:
            # 从任务队列中提取任务信息
            total_frame_count, current_frame_no, dt_box, rec_res, total_ms, default_subtitle_area = task_queue.get(block=True)
            if tbar is None:
                tbar = tqdm(total=round(total_frame_count), position=1)
            # current_frame 等于-1说明所有视频帧已经读完
            if current_frame_no == -1:
                # ocr识别队列加入结束标志，附带总帧数
                if _put((-1, frame_count, None, None)):
                    # 通过 progress_queue 提前通知总帧数，让主进程可以精确计算进度
                    progress_queue.put((-2, frame_count))
                # 更新进度条
                tbar.update(tbar.total - tbar.n)
                break
            tbar.update(round(current_frame_no - tbar.n))
            # 如果上游（精确模式）已经给了检测框和识别结果，这一帧就不需要再解码：
            # 4K 片源随机 seek 一次约 450ms，而这里只是写文件，没必要重复提取。
            if dt_box is not None and rec_res is not None:
                if not _put((current_frame_no, None, dt_box, rec_res)):
                    break
                continue
            # 设置当前视频帧
            # 如果total_ms不为空，则使用了VSF提取字幕
            if total_ms is not None:
                cap.set(cv2.CAP_PROP_POS_MSEC, total_ms)
            else:
                cap.set(cv2.CAP_PROP_POS_FRAMES, current_frame_no - 1)
            # 读取视频帧
            ret, frame = cap.read()
            # 如果读取成功
            if ret:
                frame_count += 1
                # 优先按用户选区裁剪：det 只看到字幕条，不会把选区外的招牌/台标
                # 也检出来再丢掉（快速模式下这类无谓识别曾占全部识别结果的 59%）。
                # 没有选区时才退回按上/下半帧的粗略裁剪。
                if crop_rect is not None:
                    x0, y0, x1, y1 = crop_rect
                    frame = frame[y0:y1, x0:x1]
                elif default_subtitle_area is not None:
                    frame = frame_preprocess(default_subtitle_area, frame)
                # print(f"current_frame_no: {current_frame_no}")
                if not _put((current_frame_no, frame, dt_box, rec_res)):
                    break
        except Exception as e:
            traceback.print_exc()
            break
    cap.release()


def subtitle_extract_handler(task_queue, progress_queue, video_path, raw_subtitle_path, sub_area, options):
    """
    创建并开启一个视频帧提取线程与一个ocr识别线程
    :param task_queue 任务队列，(total_frame_count总帧数, current_frame_no当前帧, dt_box检测框, rec_res识别结果, subtitle_area字幕区域)
    :param progress_queue 进度队列
    :param video_path 视频路径
    :param raw_subtitle_path 原始字幕文件路径
    :param sub_area 字幕区域
    :param options 选项
    """
    # 删除缓存
    if os.path.exists(raw_subtitle_path):
        os.remove(raw_subtitle_path)

    # 计算裁剪矩形与"平移后的选区"，供自识别回退路径使用。
    # 子进程原先只按 SubtitleArea 预设裁上/下半帧，det 会把半个 4K 画面上的招牌、
    # 台标全检出来，再逐条按"超出选区"丢掉——纯浪费算力，日志也全是这类丢弃。
    # 现在统一裁到「选区 + 16px」，坐标系一起平移到裁剪图里。
    crop_rect = None
    sub_area_local = None
    try:
        cap_probe = cv2.VideoCapture(video_path)
        if cap_probe.isOpened():
            probe_w = int(cap_probe.get(cv2.CAP_PROP_FRAME_WIDTH))
            probe_h = int(cap_probe.get(cv2.CAP_PROP_FRAME_HEIGHT))
            crop_rect = area_crop_rect(sub_area, probe_w, probe_h)
        cap_probe.release()
        if crop_rect is not None:
            ox, oy = crop_rect[0], crop_rect[1]
            sub_area_local = SubtitleAreaRect(sub_area.ymin - oy, sub_area.ymax - oy,
                                              sub_area.xmin - ox, sub_area.xmax - ox)
    except Exception:
        # 裁剪只是省算力的优化，失败就退回原有整帧/半帧路径，不能影响提取
        traceback.print_exc()
        crop_rect, sub_area_local = None, None

    # 创建一个OCR队列，大小建议值8-20
    ocr_queue = queue.Queue(20)
    # 创建一个OCR事件生产者线程
    ocr_event_producer_thread = Thread(target=ocr_task_producer,
                                       args=(ocr_queue, task_queue, progress_queue, video_path, raw_subtitle_path,
                                             crop_rect),
                                       daemon=True)
    # 创建一个OCR事件消费者提取线程
    ocr_event_consumer_thread = Thread(target=ocr_task_consumer,
                                       args=(ocr_queue, raw_subtitle_path, sub_area, video_path, options,
                                             progress_queue, sub_area_local),
                                       daemon=True)
    # 开启消费者线程
    ocr_event_producer_thread.start()
    # 开启生产者线程
    ocr_event_consumer_thread.start()
    # join方法让主线程任务结束之后，进入阻塞状态，一直等待其他的子线程执行结束之后，主线程再终止
    ocr_event_producer_thread.join()
    ocr_event_consumer_thread.join()


def async_start(video_path, raw_subtitle_path, sub_area, options):
    """
    开始进程处理异步任务
    options.REC_CHAR_TYPE
    options.DROP_SCORE
    options.SUB_AREA_DEVIATION_RATE
    options.DEBUG_OCR_LOSS
    options.HARDWARD_ACCELERATOR
    """
    assert 'REC_CHAR_TYPE' in options, "options缺少参数：REC_CHAR_TYPE"
    assert 'DROP_SCORE' in options, "options缺少参数: DROP_SCORE'"
    assert 'SUB_AREA_DEVIATION_RATE' in options, "options缺少参数: SUB_AREA_DEVIATION_RATE"
    assert 'DEBUG_OCR_LOSS' in options, "options缺少参数: DEBUG_OCR_LOSS"
    assert 'HARDWARD_ACCELERATOR' in options, "options缺少参数: HARDWARD_ACCELERATOR"
    # 创建一个任务队列
    # 任务格式为：(total_frame_count总帧数, current_frame_no当前帧, dt_box检测框, rec_res识别结果, subtitle_area字幕区域)
    task_queue = Queue()
    # 创建一个进度更新队列
    progress_queue = Queue()
    # 新建一个进程
    p = Process(target=subtitle_extract_handler,
                args=(task_queue, progress_queue, video_path, raw_subtitle_path, sub_area, SimpleNamespace(**options),))
    # 启动进程
    p.start()
    return p, task_queue, progress_queue


# 与主进程 _detect_subtitle_in_area / _iter_strip_frames 保持一致的外扩像素：
# 留白是为了不把贴着选区边界的字幕笔画切掉，det 需要看到完整字形才能识别正确。
AREA_PAD = 16


def area_crop_rect(sub_area, frame_w, frame_h, pad=AREA_PAD):
    """
    选区 → 裁剪矩形 (x0, y0, x1, y1)，外扩 pad 并夹到画面内。
    sub_area 为 None（未指定选区）时返回 None，调用方退回原有整帧/半帧逻辑。
    """
    if sub_area is None or frame_w <= 0 or frame_h <= 0:
        return None
    x0 = max(0, int(sub_area.xmin) - pad)
    y0 = max(0, int(sub_area.ymin) - pad)
    x1 = min(int(frame_w), int(sub_area.xmax) + pad)
    y1 = min(int(frame_h), int(sub_area.ymax) + pad)
    if x1 - x0 < 8 or y1 - y0 < 8:
        return None
    return x0, y0, x1, y1


def frame_preprocess(subtitle_area, frame):
    """
    将视频帧进行裁剪
    """
    # 对于分辨率大于1920*1080的视频，将其视频帧进行等比缩放至1280*720进行识别
    # paddlepaddle会将图像压缩为640*640
    # if self.frame_width > 1280:
    #     scale_rate = round(float(1280 / self.frame_width), 2)
    #     frames = cv2.resize(frames, None, fx=scale_rate, fy=scale_rate, interpolation=cv2.INTER_AREA)
    # 如果字幕出现的区域在下部分
    if subtitle_area == SubtitleArea.LOWER_PART:
        cropped = int(frame.shape[0] // 2)
        # 将视频帧切割为下半部分
        frame = frame[cropped:]
    # 如果字幕出现的区域在上半部分
    elif subtitle_area == SubtitleArea.UPPER_PART:
        cropped = int(frame.shape[0] // 2)
        # 将视频帧切割为下半部分
        frame = frame[:cropped]
    return frame


if __name__ == "__main__":
    pass
