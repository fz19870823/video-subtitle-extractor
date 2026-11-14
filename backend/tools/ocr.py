import os
from backend.config import *
import importlib
from paddleocr import PaddleOCR
from backend.tools.hardware_accelerator import HardwareAccelerator
from backend.tools.paddle_model_config import PaddleModelConfig

# 加载文本检测+识别模型
class OcrRecogniser:
    def __init__(self):
        self.recogniser = None
        # rec-only 路径用到的识别子模型（paddlex TextRecRunnerPredictor），首次使用时解析
        self._rec_model = None
        self._rec_model_checked = False
        # 必须用单例：new 一个未初始化的 HardwareAccelerator() 会让 has_cuda() 恒为 False，
        # 导致 init_model() 把 device 写成 'cpu'（精确模式下逐帧 OCR 在 CPU 上跑，约 13s/帧）。
        # subtitle_ocr 子进程会显式覆盖成 options.HARDWARD_ACCELERATOR。
        self.hardware_accelerator = HardwareAccelerator.instance()

    @staticmethod
    def y_round(y):
        y_min = y + 10 - y % 10
        y_max = y - y % 10
        if abs(y - y_min) < abs(y - y_max):
            return y_min
        else:
            return y_max

    def predict(self, image):
        if not self.recogniser:
            self.recogniser = self.init_model()

        # PaddleOCR 3.x: 使用 predict_iter 获取结果
        results = list(self.recogniser.predict_iter(image))
        if not results:
            return [], []

        res = results[0]
        dt_polys = res.get('dt_polys', [])
        rec_texts = res.get('rec_texts', [])
        rec_scores = res.get('rec_scores', [])

        if len(dt_polys) == 0:
            return [], []

        return self._format_result(dt_polys, rec_texts, rec_scores)

    def _format_result(self, dt_polys, rec_texts, rec_scores):
        """
        把 det 的多边形 + rec 的文本/置信度整理成 predict() 的历史返回格式：
        ([[(xmin,ymin),(xmax,ymin),(xmax,ymax),(xmin,ymax)], ...], [(text, score), ...])
        已按行（ymin 对齐）从小到大、行内按 xmin 排序。
        """
        # 将 dt_polys (numpy array, shape (N, points, 2)) 转换为旧的 dt_box 格式
        # 旧格式: [[(x1,y1),(x2,y2),(x3,y3),(x4,y4)], ...]
        dt_box = []
        coordinate_list = []
        for poly in dt_polys:
            points = [(int(p[0]), int(p[1])) for p in poly]
            # 取 AABB 用于排序
            xs = [p[0] for p in points]
            ys = [p[1] for p in points]
            xmin, xmax = min(xs), max(xs)
            ymin, ymax = min(ys), max(ys)
            coordinate_list.append([xmin, xmax, ymin, ymax])
            dt_box.append(points)

        # 将 rec_texts + rec_scores 转换为旧的 rec_res 格式
        rec_res = [(text, float(score)) for text, score in zip(rec_texts, rec_scores)]

        # 计算有多少行字幕，将每行字幕最小的ymin值放入lines
        lines = []
        for i in coordinate_list:
            rounded_y = self.y_round(i[2])
            if not any(abs(rounded_y - line_y) <= 10 for line_y in lines):
                lines.append(rounded_y)
        lines = sorted(lines)

        for i in coordinate_list:
            for j in lines:
                if abs(j - self.y_round(i[2])) <= 10:
                    i[2] = j

        to_rank_res = list(zip(coordinate_list, rec_res, dt_box))
        # 用sorted替代冒泡排序：先按ymin，再按xmin
        ranked_res = sorted(to_rank_res, key=lambda x: (x[0][2], x[0][0]))
        # 重建 dt_box 和 rec_res（排序后）
        sorted_dt_box = []
        sorted_rec_res = []
        for coord, rec, box in ranked_res:
            # 将 coordinate 转换回 4 点格式
            xmin, xmax, ymin, ymax = coord
            sorted_dt_box.append([(xmin, ymin), (xmax, ymin), (xmax, ymax), (xmin, ymax)])
            sorted_rec_res.append(rec)

        return sorted_dt_box, sorted_rec_res

    def _get_rec_model(self):
        """
        从已加载的 PaddleOCR 实例里取出识别子模型，复用它做 rec-only 推理，
        避免为了识别再跑一次 det（精确模式下 det 每帧要额外 ~33ms）。
        """
        if not self._rec_model_checked:
            self._rec_model_checked = True
            try:
                pipeline = self.recogniser.paddlex_pipeline._pipeline
                self._rec_model = pipeline.text_rec_model
            except AttributeError:
                self._rec_model = None
        return self._rec_model

    @staticmethod
    def _extract_crops(image, polys, pad_ratio=0.10, origin=(0, 0)):
        """
        按检测框裁图（向外扩展留识别边界余量），返回 (crops, keep_polys)。
        过小（<4px）或裁空的框会被丢弃。
        :param origin: image 左上角在整帧坐标系中的位置 (ox, oy)。polys 始终使用整帧坐标，
                       传入字幕条等局部图像时用它换算裁剪位置（origin=(0,0) 即整帧，行为不变）
        """
        h_img, w_img = image.shape[:2]
        ox, oy = origin
        crops = []
        keep = []
        for poly in polys:
            xs = [p[0] for p in poly]
            ys = [p[1] for p in poly]
            xmin, xmax = int(min(xs)), int(max(xs))
            ymin, ymax = int(min(ys)), int(max(ys))
            if xmax - xmin < 4 or ymax - ymin < 4:
                continue
            pad_y = max(2, int((ymax - ymin) * pad_ratio))
            pad_x = max(2, int(pad_y * 0.5))
            x0 = max(0, xmin - pad_x - ox)
            x1 = min(w_img, xmax + pad_x - ox)
            y0 = max(0, ymin - pad_y - oy)
            y1 = min(h_img, ymax + pad_y - oy)
            crop = image[y0:y1, x0:x1]
            if crop.size == 0:
                continue
            crops.append(crop)
            keep.append(poly)
        return crops, keep

    def recognise_polys(self, image, polys, pad_ratio=0.10, origin=(0, 0)):
        """
        复用已有检测框，只做识别（跳过 det）。
        :param image: 整帧图像（与 polys 同一坐标系）
        :param polys: 检测框序列，每个元素为 4 点多边形（numpy / list 均可）
        :param pad_ratio: 裁图时按框高的多少比例向外扩展，给识别模型留边界余量
        :param origin: image 左上角在整帧坐标系中的位置；传字幕条等局部图像时使用
        :return: (dt_box, rec_res)，格式与 predict() 完全一致
        """
        if not self.recogniser:
            self.recogniser = self.init_model()

        rec_model = self._get_rec_model()
        if rec_model is None or len(polys) == 0:
            # 拿不到识别子模型时退回完整流程，保证结果正确性
            return self.predict(image)

        crops, keep = self._extract_crops(image, polys, pad_ratio, origin)
        if not crops:
            return [], []

        # paddlex 的 rec predictor 一次可以吃多张图，内部会按 batch 处理
        results = list(rec_model(crops))
        if len(results) != len(crops):
            return self.predict(image)

        rec_texts = [r.get('rec_text', '') for r in results]
        rec_scores = [r.get('rec_score', 0.0) for r in results]
        return self._format_result(keep, rec_texts, rec_scores)

    def init_model(self):
        model_config = PaddleModelConfig(self.hardware_accelerator)

        # PaddleOCR 3.x 使用 device 参数替代 use_gpu
        if self.hardware_accelerator.has_cuda():
            device = 'gpu:0'
        else:
            device = 'cpu'

        kwargs = dict(
            use_doc_orientation_classify=False,
            use_doc_unwarping=False,
            use_textline_orientation=False,
            text_rec_score_thresh=0,
            device=device,
        )
        # Paddle 3.x 在 CPU 上启用 oneDNN + PIR 新执行器时会抛
        # NotImplementedError: ConvertPirAttribute2RuntimeAttribute not support，
        # 关闭 mkldnn 可绕过（GPU 不走该分支，无需设置）
        if device == 'cpu':
            kwargs['enable_mkldnn'] = False
        # 本地有模型目录时优先使用本地模型；缺失时不传 dir，PaddleOCR 会按 model_name 自动下载
        if model_config.DET_MODEL_PATH:
            kwargs['text_detection_model_dir'] = model_config.DET_MODEL_PATH
        if model_config.REC_MODEL_PATH:
            kwargs['text_recognition_model_dir'] = model_config.REC_MODEL_PATH
        if model_config.DET_MODEL_NAME:
            kwargs['text_detection_model_name'] = model_config.DET_MODEL_NAME
        if model_config.REC_MODEL_NAME:
            kwargs['text_recognition_model_name'] = model_config.REC_MODEL_NAME

        return PaddleOCR(**kwargs)


def get_coordinates(dt_box):
    """
    从返回的检测框中获取坐标
    :param dt_box 检测框返回结果
    :return list 坐标点列表
    """
    coordinate_list = list()
    if isinstance(dt_box, list):
        for i in dt_box:
            i = list(i)
            (x1, y1) = int(i[0][0]), int(i[0][1])
            (x2, y2) = int(i[1][0]), int(i[1][1])
            (x3, y3) = int(i[2][0]), int(i[2][1])
            (x4, y4) = int(i[3][0]), int(i[3][1])
            xmin = max(x1, x4)
            xmax = min(x2, x3)
            ymin = max(y1, y2)
            ymax = min(y3, y4)
            coordinate_list.append((xmin, xmax, ymin, ymax))
    return coordinate_list
