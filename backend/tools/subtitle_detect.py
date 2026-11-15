
import os

from backend.tools.paddle_model_config import PaddleModelConfig
from backend.tools.hardware_accelerator import HardwareAccelerator
import numpy as np

try:
    from paddleocr import TextDetection
except ImportError:
    TextDetection = None

# 长边超过这个值就把检测输入等比缩小（limit_type='max'）。0 = 不缩放（默认）。
# 实测结论（4K 样片、RTX 5060 Ti）：把 3488x480 的字幕条缩到 1920 长边，单帧 det
# 从 71ms 降到 36ms，但整片反而从 49s 变成 60s——缩小后 JPEG 噪声/边缘被放大成
# 伪文本框（识别行数 122 → 146），多出来的 rec 把省下的 det 时间吃光，还多丢了一条
# 正常字幕。所以默认保持原尺寸，只在专门需要时用 VSE_DET_LIMIT 打开。
DET_LIMIT_SIDE_LEN = 0


class SubtitleDetect:
    """
    文本框检测类，用于检测视频帧中是否存在文本框
    """

    def __init__(self, limit_side_len=None, limit_type=None):
        hardware_accelerator = HardwareAccelerator.instance()
        model_config = PaddleModelConfig(hardware_accelerator)
        # 使用 TextDetection 公开 API（PaddleOCR 3.x）
        kwargs = {}
        if model_config.DET_MODEL_PATH:
            kwargs['model_dir'] = model_config.DET_MODEL_PATH
        if model_config.DET_MODEL_NAME:
            kwargs['model_name'] = model_config.DET_MODEL_NAME
        if hardware_accelerator.has_cuda():
            kwargs['device'] = 'gpu:0'
        else:
            # 同 ocr.py：CPU 上关闭 oneDNN，规避 Paddle 3.x 的 PIR 属性转换报错
            kwargs['enable_mkldnn'] = False
        # 环境变量优先，方便 A/B 对比；0 表示不缩放（保持历史行为）
        if limit_side_len is None:
            env = os.environ.get('VSE_DET_LIMIT')
            limit_side_len = int(env) if env else DET_LIMIT_SIDE_LEN
        if limit_side_len and int(limit_side_len) > 0:
            kwargs['limit_side_len'] = int(limit_side_len)
            kwargs['limit_type'] = limit_type or 'max'
        self.text_detector = TextDetection(**kwargs)

    def detect_subtitle(self, img):
        """
        检测图像中的文本框
        :param img: 输入图像
        :return: (dt_boxes, elapse) dt_boxes为numpy数组，elapse为耗时
        """
        results = list(self.text_detector.predict(img))
        if not results:
            return np.array([]), 0
        res = results[0]
        dt_polys = res.get('dt_polys', np.array([]))
        return dt_polys, 0
