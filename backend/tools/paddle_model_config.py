import os

from backend.config import BASE_DIR, config


class PaddleModelConfig:
    """
    文本检测 / 文本识别模型配置（仅 PP-OCRv6，不再依赖 PP-OCRv5）。

    PP-OCRv6（2026-06 发布）提供 tiny / small / medium 三档模型：
      - fast 模式     -> PP-OCRv6_small  （速度对标 PP-OCRv5_mobile，精度更高）
      - accurate 模式 -> PP-OCRv6_medium （速度对标 PP-OCRv5_server，精度更高）

    medium/small 的识别模型为单一多语种模型，覆盖简体中文、繁体中文、英文、日文
    以及 46 种拉丁语系；tiny 不含日文字符集，日文场景会自动提升到 small。

    韩语、阿拉伯语系、西里尔语系、天城文、泰语、希腊语、泰米尔语、泰卢固语、
    卡纳达语、格鲁吉亚语 没有 v6 专用模型，这些语言统一走 v6 通用识别模型，
    识别效果有限，运行时会在日志里给出提示（不再回退到 PP-OCRv5）。
    """

    # 模型文件根目录
    MODEL_BASE = os.path.join(BASE_DIR, 'models')

    # PP-OCRv6 模型目录
    V6_DIR = 'V6'
    V6_TIERS = ('tiny', 'small', 'medium')
    # 识别模式 -> v6 档位
    V6_TIER_BY_MODE = {'fast': 'small', 'accurate': 'medium', 'auto': 'medium'}

    # v6 通用识别模型未覆盖的语言（无专用模型可用）
    ARABIC_LANG = ['ar', 'fa', 'ug', 'ur', 'ps', 'sd', 'bal']
    CYRILLIC_LANG = [
        'ru', 'rs_cyrillic', 'be', 'bg', 'uk', 'mn', 'abq', 'ady', 'kbd', 'ava',
        'dar', 'inh', 'che', 'lbe', 'lez', 'tab', 'cyrillic',
        'sr', 'kk', 'ky', 'tg', 'mk', 'tt', 'cv', 'ba', 'mhr', 'mo',
        'udm', 'kv', 'os', 'bua', 'xal', 'tyv', 'sah', 'kaa',
    ]
    DEVANAGARI_LANG = [
        'hi', 'mr', 'ne', 'bh', 'mai', 'ang', 'bho', 'mah', 'sck', 'new', 'gom',
        'sa', 'bgc', 'devanagari',
    ]
    OTHER_UNCOVERED_LANG = ['korean', 'ko', 'th', 'el', 'ta', 'te', 'kn', 'ka']
    UNCOVERED_LANG = frozenset(
        ARABIC_LANG + CYRILLIC_LANG + DEVANAGARI_LANG + OTHER_UNCOVERED_LANG
    )

    def __init__(self, hardware_accelerator):
        self.hardware_accelerator = hardware_accelerator
        # 设置识别语言
        self.REC_CHAR_TYPE = config.language.value

        # 实际使用的模型版本 / 档位
        self.MODEL_VERSION = 'V6'
        self.MODEL_TIER = self._resolve_tier()

        # 初始化模型路径与模型名
        # （路径为 None 表示本地缺模型，交由 PaddleOCR 按模型名自动下载）
        self.REC_MODEL_PATH = None
        self.DET_MODEL_PATH = None
        self.DET_MODEL_NAME = None
        self.REC_MODEL_NAME = None

        (
            self.DET_MODEL_PATH, self.REC_MODEL_PATH,
            self.DET_MODEL_NAME, self.REC_MODEL_NAME,
        ) = self._resolve_models()

    def describe(self):
        """用于日志输出的模型描述"""
        det = os.path.basename(self.DET_MODEL_PATH) if self.DET_MODEL_PATH \
            else f'{self.DET_MODEL_NAME}(auto)'
        rec = os.path.basename(self.REC_MODEL_PATH) if self.REC_MODEL_PATH \
            else f'{self.REC_MODEL_NAME}(auto)'
        return f'{det}  |  {rec}'

    @property
    def LANG_WARNING(self):
        """当前语言没有 PP-OCRv6 专用模型时的提示文案，否则返回 None"""
        if self.REC_CHAR_TYPE in self.UNCOVERED_LANG:
            return (f'提示：PP-OCRv6 未提供「{self.REC_CHAR_TYPE}」的专用识别模型，'
                    f'将使用多语种通用模型，该语言的识别准确率可能偏低')
        return None

    def _resolve_tier(self):
        """
        识别模式 -> v6 档位。
        auto：有 CUDA 时用 medium，否则用 small（与 README 中“CPU 用轻量、GPU 用精准”一致）。
        环境变量 VSE_OCR_TIER（tiny/small/medium）可强制指定档位，便于按机器性能调节。
        """
        forced = os.environ.get('VSE_OCR_TIER', '').strip().lower()
        if forced in self.V6_TIERS:
            tier = forced
        else:
            mode = config.mode.value
            if mode == 'auto':
                tier = 'medium' if self.hardware_accelerator.has_cuda() else 'small'
            else:
                tier = self.V6_TIER_BY_MODE.get(mode, 'medium')
        # PP-OCRv6_tiny 不含日文字符集，日文强制回退到 small
        if tier == 'tiny' and self.REC_CHAR_TYPE == 'japan':
            tier = 'small'
        return tier

    # ------------------------------------------------------------------ #
    # 路径解析
    # ------------------------------------------------------------------ #
    def _resolve_models(self):
        """解析检测 / 识别模型，返回 (det_path, rec_path, det_name, rec_name)"""
        tier = self.MODEL_TIER
        det_path, det_name = self._find_v6_model('det', tier)
        rec_path, rec_name = self._find_v6_model('rec', tier)
        return (
            det_path or None,
            rec_path or None,
            det_name or f'PP-OCRv6_{tier}_det',
            rec_name or f'PP-OCRv6_{tier}_rec',
        )

    def _find_v6_model(self, kind, tier):
        """
        在 backend/models/V6 中找到指定档位的模型目录（kind: det / rec）。
        目标档位缺失时按 medium -> small -> tiny 的精度降序兜底，返回 (path, name)。
        """
        for t in [tier] + [x for x in ('medium', 'small', 'tiny') if x != tier]:
            path = os.path.join(self.MODEL_BASE, self.V6_DIR, f'PP-OCRv6_{t}_{kind}_infer')
            if os.path.exists(path):
                name = self._read_model_name_from_yaml(path) or f'PP-OCRv6_{t}_{kind}'
                return path, name
        return None, None

    # ------------------------------------------------------------------ #
    @staticmethod
    def _read_model_name_from_yaml(model_dir):
        """从 inference.yml 中读取 Global.model_name"""
        yaml_path = os.path.join(model_dir, 'inference.yml')
        if not os.path.exists(yaml_path):
            return None
        try:
            with open(yaml_path, 'r', encoding='utf-8') as f:
                in_global = False
                for line in f:
                    stripped = line.strip()
                    if stripped == 'Global:':
                        in_global = True
                        continue
                    if in_global:
                        if stripped and not stripped.startswith('#') and ':' in stripped:
                            if stripped.startswith('model_name:'):
                                return stripped.split(':', 1)[1].strip().strip('"').strip("'")
                        # 遇到下一个顶级 section 则退出
                        if stripped and not stripped.startswith('model_name') and not stripped.startswith(' ') and stripped.endswith(':'):
                            break
        except Exception:
            pass
        return None
