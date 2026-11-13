import os

from backend.config import BASE_DIR, config


class PaddleModelConfig:
    """
    文本检测 / 文本识别模型配置。

    模型优先级：PP-OCRv6（2026-06 发布的最新一代）> PP-OCRv5（语言专用兜底）。

    PP-OCRv6 提供 tiny(1.5M) / small(7.7M) / medium(34.5M) 三档模型：
      - fast 模式     -> PP-OCRv6_small  （速度对标 PP-OCRv5_mobile，精度更高）
      - accurate 模式 -> PP-OCRv6_medium （速度对标 PP-OCRv5_server，精度更高）

    PP-OCRv6 medium/small 单一模型支持 50 种语言（简体中文、繁体中文、英文、日文
    + 46 种拉丁语系），但不包含韩语、阿拉伯语、西里尔语系、天城文、泰语、希腊语、
    泰米尔语、泰卢固语，这些语言仍回退到 PP-OCRv5 的专用 mobile 识别模型。
    """

    # 模型文件根目录
    MODEL_BASE = os.path.join(BASE_DIR, 'models')

    # PP-OCRv6 模型目录
    V6_DIR = 'V6'
    V6_TIERS = ('tiny', 'small', 'medium')
    # 识别模式 -> v6 档位
    V6_TIER_BY_MODE = {'fast': 'small', 'accurate': 'medium', 'auto': 'medium'}
    # v6 单模型覆盖的语言（medium/small 为 50 种；tiny 少日文，本处不使用 tiny）
    V6_CORE_LANG = ['ch', 'chinese_cht', 'en', 'japan']

    # PP-OCRv5 语言专用兜底模型目录
    V5_DIR = 'V5'

    LATIN_LANG = [
        'af', 'az', 'bs', 'cs', 'cy', 'da', 'de', 'es', 'et', 'fr', 'ga', 'hr',
        'hu', 'id', 'is', 'it', 'ku', 'la', 'lt', 'lv', 'mi', 'ms', 'mt', 'nl',
        'no', 'oc', 'pi', 'pl', 'pt', 'ro', 'rs_latin', 'sk', 'sl', 'sq', 'sv',
        'sw', 'tl', 'tr', 'uz', 'vi', 'latin', 'german', 'french',
        'fi', 'eu', 'gl', 'lb', 'rm', 'ca', 'qu',
    ]
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
    OTHER_LANG = [
        'ch', 'japan', 'korean', 'en', 'ta', 'kn', 'te', 'ka',
        'chinese_cht',
    ]
    MULTI_LANG = (LATIN_LANG + ARABIC_LANG + CYRILLIC_LANG
                  + DEVANAGARI_LANG + OTHER_LANG)

    def __init__(self, hardware_accelerator):
        self.hardware_accelerator = hardware_accelerator
        # 设置识别语言
        self.REC_CHAR_TYPE = config.language.value

        # PP-OCRv5 时代识别模型的输入尺寸，保留以兼容外部引用
        self.REC_IMAGE_SHAPE = '3,48,320'

        # 实际使用的模型版本 / 档位
        self.MODEL_VERSION = 'V6'
        self.MODEL_TIER = self._resolve_tier()

        # 初始化模型路径与模型名（None 路径表示本地缺模型，交由 PaddleOCR 按 name 自动下载）
        self.REC_MODEL_PATH = None
        self.DET_MODEL_PATH = None
        self.DET_MODEL_NAME = None
        self.REC_MODEL_NAME = None

        (
            self.DET_MODEL_PATH, self.REC_MODEL_PATH,
            self.DET_MODEL_NAME, self.REC_MODEL_NAME,
            self.MODEL_VERSION, self.MODEL_TIER,
        ) = self._resolve_models()

    def describe(self):
        """用于日志输出的模型描述"""
        det = os.path.basename(self.DET_MODEL_PATH) if self.DET_MODEL_PATH else f'{self.DET_MODEL_NAME}(auto)'
        rec = os.path.basename(self.REC_MODEL_PATH) if self.REC_MODEL_PATH else f'{self.REC_MODEL_NAME}(auto)'
        return f'{det}  |  {rec}'

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
        """
        解析模型，返回 (det_path, rec_path, det_name, rec_name, version, tier)。
        检测模型与识别模型可以来自不同版本：检测模型与语言无关，优先 v6；
        识别模型在 v6 覆盖的语言上优先 v6，其余语言回退 v5 专用模型。
        """
        tier = self.MODEL_TIER

        det_path, det_name, det_version = self._resolve_det(tier)
        rec_path, rec_name, rec_version = self._resolve_rec(tier)

        # 版本标记：以识别模型为准（识别模型才与语言相关）
        version = rec_version or det_version or 'V6'
        return det_path, rec_path, det_name, rec_name, version, tier

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

    def _resolve_det(self, tier):
        """检测模型：v6（同档位优先）-> v5"""
        # 1) PP-OCRv6 检测模型
        det_path, det_name = self._find_v6_model('det', tier)
        if det_path:
            return det_path, det_name, 'V6'

        # 2) 退回 v5：fast -> mobile det，其余 -> server det
        v5_base = os.path.join(self.MODEL_BASE, self.V5_DIR)
        det_dir_name = 'PP-OCRv5_mobile_det_infer' if config.mode.value == 'fast' \
            else 'PP-OCRv5_server_det_infer'
        det_path = os.path.join(v5_base, det_dir_name)
        if os.path.exists(det_path):
            return det_path, self._read_model_name_from_yaml(det_path) or det_dir_name, 'V5'

        # 3) 本地都没有：让 PaddleOCR 按模型名自动下载对应 v6 模型
        return None, f'PP-OCRv6_{tier}_det', 'V6'

    def _resolve_rec(self, tier):
        """识别模型：v6（语言覆盖内）-> v5 语言专用"""
        # 1) v6 覆盖的语言直接使用 v6 单模型
        if self.REC_CHAR_TYPE in (self.V6_CORE_LANG + self.LATIN_LANG):
            rec_path, rec_name = self._find_v6_model('rec', tier)
            if rec_path:
                return rec_path, rec_name, 'V6'
            # v6 未下载则继续尝试 v5 专用模型

        # 2) v6 未覆盖的语言 / v6 缺失：使用 v5 语言专用 mobile 模型
        v5_path = self._resolve_v5_rec()
        if v5_path:
            return v5_path, self._read_model_name_from_yaml(v5_path) or os.path.basename(v5_path), 'V5'

        # 3) 兜底：v6 通用模型（单模型覆盖 50 种语言，未覆盖语言识别效果有限）
        rec_path, rec_name = self._find_v6_model('rec', tier)
        if rec_path:
            return rec_path, rec_name, 'V6'

        return None, f'PP-OCRv6_{tier}_rec', 'V6'

    def _resolve_v5_rec(self):
        """PP-OCRv5 语言专用识别模型路径"""
        v5_base = os.path.join(self.MODEL_BASE, self.V5_DIR)
        lang = self.REC_CHAR_TYPE

        # 简繁中、英、日：通用模型，fast 用 mobile，accurate 用 server
        if lang in ('ch', 'chinese_cht', 'en', 'japan'):
            if config.mode.value == 'fast':
                path = os.path.join(v5_base, 'PP-OCRv5_mobile_rec_infer')
                if os.path.exists(path):
                    return path
            path = os.path.join(v5_base, 'PP-OCRv5_server_rec_infer')
            if os.path.exists(path):
                return path
            path = os.path.join(v5_base, 'PP-OCRv5_mobile_rec_infer')
            return path if os.path.exists(path) else None

        dir_name = self._get_v5_rec_model_name(lang)
        if not dir_name:
            return None
        path = os.path.join(v5_base, dir_name if dir_name.endswith('_infer') else f'{dir_name}_infer')
        return path if os.path.exists(path) else None

    def _get_v5_rec_model_name(self, lang):
        """
        根据语言获取 PP-OCRv5 识别模型目录名
        参考: https://www.paddleocr.ai/main/version3.x/algorithm/PP-OCRv5/PP-OCRv5_multi_languages.html
        """
        if lang in ('korean', 'ko'):
            return 'korean_PP-OCRv5_mobile_rec_infer'
        elif lang in self.LATIN_LANG:
            return 'latin_PP-OCRv5_mobile_rec_infer'
        elif lang in self.ARABIC_LANG:
            return 'arabic_PP-OCRv5_mobile_rec_infer'
        elif lang in self.CYRILLIC_LANG:
            return 'cyrillic_PP-OCRv5_mobile_rec_infer'
        elif lang in self.DEVANAGARI_LANG:
            return 'devanagari_PP-OCRv5_mobile_rec_infer'
        elif lang == 'th':
            return 'th_PP-OCRv5_mobile_rec_infer'
        elif lang == 'el':
            return 'el_PP-OCRv5_mobile_rec_infer'
        elif lang == 'ta':
            return 'ta_PP-OCRv5_mobile_rec_infer'
        elif lang == 'te':
            return 'te_PP-OCRv5_mobile_rec_infer'
        return None

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
