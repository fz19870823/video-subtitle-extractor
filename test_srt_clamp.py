"""
srt 短字幕补时长（<1s → +1s）的 clamp 回归测试。

背景：generate_subtitle_file 会把持续时间不足 1 秒的字幕补到 1 秒，
如果下一条字幕紧接着开始，补时长后的结束时间会越过下一条的起始时间，
产生时间轴重叠。修复后结束帧被 clamp 到下一条起始帧前 1 帧。

用法：python test_srt_clamp.py
"""
import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from backend.main import SubtitleExtractor


def build_extractor(raw_lines, fps=25):
    """用伪造的 raw.txt 构造一个只够跑 generate_subtitle_file 的实例"""
    se = object.__new__(SubtitleExtractor)
    se.use_vsf = False
    se.fps = fps
    se.append_output = lambda *a, **k: None
    se.update_progress = lambda *a, **k: None
    tmpdir = tempfile.mkdtemp(prefix='srt_clamp_')
    se.raw_subtitle_path = os.path.join(tmpdir, 'raw.txt')
    se.subtitle_output_path = os.path.join(tmpdir, 'out.srt')
    with open(se.raw_subtitle_path, 'w', encoding='utf-8') as f:
        for frame_no, text in raw_lines:
            f.write(f'{frame_no}\t(0, 100, 0, 50)\t{text}\n')
    return se


def parse_srt(path):
    """:return: [(start_sec, end_sec, text), ...]"""
    with open(path, encoding='utf-8') as f:
        blocks = [b.strip() for b in f.read().replace('\r\n', '\n').split('\n\n') if b.strip()]

    def to_sec(ts):
        h, m, rest = ts.split(':')
        s, ms = rest.split(',')
        return int(h) * 3600 + int(m) * 60 + int(s) + int(ms) / 1000.0

    out = []
    for b in blocks:
        lines = b.split('\n')
        start, end = [x.strip() for x in lines[1].split('-->')]
        out.append((to_sec(start), to_sec(end), ' '.join(lines[2:])))
    return out


def check_no_overlap(entries):
    """返回重叠（前一条结束时间 > 后一条开始时间）的条目对"""
    bad = []
    for i in range(len(entries) - 1):
        if entries[i][1] > entries[i + 1][0] + 1e-6:
            bad.append((i + 1, entries[i], entries[i + 1]))
    return bad


def main():
    fps = 25
    # 帧号规划（全部按升序，同文本连续帧会被合并成一条字幕）
    raw = []
    # 用例 1：短字幕 100-110（11 帧 < 25），下一条从 118 开始 → 补到 125 会撞上，应 clamp 到 117
    for f in range(100, 111):
        raw.append((f, '甲'))
    for f in range(118, 161):
        raw.append((f, '乙'))
    # 用例 2：最后一条短字幕 200-205 → 无下一条，应正常补到 225
    for f in range(200, 206):
        raw.append((f, '丙'))
    # 用例 3：长字幕 300-340（41 帧）→ 不做补时长，结束帧保持 340
    for f in range(300, 341):
        raw.append((f, '丁'))
    # 用例 4：单帧字幕 400，下一条从 402 开始 → 补时长空间不足，结束帧退化为原始结束帧 402
    raw.append((400, '戊'))
    for f in range(402, 441):
        raw.append((f, '己'))

    se = build_extractor(raw, fps=fps)
    se.generate_subtitle_file()
    entries = parse_srt(se.subtitle_output_path)

    print(f'生成条目数: {len(entries)}')
    for i, (s, e, t) in enumerate(entries, 1):
        print(f'  {i}. {s:7.3f} -> {e:7.3f}  {t}')

    failures = []

    def expect(idx, want_start_f, want_end_f, want_text, label):
        s, e, t = entries[idx]
        # 毫秒精度截断，允许 1ms 误差
        got = (round(s * fps), round(e * fps))
        want = (want_start_f, want_end_f)
        ok = abs(got[0] - want[0]) <= 1 and abs(got[1] - want[1]) <= 1 and t == want_text
        print(f'  [{"PASS" if ok else "FAIL"}] {label}: 期望 {want} {want_text!r}，实际 {got} {t!r}')
        if not ok:
            failures.append(label)

    if len(entries) != 6:
        failures.append(f'条目数应为 6，实际 {len(entries)}')
    else:
        expect(0, 100, 117, '甲', '用例1 短字幕被 clamp 到下一条前 1 帧')
        expect(1, 118, 160, '乙', '用例1 下一条不受影响')
        expect(2, 200, 225, '丙', '用例2 末条短字幕正常补到 +1s')
        expect(3, 300, 340, '丁', '用例3 长字幕不补时长')
        expect(4, 400, 402, '戊', '用例4 无补时长空间时保留原始结束帧')
        expect(5, 402, 440, '己', '用例4 下一条不受影响')

    bad = check_no_overlap(entries)
    if bad:
        for i, prev, cur in bad:
            print(f'  [FAIL] 第 {i} 与第 {i + 1} 条时间重叠: {prev[1]:.3f} > {cur[0]:.3f}')
            failures.append(f'重叠 #{i}')
    else:
        print('  [PASS] 所有相邻条目无时间重叠')

    print()
    if failures:
        print(f'FAILED: {len(failures)} 项 -> {failures}')
        return 1
    print('ALL PASS')
    return 0


if __name__ == '__main__':
    sys.exit(main())
