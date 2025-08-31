# vision_recog.py —— 图像识别 + OCR 模块（按 mod1 的 OCR 配置迁移，并做稳健性适配）
from __future__ import annotations
from pathlib import Path
import os
import re
import numpy as np
import cv2

#  OCR 运行时与配置（来自 mod1，做了可选增强） 
try:
    import pytesseract
    _HAS_TESS = True
except Exception:
    pytesseract = None  # type: ignore
    _HAS_TESS = False

# —— 这行来自 mod1.py（原封不动保留）——
_OCR_CFG = '--psm 7 -c tessedit_char_whitelist=+0123456789'

# 范围裁剪上下限（mod1 的数值场景是 1..10）
POINT_MIN, POINT_MAX = 1, 10

# 适配性增强开关（如需 100% 还原 mod1，设为 False）
USE_BAR_CROP = True        # 在 ROI 中遇到“接近整列”的细竖线时，于竖线处向左截断
USE_LEFTMOST_TOKEN = True  # 先用 image_to_data 取 token，优先最靠左的 +?数字；失败再回退 image_to_string

# ROI 几何（mod1 习惯值：右侧偏移 ≈ 0.12*w，宽度 ≈ 1.4*w）
PAD_RATIO = 0.12
RIGHT_W_FACTOR = 1.40

# 竖线检测阈值（仅在 USE_BAR_CROP=True 时生效）
_BAR_MIN_COVER_RATIO = 0.70  # 竖线在高度方向的白色像素覆盖比例阈值
_BAR_MAX_WIDTH_PX = 4        # 认为是竖线的最大宽度（像素）


def ocr_is_available() -> bool:
    """返回当前环境是否可用 OCR。"""
    return bool(_HAS_TESS)


def configure_tesseract(tesseract_exe: os.PathLike | str | None = None,
                        tessdata_dir: os.PathLike | str | None = None) -> bool:
    """
    配置 Tesseract 可执行文件与 tessdata（便携式/打包场景）。
    成功返回 True；失败返回 False。
    """
    global _HAS_TESS
    if pytesseract is None:
        _HAS_TESS = False
        return False
    try:
        if tesseract_exe:
            pytesseract.pytesseract.tesseract_cmd = str(Path(tesseract_exe))
        if tessdata_dir:
            os.environ["TESSDATA_PREFIX"] = str(Path(tessdata_dir))
        _HAS_TESS = True
        return True
    except Exception:
        _HAS_TESS = False
        return False


def set_ocr_geometry(*, pad_ratio: float | None = None, right_w_factor: float | None = None) -> None:
    """
    调整 ROI 几何参数（不改变 mod1 的 OCR 配置，仅改变取框）：pad_ratio / right_w_factor。
    """
    global PAD_RATIO, RIGHT_W_FACTOR
    if pad_ratio is not None:
        PAD_RATIO = float(pad_ratio)
    if right_w_factor is not None:
        RIGHT_W_FACTOR = float(right_w_factor)


def _ocr_cfg(psm: int | None = None) -> str:
    """
    生成传给 Tesseract 的 config。
    —— 保持与 mod1 一致（白名单 + PSM=7），这里仅允许外部调试时切 PSM（不改白名单）。
    """
    if psm is None:
        return _OCR_CFG
    # 将 _OCR_CFG 的 psm 改写成给定值；其余不变
    return re.sub(r'--psm\s+\d+', f'--psm {int(psm)}', _OCR_CFG)


#  通用图像处理 
IMG_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}
TM_THRESH = 0.78
TM_THRESH_LO = 0.70
SCALES = [0.75, 0.85, 0.90, 1.00, 1.10, 1.25, 1.40]
NMS_IOU = 0.3
ORB_NFEAT = 400
ORB_RATIO = 0.75
CONF_THR = 0.30


def cv_imread_unicode(path, flags=cv2.IMREAD_COLOR):
    """兼容中文/空格路径的 imread，返回 BGR 或 None。"""
    try:
        data = np.fromfile(str(path), dtype=np.uint8)
        if data.size == 0:
            return None
        return cv2.imdecode(data, flags)
    except Exception:
        return None


def _edge_3c(img):
    """边缘增益模板（Canny 后转回 3 通道）。"""
    g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    e = cv2.Canny(g, 80, 160)
    return cv2.cvtColor(e, cv2.COLOR_GRAY2BGR)


def crop_inner_symbol(roi):
    """裁剪图标内圈以弱化边框干扰（利于 ORB）。"""
    h, w = roi.shape[:2]
    m = int(0.18 * min(h, w))
    if h - 2*m <= 2 or w - 2*m <= 2:
        return roi
    return roi[m:h-m, m:w-m]


def load_icon_templates(icons_dir: Path):
    """载入检测/分类模板。"""
    detect_templates, classify_templates = {}, {}
    icons_dir = Path(icons_dir)
    icons_dir.mkdir(parents=True, exist_ok=True)
    for p in icons_dir.glob("*.png"):
        key = p.stem.split(".")[0]
        img = cv_imread_unicode(p, cv2.IMREAD_COLOR)
        if img is None:
            continue
        detect_templates.setdefault(key, []).append(img)
        detect_templates[key].append(_edge_3c(img))
        classify_templates.setdefault(key, []).append(crop_inner_symbol(img))
    return detect_templates, classify_templates


def _match_template_multi(img, tmpl, thr):
    """多尺度模板匹配。"""
    out = []
    th0, tw0 = tmpl.shape[:2]
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img_gray = cv2.bilateralFilter(img_gray, 5, 75, 75)
    for s in SCALES:
        hh, ww = max(16, int(th0*s)), max(16, int(tw0*s))
        t_resized = cv2.resize(tmpl, (ww, hh), interpolation=cv2.INTER_AREA)
        t_gray = cv2.cvtColor(t_resized, cv2.COLOR_BGR2GRAY)
        t_gray = cv2.bilateralFilter(t_gray, 5, 75, 75)
        res = cv2.matchTemplate(img_gray, t_gray, cv2.TM_CCOEFF_NORMED)
        yxs = np.where(res >= thr)
        for (y, x) in zip(*yxs):
            out.append((float(res[y, x]), int(x), int(y), ww, hh))
    return out


def _nms(boxes, iou_thr=0.3):
    """NMS 去重。"""
    if not boxes:
        return []
    boxes = sorted(boxes, key=lambda b: b[0], reverse=True)
    keep = []

    def iou(a, b):
        ax1, ay1, ax2, ay2 = a[1], a[2], a[1]+a[3], a[2]+a[4]
        bx1, by1, bx2, by2 = b[1], b[2], b[1]+b[3], b[2]+b[4]
        interx1, intery1 = max(ax1, bx1), max(ay1, by1)
        interx2, intery2 = min(ax2, bx2), min(ay2, by2)
        iw, ih = max(0, interx2-interx1), max(0, intery2-intery1)
        inter = iw*ih
        uni = (ax2-ax1)*(ay2-ay1) + (bx2-bx1)*(by2-by1) - inter
        return inter/uni if uni > 0 else 0.0

    while boxes:
        a = boxes.pop(0)
        keep.append(a)
        boxes = [b for b in boxes if iou(a, b) < iou_thr]
    return keep


def _estimate_icon_diameter(img):
    """估计图标直径（用于自适应缩放）。"""
    g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    g = cv2.medianBlur(g, 5)
    circles = cv2.HoughCircles(g, cv2.HOUGH_GRADIENT, dp=1.2, minDist=40,
                               param1=80, param2=22, minRadius=8, maxRadius=36)
    if circles is None:
        return None
    return float(np.median([c[2] for c in circles[0]]) * 2.0)


def _autoscale_image_to_template(img, detect_templates):
    """根据模板宽度与估计直径对截图进行一次缩放。"""
    if not detect_templates:
        return img
    widths = []
    for tmpls in detect_templates.values():
        if tmpls and tmpls[0] is not None:
            widths.append(tmpls[0].shape[1])
    if not widths:
        return img
    widths.sort()
    base_w = widths[len(widths)//2]
    diam = _estimate_icon_diameter(img)
    if not diam or diam <= 0 or base_w <= 0:
        return img
    r = diam / float(base_w)
    if 0.8 <= r <= 1.25:
        return img
    scale = 1.0 / r
    h, w = img.shape[:2]
    nh, nw = max(120, int(h*scale)), max(120, int(w*scale))
    return cv2.resize(img, (nw, nh), interpolation=cv2.INTER_AREA)


def detect_icons_in_image(img_bgr, detect_templates):
    """全图模板匹配，返回 [(label,score,x,y,w,h), ...]。"""
    hits = []
    for attr, tmpls in detect_templates.items():
        cand = []
        for t in tmpls:
            cand.extend(_match_template_multi(img_bgr, t, TM_THRESH))
        cand = _nms([(s, x, y, w, h, attr) for (s, x, y, w, h) in cand], NMS_IOU)
        hits.extend(cand)
    hits = _nms(hits, NMS_IOU)
    return [(lab, s, x, y, w, h) for (s, x, y, w, h, lab) in hits]


def _rescue_partner(img, one_hit, detect_templates):
    """同一行仅 1 个命中时，向右“救回”一个可能的搭档。"""
    attr, s, x, y, w, h = one_hit
    rx1 = x + int(0.6 * w)
    rx2 = x + int(6.0 * w)
    ry1 = y - int(0.25 * h)
    ry2 = y + int(1.25 * h)
    H, W = img.shape[:2]
    rx1 = max(0, rx1); ry1 = max(0, ry1)
    rx2 = min(W, rx2); ry2 = min(H, ry2)
    roi = img[ry1:ry2, rx1:rx2]
    if roi.size == 0:
        return None
    cand = []
    for lab, tmpls in detect_templates.items():
        for t in tmpls:
            cand.extend(_match_template_multi(roi, t, TM_THRESH_LO))
    if not cand:
        return None
    s2, x2, y2, w2, h2 = sorted(cand, key=lambda z: z[0], reverse=True)[0]
    return ("?", s2, rx1 + x2, ry1 + y2, w2, h2)


def group_hits_to_rows(hits, img=None, detect_templates=None):
    """将命中按“行”分组（并确保每行至少 2 个）。"""
    if not hits:
        return []
    hits = sorted(hits, key=lambda h: (h[3], h[2]))  # y 再 x
    rows, line = [], [hits[0]]
    avg_h = np.median([h[5] for h in hits]) if hits else 20
    thr = max(18, int(avg_h * 0.7))
    for h in hits[1:]:
        if abs(h[3] - line[-1][3]) <= thr:
            line.append(h)
        else:
            rows.append(line)
            line = [h]
    rows.append(line)

    out = []
    for line in rows:
        line = sorted(line, key=lambda h: h[1], reverse=True)[:3]
        if len(line) == 1 and img is not None and detect_templates:
            rescued = _rescue_partner(img, line[0], detect_templates)
            if rescued is not None:
                line.append(rescued)
        if len(line) < 2:
            continue
        line = sorted(line, key=lambda h: h[2])
        out.append(line)
    return out


def refine_attr_by_orb(roi_bgr, classify_templates):
    """用 ORB + BFMatcher 做类别复核。"""
    orb = cv2.ORB_create(nfeatures=ORB_NFEAT)
    gray = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2GRAY)
    k1, d1 = orb.detectAndCompute(gray, None)
    if d1 is None or len(k1) == 0:
        return None, 0.0, []
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
    scores = []
    for attr, tmpls in classify_templates.items():
        best = 0
        for t in tmpls:
            tg = cv2.cvtColor(t, cv2.COLOR_BGR2GRAY)
            k2, d2 = orb.detectAndCompute(tg, None)
            if d2 is None or len(k2) == 0:
                continue
            matches = bf.knnMatch(d1, d2, k=2)
            good = 0
            for m, n in matches:
                if m.distance < ORB_RATIO * n.distance:
                    good += 1
            best = max(best, good)
        scores.append((attr, best))
    scores.sort(key=lambda x: x[1], reverse=True)
    if not scores or scores[0][1] == 0:
        return None, 0.0, []
    best_attr, best_score = scores[0]
    second = scores[1][1] if len(scores) > 1 else 0
    conf = (best_score - second) / max(1.0, float(best_score))
    return best_attr, conf, scores[:3]


#  OCR：归一化（依据 mod1 思路，适配不再将 '|'→'1'） 
def _normalize_ocr_number(txt: str | None):
    """
    将 OCR 文本清洗为 1..10 的整数；若不存在任何数字则返回 None。
    - 纠正常见混淆：O/o→0，I/l→1  （注意：不再把“|”当作“1”）
    - 抽取连续数字串 s；
      * 若 s 以 '10' 开头，直接返回 10（保持 mod1 的快速路径）
      * 否则将 s 当整体；若 >10，则“反复去掉最后一位”直到 ≤10
    - 最终夹到 [POINT_MIN, POINT_MAX]
    """
    if not txt:
        return None
    s = txt.strip().replace(" ", "")
    # mod1 的字符替换（去除 |→1 这一步）
    s = (s.replace("O", "0").replace("o", "0")
           .replace("I", "1").replace("l", "1"))
    m = re.search(r"\d+", s)
    if not m:
        return None
    s = m.group(0)

    # mod1 的“以 10 开头则取 10”
    if s.startswith("10"):
        return 10

    # 解析并执行“超过 10 则不断去尾”的策略
    try:
        val = int(s)
    except Exception:
        return None

    while val > POINT_MAX and val >= 10:  # 保持 mod1 风格：不断去掉最后一位
        val = int(str(val)[:-1]) if len(str(val)) > 1 else val

    # 夹到范围
    if val < POINT_MIN:
        return POINT_MIN
    if val > POINT_MAX:
        return POINT_MAX
    return val


#  OCR：核心（保持 mod1 的流程 + 适配增强） 
def _crop_left_at_bar(binary_img: np.ndarray) -> np.ndarray:
    """
    在二值图中寻找近乎整列的细竖线（分隔符），若存在则在该处左侧截断。
    仅在 USE_BAR_CROP=True 时被调用。
    """
    h, w = binary_img.shape[:2]
    if w <= 2 or h <= 2:
        return binary_img
    col_sum = (binary_img > 0).sum(axis=0)  # 每列的白色像素数
    cover = col_sum / max(1, h)
    cand = np.where(cover >= _BAR_MIN_COVER_RATIO)[0]
    if cand.size == 0:
        return binary_img

    # 按连续段聚合，取最靠左且宽度小的“竖线”
    runs = []
    start = cand[0]
    prev = cand[0]
    for x in cand[1:]:
        if x == prev + 1:
            prev = x
        else:
            runs.append((start, prev))
            start = x
            prev = x
    runs.append((start, prev))
    runs = [(a, b) for (a, b) in runs if (b - a + 1) <= _BAR_MAX_WIDTH_PX]
    if not runs:
        return binary_img
    x_bar = runs[0][0]
    return binary_img[:, :max(1, x_bar)]  # 截到竖线左侧


def _ocr_leftmost_token(gray: np.ndarray) -> int | None:
    """
    先用 image_to_data 获取 token（带坐标），选 ROI 中最靠左的“+?数字(≤2位)”。
    仅在 USE_LEFTMOST_TOKEN=True 时启用。
    """
    try:
        data = pytesseract.image_to_data(
            gray, lang="eng", config=_OCR_CFG, output_type=pytesseract.Output.DICT
        )
    except Exception:
        return None
    if not data or "text" not in data:
        return None

    toks = []
    n = len(data["text"])
    for i in range(n):
        txt = (data["text"][i] or "").strip()
        if not txt:
            continue
        if re.fullmatch(r"\+?\d{1,2}", txt):
            try:
                val = int(txt.replace("+", ""))
            except Exception:
                continue
            if POINT_MIN <= val <= POINT_MAX:
                # 越靠左越好；同时保留一下 conf（越大越好）
                conf = 0.0
                try:
                    conf = float(data.get("conf", [0]*n)[i])
                except Exception:
                    pass
                toks.append((data["left"][i], conf, val))
    if not toks:
        return None
    # 先按 x 从小到大；若并列，再按 conf 从大到小
    toks.sort(key=lambda t: (t[0], -t[1]))
    return toks[0][2]


def ocr_points_right_of(img: np.ndarray, box: tuple[int, int, int, int]) -> int | None:
    """
    读取图标右侧的点数（遵循 mod1 的主流程）：
      1) 取 ROI：从 x+w+pad 开始，宽度 right_w_factor*w；
      2) 灰度 + OTSU 二值化；
      3) （适配）若启用 USE_BAR_CROP，则在 ROI 内做竖线截断；
      4) （适配）若启用 USE_LEFTMOST_TOKEN，则先 image_to_data 拿“最靠左数字”；
      5) 回退：直接 image_to_string（与 mod1 一致）；
      6) 用 _normalize_ocr_number 清洗为 1..10。
    """
    if not _HAS_TESS:
        return None
    x, y, w, h = box
    pad = max(int(PAD_RATIO * w), 6)
    rx1 = x + w + pad
    rx2 = x + w + int(RIGHT_W_FACTOR * w)
    ry1 = y
    ry2 = y + h
    rx1 = max(0, rx1); ry1 = max(0, ry1)
    rx2 = min(img.shape[1], rx2); ry2 = min(img.shape[0], ry2)
    roi = img[ry1:ry2, rx1:rx2]
    if roi.size == 0:
        return None

    # 灰度 + OTSU（保持 mod1 的阈值方式）
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]

    # 适配：竖线截断（可关闭）
    if USE_BAR_CROP:
        binary = _crop_left_at_bar(binary)

    # 适配：尝试左端优先（拿 token 的方式）
    if USE_LEFTMOST_TOKEN:
        val = _ocr_leftmost_token(binary)
        if val is not None:
            return val

    # 回退：按照 mod1 的方式直接 image_to_string + 归一化
    try:
        txt = pytesseract.image_to_string(binary, lang="eng", config=_OCR_CFG)
    except Exception:
        txt = ""
    return _normalize_ocr_number(txt)


#  高层接口 
def parse_screenshot_to_gems(image_path: Path, detect_templates, classify_templates):
    """
    解析单张截图，返回若干“词条/宝石”项：
      - 每项包含 a1/p1 与 a2/p2，若存在第三项则含 a3/p3；
      - 属性名通过模板匹配 + ORB 复核；点数通过 OCR 获得（失败则在调用处用 1 兜底）。
    """
    img = cv_imread_unicode(image_path, cv2.IMREAD_COLOR)
    if img is None:
        raise RuntimeError(f"无法读取图片：{image_path}")
    img = _autoscale_image_to_template(img, detect_templates)

    hits = detect_icons_in_image(img, detect_templates)
    if not hits:
        # 一轮失败则临时扩大尺度范围再试
        old_scales = list(SCALES)
        try:
            SCALES[:] = [0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.85, 0.90,
                         1.00, 1.10, 1.25, 1.50, 1.75, 2.00]
            hits = detect_icons_in_image(img, detect_templates)
        finally:
            SCALES[:] = old_scales

    rows = group_hits_to_rows(hits, img=img, detect_templates=detect_templates)

    gems = []
    for line in rows:
        attrs = []
        for (attr, s, x, y, w, h) in line:
            # ORB 二次确认属性
            roi_in = crop_inner_symbol(img[y:y+h, x:x+w])
            a_ref, c, _ = refine_attr_by_orb(roi_in, classify_templates)
            if a_ref and c >= CONF_THR:
                attr = a_ref
            # OCR 点数（失败返回 None，这里由上层决定是否兜底为 1）
            p = ocr_points_right_of(img, (x, y, w, h))
            attrs.append((attr, p if p is not None else 1))
        if len(attrs) >= 2:
            g = {"a1": attrs[0][0], "p1": attrs[0][1],
                 "a2": attrs[1][0], "p2": attrs[1][1]}
            if len(attrs) == 3:
                g["a3"] = attrs[2][0]; g["p3"] = attrs[2][1]
            gems.append(g)
    return gems


def parse_folder_to_gems(folder: Path, detect_templates, classify_templates, progress_cb=None):
    """批量解析文件夹内的图片。"""
    imgs = []
    folder = Path(folder)
    for p in folder.rglob("*"):
        if p.is_file() and p.suffix.lower() in IMG_EXTS:
            imgs.append(p)
    imgs.sort()

    all_gems, fails = [], []
    total = len(imgs)
    for idx, p in enumerate(imgs, 1):
        try:
            gs = parse_screenshot_to_gems(p, detect_templates, classify_templates)
            all_gems.extend(gs)
        except Exception as e:
            fails.append((p, str(e)))
        if progress_cb:
            try:
                progress_cb(idx, total, p.name)
            except Exception:
                pass
    return all_gems, total, fails


#  调试辅助 
def debug_dump_ocr_crops(img: np.ndarray, hits, out_dir: Path):
    """
    导出 OCR ROI 与 CSV 以排查问题（不改变主流程，只做观测）：
      - 输出 roi_XXX.png / roi_XXX_thr.png 以及 ocr_dump.csv
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    idx = 0
    for (attr, s, x, y, w, h) in hits:
        idx += 1
        pad = max(int(PAD_RATIO * w), 4)
        rx1 = x + w + pad
        rx2 = x + w + int(RIGHT_W_FACTOR * w)
        ry1 = y
        ry2 = y + h
        rx1 = max(0, rx1); ry1 = max(0, ry1)
        rx2 = min(img.shape[1], rx2); ry2 = min(img.shape[0], ry2)
        roi = img[ry1:ry2, rx1:rx2]
        if roi.size == 0:
            rows.append([idx, attr, x, y, w, h, "", "", "EMPTY_ROI"])
            continue
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        thr = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)[1]

        # 可视化保存
        cv2.imwrite(str(out_dir / f"roi_{idx:03d}.png"), roi)
        cv2.imwrite(str(out_dir / f"roi_{idx:03d}_thr.png"), thr)

        # 为了观测，与主流程一致：可选竖线截断 + image_to_string
        bin2 = _crop_left_at_bar(thr) if USE_BAR_CROP else thr
        txt = ''
        if _HAS_TESS:
            try:
                txt = pytesseract.image_to_string(bin2, lang="eng", config=_OCR_CFG)
            except Exception:
                txt = ''
        rows.append([idx, attr, x, y, w, h, txt.strip(), _normalize_ocr_number(txt), ""])

    try:
        import csv
        with open(out_dir / "ocr_dump.csv", "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["idx", "attr", "x", "y", "w", "h", "raw", "parsed", "note"])
            w.writerows(rows)
    except Exception:
        pass


if __name__ == "__main__":
    print("vision_recog.py 就绪（mod1 OCR 配置 + 适配增强）。")
