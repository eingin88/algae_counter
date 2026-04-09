#藻類計數與尺寸計算程式 by C44126159陳奕鈞
#內容使用大量AI輔助，請謹慎使用
import logging
logging.getLogger("cellpose").setLevel(logging.ERROR)

import warnings
warnings.filterwarnings("ignore")

import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from cellpose import models
from tqdm import tqdm

# ===============================
# 專案路徑
# ===============================
BASE_DIR = Path(r"C:")  # 請改成你的專案路徑
IMAGE_DIR = BASE_DIR / "data"        
OUT_DIR = BASE_DIR / "outputs"
OUT_OVERLAY = OUT_DIR / "overlays"
OUT_TABLE = OUT_DIR / "tables"

OUT_OVERLAY.mkdir(parents=True, exist_ok=True)
OUT_TABLE.mkdir(parents=True, exist_ok=True)

# ===============================
# 比例尺
# ===============================

PX_PER_UM = 130 / 60 # px / um for 160x

# ===============================
# 黑框 ROI 偵測參數
# ===============================
DETECT_SCALE = 1.0
BLACK_THR = 40              # 黑框閾值
MIN_AREA_RATIO = 0.05       # ROI 至少占整張圖 5%
INSET = 3                   # ROI 往內縮，避免把黑框邊線吃進分析區
RECTANGULARITY_MIN = 0.75   # 輪廓近矩形程度下限

# ===============================
# 選物種：TP / TW / PT
# ===============================
print("Available species: TP / TW / PT")
SPECIES = input("Choose species: ").strip().upper()

if SPECIES not in ["TP", "TW", "PT"]:
    raise SystemExit("Invalid species. Please choose TP, TW, or PT.")

# ===============================
# 物種尺寸範圍（µm）
# TP 用等效圓直徑、TW/PT 用長軸
# ===============================
SIZE_RANGE_UM = {
    "TP": (3, 8),
    "PT": (13, 25),
    "TW": (8, 20),
}

# ===============================
# Cellpose 參數
# diameter_px 是大概的細胞直徑/寬度，不是長度
# ===============================
SPECIES_CONFIG = {
    "TP": dict(diameter_px=None, flow=0.5, cellprob=0.0, min_area_px=None),
    "TW": dict(diameter_px=None, flow=0.6, cellprob=0.0, min_area_px=None),
    "PT": dict(diameter_px=None, flow=0.6, cellprob=0.0, min_area_px=None),
}

# ===============================
# TP 用自訓模型；TW/PT 用預設模型
# ===============================
MODEL_DICT = {
    "TP": "TP_model_V3"
}

# ===============================
# Histogram / Report 參數
# ===============================
HIST_BINS = 20
HIST_RANGE = None          
HIST_DPI = 300
HIST_FIGSIZE = (8, 5)
HIST_ALPHA = 0.85
SHOW_GRID = True

def equivalent_diameter(area_px: float) -> float:
    """等效圓直徑（px）"""
    return 2.0 * np.sqrt(area_px / np.pi)

def safe_read_bgr(path: Path):
    """安全讀圖，避免中文路徑問題"""
    img = cv2.imdecode(np.fromfile(str(path), dtype=np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError(f"Cannot read image: {path}")
    return img

def find_blackframe_roi(img_bgr: np.ndarray):
  
    h0, w0 = img_bgr.shape[:2]

    # scale=1，仍保留此寫法方便以後調整
    if DETECT_SCALE != 1.0:
        small = cv2.resize(
            img_bgr,
            (int(w0 * DETECT_SCALE), int(h0 * DETECT_SCALE)),
            interpolation=cv2.INTER_AREA
        )
    else:
        small = img_bgr.copy()

    gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)

    # 黑色區域 threshold
    black = (gray < BLACK_THR).astype(np.uint8) * 255

    cnts, _ = cv2.findContours(black, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return None, black

    img_area = small.shape[0] * small.shape[1]
    best = None
    best_score = -1

    for cnt in cnts:
        area = cv2.contourArea(cnt)
        if area < img_area * MIN_AREA_RATIO:
            continue

        x, y, w, h = cv2.boundingRect(cnt)
        rect_area = w * h
        if rect_area <= 0:
            continue

        rectangularity = area / rect_area
        if rectangularity < RECTANGULARITY_MIN:
            continue

        score = area * rectangularity
        if score > best_score:
            best_score = score
            best = (x, y, w, h)

    if best is None:
        return None, black

    x, y, w, h = best

    # 往內縮，避免吃到黑框線
    x += INSET
    y += INSET
    w -= 2 * INSET
    h -= 2 * INSET

    if w <= 0 or h <= 0:
        return None, black

    # 如果前面有縮放，映回原圖
    if DETECT_SCALE != 1.0:
        inv = 1.0 / DETECT_SCALE
        x = int(round(x * inv))
        y = int(round(y * inv))
        w = int(round(w * inv))
        h = int(round(h * inv))

    # 邊界保護
    x = max(0, x)
    y = max(0, y)
    w = min(w, w0 - x)
    h = min(h, h0 - y)

    if w <= 0 or h <= 0:
        return None, black

    return (x, y, w, h), black

def build_model(species: str):
    """依物種選模型"""
    if species in MODEL_DICT:
        model_path = MODEL_DICT[species]
        print(f"[Model] {species} -> custom model: {model_path}")
        return models.CellposeModel(pretrained_model=model_path, gpu=True)
    else:
        print(f"[Model] {species} -> default Cellpose model")
        return models.CellposeModel(gpu=True)

def plot_size_histogram(df_size: pd.DataFrame, df_count: pd.DataFrame, species: str):

    if df_size.empty:
        print("[Histogram] No data.")
        return

    # 選擇尺寸欄位
    if species == "TP":
        col = "eq_diameter_um"
        xlabel = "Equivalent diameter (µm)"
    else:
        col = "major_um"
        xlabel = "Major axis length (µm)"

    size_data = pd.to_numeric(df_size[col], errors="coerce").dropna()

    if len(size_data) == 0:
        print("[Histogram] No valid data.")
        return

    # 統計量 
    n = len(size_data)
    mean_val = size_data.mean()
    median_val = size_data.median()
    std_val = size_data.std()
    min_val = size_data.min()
    max_val = size_data.max()

    # 圖面 layout 
    fig = plt.figure(figsize=(10, 6))
    gs = fig.add_gridspec(2, 2, width_ratios=[4, 1], height_ratios=[3, 1])

    ax_hist = fig.add_subplot(gs[0, 0])
    ax_box  = fig.add_subplot(gs[1, 0], sharex=ax_hist)
    ax_text = fig.add_subplot(gs[:, 1])

    # 直方圖
    ax_hist.hist(
        size_data,
        bins=20,
        edgecolor="black",
        alpha=0.85
    )
    xmin, xmax = ax_hist.get_xlim()

    # 平均線（紅）
    ax_hist.axvline(
        mean_val,
        color="red",
        linestyle="--",
        linewidth=2,
        label=f"Mean = {mean_val:.2f}"
    )

    # 中位數線（黃）
    ax_hist.axvline(
        median_val,
        color="gold",
        linestyle="-.",
        linewidth=2,
        label=f"Median = {median_val:.2f}"
    )

    # ±1 SD 區間
    ax_hist.axvspan(
        mean_val - std_val,
        mean_val + std_val,
        alpha=0.15,
        label=f"±1 SD = {std_val:.2f}"
    )

    ax_hist.set_title(f"{species} size distribution")
    ax_hist.set_xlabel(xlabel)
    ax_hist.set_ylabel("Count")
    ax_hist.tick_params(labelbottom=True)
    ax_hist.legend()
    ax_hist.grid(alpha=0.3)

    # Boxplot
    ax_box.boxplot(
        size_data,
        vert=False,
        patch_artist=True,
        showfliers=False,
        boxprops=dict(facecolor="lightblue")
    )
    ax_box.set_xlim(xmin, xmax)

    ax_box.set_title("")
    ax_box.set_yticks([])
    ax_box.set_ylabel("")
    ax_box.grid(alpha=0.3, axis="x")

    # 統計資訊 panel
    ax_text.axis("off")

    image_counts = "\n".join(
        f"{row.image} : {row.count}"
        for row in df_count.itertuples()
    )

    n_images = len(df_count)
    total_cells = int(df_count["count"].sum())

    stats_text = (
        "Image counts\n"
        f"{image_counts}\n\n"
        f"Total images : {n_images}\n"
        f"Total cells  : {total_cells}\n\n"
        "Size statistics\n"
        f"Mean = {mean_val:.2f} µm\n"
        f"Median = {median_val:.2f} µm\n"
        f"SD = {std_val:.2f} µm\n"
        f"Min = {min_val:.2f} µm\n"
        f"Max = {max_val:.2f} µm"
    )

    ax_text.text(
        0.05,
        0.95,
        stats_text,
        va="top",
        fontsize=11
    )

    plt.tight_layout()

    out_path = OUT_DIR / f"{species}_size_report.png"
    plt.savefig(out_path, dpi=300)
    plt.close()

    print(f"[Size report] saved -> {out_path}")

def main():
    cfg = SPECIES_CONFIG[SPECIES]
    min_um, max_um = SIZE_RANGE_UM[SPECIES]

    # 找圖
    images = []
    for ext in ("*.tif", "*.tiff", "*.png", "*.jpg", "*.jpeg", "*.bmp"):
        images.extend(sorted(IMAGE_DIR.glob(ext)))

    if not images:
        print(f"No images found in: {IMAGE_DIR}")
        return

    model = build_model(SPECIES)

    # 總表
    all_size_rows = []
    count_rows = []

    for img_path in tqdm(images, desc="Processing images"):
        print(f"\n[Processing] {img_path.name}")

        img = safe_read_bgr(img_path)
        overlay = img.copy()

        # 找黑框 ROI
        roi_box, _ = find_blackframe_roi(img)
        if roi_box is None:
            print(f"[WARN] ROI not found: {img_path.name}")
            count_rows.append({
                "image": img_path.name,
                "species": SPECIES,
                "count": 0
            })

            # 輸出原圖 overlay 檢查
            stem = img_path.stem
            png_path = OUT_OVERLAY / f"{stem}_{SPECIES}_overlay.png"
            cv2.imencode(".png", overlay)[1].tofile(str(png_path))
            continue

        x, y, w, h = roi_box

        # 畫 ROI 框檢查
        cv2.rectangle(overlay, (x, y), (x + w, y + h), (255, 0, 0), 2)

        roi = img[y:y+h, x:x+w]
        gray_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)

        # Cellpose segmentation
        masks, _, _ = model.eval(
            gray_roi,
            channels=[0, 0],
            diameter=cfg["diameter_px"],
            flow_threshold=cfg["flow"],
            cellprob_threshold=cfg["cellprob"],
        )

        labels = masks.astype(np.int32)
        rows_this_image = []

        # 每顆物件逐一分析
        for lab in range(1, int(labels.max()) + 1):
            mask = (labels == lab).astype(np.uint8) * 255
            area_px = int(cv2.countNonZero(mask))
            #if area_px < cfg["min_area_px"]:
               # continue

            cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if not cnts:
                continue
            cnt = max(cnts, key=cv2.contourArea)

            # 中心點
            M = cv2.moments(cnt)
            if M["m00"] == 0:
                continue
            cx_roi = float(M["m10"] / M["m00"])
            cy_roi = float(M["m01"] / M["m00"])

            # 對應回原圖座標
            cx = cx_roi + x
            cy = cy_roi + y

            # 等效圓直徑
            eq_d_px = float(equivalent_diameter(area_px))
            eq_d_um = eq_d_px / PX_PER_UM

            major_um = None
            minor_um = None
            angle = None

            # TW/PT 用長軸，TP 用等效圓直徑
            if SPECIES in ("TW", "PT"):
                if len(cnt) < 5:
                    continue

                (ex, ey), (MA, ma), ang = cv2.fitEllipse(cnt)
                major_px = float(max(MA, ma))
                minor_px = float(min(MA, ma))
                major_um = major_px / PX_PER_UM
                minor_um = minor_px / PX_PER_UM
                angle = float(ang)

                size_um_for_filter = major_um
                label_text = f"{major_um:.1f}um"
            else:
                size_um_for_filter = eq_d_um
                label_text = f"{eq_d_um:.1f}um"

            # 統一用 size 的標準做篩選
            if not (min_um <= size_um_for_filter <= max_um ):
                continue

            # ROI contour -> 原圖 contour
            cnt_global = cnt.copy()
            cnt_global[:, 0, 0] += x
            cnt_global[:, 0, 1] += y

            # 畫輪廓 + 尺寸標註
            cv2.drawContours(overlay, [cnt_global], -1, (0, 0, 255), 1)
            cv2.putText(
                overlay,
                label_text,
                (int(cx), int(cy)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.4,
                (0, 255, 0),
                1,
                cv2.LINE_AA
            )

            rows_this_image.append({
                "image": img_path.name,
                "species": SPECIES,
                "cell_id": lab,
                "area_px": area_px,
                "eq_diameter_um": eq_d_um,
                "major_um": major_um,
                "minor_um": minor_um,
                "centroid_x": cx,
                "centroid_y": cy,
                "angle_deg": angle,
            })

        # count = 通過 size 條件後保留的數量
        count_rows.append({
            "image": img_path.name,
            "species": SPECIES,
            "count": len(rows_this_image)
        })

        all_size_rows.extend(rows_this_image)

        # 輸出 overlay
        stem = img_path.stem
        png_path = OUT_OVERLAY / f"{stem}_{SPECIES}_overlay.png"
        cv2.imencode(".png", overlay)[1].tofile(str(png_path))

        print(f"[OK] {img_path.name} -> count {len(rows_this_image)}")

    # ===============================
    # 輸出兩個 CSV
    # ===============================
    df_count = pd.DataFrame(count_rows)
    df_size = pd.DataFrame(all_size_rows)
    plot_size_histogram(df_size, df_count, SPECIES)

    count_csv_path = OUT_TABLE / f"{SPECIES}_count_summary.csv"
    size_csv_path = OUT_TABLE / f"{SPECIES}_all_size.csv"

    df_count.to_csv(count_csv_path, index=False, encoding="utf-8-sig")
    df_size.to_csv(size_csv_path, index=False, encoding="utf-8-sig")

    print("\n===============================")
    print(f"[DONE] Count CSV : {count_csv_path.name}")
    print(f"[DONE] Size  CSV : {size_csv_path.name}")
    print(f"[DONE] Please check the report in the outputs")
    print("===============================")

if __name__ == "__main__":
    main()