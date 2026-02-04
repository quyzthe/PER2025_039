import os
import numpy as np
import cv2
import matplotlib.pyplot as plt

# -------- Build event frame --------
def build_event_frame(x, y, p, H, W):
    frame = np.zeros((H, W), dtype=np.int32)
    np.add.at(frame, (y, x), p)
    return frame

# -------- Generate frames function --------
def generate_event_frames(
    x, y, p, t, 
    rgb_dir, 
    BIN_US=100_000_000,  # 100 ms
    n=1,
    plot=False           # plot frame nếu True
):
    """
    Tạo event frames và RGB frames gộp theo n time windows.

    Args:
        x, y, p, t: Dữ liệu sự kiện.
        rgb_dir: Thư mục chứa RGB images.
        BIN_US: Kích thước mỗi bin tính bằng microseconds.
        n: Số bin gộp lại thành 1 event frame.
        plot: True để hiển thị các frame.
    
    Returns:
        frames_cropped: List các event frames (numpy arrays)
        rgb_selected: List các ảnh RGB (numpy arrays)
    """
    
    rgb_dir = os.path.join(rgb_dir, "images")  # đảm bảo đúng thư mục
    rgb_files = sorted([
        f for f in os.listdir(rgb_dir)
        if f.lower().endswith((".png", ".jpg", ".jpeg"))
    ])

    t_start = 0
    t_end = t.max()
    bins = np.arange(t_start, t_end + BIN_US, BIN_US)  # +BIN_US để cover hết

    H = y.max() + 1
    W = x.max() + 1

    frames_cropped = []
    rgb_selected = []

    i = 0
    while i < len(bins) - 1:
        t0 = bins[i]
        t1 = bins[min(i+n, len(bins)-1)]  # tránh out-of-range

        mask = (t >= t0) & (t < t1)
        if mask.sum() < 20:
            i += n
            continue

        # -------- Event frame --------
        event_frame = build_event_frame(x[mask], y[mask], p[mask], H, W)
        frames_cropped.append(event_frame)

        # -------- RGB image --------
        rgb_idx = min(i, len(rgb_files)-1)  # lấy ảnh đầu trong n frames
        rgb_path = os.path.join(rgb_dir, rgb_files[rgb_idx])
        rgb_img = cv2.imread(rgb_path)
        rgb_img = cv2.cvtColor(rgb_img, cv2.COLOR_BGR2RGB)
        rgb_selected.append(rgb_img)

        # -------- Plot (nếu cần) --------
        if plot:
            fig, axs = plt.subplots(1, 2, figsize=(10, 4))
            axs[0].imshow(event_frame, cmap="gray")
            axs[0].set_title(f"Event [{t0/1e9:.2f}s – {t1/1e9:.2f}s]")
            axs[0].axis("off")

            axs[1].imshow(rgb_img)
            axs[1].set_title(f"RGB frame #{rgb_idx}")
            axs[1].axis("off")
            plt.tight_layout()
            plt.show()

        i += n

    return frames_cropped, rgb_selected