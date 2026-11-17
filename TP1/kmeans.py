import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.widgets import Button
from scipy.spatial.distance import cdist
from skimage import io

input_path = "fruits.jpg"  
number_of_classes = 2
epsilon = 1e-4
max_iter = 300


def kmeans_seti(data, K, history=None):
    data = np.asarray(data, dtype=float)
    rng = np.random.default_rng()

    unique = np.unique(data, axis=0)
    if len(unique) < K:
        raise ValueError(f"Only {len(unique)} distinct values in the data, cannot make {K} clusters.")
    Pc = unique[rng.choice(len(unique), K, replace=False)]

    for iteration in range(max_iter):
        distances = cdist(data, Pc, metric="euclidean")
        C = np.argmin(distances, axis=1)
        if history is not None:
            history.append((Pc.copy(), C.copy()))

        Pm = np.empty_like(Pc)
        for k in range(K):
            members = data[C == k]
            if len(members) > 0:
                Pm[k] = members.mean(axis=0)
            else:
                Pm[k] = data[rng.integers(len(data))]  # re-seed empty cluster

        shift = np.sum(np.abs(Pm - Pc))
        Pc = Pm
        print(f"iteration {iteration}: shift = {shift:.6f}")
        if shift < epsilon:
            break

    C = np.argmin(cdist(data, Pc, metric="euclidean"), axis=1)
    if history is not None:
        history.append((Pc.copy(), C.copy()))

    Index = [np.where(C == k)[0] for k in range(K)]

    print("centroides:\n", Pc)
    print("tailles clusters:", np.bincount(C, minlength=K))

    return C, Index, Pc


# ---------- Image <-> matrix ----------
def img_2_mat(img):
    """MxNx3 image -> (M*N) x 3 float matrix (one row per pixel)."""
    if img.ndim == 2:                       # grayscale -> 3 identical channels
        img = np.stack([img] * 3, axis=-1)
    return img[:, :, :3].reshape(-1, 3).astype(float)   # [:3] drops alpha if PNG is RGBA


def mat_2_img(Mat, img):
    """(M*N) x 3 matrix -> MxNx3 image with the same size and dtype as the original."""
    M, N = img.shape[:2]
    out = Mat.reshape(M, N, 3)
    if np.issubdtype(img.dtype, np.integer):
        info = np.iinfo(img.dtype)
        out = np.clip(np.round(out), info.min, info.max)
    return out.astype(img.dtype)


def kmeans_image(path_image, K):
    my_img = io.imread(path_image)

    # Transformation de la structure my_img NxMx3 en matrice pixels x 3
    Mat = img_2_mat(my_img)

    C, Index, Pc = kmeans_seti(Mat, K)

    # Affecter aux pixels la couleur de leur prototype
    for k in range(K):
        Mat[Index[k], ] = Pc[k, :]

    # Reconstruire la structure MxNx3 à partir de ces nouveaux pixels
    img_seg = mat_2_img(Mat, my_img)

    # os.path.splitext instead of split('.') so paths like "./img/photo.jpg" work
    out_path = os.path.splitext(path_image)[0] + "_%d.jpg" % K
    io.imsave(out_path, img_seg, check_contrast=False)
    print("image sauvegardée :", out_path)

    return my_img, img_seg


# ---------- Display ----------
def show_image_result(original, segmented, K):
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    axes[0].imshow(original, cmap="gray" if original.ndim == 2 else None)
    axes[0].set_title("Original")
    axes[1].imshow(segmented)
    axes[1].set_title(f"Segmented, K = {K}")
    for a in axes:
        a.axis("off")
    plt.tight_layout()
    plt.show()


def show_csv_iterations(data, history, K):
    # Stop matplotlib's default left/right shortcuts (zoom history) from clashing with ours
    for key in ("left", "right"):
        for km in ("keymap.back", "keymap.forward"):
            if key in plt.rcParams[km]:
                plt.rcParams[km].remove(key)

    colors = plt.cm.tab10(np.arange(K) % 10)

    # Fixed axis limits so the view doesn't jump between iterations
    margin = 0.05 * (data[:, :2].max(axis=0) - data[:, :2].min(axis=0))
    xlim = (data[:, 0].min() - margin[0], data[:, 0].max() + margin[0])
    ylim = (data[:, 1].min() - margin[1], data[:, 1].max() + margin[1])

    fig, ax = plt.subplots(figsize=(8, 7))
    plt.subplots_adjust(bottom=0.15)
    state = {"idx": 0}

    def draw(idx):
        ax.clear()
        centroids, labels = history[idx]

        for i in range(K):
            pts = data[labels == i]
            ax.scatter(pts[:, 0], pts[:, 1], color=colors[i], s=20, alpha=0.7,
                       label=f"cluster {i}")

            # Path followed by this centroid up to the current iteration
            traj = np.array([h[0][i] for h in history[:idx + 1]])
            ax.plot(traj[:, 0], traj[:, 1], "--", color=colors[i], lw=1.2, alpha=0.8)

            ax.scatter(centroids[i, 0], centroids[i, 1], color=colors[i], marker="X",
                       s=250, edgecolors="black", linewidths=1.5, zorder=3)

        last = len(history) - 1
        title = f"k-means, k = {K} | iteration {idx} / {last}"
        if idx == last:
            title += " (final)"
        ax.set_title(title)
        ax.set_xlim(xlim)
        ax.set_ylim(ylim)
        ax.set_xlabel("feature 0")
        ax.set_ylabel("feature 1")
        ax.grid(alpha=0.3)
        ax.legend(loc="upper right", fontsize=8)
        fig.canvas.draw_idle()

    def next_iter(event=None):
        if state["idx"] < len(history) - 1:
            state["idx"] += 1
            draw(state["idx"])

    def prev_iter(event=None):
        if state["idx"] > 0:
            state["idx"] -= 1
            draw(state["idx"])

    def on_key(event):
        if event.key == "right":
            next_iter()
        elif event.key == "left":
            prev_iter()

    ax_prev = fig.add_axes([0.30, 0.03, 0.15, 0.06])
    ax_next = fig.add_axes([0.55, 0.03, 0.15, 0.06])
    btn_prev = Button(ax_prev, "◀ Prev")
    btn_next = Button(ax_next, "Next ▶")
    btn_prev.on_clicked(prev_iter)
    btn_next.on_clicked(next_iter)
    fig.canvas.mpl_connect("key_press_event", on_key)
    fig._buttons = (btn_prev, btn_next)   # keep references so buttons stay alive (Jupyter)

    draw(0)
    plt.show()


# ---------- Main ----------
if __name__ == "__main__":
    ext = os.path.splitext(input_path)[1].lower()

    if ext == ".csv":
        # UTILISER LE CSV
        df = pd.read_csv(input_path, delimiter=";")
        data = df.to_numpy(dtype=float)
        history = []
        C, Index, Pc = kmeans_seti(data, number_of_classes, history=history)
        show_csv_iterations(data, history, number_of_classes)
    else:
        # UTILISER L'IMAGE
        original, segmented = kmeans_image(input_path, number_of_classes)
        show_image_result(original, segmented, number_of_classes)