import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Arc, Ellipse, FancyArrowPatch, FancyBboxPatch, Rectangle
from matplotlib.lines import Line2D


# ============================================================
# Global settings
# ============================================================

OUTPUT_DIR = "figures"
os.makedirs(OUTPUT_DIR, exist_ok=True)

PAGE_BG = "#f6f8fb"
PANEL_BG = "#ffffff"
PANEL_ALT = "#eef5ff"
PANEL_EDGE = "#c8d4e3"
TEXT = "#1f2933"
MUTED = "#5b677a"
GRID = "#dce3ec"
BLUE = "#2f5f9e"
TEAL = "#2a9d8f"
ORANGE = "#e76f51"
GOLD = "#e9b44c"
PURPLE = "#7b6dce"
GREEN = "#4f9d69"
PALETTE = [BLUE, TEAL, ORANGE, PURPLE, GREEN, GOLD]

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 11,
    "axes.titlesize": 12,
    "axes.labelsize": 10,
    "legend.fontsize": 9,
    "figure.titlesize": 13,
    "figure.facecolor": PAGE_BG,
    "axes.facecolor": PANEL_BG,
    "axes.edgecolor": PANEL_EDGE,
    "axes.labelcolor": MUTED,
    "axes.titlecolor": TEXT,
    "text.color": TEXT,
    "xtick.color": MUTED,
    "ytick.color": MUTED,
    "grid.color": GRID,
    "grid.linewidth": 0.7,
    "grid.alpha": 0.65,
    "lines.solid_capstyle": "round",
    "ps.fonttype": 42,
})


def save_figure(fig, filename_base):
    png_path = os.path.join(OUTPUT_DIR, f"{filename_base}.png")
    fig.savefig(
        png_path,
        bbox_inches="tight",
        dpi=300,
        facecolor=fig.get_facecolor(),
    )
    plt.close(fig)
    print(f"Saved: {png_path}")


def style_axis(ax, grid=True):
    ax.set_facecolor(PANEL_BG)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color(PANEL_EDGE)
    ax.spines["bottom"].set_color(PANEL_EDGE)
    ax.spines["left"].set_linewidth(0.9)
    ax.spines["bottom"].set_linewidth(0.9)
    ax.tick_params(colors=MUTED, labelsize=9)
    if grid:
        ax.grid(True)
        ax.set_axisbelow(True)


def add_rounded_panel(
    ax,
    xy,
    width,
    height,
    facecolor=PANEL_BG,
    edgecolor=PANEL_EDGE,
    linewidth=1.2,
    radius=0.025,
    shadow=True,
    zorder=1,
):
    if shadow:
        shadow_patch = FancyBboxPatch(
            (xy[0] + 0.008, xy[1] - 0.01),
            width,
            height,
            boxstyle=f"round,pad=0.014,rounding_size={radius}",
            linewidth=0,
            facecolor="#cbd5e1",
            alpha=0.28,
            zorder=zorder - 1,
        )
        ax.add_patch(shadow_patch)

    panel = FancyBboxPatch(
        xy,
        width,
        height,
        boxstyle=f"round,pad=0.014,rounding_size={radius}",
        linewidth=linewidth,
        edgecolor=edgecolor,
        facecolor=facecolor,
        zorder=zorder,
    )
    ax.add_patch(panel)
    return panel


def add_flow_arrow(ax, start, end, color=BLUE):
    arrow = FancyArrowPatch(
        start,
        end,
        arrowstyle="-|>",
        mutation_scale=18,
        linewidth=2.0,
        color=color,
        shrinkA=3,
        shrinkB=3,
        zorder=5,
    )
    ax.add_patch(arrow)


def add_caption(ax, x, y, text, fontsize=10, ha="center"):
    ax.text(x, y, text, ha=ha, va="top", fontsize=fontsize, color=MUTED)


def simple_pca(X, n_components=2):
    """
    Minimal PCA implementation using NumPy.
    X: array of shape (n_samples, n_features)
    """
    X = np.asarray(X, dtype=float)
    X_centered = X - X.mean(axis=0, keepdims=True)

    # SVD-based PCA
    _, _, Vt = np.linalg.svd(X_centered, full_matrices=False)
    return X_centered @ Vt[:n_components].T


def add_group_ellipse(ax, points, color, n_std=1.9):
    if len(points) < 3:
        return

    covariance = np.cov(points, rowvar=False)
    values, vectors = np.linalg.eigh(covariance)
    values = np.maximum(values, 1e-9)
    order = values.argsort()[::-1]
    values = values[order]
    vectors = vectors[:, order]

    angle = np.degrees(np.arctan2(vectors[1, 0], vectors[0, 0]))
    width, height = 2 * n_std * np.sqrt(values)
    center = points.mean(axis=0)

    ellipse = Ellipse(
        center,
        width=width,
        height=height,
        angle=angle,
        facecolor=color,
        edgecolor=color,
        linewidth=1.4,
        alpha=0.13,
        zorder=1,
    )
    ax.add_patch(ellipse)


# ============================================================
# Figure 1: Dimensionality Reduction
# ============================================================

def make_figure_1_dimensionality_reduction():
    rng = np.random.default_rng(42)

    fig, ax = plt.subplots(figsize=(11.6, 4.7))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    # Left: high-dimensional embedding matrix
    matrix_x, matrix_y = 0.07, 0.24
    matrix_w, matrix_h = 0.24, 0.48
    add_rounded_panel(
        ax,
        (matrix_x - 0.025, matrix_y - 0.055),
        matrix_w + 0.05,
        matrix_h + 0.16,
        facecolor=PANEL_BG,
    )

    rows, cols = 10, 12
    cell_w = matrix_w / cols
    cell_h = matrix_h / rows
    cmap = plt.get_cmap("viridis")

    for i in range(rows):
        for j in range(cols):
            value = rng.random()
            ax.add_patch(Rectangle(
                (
                    matrix_x + j * cell_w + cell_w * 0.05,
                    matrix_y + i * cell_h + cell_h * 0.06,
                ),
                cell_w * 0.88,
                cell_h * 0.82,
                facecolor=cmap(0.16 + 0.72 * value),
                edgecolor="white",
                linewidth=0.45,
                zorder=3,
            ))

    ax.text(
        matrix_x + matrix_w / 2,
        matrix_y + matrix_h + 0.07,
        "256-dimensional\nartist embeddings",
        ha="center",
        va="bottom",
        weight="bold",
        fontsize=12,
    )
    add_caption(
        ax,
        matrix_x + matrix_w / 2,
        matrix_y - 0.055,
        "High-dimensional feature space",
    )

    # Center: PCA / t-SNE block
    block_x, block_y = 0.425, 0.385
    block_w, block_h = 0.16, 0.23
    add_rounded_panel(
        ax,
        (block_x, block_y),
        block_w,
        block_h,
        facecolor=PANEL_ALT,
        edgecolor="#aac6e8",
        radius=0.03,
    )
    ax.text(
        block_x + block_w / 2,
        block_y + block_h * 0.62,
        "PCA / t-SNE",
        ha="center",
        va="center",
        weight="bold",
        fontsize=12,
        color=BLUE,
    )
    ax.text(
        block_x + block_w / 2,
        block_y + block_h * 0.34,
        "projection",
        ha="center",
        va="center",
        fontsize=9.5,
        color=MUTED,
    )

    # Right: 2D scatter plot
    scatter_x, scatter_y = 0.70, 0.22
    scatter_w, scatter_h = 0.23, 0.50
    add_rounded_panel(
        ax,
        (scatter_x - 0.025, scatter_y - 0.055),
        scatter_w + 0.05,
        scatter_h + 0.16,
        facecolor=PANEL_BG,
    )

    for frac in np.linspace(0.2, 0.8, 4):
        ax.plot(
            [scatter_x, scatter_x + scatter_w],
            [scatter_y + frac * scatter_h] * 2,
            color=GRID,
            linewidth=0.6,
            zorder=2,
        )
        ax.plot(
            [scatter_x + frac * scatter_w] * 2,
            [scatter_y, scatter_y + scatter_h],
            color=GRID,
            linewidth=0.6,
            zorder=2,
        )

    # Synthetic clustered scatter
    centers = np.array([
        [0.30, 0.65],
        [0.65, 0.62],
        [0.48, 0.32],
    ])

    for idx, c in enumerate(centers):
        pts = c + 0.06 * rng.normal(size=(18, 2))
        pts[:, 0] = np.clip(pts[:, 0], 0.08, 0.92)
        pts[:, 1] = np.clip(pts[:, 1], 0.08, 0.92)

        ax.scatter(
            scatter_x + pts[:, 0] * scatter_w,
            scatter_y + pts[:, 1] * scatter_h,
            s=28,
            alpha=0.85,
            color=PALETTE[idx],
            edgecolors="white",
            linewidths=0.55,
            zorder=4,
        )

    ax.text(
        scatter_x + scatter_w / 2,
        scatter_y + scatter_h + 0.07,
        "2D latent space\nvisualization",
        ha="center",
        va="bottom",
        weight="bold",
        fontsize=12,
    )
    add_caption(
        ax,
        scatter_x + scatter_w / 2,
        scatter_y - 0.055,
        "Artists projected into two dimensions",
    )

    # Arrows
    add_flow_arrow(
        ax,
        (matrix_x + matrix_w + 0.03, 0.50),
        (block_x - 0.03, 0.50),
        color=BLUE,
    )
    add_flow_arrow(
        ax,
        (block_x + block_w + 0.03, 0.50),
        (scatter_x - 0.03, 0.50),
        color=BLUE,
    )

    fig.suptitle(
        "Dimensionality Reduction for Artist Embeddings",
        y=0.98,
        color=TEXT,
        weight="bold",
    )
    save_figure(fig, "fig1_dimensionality_reduction")


# ============================================================
# Figure 2: Activation Functions
# ============================================================

def make_figure_2_activation_functions():
    x = np.linspace(-5, 5, 600)

    linear = x
    sigmoid = 1 / (1 + np.exp(-x))
    tanh = np.tanh(x)
    relu = np.maximum(0, x)

    # GELU tanh approximation
    gelu = 0.5 * x * (
        1 + np.tanh(np.sqrt(2 / np.pi) * (x + 0.044715 * x**3))
    )

    functions = [
        ("Linear", linear, r"$f(x)=x$", BLUE, (-5.4, 5.4)),
        ("Sigmoid", sigmoid, r"$f(x)=\frac{1}{1+e^{-x}}$", TEAL, (-0.1, 1.1)),
        ("Tanh", tanh, r"$f(x)=\tanh(x)$", ORANGE, (-1.2, 1.2)),
        ("ReLU", relu, r"$f(x)=\max(0,x)$", PURPLE, (-0.4, 5.4)),
        (
            "GELU",
            gelu,
            r"$f(x)\approx0.5x(1+\tanh(\cdots))$",
            GREEN,
            (-0.5, 5.4),
        ),
    ]

    fig, axes = plt.subplots(2, 3, figsize=(11.4, 6.9))
    axes = axes.ravel()

    for idx, (name, y, formula, color, ylim) in enumerate(functions):
        ax = axes[idx]
        style_axis(ax)
        ax.plot(x, y, linewidth=2.7, color=color)
        ax.axhline(0, linewidth=0.9, color="#9aa8b8")
        ax.axvline(0, linewidth=0.9, color="#9aa8b8")
        ax.set_xlim(-5, 5)
        ax.set_ylim(*ylim)
        ax.set_title(name, weight="bold", pad=10)
        ax.set_xlabel("x")
        ax.set_ylabel("f(x)")
        ax.text(
            0.05,
            0.90,
            formula,
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize=9,
            color=TEXT,
            bbox={
                "boxstyle": "round,pad=0.28",
                "facecolor": "#f8fafc",
                "edgecolor": PANEL_EDGE,
                "linewidth": 0.7,
                "alpha": 0.95,
            },
        )

    # Hide the empty sixth subplot
    axes[-1].axis("off")
    add_rounded_panel(
        axes[-1],
        (0.12, 0.28),
        0.76,
        0.44,
        facecolor="#f8fafc",
        edgecolor=PANEL_EDGE,
        radius=0.035,
        shadow=False,
        zorder=2,
    )
    axes[-1].text(
        0.5,
        0.5,
        "Activation functions introduce\nnon-linear transformations",
        ha="center",
        va="center",
        fontsize=11,
        color=MUTED,
        transform=axes[-1].transAxes,
    )

    fig.suptitle(
        "Common Activation Functions in Neural Networks",
        y=0.99,
        color=TEXT,
        weight="bold",
    )
    fig.tight_layout(rect=[0, 0.02, 1, 0.95], w_pad=2.0, h_pad=2.2)
    save_figure(fig, "fig2_activation_functions")


# ============================================================
# Figure 8: Cosine Similarity Geometry
# ============================================================

def draw_vector(ax, start, end, label, color, label_offset=(0.05, 0.03)):
    arrow = FancyArrowPatch(
        start,
        end,
        arrowstyle="-|>",
        mutation_scale=16,
        linewidth=2.4,
        color=color,
        zorder=4,
    )
    ax.add_patch(arrow)
    ax.text(
        end[0] + label_offset[0],
        end[1] + label_offset[1],
        label,
        fontsize=11,
        weight="bold",
        color=color,
    )


def make_figure_8_cosine_similarity_geometry():
    fig, axes = plt.subplots(1, 2, figsize=(10.8, 4.7))

    examples = [
        {
            "title": "High cosine similarity",
            "v1": np.array([1.0, 0.25]),
            "v2": np.array([0.95, 0.55]),
            "theta_label": r"small $\theta$",
        },
        {
            "title": "Low cosine similarity",
            "v1": np.array([1.0, 0.20]),
            "v2": np.array([0.15, 1.0]),
            "theta_label": r"large $\theta$",
        },
    ]

    for ax, ex in zip(axes, examples):
        style_axis(ax)
        ax.set_aspect("equal")
        ax.set_xlim(-0.1, 1.35)
        ax.set_ylim(-0.1, 1.35)

        ax.axhline(0, linewidth=1.0, color="#9aa8b8")
        ax.axvline(0, linewidth=1.0, color="#9aa8b8")
        ax.add_patch(
            plt.Circle(
                (0, 0),
                1.0,
                fill=False,
                linestyle="--",
                linewidth=1.0,
                edgecolor=GRID,
                zorder=1,
            )
        )

        ax.set_xlabel("Embedding dimension 1")
        ax.set_ylabel("Embedding dimension 2")
        ax.set_title(ex["title"], weight="bold", pad=10)

        origin = np.array([0.0, 0.0])
        v1 = ex["v1"]
        v2 = ex["v2"]

        draw_vector(ax, origin, v1, r"$\mathbf{u}$", BLUE)
        draw_vector(ax, origin, v2, r"$\mathbf{v}$", ORANGE)

        # Angle arc
        angle1 = np.degrees(np.arctan2(v1[1], v1[0]))
        angle2 = np.degrees(np.arctan2(v2[1], v2[0]))
        theta1, theta2 = sorted([angle1, angle2])

        arc = Arc(
            (0, 0),
            width=0.55,
            height=0.55,
            angle=0,
            theta1=theta1,
            theta2=theta2,
            linewidth=2.0,
            color=GOLD,
            zorder=3,
        )
        ax.add_patch(arc)

        mid_angle = np.radians((theta1 + theta2) / 2)
        ax.text(
            0.38 * np.cos(mid_angle),
            0.38 * np.sin(mid_angle),
            ex["theta_label"],
            ha="center",
            va="center",
            fontsize=10,
            color=TEXT,
            bbox={
                "boxstyle": "round,pad=0.2",
                "facecolor": "#fff7df",
                "edgecolor": "#efd28a",
                "linewidth": 0.7,
                "alpha": 0.95,
            },
        )

        similarity = np.dot(v1, v2) / (
            np.linalg.norm(v1) * np.linalg.norm(v2)
        )

        ax.text(
            0.05,
            1.18,
            rf"$\cos(\theta) = {similarity:.2f}$",
            fontsize=11,
            weight="bold",
            color=TEXT,
            bbox={
                "boxstyle": "round,pad=0.3",
                "facecolor": "#f8fafc",
                "edgecolor": PANEL_EDGE,
                "linewidth": 0.8,
            },
        )

        ax.legend(
            handles=[
                Line2D([0], [0], color=BLUE, lw=2.4, label=r"$\mathbf{u}$"),
                Line2D([0], [0], color=ORANGE, lw=2.4, label=r"$\mathbf{v}$"),
            ],
            loc="lower right",
            frameon=True,
            facecolor=PANEL_BG,
            edgecolor=PANEL_EDGE,
        )

    fig.text(
        0.5,
        -0.01,
        "Cosine similarity depends on vector direction rather than magnitude.",
        ha="center",
        fontsize=10.5,
        color=MUTED,
    )

    fig.suptitle(
        "Geometric Interpretation of Cosine Similarity",
        y=1.02,
        color=TEXT,
        weight="bold",
    )
    fig.tight_layout(w_pad=2.6)
    save_figure(fig, "fig8_cosine_similarity_geometry")


# ============================================================
# Figure 9: Latent Space Visualization
# ============================================================

def make_synthetic_artist_embeddings(
    n_artists=180,
    embedding_dim=256,
    n_groups=4,
    random_state=7
):
    """
    Creates synthetic 256-dimensional artist embeddings for visualization.
    Replace this with real model embeddings when available.
    """
    rng = np.random.default_rng(random_state)

    embeddings = []
    labels = []

    artists_per_group = n_artists // n_groups

    for group_id in range(n_groups):
        center = rng.normal(size=(embedding_dim,))
        center = center / np.linalg.norm(center)

        group_embeddings = center + 0.22 * rng.normal(
            size=(artists_per_group, embedding_dim)
        )

        # L2 normalize to imitate cosine-based artist embeddings
        group_embeddings = group_embeddings / np.linalg.norm(
            group_embeddings,
            axis=1,
            keepdims=True
        )

        embeddings.append(group_embeddings)
        labels.extend([f"Genre {chr(65 + group_id)}"] * artists_per_group)

    embeddings = np.vstack(embeddings)
    labels = np.array(labels)

    return embeddings, labels


def make_figure_9_latent_space_visualization():
    # ========================================================
    # Option A: synthetic example
    # ========================================================
    embeddings, labels = make_synthetic_artist_embeddings()

    # ========================================================
    # Option B: use real embeddings
    # Uncomment and adapt these lines when you have real data:
    #
    # embeddings = np.load("artist_embeddings.npy")
    # labels = np.load("artist_labels.npy", allow_pickle=True)
    #
    # Expected:
    # embeddings.shape == (num_artists, 256)
    # labels.shape == (num_artists,)
    # ========================================================

    coords = simple_pca(embeddings, n_components=2)

    fig, ax = plt.subplots(figsize=(7.7, 6.6))
    style_axis(ax)

    unique_labels = list(dict.fromkeys(labels))
    legend_handles = []

    for idx, label in enumerate(unique_labels):
        mask = labels == label
        color = PALETTE[idx % len(PALETTE)]
        points = coords[mask]

        add_group_ellipse(ax, points, color)
        ax.scatter(
            points[:, 0],
            points[:, 1],
            s=42,
            alpha=0.82,
            color=color,
            edgecolors=PANEL_BG,
            linewidths=0.55,
            label=label
        )

        centroid = points.mean(axis=0)
        ax.text(
            centroid[0],
            centroid[1],
            label,
            ha="center",
            va="center",
            fontsize=9,
            weight="bold",
            color=color,
            bbox={
                "boxstyle": "round,pad=0.25",
                "facecolor": PANEL_BG,
                "edgecolor": color,
                "linewidth": 0.7,
                "alpha": 0.9,
            },
            zorder=5,
        )
        legend_handles.append(
            Line2D(
                [0],
                [0],
                marker="o",
                color="none",
                markerfacecolor=color,
                markeredgecolor=PANEL_BG,
                markersize=8,
                label=label,
            )
        )

    ax.axhline(0, linewidth=0.9, color="#9aa8b8", alpha=0.7)
    ax.axvline(0, linewidth=0.9, color="#9aa8b8", alpha=0.7)
    ax.set_title("Artist Latent Space Visualization", weight="bold", pad=12)
    ax.set_xlabel("Component 1")
    ax.set_ylabel("Component 2")
    ax.legend(
        handles=legend_handles,
        title="Metadata group",
        frameon=True,
        facecolor=PANEL_BG,
        edgecolor=PANEL_EDGE,
        loc="upper right",
    )

    fig.text(
        0.5,
        0.015,
        "Each point represents one artist embedding projected from 256 dimensions to 2 dimensions.",
        ha="center",
        va="bottom",
        fontsize=9,
        color=MUTED,
    )

    fig.tight_layout(rect=[0, 0.045, 1, 1])
    save_figure(fig, "fig9_latent_space_visualization")


# ============================================================
# Main
# ============================================================

if __name__ == "__main__":
    make_figure_1_dimensionality_reduction()
    make_figure_2_activation_functions()
    make_figure_8_cosine_similarity_geometry()
    make_figure_9_latent_space_visualization()
