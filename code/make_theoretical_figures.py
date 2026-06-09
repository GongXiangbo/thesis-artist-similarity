import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, Arc, Rectangle
from matplotlib.lines import Line2D


# ============================================================
# Global settings
# ============================================================

OUTPUT_DIR = "figures"
os.makedirs(OUTPUT_DIR, exist_ok=True)

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 11,
    "axes.titlesize": 12,
    "axes.labelsize": 10,
    "legend.fontsize": 9,
    "figure.titlesize": 13,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})


def save_figure(fig, filename_base):
    pdf_path = os.path.join(OUTPUT_DIR, f"{filename_base}.pdf")
    png_path = os.path.join(OUTPUT_DIR, f"{filename_base}.png")
    fig.savefig(pdf_path, bbox_inches="tight")
    fig.savefig(png_path, bbox_inches="tight", dpi=300)
    plt.close(fig)
    print(f"Saved: {pdf_path}")
    print(f"Saved: {png_path}")


def simple_pca(X, n_components=2):
    """
    Minimal PCA implementation using NumPy.
    X: array of shape (n_samples, n_features)
    """
    X = np.asarray(X, dtype=float)
    X_centered = X - X.mean(axis=0, keepdims=True)

    # SVD-based PCA
    U, S, Vt = np.linalg.svd(X_centered, full_matrices=False)
    return X_centered @ Vt[:n_components].T


# ============================================================
# Figure 1: Dimensionality Reduction
# ============================================================

def make_figure_1_dimensionality_reduction():
    rng = np.random.default_rng(42)

    fig, ax = plt.subplots(figsize=(11, 4.2))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    # Left: high-dimensional embedding matrix
    matrix_x, matrix_y = 0.07, 0.23
    matrix_w, matrix_h = 0.24, 0.52

    ax.add_patch(Rectangle(
        (matrix_x, matrix_y),
        matrix_w,
        matrix_h,
        fill=False,
        linewidth=1.5
    ))

    rows, cols = 10, 12
    cell_w = matrix_w / cols
    cell_h = matrix_h / rows

    for i in range(rows):
        for j in range(cols):
            value = rng.random()
            shade = 0.25 + 0.65 * value
            ax.add_patch(Rectangle(
                (matrix_x + j * cell_w, matrix_y + i * cell_h),
                cell_w * 0.9,
                cell_h * 0.85,
                facecolor=str(shade),
                edgecolor="white",
                linewidth=0.4
            ))

    ax.text(
        matrix_x + matrix_w / 2,
        matrix_y + matrix_h + 0.075,
        "256-dimensional\nartist embeddings",
        ha="center",
        va="bottom",
        weight="bold"
    )

    ax.text(
        matrix_x + matrix_w / 2,
        matrix_y - 0.07,
        "High-dimensional feature space",
        ha="center",
        va="top"
    )

    # Center: PCA / t-SNE block
    block_x, block_y = 0.43, 0.42
    block_w, block_h = 0.16, 0.16

    ax.add_patch(Rectangle(
        (block_x, block_y),
        block_w,
        block_h,
        fill=False,
        linewidth=1.5
    ))

    ax.text(
        block_x + block_w / 2,
        block_y + block_h / 2,
        "PCA / t-SNE",
        ha="center",
        va="center",
        weight="bold"
    )

    # Right: 2D scatter plot
    scatter_x, scatter_y = 0.70, 0.20
    scatter_w, scatter_h = 0.23, 0.56

    ax.add_patch(Rectangle(
        (scatter_x, scatter_y),
        scatter_w,
        scatter_h,
        fill=False,
        linewidth=1.2
    ))

    # Synthetic clustered scatter
    centers = np.array([
        [0.30, 0.65],
        [0.65, 0.62],
        [0.48, 0.32],
    ])

    for c in centers:
        pts = c + 0.06 * rng.normal(size=(18, 2))
        pts[:, 0] = np.clip(pts[:, 0], 0.08, 0.92)
        pts[:, 1] = np.clip(pts[:, 1], 0.08, 0.92)

        ax.scatter(
            scatter_x + pts[:, 0] * scatter_w,
            scatter_y + pts[:, 1] * scatter_h,
            s=28,
            alpha=0.85,
            edgecolors="black",
            linewidths=0.3
        )

    ax.text(
        scatter_x + scatter_w / 2,
        scatter_y + scatter_h + 0.075,
        "2D latent space\nvisualization",
        ha="center",
        va="bottom",
        weight="bold"
    )

    ax.text(
        scatter_x + scatter_w / 2,
        scatter_y - 0.07,
        "Artists projected into two dimensions",
        ha="center",
        va="top"
    )

    # Arrows
    arrow1 = FancyArrowPatch(
        (matrix_x + matrix_w + 0.03, 0.50),
        (block_x - 0.03, 0.50),
        arrowstyle="-|>",
        mutation_scale=15,
        linewidth=1.5
    )
    arrow2 = FancyArrowPatch(
        (block_x + block_w + 0.03, 0.50),
        (scatter_x - 0.03, 0.50),
        arrowstyle="-|>",
        mutation_scale=15,
        linewidth=1.5
    )

    ax.add_patch(arrow1)
    ax.add_patch(arrow2)

    fig.suptitle("Dimensionality Reduction for Artist Embeddings", y=0.98)
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
        ("Linear", linear, r"$f(x)=x$"),
        ("Sigmoid", sigmoid, r"$f(x)=\frac{1}{1+e^{-x}}$"),
        ("Tanh", tanh, r"$f(x)=\tanh(x)$"),
        ("ReLU", relu, r"$f(x)=\max(0,x)$"),
        ("GELU", gelu, r"$f(x)\approx0.5x(1+\tanh(\sqrt{2/\pi}(x+0.044715x^3)))$"),
    ]

    fig, axes = plt.subplots(2, 3, figsize=(11, 6.6))
    axes = axes.ravel()

    for idx, (name, y, formula) in enumerate(functions):
        ax = axes[idx]
        ax.plot(x, y, linewidth=2)
        ax.axhline(0, linewidth=0.8)
        ax.axvline(0, linewidth=0.8)
        ax.grid(True, linewidth=0.4, alpha=0.35)
        ax.set_title(name, weight="bold")
        ax.set_xlabel("x")
        ax.set_ylabel("f(x)")
        ax.text(
            0.5,
            -0.28,
            formula,
            transform=ax.transAxes,
            ha="center",
            va="top",
            fontsize=9
        )

    # Hide the empty sixth subplot
    axes[-1].axis("off")

    fig.suptitle("Common Activation Functions in Neural Networks", y=0.99)
    fig.tight_layout(rect=[0, 0.02, 1, 0.95])
    save_figure(fig, "fig2_activation_functions")


# ============================================================
# Figure 8: Cosine Similarity Geometry
# ============================================================

def draw_vector(ax, start, end, label):
    arrow = FancyArrowPatch(
        start,
        end,
        arrowstyle="-|>",
        mutation_scale=16,
        linewidth=2
    )
    ax.add_patch(arrow)
    ax.text(
        end[0] + 0.05,
        end[1] + 0.03,
        label,
        fontsize=11,
        weight="bold"
    )


def make_figure_8_cosine_similarity_geometry():
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.4))

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
        ax.set_aspect("equal")
        ax.set_xlim(-0.1, 1.35)
        ax.set_ylim(-0.1, 1.35)

        ax.axhline(0, linewidth=0.8)
        ax.axvline(0, linewidth=0.8)

        ax.set_xlabel("Embedding dimension 1")
        ax.set_ylabel("Embedding dimension 2")
        ax.set_title(ex["title"], weight="bold")

        origin = np.array([0.0, 0.0])
        v1 = ex["v1"]
        v2 = ex["v2"]

        draw_vector(ax, origin, v1, r"$\mathbf{u}$")
        draw_vector(ax, origin, v2, r"$\mathbf{v}$")

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
            linewidth=1.5
        )
        ax.add_patch(arc)

        mid_angle = np.radians((theta1 + theta2) / 2)
        ax.text(
            0.38 * np.cos(mid_angle),
            0.38 * np.sin(mid_angle),
            ex["theta_label"],
            ha="center",
            va="center"
        )

        similarity = np.dot(v1, v2) / (
            np.linalg.norm(v1) * np.linalg.norm(v2)
        )

        ax.text(
            0.03,
            1.22,
            rf"$\cos(\theta) = {similarity:.2f}$",
            fontsize=11
        )

        ax.grid(True, linewidth=0.4, alpha=0.35)

    fig.text(
        0.5,
        -0.02,
        "Cosine similarity depends on vector direction rather than magnitude.",
        ha="center",
        fontsize=11
    )

    fig.suptitle("Geometric Interpretation of Cosine Similarity", y=1.02)
    fig.tight_layout()
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

    fig, ax = plt.subplots(figsize=(7.2, 6.2))

    unique_labels = list(dict.fromkeys(labels))

    for label in unique_labels:
        mask = labels == label
        ax.scatter(
            coords[mask, 0],
            coords[mask, 1],
            s=34,
            alpha=0.78,
            edgecolors="black",
            linewidths=0.25,
            label=label
        )

    ax.set_title("Artist Latent Space Visualization", weight="bold")
    ax.set_xlabel("Component 1")
    ax.set_ylabel("Component 2")
    ax.grid(True, linewidth=0.4, alpha=0.35)
    ax.legend(title="Metadata group", frameon=True)

    ax.text(
        0.02,
        -0.12,
        "Each point represents one artist embedding projected from 256 dimensions to 2 dimensions.",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=9
    )

    fig.tight_layout()
    save_figure(fig, "fig9_latent_space_visualization")


# ============================================================
# Main
# ============================================================

if __name__ == "__main__":
    make_figure_1_dimensionality_reduction()
    make_figure_2_activation_functions()
    make_figure_8_cosine_similarity_geometry()
    make_figure_9_latent_space_visualization()