import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns


def plot_embeddings_umap(embeddings: np.ndarray, labels: list, title: str, save_path: str):
    """ Plots UMAP/t-SNE layout of syllable embeddings. """
    # Placeholder plotting code
    plt.figure(figsize=(8, 6))
    plt.scatter(embeddings[:, 0], embeddings[:, 1], c='purple', alpha=0.7)
    for i, txt in enumerate(labels):
        plt.annotate(txt, (embeddings[i, 0], embeddings[i, 1]), fontsize=9)
    plt.title(title)
    plt.xlabel("UMAP 1")
    plt.ylabel("UMAP 2")
    sns.despine()
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"Saved embedding UMAP plot to {save_path}")
