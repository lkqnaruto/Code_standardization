"""
LOB-Level Document Embedding Divergence Analysis
=================================================
Analyzes whether documents differ systematically at the LOB (Line of Business)
level versus the overall corpus level using BERT embeddings and multiple
statistical methods.

Methods included:
  1. Centroid Distance (cosine & euclidean) between LOB and global mean embeddings
  2. Permutation Test for statistical significance of centroid distances
  3. MMD (Maximum Mean Discrepancy) - kernel-based distribution comparison
  4. Multivariate Hotelling's T² Test - parametric test for mean differences
  5. Intra-LOB vs Inter-LOB Cohesion (silhouette-style analysis)
  6. PCA / t-SNE Visualization

Requirements:
  pip install torch transformers pandas numpy scipy scikit-learn matplotlib seaborn tqdm
"""

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, field

# ---------------------------------------------------------------------------
# 1. Embedding computation
# ---------------------------------------------------------------------------

def load_model(model_name: str = "bert-base-uncased", device: Optional[str] = None):
    """Load a BERT-based model and tokenizer.

    Args:
        model_name: Any HuggingFace model name compatible with AutoModel.
            Examples: 'bert-base-uncased', 'sentence-transformers/all-MiniLM-L6-v2',
                      'dmis-lab/biobert-base-cased-v1.2'
        device: 'cpu', 'cuda', or None (auto-detect).

    Returns:
        (tokenizer, model, device)
    """
    import torch
    from transformers import AutoTokenizer, AutoModel

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name).to(device)
    model.eval()

    print(f"Loaded '{model_name}' on {device}  |  hidden_size={model.config.hidden_size}")
    return tokenizer, model, device


def compute_embeddings(
    documents: List[str],
    tokenizer,
    model,
    device: str,
    batch_size: int = 32,
    max_length: int = 512,
    pooling: str = "mean",          # 'mean' | 'cls'
) -> np.ndarray:
    """Encode documents into embeddings using mean-pooling (default) or [CLS].

    Args:
        documents: list of document strings.
        pooling: 'mean' for mean of all token embeddings (recommended),
                 'cls' for the [CLS] token embedding.

    Returns:
        np.ndarray of shape (n_documents, hidden_size).
    """
    import torch
    from tqdm import tqdm

    all_embeddings = []

    for start in tqdm(range(0, len(documents), batch_size), desc="Encoding"):
        batch_texts = documents[start : start + batch_size]
        encoded = tokenizer(
            batch_texts,
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        ).to(device)

        with torch.no_grad():
            outputs = model(**encoded)

        if pooling == "cls":
            emb = outputs.last_hidden_state[:, 0, :]          # [CLS] token
        else:
            # Mean pooling over non-padding tokens
            attention_mask = encoded["attention_mask"].unsqueeze(-1)
            token_embs = outputs.last_hidden_state * attention_mask
            emb = token_embs.sum(dim=1) / attention_mask.sum(dim=1)

        all_embeddings.append(emb.cpu().numpy())

    return np.vstack(all_embeddings)


# ---------------------------------------------------------------------------
# 2. Core analysis: centroid distances
# ---------------------------------------------------------------------------

@dataclass
class LOBResult:
    """Results for a single LOB."""
    lob: str
    n_docs: int
    cosine_distance: float
    euclidean_distance: float
    cosine_similarity: float
    # Permutation test
    perm_p_value: Optional[float] = None
    # MMD
    mmd_value: Optional[float] = None
    mmd_p_value: Optional[float] = None
    # Hotelling T²
    hotelling_t2: Optional[float] = None
    hotelling_p_value: Optional[float] = None
    # Cohesion
    intra_cohesion: Optional[float] = None
    inter_cohesion: Optional[float] = None
    cohesion_ratio: Optional[float] = None


def cosine_distance(a: np.ndarray, b: np.ndarray) -> float:
    """Cosine distance = 1 - cosine_similarity."""
    from numpy.linalg import norm
    cos_sim = np.dot(a, b) / (norm(a) * norm(b) + 1e-10)
    return float(1.0 - cos_sim)


def compute_centroid_distances(
    embeddings: np.ndarray,
    lob_labels: np.ndarray,
) -> Dict[str, LOBResult]:
    """Compute distance between each LOB centroid and the global centroid."""
    from numpy.linalg import norm

    global_mean = embeddings.mean(axis=0)
    unique_lobs = np.unique(lob_labels)
    results: Dict[str, LOBResult] = {}

    for lob in unique_lobs:
        mask = lob_labels == lob
        lob_embs = embeddings[mask]
        lob_mean = lob_embs.mean(axis=0)

        cos_dist = cosine_distance(lob_mean, global_mean)
        euc_dist = float(norm(lob_mean - global_mean))
        cos_sim = 1.0 - cos_dist

        results[lob] = LOBResult(
            lob=lob,
            n_docs=int(mask.sum()),
            cosine_distance=cos_dist,
            euclidean_distance=euc_dist,
            cosine_similarity=cos_sim,
        )

    return results


# ---------------------------------------------------------------------------
# 3. Permutation test for centroid distance significance
# ---------------------------------------------------------------------------

def permutation_test_centroid(
    embeddings: np.ndarray,
    lob_labels: np.ndarray,
    lob: str,
    n_permutations: int = 1000,
    metric: str = "cosine",
    seed: int = 42,
) -> float:
    """Test H0: LOB centroid is no further from global mean than random subsets.

    Randomly reassigns documents to a pseudo-LOB of the same size and measures
    the centroid distance. Returns a p-value.
    """
    rng = np.random.RandomState(seed)
    mask = lob_labels == lob
    n_lob = mask.sum()
    global_mean = embeddings.mean(axis=0)

    lob_mean = embeddings[mask].mean(axis=0)
    if metric == "cosine":
        observed = cosine_distance(lob_mean, global_mean)
    else:
        observed = float(np.linalg.norm(lob_mean - global_mean))

    count_ge = 0
    for _ in range(n_permutations):
        idx = rng.choice(len(embeddings), size=n_lob, replace=False)
        rand_mean = embeddings[idx].mean(axis=0)
        if metric == "cosine":
            d = cosine_distance(rand_mean, global_mean)
        else:
            d = float(np.linalg.norm(rand_mean - global_mean))
        if d >= observed:
            count_ge += 1

    return (count_ge + 1) / (n_permutations + 1)   # conservative p-value


# ---------------------------------------------------------------------------
# 4. MMD (Maximum Mean Discrepancy)
# ---------------------------------------------------------------------------

def compute_mmd(
    X: np.ndarray,
    Y: np.ndarray,
    kernel: str = "rbf",
    gamma: Optional[float] = None,
) -> float:
    """Compute MMD² between two sets of samples using an RBF kernel.

    MMD measures distributional distance — useful when LOB might differ in
    spread/shape, not just centroid location.
    """
    from sklearn.metrics.pairwise import rbf_kernel

    if gamma is None:
        # Median heuristic
        from scipy.spatial.distance import pdist
        combined = np.vstack([X, Y])
        gamma = 1.0 / np.median(pdist(combined, "sqeuclidean"))

    K_xx = rbf_kernel(X, X, gamma=gamma)
    K_yy = rbf_kernel(Y, Y, gamma=gamma)
    K_xy = rbf_kernel(X, Y, gamma=gamma)

    mmd2 = K_xx.mean() + K_yy.mean() - 2 * K_xy.mean()
    return float(max(mmd2, 0.0))


def mmd_permutation_test(
    X: np.ndarray,
    Y: np.ndarray,
    n_permutations: int = 500,
    seed: int = 42,
) -> Tuple[float, float]:
    """Permutation test for MMD significance."""
    rng = np.random.RandomState(seed)
    observed_mmd = compute_mmd(X, Y)
    combined = np.vstack([X, Y])
    nx = len(X)

    count_ge = 0
    for _ in range(n_permutations):
        perm = rng.permutation(len(combined))
        X_perm = combined[perm[:nx]]
        Y_perm = combined[perm[nx:]]
        if compute_mmd(X_perm, Y_perm) >= observed_mmd:
            count_ge += 1

    p_value = (count_ge + 1) / (n_permutations + 1)
    return observed_mmd, p_value


# ---------------------------------------------------------------------------
# 5. Hotelling's T² test
# ---------------------------------------------------------------------------

def hotelling_t2_test(
    X: np.ndarray,
    Y: np.ndarray,
    n_components: int = 50,
) -> Tuple[float, float]:
    """Hotelling's T² test for multivariate mean equality.

    Since BERT embeddings are 768-d (or more), we PCA-reduce to n_components
    first to make the covariance matrix invertible.

    Returns (T², p_value).
    """
    from sklearn.decomposition import PCA
    from scipy.stats import f as f_dist

    # Reduce dimensionality
    combined = np.vstack([X, Y])
    n_comp = min(n_components, combined.shape[0] - 2, combined.shape[1])
    pca = PCA(n_components=n_comp)
    combined_pca = pca.fit_transform(combined)
    X_r = combined_pca[: len(X)]
    Y_r = combined_pca[len(X) :]

    n1, p = X_r.shape
    n2 = Y_r.shape[0]

    mean_diff = X_r.mean(axis=0) - Y_r.mean(axis=0)

    # Pooled covariance
    S1 = np.cov(X_r, rowvar=False, ddof=1) if n1 > 1 else np.zeros((p, p))
    S2 = np.cov(Y_r, rowvar=False, ddof=1) if n2 > 1 else np.zeros((p, p))
    S_pool = ((n1 - 1) * S1 + (n2 - 1) * S2) / (n1 + n2 - 2)

    # Regularize
    S_pool += np.eye(p) * 1e-6

    S_inv = np.linalg.inv(S_pool)
    t2 = (n1 * n2) / (n1 + n2) * mean_diff @ S_inv @ mean_diff

    # Convert to F-statistic
    df1 = p
    df2 = n1 + n2 - p - 1
    if df2 <= 0:
        return float(t2), np.nan
    f_stat = t2 * df2 / (df1 * (n1 + n2 - 2))
    p_value = 1.0 - f_dist.cdf(f_stat, df1, df2)

    return float(t2), float(p_value)


# ---------------------------------------------------------------------------
# 6. Intra-LOB cohesion vs. inter-LOB distance
# ---------------------------------------------------------------------------

def compute_cohesion_metrics(
    embeddings: np.ndarray,
    lob_labels: np.ndarray,
) -> Dict[str, Tuple[float, float, float]]:
    """For each LOB compute:
      - intra_cohesion:  mean pairwise cosine sim WITHIN the LOB
      - inter_cohesion:  mean cosine sim between LOB docs and ALL other docs
      - ratio:           intra / inter  (>1 means LOB is more internally coherent)
    """
    from sklearn.metrics.pairwise import cosine_similarity

    sim_matrix = cosine_similarity(embeddings)
    unique_lobs = np.unique(lob_labels)
    results = {}

    for lob in unique_lobs:
        mask = lob_labels == lob
        idx_in = np.where(mask)[0]
        idx_out = np.where(~mask)[0]

        if len(idx_in) < 2 or len(idx_out) == 0:
            results[lob] = (np.nan, np.nan, np.nan)
            continue

        intra_sims = sim_matrix[np.ix_(idx_in, idx_in)]
        # Exclude diagonal
        np.fill_diagonal(intra_sims, np.nan)
        intra_mean = float(np.nanmean(intra_sims))

        inter_sims = sim_matrix[np.ix_(idx_in, idx_out)]
        inter_mean = float(np.nanmean(inter_sims))

        ratio = intra_mean / (inter_mean + 1e-10)
        results[lob] = (intra_mean, inter_mean, ratio)

    return results


# ---------------------------------------------------------------------------
# 7. Visualization helpers
# ---------------------------------------------------------------------------

def plot_pca(
    embeddings: np.ndarray,
    lob_labels: np.ndarray,
    save_path: str = "pca_lob.png",
):
    """2-D PCA scatter colored by LOB with centroids marked."""
    from sklearn.decomposition import PCA
    import matplotlib.pyplot as plt

    pca = PCA(n_components=2)
    coords = pca.fit_transform(embeddings)

    unique_lobs = np.unique(lob_labels)
    fig, ax = plt.subplots(figsize=(10, 7))
    cmap = plt.cm.get_cmap("tab10", len(unique_lobs))

    for i, lob in enumerate(unique_lobs):
        mask = lob_labels == lob
        ax.scatter(
            coords[mask, 0], coords[mask, 1],
            label=lob, alpha=0.5, s=30, color=cmap(i),
        )
        centroid = coords[mask].mean(axis=0)
        ax.scatter(*centroid, marker="X", s=200, edgecolors="black",
                   linewidths=1.5, color=cmap(i), zorder=5)

    # Global centroid
    global_c = coords.mean(axis=0)
    ax.scatter(*global_c, marker="*", s=400, color="black", zorder=6,
               label="Global Centroid")

    var_explained = pca.explained_variance_ratio_
    ax.set_xlabel(f"PC1 ({var_explained[0]:.1%} variance)")
    ax.set_ylabel(f"PC2 ({var_explained[1]:.1%} variance)")
    ax.set_title("PCA of Document Embeddings by LOB")
    ax.legend(bbox_to_anchor=(1.05, 1), loc="upper left", fontsize=8)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"PCA plot saved to {save_path}")


def plot_tsne(
    embeddings: np.ndarray,
    lob_labels: np.ndarray,
    save_path: str = "tsne_lob.png",
    perplexity: int = 30,
    seed: int = 42,
):
    """2-D t-SNE scatter colored by LOB."""
    from sklearn.manifold import TSNE
    import matplotlib.pyplot as plt

    tsne = TSNE(n_components=2, perplexity=min(perplexity, len(embeddings) - 1),
                random_state=seed, init="pca", learning_rate="auto")
    coords = tsne.fit_transform(embeddings)

    unique_lobs = np.unique(lob_labels)
    fig, ax = plt.subplots(figsize=(10, 7))
    cmap = plt.cm.get_cmap("tab10", len(unique_lobs))

    for i, lob in enumerate(unique_lobs):
        mask = lob_labels == lob
        ax.scatter(coords[mask, 0], coords[mask, 1],
                   label=lob, alpha=0.5, s=30, color=cmap(i))

    ax.set_title("t-SNE of Document Embeddings by LOB")
    ax.legend(bbox_to_anchor=(1.05, 1), loc="upper left", fontsize=8)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"t-SNE plot saved to {save_path}")


def plot_distance_heatmap(
    embeddings: np.ndarray,
    lob_labels: np.ndarray,
    save_path: str = "lob_distance_heatmap.png",
):
    """Heatmap of pairwise cosine distances between LOB centroids."""
    import matplotlib.pyplot as plt
    import seaborn as sns

    unique_lobs = sorted(np.unique(lob_labels))
    centroids = np.array([
        embeddings[lob_labels == lob].mean(axis=0) for lob in unique_lobs
    ])

    # Pairwise cosine distance
    from sklearn.metrics.pairwise import cosine_distances
    dist_matrix = cosine_distances(centroids)

    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(
        dist_matrix,
        xticklabels=unique_lobs,
        yticklabels=unique_lobs,
        annot=True, fmt=".4f", cmap="YlOrRd", ax=ax,
    )
    ax.set_title("Pairwise Cosine Distance Between LOB Centroids")
    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Heatmap saved to {save_path}")


# ---------------------------------------------------------------------------
# 8. Full pipeline
# ---------------------------------------------------------------------------

def run_analysis(
    df: pd.DataFrame,
    doc_col: str = "A",
    lob_col: str = "B",
    model_name: str = "bert-base-uncased",
    n_permutations: int = 1000,
    run_mmd: bool = True,
    run_hotelling: bool = True,
    run_visualizations: bool = True,
    output_dir: str = ".",
) -> pd.DataFrame:
    """End-to-end analysis pipeline.

    Args:
        df: DataFrame with document text and LOB labels.
        doc_col: column name for documents.
        lob_col: column name for LOB labels.
        model_name: HuggingFace model to use for embeddings.
        n_permutations: number of permutations for significance tests.
        run_mmd: whether to compute MMD (can be slow for large N).
        run_hotelling: whether to run Hotelling's T² test.
        run_visualizations: whether to generate plots.
        output_dir: directory for saving plots and CSV.

    Returns:
        Summary DataFrame with one row per LOB.
    """
    import os

    documents = df[doc_col].astype(str).tolist()
    lob_labels = df[lob_col].astype(str).values

    # --- Embeddings ---
    tokenizer, model, device = load_model(model_name)
    embeddings = compute_embeddings(documents, tokenizer, model, device)
    print(f"Embedding matrix shape: {embeddings.shape}")

    global_mean = embeddings.mean(axis=0)
    print(f"Global mean embedding norm: {np.linalg.norm(global_mean):.4f}")

    # --- Centroid distances ---
    results = compute_centroid_distances(embeddings, lob_labels)

    # --- Permutation tests ---
    print("\nRunning permutation tests...")
    for lob in results:
        p = permutation_test_centroid(
            embeddings, lob_labels, lob,
            n_permutations=n_permutations, metric="cosine",
        )
        results[lob].perm_p_value = p

    # --- MMD ---
    if run_mmd:
        print("Running MMD tests...")
        for lob in results:
            mask = lob_labels == lob
            lob_embs = embeddings[mask]
            other_embs = embeddings[~mask]
            # Subsample other if very large (MMD is O(n²))
            if len(other_embs) > 2000:
                idx = np.random.choice(len(other_embs), 2000, replace=False)
                other_embs = other_embs[idx]
            mmd_val, mmd_p = mmd_permutation_test(
                lob_embs, other_embs, n_permutations=min(n_permutations, 500),
            )
            results[lob].mmd_value = mmd_val
            results[lob].mmd_p_value = mmd_p

    # --- Hotelling T² ---
    if run_hotelling:
        print("Running Hotelling's T² tests...")
        for lob in results:
            mask = lob_labels == lob
            if mask.sum() < 3:
                continue
            lob_embs = embeddings[mask]
            other_embs = embeddings[~mask]
            t2, p = hotelling_t2_test(lob_embs, other_embs)
            results[lob].hotelling_t2 = t2
            results[lob].hotelling_p_value = p

    # --- Cohesion ---
    print("Computing cohesion metrics...")
    cohesion = compute_cohesion_metrics(embeddings, lob_labels)
    for lob, (intra, inter, ratio) in cohesion.items():
        results[lob].intra_cohesion = intra
        results[lob].inter_cohesion = inter
        results[lob].cohesion_ratio = ratio

    # --- Build summary table ---
    rows = []
    for lob in sorted(results.keys()):
        r = results[lob]
        rows.append({
            "LOB": r.lob,
            "N_Docs": r.n_docs,
            "Cosine_Distance": round(r.cosine_distance, 6),
            "Euclidean_Distance": round(r.euclidean_distance, 4),
            "Cosine_Similarity": round(r.cosine_similarity, 6),
            "Perm_PValue": round(r.perm_p_value, 4) if r.perm_p_value is not None else None,
            "MMD": round(r.mmd_value, 6) if r.mmd_value is not None else None,
            "MMD_PValue": round(r.mmd_p_value, 4) if r.mmd_p_value is not None else None,
            "Hotelling_T2": round(r.hotelling_t2, 2) if r.hotelling_t2 is not None else None,
            "Hotelling_PValue": round(r.hotelling_p_value, 4) if r.hotelling_p_value is not None else None,
            "Intra_Cohesion": round(r.intra_cohesion, 4) if r.intra_cohesion is not None else None,
            "Inter_Cohesion": round(r.inter_cohesion, 4) if r.inter_cohesion is not None else None,
            "Cohesion_Ratio": round(r.cohesion_ratio, 4) if r.cohesion_ratio is not None else None,
        })

    summary_df = pd.DataFrame(rows)

    # --- Visualizations ---
    if run_visualizations:
        print("\nGenerating visualizations...")
        plot_pca(embeddings, lob_labels,
                 save_path=os.path.join(output_dir, "pca_lob.png"))
        if len(embeddings) >= 10:
            plot_tsne(embeddings, lob_labels,
                      save_path=os.path.join(output_dir, "tsne_lob.png"))
        plot_distance_heatmap(embeddings, lob_labels,
                              save_path=os.path.join(output_dir, "lob_distance_heatmap.png"))

    # --- Save CSV ---
    csv_path = os.path.join(output_dir, "lob_analysis_results.csv")
    summary_df.to_csv(csv_path, index=False)
    print(f"\nResults saved to {csv_path}")

    # --- Print summary ---
    print("\n" + "=" * 80)
    print("LOB EMBEDDING DIVERGENCE SUMMARY")
    print("=" * 80)
    print(summary_df.to_string(index=False))
    print("\nInterpretation guide:")
    print("  Cosine_Distance   : 0 = identical direction, 2 = opposite. Larger = more divergent.")
    print("  Perm_PValue       : < 0.05 suggests LOB centroid is significantly different from random.")
    print("  MMD               : Distributional distance. Higher = more different spread/shape.")
    print("  Hotelling_PValue  : < 0.05 suggests LOB mean embedding significantly differs.")
    print("  Cohesion_Ratio    : > 1.0 means LOB docs are more similar to each other than to others.")

    return summary_df


# ---------------------------------------------------------------------------
# 9. Demo with synthetic data
# ---------------------------------------------------------------------------

def create_demo_data(n_per_lob: int = 50) -> pd.DataFrame:
    """Create a small synthetic dataset for demonstration."""
    lobs = ["Commercial Lending", "Retail Banking", "Wealth Management", "Insurance"]
    docs = {
        "Commercial Lending": [
            "The commercial loan application was reviewed for credit risk assessment.",
            "Corporate borrowers must provide audited financial statements.",
            "Loan covenants require maintaining a debt-to-equity ratio below 3.",
            "The syndicated lending facility was structured with multiple tranches.",
            "Underwriting standards for commercial real estate loans were tightened.",
        ],
        "Retail Banking": [
            "The customer opened a new checking account with direct deposit.",
            "Mobile banking adoption has increased significantly among millennials.",
            "ATM transaction limits were updated in the latest policy revision.",
            "Overdraft protection enrollment is available for eligible accounts.",
            "The branch network expanded with three new locations this quarter.",
        ],
        "Wealth Management": [
            "The portfolio allocation was rebalanced toward fixed income securities.",
            "High-net-worth clients received tailored estate planning advice.",
            "Alternative investment strategies including hedge funds were discussed.",
            "The trust administration process requires annual beneficiary reviews.",
            "Tax-loss harvesting opportunities were identified in the equity portfolio.",
        ],
        "Insurance": [
            "The underwriting team evaluated the property casualty risk profile.",
            "Claims processing time decreased following the new automation system.",
            "Actuarial models were updated with the latest mortality tables.",
            "Reinsurance treaty negotiations for the upcoming year have begun.",
            "Policyholder retention rates improved with the new loyalty program.",
        ],
    }

    rows = []
    for lob in lobs:
        base_docs = docs[lob]
        for i in range(n_per_lob):
            doc = base_docs[i % len(base_docs)]
            # Add slight variation
            if i >= len(base_docs):
                doc = doc + f" (variant {i})"
            rows.append({"A": doc, "B": lob})

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse, os

    parser = argparse.ArgumentParser(description="LOB Embedding Divergence Analysis")
    parser.add_argument("--input", type=str, default=None,
                        help="Path to input CSV. Must have columns specified by --doc-col and --lob-col.")
    parser.add_argument("--doc-col", type=str, default="A",
                        help="Column name for documents (default: 'A').")
    parser.add_argument("--lob-col", type=str, default="B",
                        help="Column name for LOB labels (default: 'B').")
    parser.add_argument("--model", type=str, default="bert-base-uncased",
                        help="HuggingFace model name (default: 'bert-base-uncased').")
    parser.add_argument("--permutations", type=int, default=1000,
                        help="Number of permutations for significance tests.")
    parser.add_argument("--output-dir", type=str, default="./results",
                        help="Directory for output files.")
    parser.add_argument("--demo", action="store_true",
                        help="Run with synthetic demo data.")

    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    if args.demo or args.input is None:
        print("Running with demo data...\n")
        df = create_demo_data(n_per_lob=30)
    else:
        df = pd.read_csv(args.input)

    print(f"Dataset: {len(df)} documents, {df[args.lob_col].nunique()} LOBs")
    print(f"LOB distribution:\n{df[args.lob_col].value_counts().to_string()}\n")

    summary = run_analysis(
        df,
        doc_col=args.doc_col,
        lob_col=args.lob_col,
        model_name=args.model,
        n_permutations=args.permutations,
        output_dir=args.output_dir,
    )
