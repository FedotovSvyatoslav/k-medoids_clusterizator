"""
Иерархическая агломеративная кластеризация.

Мера схожести кластеров (--linkage):
  single   - одиночная связь (расстояние между ближайшими объектами),
  complete - полная связь (между самыми дальними объектами),
  average  - групповое среднее (среднее попарных расстояний),
  ward     - расстояние Уорда (минимальный прирост внутрикластерной суммы квадратов).
Расстояние между объектами - евклидово. Пересчёт расстояний после слияния кластеров выполняется
по формуле Ланса-Уильямса. Результат - матрица слияний Z (строка: номера объединяемых кластеров,
высота слияния, размер нового кластера; формат как у scipy.cluster.hierarchy.linkage).

Предобработка и загрузка данных - те же, что в clustering.py (стандартизация признаков, удаление строк
с пропусками), поэтому результаты можно напрямую сравнивать с k-Means и k-Medoids из clustering.py.

Выход
  dendrogram_<linkage>.png      - дендрограмма (цвет ветвей - кластеры при разрезе на --k кластеров,
                                  пунктир - уровень разреза);
  clusters_<linkage>_k<k>.csv   - координаты объектов и номера кластеров;
  при нескольких мерах: dendrograms_all.png, linkage_metrics.csv, ari_between_linkages.csv/.png,
  quality_vs_k.png, scatter_k<k>.png (разбиения всех мер и k-Means/k-Medoids в проекции на 2 главные
  компоненты), profiles_k<k>.csv (средние значения признаков по кластерам).
Для сравнения с программой clustering.py в метрики добавляются k-Means/k-Medoids при том же k.

Примеры
  python hierarchical.py --data customers.csv --linkage ward --k 3
  python hierarchical.py --data customers.csv --linkage all --k 3
"""
import argparse
import os
import sys

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from clustering import (load_data, standardize, sqdist, ari, evaluate, Projector,  # noqa: E402
                        draw_scatter, ALG_RU)

LINKAGES = ["single", "complete", "average", "ward"]
LINK_RU = {"single": "Single linkage", "complete": "Complete linkage",
           "average": "Group average", "ward": "Ward distance"}


# ----------------------------------------------------------------- алгоритм
def linkage_matrix(X, method):
    """Агломеративная кластеризация; возвращает матрицу слияний Z размера (n-1) x 4."""
    n = len(X)
    d2 = sqdist(X, X)
    M = d2.copy() if method == "ward" else np.sqrt(d2)      # Ward - на квадратах расстояний
    np.fill_diagonal(M, np.inf)
    size = np.ones(n)
    ids = np.arange(n)
    active = np.ones(n, dtype=bool)
    Z = np.zeros((n - 1, 4))
    for step in range(n - 1):
        i, j = divmod(int(np.argmin(M)), n)
        if i > j:
            i, j = j, i
        dij = M[i, j]
        a, b = sorted((ids[i], ids[j]))
        Z[step] = [a, b, np.sqrt(dij) if method == "ward" else dij, size[i] + size[j]]
        ni, nj = size[i], size[j]
        if method == "single":
            new = np.minimum(M[i], M[j])
        elif method == "complete":
            new = np.maximum(M[i], M[j])
        elif method == "average":
            new = (ni * M[i] + nj * M[j]) / (ni + nj)
        else:  # ward (формула Ланса-Уильямса для квадратов расстояний)
            new = ((ni + size) * M[i] + (nj + size) * M[j] - size * dij) / (ni + nj + size)
        active[j] = False
        new[~active] = np.inf
        new[i] = np.inf
        M[i, :] = new
        M[:, i] = new
        M[j, :] = np.inf
        M[:, j] = np.inf
        size[i] = ni + nj
        ids[i] = n + step
    return Z


def cut_tree(Z, n, k):
    """Номера кластеров (0..k-1) при разрезе дендрограммы на k кластеров."""
    members = {i: [i] for i in range(n)}
    for step in range(n - k):
        a, b = int(Z[step, 0]), int(Z[step, 1])
        members[n + step] = members.pop(a) + members.pop(b)
    labels = np.empty(n, dtype=int)
    for c, pts in enumerate(sorted(members.values(), key=lambda p: min(p))):
        labels[pts] = c
    return labels


def cophenetic_corr(Z, D):
    """Кофенетическая корреляция: насколько дендрограмма сохраняет исходные расстояния."""
    n = len(D)
    C = np.zeros((n, n))
    members = {i: [i] for i in range(n)}
    for step in range(n - 1):
        a, b = int(Z[step, 0]), int(Z[step, 1])
        la, lb = members.pop(a), members.pop(b)
        C[np.ix_(la, lb)] = Z[step, 2]
        C[np.ix_(lb, la)] = Z[step, 2]
        members[n + step] = la + lb
    iu = np.triu_indices(n, 1)
    return float(np.corrcoef(C[iu], D[iu])[0, 1])


# ----------------------------------------------------------------- качество разбиения
def sse_of(Xs, labels):
    return float(sum(((Xs[labels == c] - Xs[labels == c].mean(axis=0)) ** 2).sum()
                     for c in np.unique(labels)))


def silhouette(D, labels):
    n = len(labels)
    cl = np.unique(labels)
    sums = np.stack([D[:, labels == c].sum(axis=1) for c in cl], axis=1)
    cnt = np.array([(labels == c).sum() for c in cl])
    own = np.searchsorted(cl, labels)
    a = sums[np.arange(n), own] / np.maximum(cnt[own] - 1, 1)
    means = sums / cnt
    means[np.arange(n), own] = np.inf
    b = means.min(axis=1)
    s = (b - a) / np.maximum(a, b)
    s[cnt[own] == 1] = 0.0
    return float(np.nanmean(s))


# ----------------------------------------------------------------- дендрограмма
def draw_dendrogram(ax, Z, n, k, title, ylabel):
    labels = cut_tree(Z, n, k)
    left_leaf = {}
    # порядок листьев (обход в глубину без рекурсии)
    order, stack = [], [2 * n - 2]
    while stack:
        node = stack.pop()
        if node < n:
            order.append(node)
        else:
            stack.append(int(Z[node - n, 1]))
            stack.append(int(Z[node - n, 0]))
    xpos = {leaf: i for i, leaf in enumerate(order)}
    ypos = {leaf: 0.0 for leaf in range(n)}
    rep = {leaf: leaf for leaf in range(n)}
    segs, cols = [], []
    palette = plt.get_cmap("tab10" if k <= 10 else "tab20")
    for step in range(n - 1):
        a, b = int(Z[step, 0]), int(Z[step, 1])
        h = Z[step, 2]
        segs.append([(xpos[a], ypos[a]), (xpos[a], h), (xpos[b], h), (xpos[b], ypos[b])])
        node = n + step
        xpos[node] = (xpos[a] + xpos[b]) / 2.0
        ypos[node] = h
        rep[node] = rep[a]
        cols.append(palette(labels[rep[node]] % palette.N) if step < n - k else "#8c8c8c")
    ax.add_collection(LineCollection(segs, colors=cols, linewidths=0.9))
    thr = (Z[n - k - 1, 2] + Z[n - k, 2]) / 2.0 if k > 1 else Z[-1, 2]
    ax.axhline(thr, color="r", ls="--", lw=1, label=f"разрез на k = {k} кластера(ов)")
    ax.set_xlim(-1, n)
    ax.set_ylim(0, Z[-1, 2] * 1.05)
    ax.set_xticks([])
    ax.set_xlabel(f"объекты (n = {n}), порядок листьев определяется деревом")
    ax.set_ylabel(ylabel)
    ax.set_title(title, fontsize=11)
    ax.legend(loc="upper left", fontsize=8)


# ----------------------------------------------------------------- основной код
def main():
    ap = argparse.ArgumentParser(description="Иерархическая кластеризация (агломеративная)")
    ap.add_argument("--data", required=True, help="customers.csv (первая строка - заголовок)")
    ap.add_argument("--linkage", default="ward", choices=LINKAGES + ["all"])
    ap.add_argument("--k", type=int, default=3, help="число кластеров при разрезе дендрограммы")
    ap.add_argument("--k-min", type=int, default=2)
    ap.add_argument("--k-max", type=int, default=10)
    ap.add_argument("--features", nargs="+", default=None)
    ap.add_argument("--no-scale", action="store_true")
    ap.add_argument("--no-compare", action="store_true", help="не сравнивать с k-Means/k-Medoids")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default="results_hierarchical")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    df, cols = load_data(args.data, args.features)
    X = df[cols].to_numpy(dtype=float)
    Xs, mu, sd = standardize(X, not args.no_scale)
    n = len(Xs)
    D = np.sqrt(sqdist(Xs, Xs))
    links = LINKAGES if args.linkage == "all" else [args.linkage]
    ks = list(range(args.k_min, args.k_max + 1))
    print(f"объектов: {n}, признаки: {cols}")

    Zs, rows, part = {}, [], {}
    for lk in links:
        Z = linkage_matrix(Xs, lk)
        Zs[lk] = Z
        coph = cophenetic_corr(Z, D)
        for k in sorted(set(ks + [args.k])):
            lab = cut_tree(Z, n, k)
            sizes = sorted(np.bincount(lab).tolist(), reverse=True)
            rows.append({"linkage": lk, "k": k, "sse": sse_of(Xs, lab), "silhouette": silhouette(D, lab),
                         "cophenetic_corr": coph, "sizes": " ".join(map(str, sizes)),
                         "max_cluster_share": sizes[0] / n})
            if k == args.k:
                part[lk] = lab
        lab = part[lk]
        out = df.copy()
        out["cluster"] = lab
        out.to_csv(os.path.join(args.out, f"clusters_{lk}_k{args.k}.csv"), index=False)
        r = [x for x in rows if x["linkage"] == lk and x["k"] == args.k][0]
        print(f"\n=== {LINK_RU[lk]}, k = {args.k} ===")
        print(f"размеры кластеров: {r['sizes']}; SSE = {r['sse']:.1f}; силуэт = {r['silhouette']:.3f}; "
              f"кофенетическая корреляция = {coph:.3f}")
        print("высоты последних слияний:", np.round(Z[-6:, 2], 3).tolist())
        fig, ax = plt.subplots(figsize=(11, 5.2))
        draw_dendrogram(ax, Z, n, args.k, f"Дендрограмма: {LINK_RU[lk]} (евклидово расстояние)",
                        "высота слияния" + (" (расстояние Уорда)" if lk == "ward" else ""))
        fig.tight_layout()
        fig.savefig(os.path.join(args.out, f"dendrogram_{lk}.png"), dpi=150)
        plt.close(fig)

    # --- k-Means / k-Medoids (программа clustering.py) для сравнения
    pa = {}
    if not args.no_compare:
        res, _, _ = evaluate(X, sorted(set(ks + [args.k])), ["kmeans", "kmedoids"], args.seed, not args.no_scale)
        for (alg, k), r in res.items():
            lab = r["labels"]
            rows.append({"linkage": alg, "k": k, "sse": sse_of(Xs, lab), "silhouette": silhouette(D, lab),
                         "cophenetic_corr": np.nan, "sizes": " ".join(map(str, sorted(np.bincount(lab).tolist(), reverse=True))),
                         "max_cluster_share": np.bincount(lab).max() / n})
            if k == args.k:
                pa[alg] = (lab, r)
        # ARI между иерархическими разбиениями и k-Means/k-Medoids
        for row in rows:
            if row["linkage"] in links:
                lab = cut_tree(Zs[row["linkage"]], n, row["k"])
                for alg in ("kmeans", "kmedoids"):
                    row[f"ari_vs_{alg}"] = ari(lab, res[(alg, row["k"])]["labels"])
        print("\nARI иерархических разбиений относительно k-Means / k-Medoids (k = %d):" % args.k)
        for lk in links:
            r = [x for x in rows if x["linkage"] == lk and x["k"] == args.k][0]
            print(f"  {LINK_RU[lk]:17s} {r['ari_vs_kmeans']:.3f} / {r['ari_vs_kmedoids']:.3f}")
    met = pd.DataFrame(rows)
    met.to_csv(os.path.join(args.out, "linkage_metrics.csv"), index=False)

    if len(links) > 1:
        # дендрограммы всех мер
        fig, axs = plt.subplots(2, 2, figsize=(15, 9))
        for ax, lk in zip(axs.ravel(), links):
            draw_dendrogram(ax, Zs[lk], n, args.k, LINK_RU[lk], "высота слияния")
        fig.tight_layout()
        fig.savefig(os.path.join(args.out, "dendrograms_all.png"), dpi=140)
        plt.close(fig)

        # ARI между мерами схожести
        A = np.array([[ari(part[a], part[b]) for b in links] for a in links])
        pd.DataFrame(A, index=links, columns=links).round(3).to_csv(os.path.join(args.out, "ari_between_linkages.csv"))
        fig, ax = plt.subplots(figsize=(5.4, 4.6))
        im = ax.imshow(A, vmin=0, vmax=1, cmap="viridis")
        ax.set_xticks(range(len(links)), [LINK_RU[x] for x in links], rotation=30, ha="right", fontsize=8)
        ax.set_yticks(range(len(links)), [LINK_RU[x] for x in links], fontsize=8)
        for i in range(len(links)):
            for j in range(len(links)):
                ax.text(j, i, f"{A[i, j]:.2f}", ha="center", va="center", color="w" if A[i, j] < 0.6 else "k")
        ax.set_title(f"ARI между разбиениями на k = {args.k} кластера(ов)")
        fig.colorbar(im)
        fig.tight_layout()
        fig.savefig(os.path.join(args.out, "ari_between_linkages.png"), dpi=150)
        plt.close(fig)

        # качество в зависимости от k
        fig, axs = plt.subplots(1, 2, figsize=(13, 4.8))
        for lk in links + ([] if args.no_compare else ["kmeans", "kmedoids"]):
            g = met[(met.linkage == lk) & met.k.isin(ks)].sort_values("k")
            name = LINK_RU.get(lk, ALG_RU.get(lk))
            style = "--" if lk in ("kmeans", "kmedoids") else "-"
            axs[0].plot(g.k, g.sse, style, marker="o", label=name)
            axs[1].plot(g.k, g.silhouette, style, marker="o", label=name)
        axs[0].set_ylabel("Ошибка кластеризации (SSE)")
        axs[1].set_ylabel("Средний коэффициент силуэта")
        for ax in axs:
            ax.set_xlabel("число кластеров k")
            ax.set_xticks(ks)
            ax.grid(alpha=0.3)
            ax.legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(os.path.join(args.out, "quality_vs_k.png"), dpi=150)
        plt.close(fig)

        # точечные графики и профили кластеров
        proj = Projector(X, cols, None, not args.no_scale)
        panels = [(LINK_RU[lk], part[lk], None) for lk in links]
        if pa:
            for alg in ("kmeans", "kmedoids"):
                lab, r = pa[alg]
                panels.append((ALG_RU[alg], lab, r["centers"] * sd + mu))
        ncol = 3 if len(panels) > 4 else 2
        nrow = int(np.ceil(len(panels) / ncol))
        fig, axs = plt.subplots(nrow, ncol, figsize=(5.4 * ncol, 4.6 * nrow), squeeze=False)
        for ax in axs.ravel()[len(panels):]:
            ax.axis("off")
        for ax, (name, lab, cen) in zip(axs.ravel(), panels):
            cen = np.array([X[lab == c].mean(axis=0) for c in range(args.k)]) if cen is None else cen
            draw_scatter(ax, proj, X, lab, cen, None,
                         f"{name}, k = {args.k}\nSSE = {sse_of(Xs, lab):.0f}", args.k)
        fig.tight_layout()
        fig.savefig(os.path.join(args.out, f"scatter_k{args.k}.png"), dpi=140)
        plt.close(fig)

        prof = []
        for name, lab, _ in panels:
            g = df[cols].assign(cluster=lab).groupby("cluster").mean()
            g.insert(0, "n", np.bincount(lab))
            g.insert(0, "method", name)
            prof.append(g.reset_index())
        pd.concat(prof).round(3).to_csv(os.path.join(args.out, f"profiles_k{args.k}.csv"), index=False)


if __name__ == "__main__":
    main()
