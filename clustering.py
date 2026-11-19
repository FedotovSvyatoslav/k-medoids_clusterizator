"""
Кластеризация данных: k-Means и k-Medoids.

Алгоритмы
  kmeans   - алгоритм Ллойда, инициализация k-means++, n_init перезапусков (выбирается лучший по SSE);
  kmedoids - метод k-медоидов (итерации «назначение - пересчёт медоидов», как в Voronoi iteration),
             инициализация в стиле k-means++, n_init перезапусков; евклидово расстояние.
Ошибка кластеризации
  error = SSE - сумма квадратов евклидовых расстояний от объектов до представителей своих кластеров
          (центроидов для k-Means, медоидов для k-Medoids); эта величина одинаково определена для обоих
          алгоритмов, поэтому их можно сравнивать. Для k-Medoids дополнительно выводится его целевая
          функция cost - сумма (не квадратов) расстояний до медоидов.
Предобработка
  Признаки имеют разные масштабы (Income, Age, ...), поэтому по умолчанию они стандартизуются
  (z-score; отключается --no-scale). Строки с пропусками в выбранных признаках отбрасываются.
  По умолчанию в кластеризации участвуют все числовые атрибуты, кроме Row и CustomerId (--features).
Шум
  --noise-frac p: у round(p*n) случайно выбранных объектов к 1..d случайным координатам прибавляется или
  вычитается случайная величина  ±U(0.5, 1.5) * noise-scale * std(признака)  (в исходных единицах);
  значения неотрицательных признаков не опускаются ниже нуля. Шум вносится до стандартизации.

Режимы
  (по умолчанию)     один запуск: заданные --algorithm, --k, --noise-frac. Выдаёт CSV с координатами точек и
                     номерами кластеров, ошибку, точечный график.
  --sweep-k          ошибка в зависимости от k (--k-min..--k-max) при заданном --noise-frac.
  --noise-experiment доли шума 0%, 1%, 3%, 5%, 10% x k = --k-min..--k-max x алгоритмы: графики ошибки
                     от k для каждой доли шума, устойчивость разбиения (ARI относительно разбиения
                     незашумлённых данных), точечные графики для k = --k.

Примеры
  c
  python clustering.py --data customers.csv --algorithm both --sweep-k --k-min 2 --k-max 10
  python clustering.py --data customers.csv --algorithm kmeans --k 3 --noise-frac 0.05
  python clustering.py --data customers.csv --algorithm both --noise-experiment --k 3
"""
import argparse
import json
import os

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ALGS = ["kmeans", "kmedoids"]
ALG_RU = {"kmeans": "k-Means", "kmedoids": "k-Medoids"}


# ----------------------------------------------------------------- данные
def load_data(path, features=None):
    df = pd.read_csv(path, sep=None, engine="python")
    df.columns = [c.strip() for c in df.columns]
    if features is None:
        features = [c for c in df.columns if c not in ("Row", "CustomerId")
                    and pd.api.types.is_numeric_dtype(df[c])]
    missing = [f for f in features if f not in df.columns]
    if missing:
        raise SystemExit(f"в файле нет столбцов: {missing}; есть: {list(df.columns)}")
    before = len(df)
    df = df.dropna(subset=features).reset_index(drop=True)
    if len(df) < before:
        print(f"удалено строк с пропусками: {before - len(df)} (осталось {len(df)})")
    return df, features


def add_noise(X, frac, scale, rng):
    """Возвращает зашумлённую копию X и булеву маску зашумлённых объектов."""
    Xn = X.copy()
    n, d = X.shape
    mask = np.zeros(n, dtype=bool)
    m = int(round(frac * n))
    if m == 0:
        return Xn, mask
    std = X.std(axis=0)
    nonneg = X.min(axis=0) >= 0
    idx = rng.choice(n, m, replace=False)
    mask[idx] = True
    for i in idx:
        nf = rng.randint(1, d + 1)
        cols = rng.choice(d, nf, replace=False)
        Xn[i, cols] += rng.choice([-1.0, 1.0], nf) * rng.uniform(0.5, 1.5, nf) * scale * std[cols]
    Xn[:, nonneg] = np.maximum(Xn[:, nonneg], 0.0)
    return Xn, mask


def standardize(X, scale=True):
    if not scale:
        return X.copy(), np.zeros(X.shape[1]), np.ones(X.shape[1])
    mu, sd = X.mean(axis=0), X.std(axis=0)
    sd[sd == 0] = 1.0
    return (X - mu) / sd, mu, sd


# ----------------------------------------------------------------- алгоритмы
def sqdist(X, C):
    return ((X[:, None, :] - C[None, :, :]) ** 2).sum(axis=2)


def kmeans(X, k, rng, n_init=10, max_iter=300):
    n = len(X)
    best = None
    for _ in range(n_init):
        # k-means++
        C = [X[rng.randint(n)]]
        for _ in range(1, k):
            d2 = sqdist(X, np.array(C)).min(axis=1)
            p = d2 / d2.sum() if d2.sum() > 0 else None
            C.append(X[rng.choice(n, p=p)])
        C = np.array(C)
        for _ in range(max_iter):
            d2 = sqdist(X, C)
            lab = d2.argmin(axis=1)
            newC = C.copy()
            for j in range(k):
                pts = X[lab == j]
                if len(pts):
                    newC[j] = pts.mean(axis=0)
                else:                      # пустой кластер: берём самый далёкий объект
                    newC[j] = X[d2.min(axis=1).argmax()]
            if np.allclose(newC, C):
                C = newC
                break
            C = newC
        d2 = sqdist(X, C)
        lab = d2.argmin(axis=1)
        sse = float(d2[np.arange(n), lab].sum())
        if best is None or sse < best["sse"]:
            best = {"labels": lab, "centers": C, "sse": sse, "cost": float(np.sqrt(d2[np.arange(n), lab]).sum())}
    return best


def kmedoids(X, k, rng, D=None, n_init=10, max_iter=300):
    n = len(X)
    if D is None:
        D = np.sqrt(sqdist(X, X))
    best = None
    for _ in range(n_init):
        med = [rng.randint(n)]
        for _ in range(1, k):
            d = D[:, med].min(axis=1) ** 2
            p = d / d.sum() if d.sum() > 0 else None
            med.append(rng.choice(n, p=p))
        med = np.array(med)
        for _ in range(max_iter):
            lab = D[:, med].argmin(axis=1)
            new = med.copy()
            for j in range(k):
                members = np.nonzero(lab == j)[0]
                if len(members):
                    new[j] = members[D[np.ix_(members, members)].sum(axis=1).argmin()]
            if np.array_equal(new, med):
                break
            med = new
        lab = D[:, med].argmin(axis=1)
        dist = D[np.arange(n), med[lab]]
        sse, cost = float((dist ** 2).sum()), float(dist.sum())
        if best is None or cost < best["cost"]:
            best = {"labels": lab, "centers": X[med], "sse": sse, "cost": cost, "medoid_idx": med}
    return best


def run_algorithm(alg, Xs, k, rng, D=None):
    return kmeans(Xs, k, rng) if alg == "kmeans" else kmedoids(Xs, k, rng, D)


def ari(a, b):
    """Скорректированный индекс Рэнда между двумя разбиениями."""
    n = len(a)
    _, ai = np.unique(a, return_inverse=True)
    _, bi = np.unique(b, return_inverse=True)
    cont = np.zeros((ai.max() + 1, bi.max() + 1))
    np.add.at(cont, (ai, bi), 1)
    c2 = lambda x: x * (x - 1) / 2.0
    sij, sa, sb = c2(cont).sum(), c2(cont.sum(1)).sum(), c2(cont.sum(0)).sum()
    exp = sa * sb / c2(n)
    mx = (sa + sb) / 2.0
    return 1.0 if mx == exp else float((sij - exp) / (mx - exp))


# ----------------------------------------------------------------- графики
class Projector:
    """Проекция на плоскость: два выбранных признака либо первые две главные компоненты."""

    def __init__(self, X, cols, plot_features=None, scale=True):
        if plot_features:
            ix = [cols.index(f) for f in plot_features]
            self.fn = lambda A: A[:, ix]
            self.labels = plot_features
        else:
            Xs, self.mu, self.sd = standardize(X, scale)
            self.mean = Xs.mean(axis=0)
            _, _, Vt = np.linalg.svd(Xs - self.mean, full_matrices=False)
            V = Vt[:2].T
            self.fn = lambda A: ((A - self.mu) / self.sd - self.mean) @ V
            self.labels = ["PC1 (главная компонента 1)", "PC2 (главная компонента 2)"]


def draw_scatter(ax, proj, X, labels, centers_orig, mask, title, k):
    P = proj.fn(X)
    cmap = "tab10" if k <= 10 else "tab20"
    ax.scatter(P[:, 0], P[:, 1], c=labels, cmap=cmap, s=12, alpha=0.8, vmin=0, vmax=9 if k <= 10 else 19)
    if mask is not None and mask.any():
        ax.scatter(P[mask, 0], P[mask, 1], facecolors="none", edgecolors="k", s=40, linewidths=0.8,
                   label="зашумлённые объекты")
        ax.legend(fontsize=7, loc="best")
    C = proj.fn(centers_orig)
    ax.scatter(C[:, 0], C[:, 1], c="k", marker="X", s=130, edgecolors="w", linewidths=1.2, zorder=5)
    ax.set_title(title, fontsize=10)
    ax.set_xlabel(proj.labels[0])
    ax.set_ylabel(proj.labels[1])
    ax.grid(alpha=0.25)


def centers_to_orig(res, mu, sd):
    return res["centers"] * sd + mu


# ----------------------------------------------------------------- основной код
def evaluate(X, k_list, algs, seed, scale):
    """Кластеризует X для всех k и алгоритмов; возвращает {(alg,k): результат}, mu, sd."""
    Xs, mu, sd = standardize(X, scale)
    D = np.sqrt(sqdist(Xs, Xs)) if "kmedoids" in algs else None
    out = {}
    for alg in algs:
        for k in k_list:
            rng = np.random.RandomState(seed + 17 * k)
            out[(alg, k)] = run_algorithm(alg, Xs, k, rng, D)
    return out, mu, sd


def main():
    ap = argparse.ArgumentParser(description="Кластеризация: k-Means и k-Medoids")
    ap.add_argument("--data", required=True, help="customers.csv (первая строка - заголовок)")
    ap.add_argument("--algorithm", default="both", choices=ALGS + ["both"])
    ap.add_argument("--k", type=int, default=3, help="число кластеров")
    ap.add_argument("--noise-frac", type=float, default=0.0, help="доля зашумлённых объектов, 0..1")
    ap.add_argument("--noise-scale", type=float, default=1.0,
                    help="масштаб шума в долях стандартного отклонения признака")
    ap.add_argument("--features", nargs="+", default=None, help="признаки для кластеризации")
    ap.add_argument("--no-scale", action="store_true", help="не стандартизовать признаки")
    ap.add_argument("--plot-features", nargs=2, default=None, metavar=("X", "Y"),
                    help="признаки для осей точечного графика (по умолчанию - главные компоненты)")
    ap.add_argument("--sweep-k", action="store_true", help="ошибка в зависимости от k")
    ap.add_argument("--noise-experiment", action="store_true", help="доли шума 0/1/3/5/10%% x k")
    ap.add_argument("--k-min", type=int, default=3)
    ap.add_argument("--k-max", type=int, default=9)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default="results_clustering")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    df, cols = load_data(args.data, args.features)
    X = df[cols].to_numpy(dtype=float)
    scale = not args.no_scale
    algs = ALGS if args.algorithm == "both" else [args.algorithm]
    ks = list(range(args.k_min, args.k_max + 1))
    proj = Projector(X, cols, args.plot_features, scale)
    print(f"объектов: {len(X)}, признаки: {cols}")

    # ---- эксперимент с шумом
    if args.noise_experiment:
        fracs = [0.0, 0.01, 0.03, 0.05, 0.10]
        k_list = sorted(set(ks + [args.k]))
        rows, runs, clean = [], {}, None
        for p in fracs:
            rng = np.random.RandomState(args.seed + int(p * 1000))
            Xn, mask = add_noise(X, p, args.noise_scale, rng)
            res, mu, sd = evaluate(Xn, k_list, algs, args.seed, scale)
            if p == 0.0:
                clean = res
            for (alg, k), r in res.items():
                a = ari(r["labels"], clean[(alg, k)]["labels"])
                rows.append({"algorithm": alg, "noise_frac": p, "k": k, "error_sse": r["sse"],
                             "cost_sum_dist": r["cost"], "ari_vs_clean": a, "n_noisy": int(mask.sum())})
                print(f"{alg:9s} noise={p:4.0%} k={k}  SSE={r['sse']:.2f}  ARI={a:.3f}")
                runs[(alg, p, k)] = (r, Xn, mask, mu, sd)
        res_df = pd.DataFrame(rows)
        res_df.to_csv(os.path.join(args.out, "noise_experiment.csv"), index=False)

        # ошибка от k для каждой доли шума + ARI
        fig, axs = plt.subplots(2, len(algs), figsize=(6.2 * len(algs), 8.5), squeeze=False)
        for c, alg in enumerate(algs):
            for p in fracs:
                g = res_df[(res_df.algorithm == alg) & (res_df.noise_frac == p) & (res_df.k.isin(ks))]
                axs[0, c].plot(g.k, g.error_sse, marker="o", label=f"шум {p:.0%}")
                if p > 0:
                    axs[1, c].plot(g.k, g.ari_vs_clean, marker="o", label=f"шум {p:.0%}")
            axs[0, c].set_title(f"{ALG_RU[alg]}: ошибка кластеризации (SSE)")
            axs[0, c].set_ylabel("SSE")
            axs[1, c].set_title(f"{ALG_RU[alg]}: сходство с разбиением без шума (ARI)")
            axs[1, c].set_ylabel("ARI")
            for r_ in (0, 1):
                axs[r_, c].set_xlabel("число кластеров k")
                axs[r_, c].set_xticks(ks)
                axs[r_, c].grid(alpha=0.3)
                axs[r_, c].legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(os.path.join(args.out, "noise_error_vs_k.png"), dpi=150)
        plt.close(fig)

        # точечные графики для k = --k
        fig, axs = plt.subplots(len(algs), len(fracs), figsize=(4.4 * len(fracs), 4.2 * len(algs)), squeeze=False)
        for r_, alg in enumerate(algs):
            for c, p in enumerate(fracs):
                r, Xn, mask, mu, sd = runs[(alg, p, args.k)]
                draw_scatter(axs[r_, c], proj, Xn, r["labels"], centers_to_orig(r, mu, sd), mask,
                             f"{ALG_RU[alg]}, шум {p:.0%}, k={args.k}\nSSE={r['sse']:.1f}", args.k)
        fig.tight_layout()
        fig.savefig(os.path.join(args.out, f"noise_scatter_k{args.k}.png"), dpi=140)
        plt.close(fig)
        return

    # ---- зависимость ошибки от k
    rng = np.random.RandomState(args.seed)
    Xn, mask = add_noise(X, args.noise_frac, args.noise_scale, rng)
    if args.sweep_k:
        k_list = sorted(set(ks + [args.k]))
        res, mu, sd = evaluate(Xn, k_list, algs, args.seed, scale)
        rows = [{"algorithm": a, "k": k, "noise_frac": args.noise_frac, "error_sse": r["sse"],
                 "cost_sum_dist": r["cost"]} for (a, k), r in res.items()]
        pd.DataFrame(rows).to_csv(os.path.join(args.out, "error_vs_k.csv"), index=False)
        fig, ax = plt.subplots(figsize=(7, 4.6))
        for alg in algs:
            g = [(k, res[(alg, k)]["sse"]) for k in ks]
            ax.plot(*zip(*g), marker="o", label=ALG_RU[alg])
        ax.set_xlabel("число кластеров k")
        ax.set_ylabel("Ошибка кластеризации (SSE)")
        ax.set_title(f"Ошибка кластеризации в зависимости от k (шум {args.noise_frac:.0%})")
        ax.set_xticks(ks)
        ax.grid(alpha=0.3)
        ax.legend()
        fig.tight_layout()
        fig.savefig(os.path.join(args.out, "error_vs_k.png"), dpi=150)
        plt.close(fig)
        for (a, k), r in sorted(res.items()):
            print(f"{a:9s} k={k}  SSE={r['sse']:.2f}")
    else:
        res, mu, sd = evaluate(Xn, [args.k], algs, args.seed, scale)

    # ---- одиночный запуск: координаты, кластеры, ошибка, график
    fig, axs = plt.subplots(1, len(algs), figsize=(6.4 * len(algs), 5.2), squeeze=False)
    summary = []
    for c, alg in enumerate(algs):
        r = res[(alg, args.k)]
        out = df.copy()
        out[cols] = Xn
        out["cluster"] = r["labels"]
        out["is_noisy"] = mask
        out.to_csv(os.path.join(args.out, f"clusters_{alg}_k{args.k}_noise{int(args.noise_frac*100)}.csv"),
                   index=False)
        centers = pd.DataFrame(centers_to_orig(r, mu, sd), columns=cols)
        print(f"\n=== {ALG_RU[alg]}, k={args.k}, шум {args.noise_frac:.0%} ===")
        print(f"ошибка кластеризации (SSE) = {r['sse']:.4f}; сумма расстояний до представителей = {r['cost']:.4f}")
        print("размеры кластеров:", np.bincount(r["labels"], minlength=args.k).tolist())
        print("представители кластеров (исходные единицы):")
        print(centers.round(3).to_string())
        print("координаты точек и кластеры (первые 10):")
        print(out[cols + ["cluster"]].head(10).round(3).to_string())
        summary.append({"algorithm": alg, "k": args.k, "noise_frac": args.noise_frac,
                        "error_sse": r["sse"], "cost_sum_dist": r["cost"]})
        draw_scatter(axs[0, c], proj, Xn, r["labels"], centers_to_orig(r, mu, sd), mask,
                     f"{ALG_RU[alg]}, k={args.k}, шум {args.noise_frac:.0%}\nSSE={r['sse']:.1f}", args.k)
    fig.tight_layout()
    fig.savefig(os.path.join(args.out, f"scatter_k{args.k}_noise{int(args.noise_frac*100)}.png"), dpi=150)
    plt.close(fig)
    with open(os.path.join(args.out, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()