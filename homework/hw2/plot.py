import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import poisson

# fixed parameters
n = 1024
lambdas = [0.1, 0.05]
Ls = [10, 50, 100, 1000]
num_samples = 10000  # Number of frame positions for histogram

for lam in lambdas:
    N = int(lam * n**2)
    points_x = np.random.uniform(0, n, N)
    points_y = np.random.uniform(0, n, N)
    for L in Ls:
        # Slide frames: sample num_samples random left-bottom corners
        frame_origins_x = np.random.uniform(0, n-L, num_samples)
        frame_origins_y = np.random.uniform(0, n-L, num_samples)
        counts = []
        for ox, oy in zip(frame_origins_x, frame_origins_y):
            mask = (
                (points_x >= ox) & (points_x < ox + L) &
                (points_y >= oy) & (points_y < oy + L)
            )
            counts.append(np.sum(mask))
        counts = np.array(counts)
        # Empirical histogram
        bins = np.arange(0, counts.max()+2) - 0.5
        hist_vals, _ = np.histogram(counts, bins=bins, density=True)
        centers = (bins[:-1] + bins[1:]) / 2
        # Theoretical Poisson
        mu = lam * L**2
        pois_probs = poisson.pmf(np.arange(int(2*centers.max())+1), mu)
        # Plot
        plt.figure(figsize=(8,6))
        if(L==1000):

            plt.bar(centers, hist_vals, width=1, alpha=1, label='Empirical Histogram')
        else:
            plt.bar(centers, hist_vals, width=1, alpha=0.6, label='Empirical Histogram')
        if L==1000 and lam==0.1:
            plt.xlim(91000, 109000)
        elif L==1000 and lam==0.05:
            plt.xlim(40000, 60000)
        plt.plot(np.arange(int(2*centers.max())+1), pois_probs, 'ro--', lw=2, label=f'Poisson (λ={lam}, L={L})',markersize=2)
        plt.xlabel('k (number of points in frame)')
        plt.ylabel('Probability')
        plt.title(f'$n$={n}, λ={lam}, L={L}, μ={mu:.1f}')
        plt.legend(loc='upper left')
        plt.tight_layout()
        plt.savefig(f'poisson_hist_lambda{lam}_L{L}.png')
        print(f'Saved plot for λ={lam}, L={L} as poisson_hist_lambda{lam}_L{L}.png')
        #plt.show()