"""
TFG - The ATM Implied Volatility Skew under the Heston Model
Replication of the numerical experiment of the textbook chapter on Heston (Alos and Merino, Introduction to Financial Derivatives with Python).

Contents
  1. Black-Scholes-Merton call price
  2. Heston semi-analytical price (Little Heston Trap formulation)
  3. Decomposition formula (Black-Scholes price + two correction terms)
  4. Conditional Monte Carlo with the Milstein scheme (vectorised over paths)
  5. Implied volatility curves: analytical Heston vs decomposition approximation

Parameters of the book experiment:
  S0 = K = 100, T = 1, r = 0, kappa = 1, theta = 0.09, sigma0^2 = 0.04, nu = 0.3, rho = -0.5
Book results: analytical 9.2264, decomposition 9.2165, conditional Milstein ~9.227
"""
import time

import matplotlib.pyplot as plt
import numpy as np
from scipy import stats as ss
from scipy.integrate import quad
from scipy.optimize import brentq


# ---------------------------------------------------------------------------
# 1. Black-Scholes-Merton
# ---------------------------------------------------------------------------
def bs_d1(S0, K, r, sigma, T):
    return (np.log(S0 / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))


def bs_d2(S0, K, r, sigma, T):
    return bs_d1(S0, K, r, sigma, T) - sigma * np.sqrt(T)


def bs_call(S0, K, r, sigma, T):
    """Black-Scholes-Merton European call (works with numpy arrays)."""
    return S0 * ss.norm.cdf(bs_d1(S0, K, r, sigma, T)) - K * np.exp(-r * T) * ss.norm.cdf(
        bs_d2(S0, K, r, sigma, T)
    )


def implied_vol(price, S0, K, r, T):
    """Invert Black-Scholes numerically."""
    return brentq(lambda s: bs_call(S0, K, r, s, T) - price, 1e-4, 5.0)


# ---------------------------------------------------------------------------
# 2. Heston semi-analytical price (Little Heston Trap)
# ---------------------------------------------------------------------------
def heston_cf(u, S0, v0, r, T, kappa, theta, nu, rho, j):
    """Characteristic function phi_j (j = 1, 2) in the Little Heston Trap form."""
    x = np.log(S0)
    uj = 0.5 if j == 1 else -0.5
    bj = kappa - rho * nu if j == 1 else kappa
    iu = 1j * u
    d = np.sqrt((rho * nu * iu - bj) ** 2 - nu**2 * (2 * uj * iu - u**2))
    g = (bj - rho * nu * iu - d) / (bj - rho * nu * iu + d)
    e = np.exp(-d * T)
    C = r * iu * T + (kappa * theta / nu**2) * (
        (bj - rho * nu * iu - d) * T - 2 * np.log((1 - g * e) / (1 - g))
    )
    D = ((bj - rho * nu * iu - d) / nu**2) * ((1 - e) / (1 - g * e))
    return np.exp(C + D * v0 + iu * x)


def heston_prob(j, S0, K, v0, r, T, kappa, theta, nu, rho, upper=200.0):
    """Risk-neutral probability P_j by Fourier inversion."""

    def integrand(u):
        phi = heston_cf(u, S0, v0, r, T, kappa, theta, nu, rho, j)
        return np.real(np.exp(-1j * u * np.log(K)) * phi / (1j * u))

    integral, _ = quad(integrand, 1e-8, upper, limit=500)
    return 0.5 + integral / np.pi


def heston_call(S0, K, v0, r, T, kappa, theta, nu, rho):
    """European call price in the Heston model: S0*P1 - K*exp(-rT)*P2."""
    P1 = heston_prob(1, S0, K, v0, r, T, kappa, theta, nu, rho)
    P2 = heston_prob(2, S0, K, v0, r, T, kappa, theta, nu, rho)
    return S0 * P1 - K * np.exp(-r * T) * P2


# ---------------------------------------------------------------------------
# 3. Decomposition formula
#    V ~ BS(vbar) + rho/2 * H * E[int sigma d<M,B>] + 1/8 * Kfun * E[int d<M,M>]
#    with dM_t = (nu/kappa) sigma_t (1 - exp(-kappa (T-t))) dB_t
# ---------------------------------------------------------------------------
def heston_decomposition(S0, K, v0, r, T, kappa, theta, nu, rho):
    exp_var = lambda s: theta + (v0 - theta) * np.exp(-kappa * s)  # E[sigma_s^2]
    w = lambda s: 1.0 - np.exp(-kappa * (T - s))

    # future average volatility
    int_var = theta * T + (v0 - theta) / kappa * (1.0 - np.exp(-kappa * T))
    vbar = np.sqrt(int_var / T)

    # E[int sigma d<M,B>] and E[int d<M,M>] (computed by quadrature)
    cov = (nu / kappa) * quad(lambda s: exp_var(s) * w(s), 0.0, T)[0]
    qv = (nu / kappa) ** 2 * quad(lambda s: exp_var(s) * w(s) ** 2, 0.0, T)[0]

    d1 = bs_d1(S0, K, r, vbar, T)
    pref = S0 / (vbar * np.sqrt(2 * np.pi * T)) * np.exp(-0.5 * d1**2)
    H = pref * (1.0 - d1 / (vbar * np.sqrt(T)))
    Kfun = pref * (d1**2 - vbar * d1 * np.sqrt(T) - 1.0) / (vbar**2 * T)

    return bs_call(S0, K, r, vbar, T) + 0.5 * rho * H * cov + 0.125 * Kfun * qv


# ---------------------------------------------------------------------------
# 4. Conditional Monte Carlo (Milstein), vectorised over paths
# ---------------------------------------------------------------------------
def heston_mc_conditional(S0, K, v0, r, T, kappa, theta, nu, rho, m, n, seed=0):
    """Only the Brownian motion B driving the variance is simulated; the
    independent part is integrated analytically through Black-Scholes."""
    rng = np.random.default_rng(seed)
    dt = T / m
    v = np.full(n, v0, dtype=float)
    int_var = np.zeros(n)
    int_stoch = np.zeros(n)
    for _ in range(m):
        dB = np.sqrt(dt) * rng.standard_normal(n)
        sig = np.sqrt(v)
        int_var += v * dt
        int_stoch += sig * dB
        v = v + kappa * (theta - v) * dt + nu * sig * dB + 0.25 * nu**2 * (dB**2 - dt)
        v = np.abs(v)  # reflecting principle
    S0p = S0 * np.exp(rho * int_stoch - 0.5 * rho**2 * int_var)
    sig_eff = np.sqrt((1.0 - rho**2) * int_var / T)
    prices = bs_call(S0p, K, r, sig_eff, T)
    return prices.mean(), prices.std(ddof=1) / np.sqrt(n)


# ---------------------------------------------------------------------------
# Main experiment
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    plt.switch_backend("Agg")  # only when run as a script: save figures without opening windows
    S0, K, T, r = 100.0, 100.0, 1.0, 0.0
    kappa, theta, v0, nu, rho = 1.0, 0.09, 0.04, 0.3, -0.5

    print("=== Experiment from the book (S0=K=100, T=1, r=0) ===")
    t0 = time.perf_counter()
    p_an = heston_call(S0, K, v0, r, T, kappa, theta, nu, rho)
    t_an = time.perf_counter() - t0
    t0 = time.perf_counter()
    p_dec = heston_decomposition(S0, K, v0, r, T, kappa, theta, nu, rho)
    t_dec = time.perf_counter() - t0
    t0 = time.perf_counter()
    p_mc, se_mc = heston_mc_conditional(S0, K, v0, r, T, kappa, theta, nu, rho, m=1000, n=5000, seed=1)
    t_mc = time.perf_counter() - t0
    print(f"Analytical (Fourier)       : {p_an:8.4f}   ({t_an:.4f} s)   book: 9.2264")
    print(f"Decomposition              : {p_dec:8.4f}   ({t_dec:.4f} s)   book: 9.2165")
    print(f"Conditional MC (5k, 1000)  : {p_mc:8.4f} +- {se_mc:.4f}   ({t_mc:.2f} s)   book: ~9.227")

    p_mc2, se_mc2 = heston_mc_conditional(S0, K, v0, r, T, kappa, theta, nu, rho, m=250, n=200_000, seed=2)
    print(f"Conditional MC (200k, 250) : {p_mc2:8.4f} +- {se_mc2:.4f}")

    # Sanity checks
    print("\n=== Sanity checks ===")
    p_h = heston_call(S0, K, 0.09, r, T, 1.0, 0.09, 0.01, 0.0)
    p_b = bs_call(S0, K, r, 0.3, T)
    print(f"nu -> 0, v0=theta: Heston {p_h:.5f} vs Black-Scholes(0.3) {p_b:.5f}")

    # Implied vol curves (Figure 2 of the chapter): sigma0 = 0.2, T = 0.5 and T = 1
    print("\n=== Implied volatility: analytical vs decomposition ===")
    strikes = np.arange(80.0, 120.1, 2.5)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), sharey=True)
    for ax, TT in zip(axes, (0.5, 1.0)):
        iv_an, iv_dec = [], []
        for Kk in strikes:
            iv_an.append(implied_vol(heston_call(S0, Kk, v0, r, TT, kappa, theta, nu, rho), S0, Kk, r, TT))
            iv_dec.append(implied_vol(heston_decomposition(S0, Kk, v0, r, TT, kappa, theta, nu, rho), S0, Kk, r, TT))
        iv_an, iv_dec = np.array(iv_an), np.array(iv_dec)
        atm = np.argmin(np.abs(strikes - 100))
        print(f"T={TT}: ATM IV analytical {iv_an[atm]:.4f}, decomposition {iv_dec[atm]:.4f}, "
              f"max |diff| over strikes {np.max(np.abs(iv_an - iv_dec)):.4f}")
        ax.plot(strikes, iv_an, "k-", label="Heston (analytical)")
        ax.plot(strikes, iv_dec, "r--", label="Decomposition formula")
        ax.set_title(f"T = {TT}")
        ax.set_xlabel("Strike")
        ax.grid(alpha=0.3)
    axes[0].set_ylabel("Implied volatility")
    axes[0].legend()
    plt.tight_layout()
    plt.savefig("heston_implied_vol.png", dpi=150)
    print("Figure saved to heston_implied_vol.png")
