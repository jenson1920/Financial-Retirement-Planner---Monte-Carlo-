"""
Monte Carlo Retirement Planner
------------------------------
Simulates thousands of possible investment paths to estimate:
  1. How large your pot could be at retirement (10th / 50th / 90th percentile)
  2. The probability your money lasts through retirement

All figures are in today's money (real terms), so inflation is already allowed for.

Run:  python retirement_planner.py
"""

from dataclasses import dataclass

import matplotlib.pyplot as plt
import numpy as np


@dataclass
class Plan:
    current_age: int = 21
    retirement_age: int = 65
    end_age: int = 90                 # age the money needs to last until
    current_savings: float = 2_000
    annual_contribution: float = 3_600   # in today's money
    target_income: float = 25_000        # yearly spending in retirement, today's money
    inflation: float = 0.02


# Simple strategies: (expected nominal return, volatility). Illustrative assumptions -
# replace with your own research and cite your sources in the README.
STRATEGIES = {
    "100% equities": (0.07, 0.16),
    "60/40 equities/bonds": (0.055, 0.10),
    "Cautious (30/70)": (0.04, 0.06),
}


def simulate(plan: Plan, mean: float, vol: float, n_sims: int = 10_000, seed: int = 42):
    """Return pot values by year for every simulation, plus the success flags."""
    rng = np.random.default_rng(seed)

    # Convert to real terms. Log-normal returns keep the pot from going negative
    # purely through the return draw.
    real_mean = (1 + mean) / (1 + plan.inflation) - 1
    mu = np.log(1 + real_mean) - 0.5 * vol**2      # drift adjustment for log-normal
    sigma = vol

    years_saving = plan.retirement_age - plan.current_age
    years_retired = plan.end_age - plan.retirement_age
    total_years = years_saving + years_retired

    returns = np.exp(rng.normal(mu, sigma, size=(n_sims, total_years)))

    pots = np.zeros((n_sims, total_years + 1))
    pots[:, 0] = plan.current_savings

    for t in range(total_years):
        if t < years_saving:   # accumulation: add savings, then grow
            pots[:, t + 1] = (pots[:, t] + plan.annual_contribution) * returns[:, t]
        else:                  # retirement: withdraw income, then grow
            pots[:, t + 1] = np.maximum(pots[:, t] - plan.target_income, 0) * returns[:, t]

    pot_at_retirement = pots[:, years_saving]
    success = pots[:, -1] > 0
    return pots, pot_at_retirement, success


def summarise(plan: Plan, label: str, pot_at_ret: np.ndarray, success: np.ndarray) -> None:
    p10, p50, p90 = np.percentile(pot_at_ret, [10, 50, 90])
    print(f"\n{label}")
    print(f"  Pot at {plan.retirement_age} (today's money):  "
          f"10th £{p10:,.0f} | median £{p50:,.0f} | 90th £{p90:,.0f}")
    print(f"  Chance money lasts to age {plan.end_age}: {success.mean():.1%}")


def plot_results(plan: Plan, pots: np.ndarray, pot_at_ret: np.ndarray, label: str) -> None:
    ages = np.arange(plan.current_age, plan.end_age + 1)
    p10, p50, p90 = np.percentile(pots, [10, 50, 90], axis=0)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

    axes[0].fill_between(ages, p10, p90, alpha=0.25, label="10th-90th percentile")
    axes[0].plot(ages, p50, linewidth=2, label="Median")
    axes[0].axvline(plan.retirement_age, linestyle="--", color="grey")
    axes[0].set(title=f"Pot over time - {label}", xlabel="Age", ylabel="Pot (£, today's money)")
    axes[0].legend()

    axes[1].hist(pot_at_ret, bins=60)
    axes[1].set(title=f"Pot at age {plan.retirement_age}", xlabel="£ (today's money)", ylabel="Simulations")

    plt.tight_layout()
    plt.savefig("results.png", dpi=150)
    print("\nChart saved as results.png")


def main() -> None:
    plan = Plan()

    best_label, best_success, best_data = None, -1.0, None
    for label, (mean, vol) in STRATEGIES.items():
        pots, pot_at_ret, success = simulate(plan, mean, vol)
        summarise(plan, label, pot_at_ret, success)
        if success.mean() > best_success:
            best_label, best_success, best_data = label, success.mean(), (pots, pot_at_ret)

    print(f"\nHighest chance of success: {best_label} ({best_success:.1%})")
    plot_results(plan, best_data[0], best_data[1], best_label)


if __name__ == "__main__":
    main()
