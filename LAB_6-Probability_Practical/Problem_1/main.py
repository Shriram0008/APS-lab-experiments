import numpy as np
import matplotlib.pyplot as plt


# ============================================================
# PROBLEM 1: SIMULATING CUSTOMER CLICKS
# ============================================================

# Given probability
p = 0.30

# Number of customers
sample_size = 20


# ============================================================
# STEP 1: GENERATE BERNOULLI OBSERVATIONS
# ============================================================

samples = np.random.binomial(
    n=1,
    p=p,
    size=sample_size
)

print("Generated observations:")
print(samples)


# ============================================================
# STEP 2: COUNT CLICKS AND NON-CLICKS
# ============================================================

number_of_clicks = np.sum(samples == 1)
number_of_non_clicks = np.sum(samples == 0)

print("\nNumber of clicks:", number_of_clicks)
print("Number of non-clicks:", number_of_non_clicks)

print(
    "Total observations:",
    number_of_clicks + number_of_non_clicks
)


# ============================================================
# STEP 3: CALCULATE EMPIRICAL PMF
# ============================================================

empirical_p_0 = number_of_non_clicks / sample_size
empirical_p_1 = number_of_clicks / sample_size

print("\nEmpirical PMF")
print("P(X = 0):", empirical_p_0)
print("P(X = 1):", empirical_p_1)


# ============================================================
# STEP 4: THEORETICAL PMF
# ============================================================

theoretical_p_0 = 1 - p
theoretical_p_1 = p

print("\nTheoretical PMF")
print("P(X = 0):", theoretical_p_0)
print("P(X = 1):", theoretical_p_1)


# ============================================================
# STEP 5: EXPECTATION
# ============================================================

# Theoretical expectation of Bernoulli random variable
theoretical_mean = p

# Empirical expectation
empirical_mean = np.mean(samples)

print("\nExpectation")
print("Theoretical:", theoretical_mean)
print("Empirical:", empirical_mean)


# ============================================================
# STEP 6: VARIANCE
# ============================================================

# Theoretical variance
theoretical_variance = p * (1 - p)

# Empirical variance
empirical_variance = np.var(samples)

print("\nVariance")
print("Theoretical:", theoretical_variance)
print("Empirical:", empirical_variance)


# ============================================================
# STEP 7: FINAL RESULT TABLE
# ============================================================

mean_difference = abs(
    theoretical_mean - empirical_mean
)

variance_difference = abs(
    theoretical_variance - empirical_variance
)

print("\nFinal Result Table")
print("-" * 60)

print(
    f"{'Measure':<15}"
    f"{'Theoretical':<15}"
    f"{'Empirical':<15}"
    f"{'Difference':<15}"
)

print("-" * 60)

print(
    f"{'Expectation':<15}"
    f"{theoretical_mean:<15.4f}"
    f"{empirical_mean:<15.4f}"
    f"{mean_difference:<15.4f}"
)

print(
    f"{'Variance':<15}"
    f"{theoretical_variance:<15.4f}"
    f"{empirical_variance:<15.4f}"
    f"{variance_difference:<15.4f}"
)


# ============================================================
# STEP 8: PLOT THEORETICAL VS EMPIRICAL PMF
# ============================================================

x = [0, 1]

theoretical_pmf = [
    theoretical_p_0,
    theoretical_p_1
]

empirical_pmf = [
    empirical_p_0,
    empirical_p_1
]

width = 0.35

plt.bar(
    np.array(x) - width / 2,
    theoretical_pmf,
    width=width,
    label="Theoretical"
)

plt.bar(
    np.array(x) + width / 2,
    empirical_pmf,
    width=width,
    label="Empirical"
)

plt.xlabel("X")
plt.ylabel("Probability")
plt.title("Theoretical vs Empirical PMF")
plt.xticks(x)
plt.legend()

plt.show()


# ============================================================
# STEP 9: EFFECT OF SAMPLE SIZE
# ============================================================

sample_sizes = [10, 100, 1000, 10000]

print("\nEffect of Sample Size")
print("-" * 75)

print(
    f"{'Sample Size':<15}"
    f"{'P(X=0)':<15}"
    f"{'P(X=1)':<15}"
    f"{'Mean':<15}"
    f"{'Variance':<15}"
)

print("-" * 75)

for n in sample_sizes:

    samples_n = np.random.binomial(
        n=1,
        p=p,
        size=n
    )

    count_0 = np.sum(samples_n == 0)
    count_1 = np.sum(samples_n == 1)

    probability_0 = count_0 / n
    probability_1 = count_1 / n

    mean = np.mean(samples_n)
    variance = np.var(samples_n)

    print(
        f"{n:<15}"
        f"{probability_0:<15.4f}"
        f"{probability_1:<15.4f}"
        f"{mean:<15.4f}"
        f"{variance:<15.4f}"
    )