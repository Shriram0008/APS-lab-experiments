import numpy as np


# ============================================================
# PROBLEM 2: NUMBER OF FRAUDULENT TRANSACTIONS FLAGGED
# ============================================================

# Given in the problem
TRANSACTIONS_PER_HOUR = 50
FLAGGING_PROBABILITY = 0.04
NUMBER_OF_HOURS = 20_000
MAXIMUM_CAPACITY = 4


# ============================================================
# STEP 1: DEFINE THE BINOMIAL RANDOM VARIABLE
# ============================================================

# X = total number of flagged transactions in one hour
#
# X ~ Binomial(n=50, p=0.04)


# ============================================================
# STEP 2: SIMULATE 20,000 INDEPENDENT ONE-HOUR PERIODS
# ============================================================

flagged_transactions = np.random.binomial(
    n=TRANSACTIONS_PER_HOUR,
    p=FLAGGING_PROBABILITY,
    size=NUMBER_OF_HOURS
)


# ============================================================
# STEP 3: DISPLAY SIMULATION INFORMATION
# ============================================================

print("=" * 60)
print("PROBLEM 2: FRAUD DETECTION SIMULATION")
print("=" * 60)

print("\nGiven Parameters")
print("-" * 60)

print(
    "Transactions examined per hour:",
    TRANSACTIONS_PER_HOUR
)

print(
    "Probability of a transaction being flagged:",
    FLAGGING_PROBABILITY
)

print(
    "Number of simulated one-hour periods:",
    NUMBER_OF_HOURS
)

print(
    "Maximum investigation capacity per hour:",
    MAXIMUM_CAPACITY
)


print("\nFirst 20 Simulated Hours")
print("-" * 60)

print(flagged_transactions[:20])

print(
    "\nTotal number of simulated hours:",
    len(flagged_transactions)
)


# ============================================================
# STEP 4: BASIC SIMULATION RESULTS
# ============================================================

minimum_flagged = np.min(flagged_transactions)
maximum_flagged = np.max(flagged_transactions)
empirical_mean = np.mean(flagged_transactions)
empirical_variance = np.var(flagged_transactions)


print("\nBasic Simulation Results")
print("-" * 60)

print(
    "Minimum flagged transactions in an hour:",
    minimum_flagged
)

print(
    "Maximum flagged transactions in an hour:",
    maximum_flagged
)

print(
    "Empirical mean:",
    empirical_mean
)

print(
    "Empirical variance:",
    empirical_variance
)


# ============================================================
# STEP 5: CHECK INVESTIGATION TEAM CAPACITY
# ============================================================

# "At most 4" means:
#
# X <= 4
#
# Therefore, hours with 0, 1, 2, 3 or 4 flagged
# transactions are within the team's capacity.

hours_within_capacity = np.sum(
    flagged_transactions <= MAXIMUM_CAPACITY
)


# If more than 4 transactions are flagged:
#
# X > 4
#
# the team's capacity is exceeded.

hours_exceeding_capacity = np.sum(
    flagged_transactions > MAXIMUM_CAPACITY
)


print("\nInvestigation Capacity")
print("-" * 60)

print(
    "Hours with at most 4 flagged transactions:",
    hours_within_capacity
)

print(
    "Hours with more than 4 flagged transactions:",
    hours_exceeding_capacity
)


# ============================================================
# STEP 6: EMPIRICAL PROBABILITIES
# ============================================================

empirical_probability_within_capacity = (
    hours_within_capacity / NUMBER_OF_HOURS
)

empirical_probability_exceeding_capacity = (
    hours_exceeding_capacity / NUMBER_OF_HOURS
)


print("\nEmpirical Probabilities")
print("-" * 60)

print(
    "Empirical P(X <= 4):",
    empirical_probability_within_capacity
)

print(
    "Empirical P(X > 4):",
    empirical_probability_exceeding_capacity
)


# ============================================================
# STEP 7: THEORETICAL EXPECTATION
# ============================================================

# For Binomial distribution:
#
# E[X] = n * p

theoretical_mean = (
    TRANSACTIONS_PER_HOUR
    * FLAGGING_PROBABILITY
)


print("\nTheoretical Expectation")
print("-" * 60)

print(
    "E[X] =",
    theoretical_mean
)


# ============================================================
# STEP 8: THEORETICAL VARIANCE
# ============================================================

# For Binomial distribution:
#
# Var(X) = n * p * (1 - p)

theoretical_variance = (
    TRANSACTIONS_PER_HOUR
    * FLAGGING_PROBABILITY
    * (1 - FLAGGING_PROBABILITY)
)


print("\nTheoretical Variance")
print("-" * 60)

print(
    "Var(X) =",
    theoretical_variance
)


# ============================================================
# STEP 9: COMPARE THEORETICAL AND EMPIRICAL VALUES
# ============================================================

mean_difference = abs(
    theoretical_mean - empirical_mean
)

variance_difference = abs(
    theoretical_variance - empirical_variance
)


print("\nTheoretical vs Empirical Results")
print("-" * 60)

print(
    f"{'Measure':<20}"
    f"{'Theoretical':<15}"
    f"{'Empirical':<15}"
    f"{'Difference':<15}"
)

print("-" * 60)

print(
    f"{'Expectation':<20}"
    f"{theoretical_mean:<15.4f}"
    f"{empirical_mean:<15.4f}"
    f"{mean_difference:<15.4f}"
)

print(
    f"{'Variance':<20}"
    f"{theoretical_variance:<15.4f}"
    f"{empirical_variance:<15.4f}"
    f"{variance_difference:<15.4f}"
)


# ============================================================
# STEP 10: FINAL SUMMARY
# ============================================================

print("\n" + "=" * 60)
print("FINAL SUMMARY")
print("=" * 60)

print(
    "Random Variable: X = number of flagged transactions per hour"
)

print(
    "Distribution: Binomial"
)

print(
    f"Parameters: n = {TRANSACTIONS_PER_HOUR}, "
    f"p = {FLAGGING_PROBABILITY}"
)

print(
    f"Theoretical Mean: {theoretical_mean:.4f}"
)

print(
    f"Empirical Mean: {empirical_mean:.4f}"
)

print(
    f"Theoretical Variance: {theoretical_variance:.4f}"
)

print(
    f"Empirical Variance: {empirical_variance:.4f}"
)

print(
    f"Hours within capacity (X <= 4): "
    f"{hours_within_capacity}"
)

print(
    f"Hours exceeding capacity (X > 4): "
    f"{hours_exceeding_capacity}"
)

print(
    f"Empirical P(X <= 4): "
    f"{empirical_probability_within_capacity:.4f}"
)

print(
    f"Empirical P(X > 4): "
    f"{empirical_probability_exceeding_capacity:.4f}"
)

print("=" * 60)