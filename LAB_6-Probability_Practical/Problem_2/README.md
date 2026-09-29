# Problem 2: Number of Fraudulent Transactions Flagged

## 1. Problem Statement

A fraud-detection model examines 50 transactions every hour.

Each transaction has a 4% probability of being flagged as potentially fraudulent.

Initially, individual flagging decisions are assumed to be independent.

The fraud investigation team can examine at most four flagged transactions per hour.

The task is to simulate the fraud-detection system for 20,000 independent one-hour periods.

For every simulated hour:

* Exactly 50 transactions are examined.
* Each transaction has a flagging probability of 0.04.
* \(X\) records the total number of flagged transactions.

The simulation uses NumPy's `np.random.binomial()` function.

---

# 2. Given Values

| Parameter                          |                       Value |
| ---------------------------------- | --------------------------: |
| Transactions per hour              |                          50 |
| Flagging probability               |                        0.04 |
| Flagging probability in percentage |                          4% |
| Simulated one-hour periods         |                      20,000 |
| Maximum capacity                   | 4 flagged transactions/hour |

---

# 3. Random Variable

Define:

$$
X=\text{number of flagged transactions in one hour}
$$

Since there are 50 independent transactions and each transaction has two possible outcomes:

* Flagged
* Not flagged

and we count the number of flagged transactions, \(X\) follows a Binomial distribution.

Therefore:

$$
\boxed{X\sim Binomial(50,0.04)}
$$

---

# 4. Why Binomial Distribution?

The problem satisfies the conditions of a binomial experiment:

1. There is a fixed number of trials.
2. There are 50 transactions.
3. Each transaction has two outcomes:

   * Flagged
   * Not flagged
4. The probability of being flagged is constant:

   $$
   p=0.04
   $$
5. Individual flagging decisions are assumed independent.
6. We count the number of flagged transactions.

Therefore:

$$
X\sim Binomial(n=50,p=0.04)
$$

---

# 5. Step 1 — Simulate 20,000 One-Hour Periods

The PDF asks us to simulate 20,000 independent one-hour periods.

We use:

```python
flagged_transactions = np.random.binomial(
    n=50,
    p=0.04,
    size=20000
)
```

Explanation:

* `n=50` → 50 transactions are examined during each hour.
* `p=0.04` → Each transaction has a 4% probability of being flagged.
* `size=20000` → 20,000 independent hours are simulated.

The resulting array contains one value for each simulated hour.

For example:

```text
[1, 2, 0, 3, 1, 2, 4, 0, ...]
```

A value of `3` means that three transactions were flagged during that simulated hour.

---

# 6. Step 2 — Number of Simulated Hours

The simulation should contain:

$$
\boxed{20,000}
$$

observations.

Python:

```python
len(flagged_transactions)
```

should return:

```text
20000
```

---

# 7. Step 3 — Investigation Team Capacity

The team can examine at most four flagged transactions per hour.

"At most 4" means:

$$
X\le4
$$

Therefore:

### Within capacity

```python
flagged_transactions <= 4
```

### Capacity exceeded

```python
flagged_transactions > 4
```

Possible values within capacity are:

$$
0,1,2,3,4
$$

Values exceeding capacity are:

$$
5,6,7,\ldots
$$

---

# 8. Step 4 — Empirical Probability of Exceeding Capacity

The number of simulated hours in which more than four transactions were flagged is:

```python
hours_exceeding_capacity = np.sum(
    flagged_transactions > 4
)
```

The empirical probability is:

$$
P_{empirical}(X>4)
=
\frac{\text{Number of hours with }X>4}
{20000}
$$

Python:

```python
empirical_probability_exceeding_capacity = (
    hours_exceeding_capacity / 20000
)
```

---

# 9. Step 5 — Empirical Probability of Staying Within Capacity

The number of hours where the team can handle all flagged transactions is:

```python
hours_within_capacity = np.sum(
    flagged_transactions <= 4
)
```

The empirical probability is:

$$
P_{empirical}(X\le4)
=
\frac{\text{Number of hours with }X\le4}
{20000}
$$

Python:

```python
empirical_probability_within_capacity = (
    hours_within_capacity / 20000
)
```

The two probabilities should add to approximately:

$$
1
$$

because every simulated hour either:

$$
X\le4
$$

or:

$$
X>4
$$

---

# 10. Step 6 — Theoretical Expectation

For a Binomial random variable:

$$
E[X]=np
$$

Substituting:

$$
E[X]=50(0.04)
$$

$$
\boxed{E[X]=2}
$$

Therefore, theoretically, the system produces an average of approximately two flagged transactions per hour.

---

# 11. Step 7 — Empirical Expectation

The empirical mean is calculated using:

```python
empirical_mean = np.mean(flagged_transactions)
```

It should generally be close to the theoretical value:

$$
E[X]=2
$$

The exact value changes because the data are generated randomly.

---

# 12. Step 8 — Theoretical Variance

For a Binomial random variable:

$$
Var(X)=np(1-p)
$$

Substituting:

$$
Var(X)=50(0.04)(1-0.04)
$$

$$
=50(0.04)(0.96)
$$

$$
=1.92
$$

Therefore:

$$
\boxed{Var(X)=1.92}
$$

---

# 13. Step 9 — Empirical Variance

The empirical variance is calculated using:

```python
empirical_variance = np.var(
    flagged_transactions
)
```

The empirical variance should generally be close to the theoretical variance:

$$
1.92
$$

because 20,000 hours are simulated.

---

# 14. Theoretical vs Empirical Results

| Measure                            | Theoretical Value |       Empirical Value |
| ---------------------------------- | ----------------: | --------------------: |
| Expected flagged transactions/hour |              2.00 | Depends on simulation |
| Variance                           |              1.92 | Depends on simulation |

The empirical values will not necessarily be exactly equal to the theoretical values because the simulation is random.

---

# 15. Complete Python Program

```python
import numpy as np


# ============================================================
# PROBLEM 2: NUMBER OF FRAUDULENT TRANSACTIONS FLAGGED
# ============================================================

transactions_per_hour = 50
p = 0.04
number_of_hours = 20000
maximum_capacity = 4


# ============================================================
# STEP 1: SIMULATE 20,000 ONE-HOUR PERIODS
# ============================================================

flagged_transactions = np.random.binomial(
    n=transactions_per_hour,
    p=p,
    size=number_of_hours
)

print("First 20 simulated hours:")
print(flagged_transactions[:20])

print("\nTotal simulated hours:", len(flagged_transactions))


# ============================================================
# STEP 2: BASIC STATISTICS
# ============================================================

minimum_flagged = np.min(flagged_transactions)
maximum_flagged = np.max(flagged_transactions)
average_flagged = np.mean(flagged_transactions)

print("\nBasic Statistics")
print("-" * 40)

print(
    "Minimum flagged transactions:",
    minimum_flagged
)

print(
    "Maximum flagged transactions:",
    maximum_flagged
)

print(
    "Average flagged transactions:",
    average_flagged
)


# ============================================================
# STEP 3: CAPACITY CHECK
# ============================================================

hours_within_capacity = np.sum(
    flagged_transactions <= maximum_capacity
)

hours_exceeding_capacity = np.sum(
    flagged_transactions > maximum_capacity
)

print("\nInvestigation Capacity")
print("-" * 40)

print(
    "Hours with at most 4 flagged transactions:",
    hours_within_capacity
)

print(
    "Hours with more than 4 flagged transactions:",
    hours_exceeding_capacity
)


# ============================================================
# STEP 4: EMPIRICAL PROBABILITIES
# ============================================================

empirical_probability_within_capacity = (
    hours_within_capacity / number_of_hours
)

empirical_probability_exceeding_capacity = (
    hours_exceeding_capacity / number_of_hours
)

print("\nEmpirical Probabilities")
print("-" * 40)

print(
    "Empirical P(X <= 4):",
    empirical_probability_within_capacity
)

print(
    "Empirical P(X > 4):",
    empirical_probability_exceeding_capacity
)


# ============================================================
# STEP 5: THEORETICAL EXPECTATION
# ============================================================

theoretical_mean = (
    transactions_per_hour * p
)

print("\nTheoretical Expectation")
print("-" * 40)

print(
    "E[X] =",
    theoretical_mean
)


# ============================================================
# STEP 6: EMPIRICAL EXPECTATION
# ============================================================

empirical_mean = np.mean(
    flagged_transactions
)

print("\nEmpirical Expectation")
print("-" * 40)

print(
    "Empirical mean =",
    empirical_mean
)


# ============================================================
# STEP 7: THEORETICAL VARIANCE
# ============================================================

theoretical_variance = (
    transactions_per_hour
    * p
    * (1 - p)
)

print("\nTheoretical Variance")
print("-" * 40)

print(
    "Var(X) =",
    theoretical_variance
)


# ============================================================
# STEP 8: EMPIRICAL VARIANCE
# ============================================================

empirical_variance = np.var(
    flagged_transactions
)

print("\nEmpirical Variance")
print("-" * 40)

print(
    "Empirical variance =",
    empirical_variance
)


# ============================================================
# STEP 9: FINAL RESULT TABLE
# ============================================================

mean_difference = abs(
    theoretical_mean - empirical_mean
)

variance_difference = abs(
    theoretical_variance - empirical_variance
)

print("\nFinal Result Table")
print("-" * 65)

print(
    f"{'Measure':<20}"
    f"{'Theoretical':<15}"
    f"{'Empirical':<15}"
    f"{'Difference':<15}"
)

print("-" * 65)

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
```

---

# 16. How to Run

From the project root:

```powershell
python problem_2_fraud_detection/main.py
```

Or:

```powershell
cd problem_2_fraud_detection
python main.py
```

---

# 17. Important Concepts

### Bernoulli vs Binomial

| Problem   | Situation                            | Distribution |
| --------- | ------------------------------------ | ------------ |
| Problem 1 | One customer clicks or doesn't click | Bernoulli    |
| Problem 2 | Count flagged transactions among 50  | Binomial     |

### Problem 2 formula

$$
\boxed{X\sim Binomial(50,0.04)}
$$

### Expected value

$$
\boxed{E[X]=2}
$$

### Variance

$$
\boxed{Var(X)=1.92}
$$

### Team capacity

$$
\boxed{X\le4}
$$

### Capacity exceeded

$$
\boxed{X>4}
$$

### Number of simulations

$$
\boxed{20,000}
$$

---

# 18. Final Conclusion

The fraud-detection problem is modeled using a **Binomial random variable** because each hour contains 50 independent transaction-level flagging decisions, each with probability 0.04 of being flagged.

Thus:

$$
X\sim Binomial(50,0.04)
$$

The theoretical expected number of flagged transactions per hour is:

$$
\boxed{2}
$$

and the theoretical variance is:

$$
\boxed{1.92}
$$

The simulation generates 20,000 independent hourly observations. The empirical mean and variance should generally be close to their theoretical values, while the exact empirical results vary from run to run because the simulation is random.
