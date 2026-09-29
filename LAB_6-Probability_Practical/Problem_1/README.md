# Problem 1: Simulating Customer Clicks Using a Bernoulli Random Variable

## 1. Problem Statement

An organization sends promotional emails.

Historical data indicate that a customer clicks the link with probability:

$$
p = 0.30
$$

We define a random variable \(X\) to represent whether a customer clicks the promotional email link.

The objective is to simulate customer clicks using a Bernoulli random variable, calculate the empirical and theoretical PMFs, calculate expectation and variance, compare the results, and study the effect of increasing the sample size.

---

# 2. Random Variable Definition

We define:

$$
X =
\begin{cases}
1, & \text{if the customer clicks the link}\\
0, & \text{if the customer does not click the link}
\end{cases}
$$

Therefore:

* \(X = 1\) → Customer clicks
* \(X = 0\) → Customer does not click
* Possible values of \(X\) are \(\{0,1\}\)

Since \(X\) can take only discrete values, it is a **discrete random variable**.

The suitable probability distribution is the **Bernoulli distribution** because there are two possible outcomes: click or no click.

---

# 3. Step 1 — Identify the Random Variable

## Question 1: What does X = 1 represent?

### Answer:

\(X=1\) represents that the customer **clicks the promotional email link**.

---

## Question 2: What does X = 0 represent?

### Answer:

\(X=0\) represents that the customer **does not click the promotional email link**.

---

## Question 3: What are the possible values of X?

### Answer:

The possible values are:

$$
X \in \{0,1\}
$$

---

## Question 4: Is X a discrete or continuous random variable?

### Answer:

\(X\) is a **discrete random variable** because it can take only the values 0 and 1.

---

## Question 5: Which probability distribution is suitable for X?

### Answer:

The **Bernoulli distribution** is suitable because there is one trial with two possible outcomes:

* Success → Click
* Failure → No click

---

# 4. Step 2 — Theoretical PMF

For a Bernoulli random variable:

$$
P(X=x)=p^x(1-p)^{1-x}
$$

Given:

$$
p=0.30
$$

Therefore:

### For X = 0

$$
P(X=0)=1-p
$$

$$
P(X=0)=1-0.30=0.70
$$

### For X = 1

$$
P(X=1)=p
$$

$$
P(X=1)=0.30
$$

Therefore, the theoretical PMF is:

| X | Meaning  | Theoretical Probability |
| - | -------- | ----------------------: |
| 0 | No click |                    0.70 |
| 1 | Click    |                    0.30 |

The probabilities add up to 1:

$$
0.70+0.30=1
$$

---

# 5. Step 3 — Generate Outcomes for 20 Customers

We generate observations for 20 customers.

The Python function used is:

```python
np.random.binomial(n=1, p=p, size=sample_size)
```

Here:

* `n=1` → One Bernoulli trial
* `p=0.30` → Probability of success/click
* `size=20` → Generate observations for 20 customers

Example:

```text
[0 1 0 0 1 0 0 1 0 0 0 1 0 0 1 0 0 0 1 0]
```

The exact output changes each time because the observations are randomly generated.

---

## Question 1: How many observations were generated?

### Answer:

20 observations were generated because:

```python
sample_size = 20
```

---

## Question 2: What does each zero represent?

### Answer:

Each `0` represents a customer who **did not click** the link.

---

## Question 3: What does each one represent?

### Answer:

Each `1` represents a customer who **clicked** the link.

---

## Question 4: Are your generated observations the same as the theoretical probabilities?

### Answer:

Not necessarily.

The theoretical probability of a click is:

$$
P(X=1)=0.30
$$

However, the empirical probability obtained from only 20 randomly generated observations may be different from 0.30.

This difference occurs because of random variation in a finite sample.

As the sample size increases, the empirical probability generally becomes closer to the theoretical probability.

---

# 6. Step 4 — Count Clicks and Non-Clicks

We count:

* `1` → Click
* `0` → No click

Python:

```python
number_of_clicks = np.sum(samples == 1)
number_of_non_clicks = np.sum(samples == 0)
```

The complete section is:

```python
number_of_clicks = np.sum(samples == 1)
number_of_non_clicks = np.sum(samples == 0)

print("Number of clicks:", number_of_clicks)
print("Number of non-clicks:", number_of_non_clicks)
```

---

## Question 1: How many customers clicked?

### Answer:

The answer depends on the randomly generated sample.

It is obtained using:

```python
number_of_clicks = np.sum(samples == 1)
```

For example, if the generated sample contains five `1`s:

$$
\text{Number of clicks}=5
$$

---

## Question 2: How many customers did not click?

### Answer:

It is obtained using:

```python
number_of_non_clicks = np.sum(samples == 0)
```

For example, if there are five clicks out of 20:

$$
20-5=15
$$

Therefore:

$$
\text{Number of non-clicks}=15
$$

---

## Question 3: Do the two counts add up to 20?

### Answer:

Yes.

Every customer must either click or not click.

Therefore:

$$
\text{Clicks}+\text{Non-clicks}=20
$$

---

# 7. Step 5 — Calculate the Empirical PMF

The empirical probability is calculated from the observed data.

For \(X=0\):

$$
P_{empirical}(X=0)
=
\frac{\text{Number of non-clicks}}{20}
$$

For \(X=1\):

$$
P_{empirical}(X=1)
=
\frac{\text{Number of clicks}}{20}
$$

Python:

```python
empirical_p_0 = number_of_non_clicks / sample_size
empirical_p_1 = number_of_clicks / sample_size
```

For example, if there are:

* 15 non-clicks
* 5 clicks

then:

$$
P_{empirical}(X=0)=\frac{15}{20}=0.75
$$

$$
P_{empirical}(X=1)=\frac{5}{20}=0.25
$$

---

# 8. Step 6 — Compare Empirical and Theoretical PMFs

The theoretical PMF is:

| X | Theoretical |
| - | ----------: |
| 0 |        0.70 |
| 1 |        0.30 |

The empirical PMF depends on the generated sample.

For example:

| X | Theoretical | Empirical |
| - | ----------: | --------: |
| 0 |        0.70 |      0.75 |
| 1 |        0.30 |      0.25 |

---

## Question 1: Is the empirical probability of a click exactly 0.30?

### Answer:

Not necessarily.

The theoretical probability is:

$$
P(X=1)=0.30
$$

The empirical probability depends on the randomly generated observations.

It may be 0.25, 0.30, 0.35, etc.

---

## Question 2: Which outcome has the larger probability?

### Answer:

The outcome \(X=0\), representing **no click**, has the larger theoretical probability.

$$
P(X=0)=0.70
$$

while:

$$
P(X=1)=0.30
$$

Therefore:

$$
0.70>0.30
$$

---

## Question 3: Why are the empirical and theoretical probabilities different?

### Answer:

The theoretical probability represents the assumed underlying probability.

The empirical probability is calculated from a finite random sample.

Because the sample is random and only contains a limited number of observations, the empirical probability may differ from the theoretical probability.

With a larger sample size, the empirical probability generally gets closer to the theoretical probability.

---

# 9. Step 7 — Plot Empirical and Theoretical PMFs

We use Matplotlib to compare the two PMFs.

The graph contains:

* Theoretical probability
* Empirical probability
* \(X=0\)
* \(X=1\)

Python:

```python
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
```

---

# 10. Step 8 — Calculate Theoretical Expectation

For a Bernoulli random variable:

$$
E[X]=p
$$

Given:

$$
p=0.30
$$

Therefore:

$$
\boxed{E[X]=0.30}
$$

Python:

```python
theoretical_mean = p
```

---

# 11. Step 9 — Calculate Empirical Expectation

The empirical expectation is the sample mean:

$$
\bar X=\frac{1}{n}\sum_{i=1}^{n}X_i
$$

Python:

```python
empirical_mean = np.mean(samples)
```

Since the observations contain only 0 and 1, the empirical mean is also equal to the observed proportion of clicks.

For example, if there are 5 clicks out of 20:

$$
\bar X=\frac{5}{20}=0.25
$$

---

# 12. Step 10 — Calculate Theoretical Variance

For a Bernoulli random variable:

$$
Var(X)=p(1-p)
$$

Given:

$$
p=0.30
$$

Therefore:

$$
Var(X)=0.30(1-0.30)
$$

$$
=0.30(0.70)
$$

$$
\boxed{Var(X)=0.21}
$$

Python:

```python
theoretical_variance = p * (1 - p)
```

---

# 13. Step 11 — Calculate Empirical Variance

The empirical variance is calculated from the generated observations.

Python:

```python
empirical_variance = np.var(samples)
```

The exact value depends on the random sample.

---

# 14. Step 12 — Final Result Table

The required measures are:

* Expectation
* Variance

We compare:

* Theoretical value
* Empirical value
* Absolute difference

The absolute difference is:

$$
|\text{Theoretical}-\text{Empirical}|
$$

Python:

```python
mean_difference = abs(
    theoretical_mean - empirical_mean
)

variance_difference = abs(
    theoretical_variance - empirical_variance
)
```

The final table has the form:

| Measure     | Theoretical Value |       Empirical Value |   Absolute Difference |
| ----------- | ----------------: | --------------------: | --------------------: |
| Expectation |              0.30 | Depends on simulation | Depends on simulation |
| Variance    |              0.21 | Depends on simulation | Depends on simulation |

The empirical values will change each time the program is run because the observations are randomly generated.

---

# 15. Step 13 — Effect of Sample Size

The experiment is repeated for:

$$
n=10,\;100,\;1000,\;10000
$$

For every sample size, we calculate:

* Empirical \(P(X=0)\)
* Empirical \(P(X=1)\)
* Empirical mean
* Empirical variance

Python:

```python
sample_sizes = [10, 100, 1000, 10000]

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
        n,
        probability_0,
        probability_1,
        mean,
        variance
    )
```

---

# 16. Sample Size Table

The result will have this structure:

| Sample Size | Empirical P(X=0) | Empirical P(X=1) | Empirical Mean | Empirical Variance |
| ----------: | ---------------: | ---------------: | -------------: | -----------------: |
|          10 |           Random |           Random |         Random |             Random |
|         100 |           Random |           Random |         Random |             Random |
|       1,000 |           Random |           Random |         Random |             Random |
|      10,000 |           Random |           Random |         Random |             Random |

The exact values should **not** be hard-coded because they depend on the random simulation.

As the sample size increases, the empirical values generally approach the theoretical values:

$$
P(X=0)=0.70
$$

$$
P(X=1)=0.30
$$

$$
E[X]=0.30
$$

$$
Var(X)=0.21
$$

This demonstrates the effect of increasing the sample size.

---

# 17. Complete Project Structure

```text
LAB_6-Probability_Practical/
│
├── Problem_1/
   │
   ├── main.py
   └── README.md

```
---

# 18. Required Python Libraries

Install:

```bash
pip install numpy matplotlib
```

Libraries used:

### NumPy

Used for:

* Random simulation
* Counting observations
* Mean
* Variance
* Numerical calculations

### Matplotlib

Used for:

* Plotting theoretical PMF
* Plotting empirical PMF
* Comparing the two distributions

---

# 19. How to Run Problem 1

Open the terminal in the main project folder:

```bash
cd LAB_6-Probability_Practical
```

Run:

```bash
python Problem_1/main.py
```

Alternatively:

```bash
cd Problem_1
python main.py
```

---

# 20. Important Formulas

### Bernoulli PMF

$$
P(X=x)=p^x(1-p)^{1-x}
$$

### Probability of no click

$$
P(X=0)=1-p=0.70
$$

### Probability of click

$$
P(X=1)=p=0.30
$$

### Theoretical expectation

$$
E[X]=p=0.30
$$

### Theoretical variance

$$
Var(X)=p(1-p)=0.21
$$

### Empirical probability

$$
P_{empirical}(X=x)
=
\frac{\text{Number of observations with }X=x}{n}
$$

### Empirical expectation

$$
\bar X=\frac{1}{n}\sum_{i=1}^{n}X_i
$$

### Absolute difference

$$
|\text{Theoretical}-\text{Empirical}|
$$

---

# 21. Key Concepts to Remember

1. **Bernoulli distribution** has two possible outcomes.
2. `1` represents a click.
3. `0` represents no click.
4. The theoretical click probability is **0.30**.
5. The theoretical no-click probability is **0.70**.
6. The theoretical expectation is **0.30**.
7. The theoretical variance is **0.21**.
8. Empirical values come from simulated observations.
9. Empirical values can differ from theoretical values because of random variation.
10. Increasing the sample size generally makes empirical values closer to theoretical values.

---

# 22. Conclusion

The customer-click problem is modeled using a Bernoulli random variable because each customer has two possible outcomes: click or no click.

For the given probability \(p=0.30\):

$$
P(X=0)=0.70
$$

$$
P(X=1)=0.30
$$

$$
E[X]=0.30
$$

$$
Var(X)=0.21
$$

Simulation with different sample sizes demonstrates how empirical probabilities, expectation, and variance behave compared with their theoretical values.