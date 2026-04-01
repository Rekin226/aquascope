# AquaScope Theory Guide

A mathematical reference for every method in AquaScope. Each section states the
equations, assumptions, and key references with DOI links so you can trace every
result back to the peer-reviewed literature.

---

## Table of Contents

1. [Flood Frequency Analysis](#1-flood-frequency-analysis)
2. [Baseflow Separation](#2-baseflow-separation)
3. [Flow Duration Curves](#3-flow-duration-curves)
4. [Hydrological Signatures](#4-hydrological-signatures)
5. [Extreme Value Theory (Extended)](#5-extreme-value-theory-extended)
6. [Copula Analysis](#6-copula-analysis)
7. [Change-Point Detection](#7-change-point-detection)
8. [Bayesian Uncertainty Quantification](#8-bayesian-uncertainty-quantification)
9. [Rating Curves](#9-rating-curves)
10. [Transfer Learning for Ungauged Basins](#10-transfer-learning-for-ungauged-basins)
11. [Decision Trees — Choosing the Right Method](#11-decision-trees--choosing-the-right-method)

---

## 1. Flood Frequency Analysis

### 1.1 Generalized Extreme Value (GEV)

The GEV distribution unifies the Gumbel, Fréchet, and Weibull extreme-value
families into a single three-parameter model.

**CDF:**

$$F(x) = \exp\!\Bigl\{-\Bigl[1 + \xi\,\frac{x - \mu}{\sigma}\Bigr]^{-1/\xi}\Bigr\}, \qquad 1 + \xi\,\frac{x-\mu}{\sigma} > 0$$

where $\mu$ is the location, $\sigma > 0$ the scale, and $\xi$ the shape
parameter. When $\xi \to 0$ the distribution reduces to the Gumbel.

**Quantile function (return-period estimate):**

$$x_p = \mu - \frac{\sigma}{\xi}\bigl[1 - (-\ln p)^{-\xi}\bigr], \qquad p = 1 - 1/T$$

where $T$ is the return period in years.

**Parameter estimation:** AquaScope uses maximum-likelihood estimation (MLE) via
`scipy.stats.genextreme.fit()`. Bootstrap confidence intervals are computed by
resampling the annual maximum series 1 000 times.

> **Reference:** Coles, S. (2001). *An Introduction to Statistical Modeling of
> Extreme Values.* Springer.
> [doi:10.1007/978-1-4471-3675-0](https://doi.org/10.1007/978-1-4471-3675-0)

### 1.2 Log-Pearson Type III (LP3)

The US standard method (Bulletin 17C). The annual maximum series $Q$ is
log-transformed:

$$Y = \log_{10}(Q)$$

A Pearson Type III distribution is fitted to $Y$ with moments:

- $\bar{Y} = \frac{1}{n}\sum Y_i$ (mean)
- $S_Y = \sqrt{\frac{1}{n-1}\sum(Y_i - \bar{Y})^2}$ (standard deviation)
- $G_s = \frac{n}{(n-1)(n-2)\,S_Y^3}\sum(Y_i - \bar{Y})^3$ (station skew)

**Bulletin 17C weighted skew:**

$$G_w = w_1\,G_s + w_2\,G_r$$

where $G_r$ is the regional (generalized) skew and the weights are
inverse-variance:

$$w_1 = \frac{\text{MSE}_{G_r}}{\text{MSE}_{G_s} + \text{MSE}_{G_r}}, \qquad w_2 = 1 - w_1$$

Station skew MSE is approximated as:

$$\text{MSE}_{G_s} \approx \frac{6}{n}\Bigl(1 + \frac{9}{6}G_s^2 + \frac{15}{48}G_s^4\Bigr)$$

The default regional skew MSE is 0.302 (USGS national value).

**Return-period quantile:**

$$Q_T = 10^{\bar{Y} + K_T \cdot S_Y}$$

where $K_T$ is the Pearson Type III frequency factor for exceedance probability
$1/T$ and skew $G_w$.

> **Reference:** England, J. F., Jr., et al. (2018). *Guidelines for
> Determining Flood Flow Frequency — Bulletin 17C.* USGS Techniques and
> Methods, Book 4, Chapter B5.
> [doi:10.3133/tm4B5](https://doi.org/10.3133/tm4B5)

### 1.3 Expected Moments Algorithm (EMA)

EMA extends LP3 to handle **censored observations** — zero-flow years,
low outliers detected by the Multiple Grubbs-Beck test, and historical
(paleoflood) data.

Each observation $i$ is associated with a **perception threshold** interval
$[T_{L,i},\, T_{U,i}]$:

- Systematic record above threshold: $T_L = 0,\ T_U = \infty$
- Zero-flow year: $T_L = 0,\ T_U = Q_{\text{thresh}}$ (left-censored)
- Historical flood: $T_L = Q_{\text{perception}},\ T_U = \infty$

The algorithm iteratively adjusts the first three moments of $\log_{10}(Q)$ to
account for the probability mass below/above the perception thresholds.

**Censored-adjusted mean:**

$$\bar{Y}_{\text{adj}} = \frac{1}{n_{\text{eff}}}\Bigl[\sum_{i \in \text{obs}} Y_i + n_c\,\hat{\mu}_c\Bigr]$$

where $n_c$ is the number of censored observations and $\hat{\mu}_c$ is the
expected value of $Y$ conditional on $Y \leq \log_{10}(T_U)$ under the current
LP3 fit.

> **Reference:** Cohn, T. A., Lane, W. L., & Baier, W. G. (1997). An algorithm
> for computing moments-based flood quantile estimates when historical flood
> information is available. *Water Resources Research*, 33(9), 2089–2096.
> [doi:10.1029/96WR03706](https://doi.org/10.1029/96WR03706)

### 1.4 Multiple Grubbs-Beck Test

Identifies **low outliers** that should be treated as censored in the EMA.

For sample size $n$, the test statistic for the $k$-th smallest value is:

$$G_k = \frac{\bar{Y}_{(-k)} - Y_{(k)}}{S_{(-k)}}$$

where $\bar{Y}_{(-k)}$ and $S_{(-k)}$ are the mean and standard deviation of
the sample excluding the $k$ smallest values. Values exceeding the critical
value at significance level $\alpha$ are flagged as low outliers.

> **Reference:** Grubbs, F. E., & Beck, G. (1972). Extension of sample sizes
> and percentage points for significance tests of outlying observations.
> *Technometrics*, 14(4), 847–854.
> [doi:10.1080/00401706.1972.10488981](https://doi.org/10.1080/00401706.1972.10488981)

### 1.5 L-Moments

L-moments are linear combinations of probability-weighted moments (PWMs), more
robust than conventional moments for small samples.

**Probability-weighted moments:**

$$\beta_r = \frac{1}{n}\sum_{i=1}^{n} \frac{\binom{i-1}{r}}{\binom{n-1}{r}}\,x_{(i)}$$

**First four L-moments:**

$$\lambda_1 = \beta_0, \qquad \lambda_2 = 2\beta_1 - \beta_0$$
$$\lambda_3 = 6\beta_2 - 6\beta_1 + \beta_0, \qquad \lambda_4 = 20\beta_3 - 30\beta_2 + 12\beta_1 - \beta_0$$

**L-moment ratios:**

$$\tau = \lambda_2 / \lambda_1 \text{ (L-CV)}, \qquad \tau_3 = \lambda_3 / \lambda_2 \text{ (L-skewness)}, \qquad \tau_4 = \lambda_4 / \lambda_2 \text{ (L-kurtosis)}$$

**GEV parameters from L-moments:**

$$c = \frac{2}{3 + \tau_3} - \frac{\ln 2}{\ln 3}, \qquad \xi \approx 7.8590c + 2.9554c^2$$
$$\sigma = \frac{\lambda_2\,\xi}{(1 - 2^{-\xi})\,\Gamma(1+\xi)}, \qquad \mu = \lambda_1 - \frac{\sigma}{\xi}\bigl[\Gamma(1+\xi) - 1\bigr]$$

> **Reference:** Hosking, J. R. M. (1990). L-moments: analysis and estimation
> of distributions using linear combinations of order statistics.
> *Journal of the Royal Statistical Society B*, 52(1), 105–124.
> [doi:10.1111/j.2517-6161.1990.tb01775.x](https://doi.org/10.1111/j.2517-6161.1990.tb01775.x)

### 1.6 Non-Stationary GEV

For climate-affected flood series, the location parameter varies with time:

$$\mu(t) = \mu_0 + \mu_1\,t$$

The log-likelihood is maximised over $(\mu_0, \mu_1, \sigma, \xi)$ using
Nelder-Mead optimisation. AquaScope reports trend significance and
per-year quantile estimates.

### 1.7 Regional Frequency Analysis

The **index-flood method** (Hosking & Wallis, 1997) pools data from multiple
sites to improve quantile estimates at each site.

1. **Discordancy measure** $D_i$: identifies sites whose L-moment ratios are
   outliers relative to the regional average.
2. **Heterogeneity measure** $H$: tests whether sites form a homogeneous region.
   $H < 1$: acceptably homogeneous; $1 \leq H < 2$: possibly heterogeneous;
   $H \geq 2$: definitely heterogeneous.
3. **Regional growth curve**: a common dimensionless frequency curve scaled by
   each site's index flood (typically the sample mean).

> **Reference:** Hosking, J. R. M., & Wallis, J. R. (1997). *Regional
> Frequency Analysis: An Approach Based on L-Moments.* Cambridge University
> Press. ISBN 978-0521019408.

---

## 2. Baseflow Separation

### 2.1 Lyne-Hollick Digital Filter

A one-parameter recursive filter applied to the total streamflow $Q(t)$.

**Quickflow equation:**

$$q_f(t) = \alpha\,q_f(t-1) + \frac{1+\alpha}{2}\bigl[Q(t) - Q(t-1)\bigr]$$

with the constraint $0 \leq q_f(t) \leq Q(t)$.

**Baseflow:**

$$q_b(t) = Q(t) - q_f(t)$$

The filter parameter $\alpha$ controls smoothness (typical value: 0.925).
Multiple forward-backward passes improve separation quality.

> **Reference:** Nathan, R. J., & McMahon, T. A. (1990). Evaluation of
> automated techniques for base flow and recession analyses. *Water Resources
> Research*, 26(7), 1465–1473.
> [doi:10.1016/0022-1694(90)90259-2](https://doi.org/10.1016/0022-1694(90)90259-2)

### 2.2 Eckhardt Two-Parameter Filter

$$q_b(t) = \frac{(1 - \text{BFI}_{\max})\,\alpha\,q_b(t-1) + (1-\alpha)\,\text{BFI}_{\max}\,Q(t)}{1 - \alpha\,\text{BFI}_{\max}}$$

with $q_b(t) \leq Q(t)$.

The $\text{BFI}_{\max}$ parameter depends on aquifer type:

| Aquifer type | Recommended BFI_max |
|---|---|
| Perennial streams, porous aquifer | 0.80 |
| Ephemeral streams, porous aquifer | 0.50 |
| Perennial streams, hard rock | 0.25 |

> **Reference:** Eckhardt, K. (2005). How to construct recursive digital
> filters for baseflow separation. *Hydrological Processes*, 19(2), 507–515.
> [doi:10.1016/j.jhydrol.2005.07.017](https://doi.org/10.1016/j.jhydrol.2005.07.017)

---

## 3. Flow Duration Curves

The FDC shows the percentage of time a given discharge is equalled or exceeded.

**Weibull plotting position:**

$$p_i = \frac{i}{n + 1}$$

where $i$ is the rank (1 = largest) and $n$ is the sample size. This is
an exceedance probability.

**FDC slope** (between the 33rd and 66th percentiles on a log scale):

$$S_{\text{FDC}} = \frac{\ln Q_{33} - \ln Q_{66}}{0.66 - 0.33}$$

A steep slope indicates a flashy regime; a flat slope indicates sustained
baseflow.

> **Reference:** Vogel, R. M., & Fennessey, N. M. (1995). Flow duration curves
> II: A review of applications in water resources planning.
> *Water Resources Bulletin*, 31(6), 1029–1039.
> [doi:10.1111/j.1752-1688.1995.tb04392.x](https://doi.org/10.1111/j.1752-1688.1995.tb04392.x)

---

## 4. Hydrological Signatures

AquaScope computes 22 catchment signatures from daily discharge. These
characterise magnitude, variability, frequency, duration, timing, and rate of
change of streamflow.

| Signature | Formula | Unit |
|---|---|---|
| Mean discharge ($\bar{Q}$) | $\frac{1}{n}\sum Q_i$ | m³/s |
| Median discharge ($Q_{50}$) | 50th percentile | m³/s |
| Coefficient of variation | $\text{CV} = S_Q / \bar{Q}$ | — |
| Skewness | $\frac{n}{(n-1)(n-2)}\sum\bigl(\frac{Q_i - \bar{Q}}{S_Q}\bigr)^3$ | — |
| Low flow (Q95) | 95th exceedance percentile | m³/s |
| High flow (Q5) | 5th exceedance percentile | m³/s |
| Baseflow index (BFI) | $\sum q_b / \sum Q$ | — |
| Runoff ratio | $\sum Q / \sum P$ | — |
| FDC slope | See §3 | — |
| Flashiness index | $\sum\lvert Q_i - Q_{i-1}\rvert / \sum Q_i$ | — |
| Rising limb density | (# rising days) / (# peaks) | d/peak |
| Falling limb density | (# falling days) / (# peaks) | d/peak |
| Zero-flow fraction | (# days with Q = 0) / n | — |
| High-flow frequency | (# days with Q > 9 × $Q_{50}$) / n | — |
| High-flow duration | mean consecutive days above 9 × $Q_{50}$ | d |
| Low-flow frequency | (# days with Q < 0.2 × $\bar{Q}$) / n | — |
| Low-flow duration | mean consecutive days below 0.2 × $\bar{Q}$ | d |
| Recession constant ($k$) | slope of $\ln Q$ during recessions | 1/d |
| Peak month | month with highest mean Q | month |
| Seasonality index | circular variance of monthly Q | — |
| Flow elasticity | $\text{median}\bigl(\frac{\Delta Q/\bar{Q}}{\Delta P/\bar{P}}\bigr)$ | — |
| Mean half-flow date | day of year at which 50 % of annual flow has passed | DOY |

> **Reference:** McMillan, H. (2020). Linking hydrologic signatures to
> hydrologic processes: A review. *WIREs Water*, 7(6), e1481.
> [doi:10.1002/wat2.1481](https://doi.org/10.1002/wat2.1481)

---

## 5. Extreme Value Theory (Extended)

### 5.1 Gumbel Distribution

A special case of GEV with $\xi = 0$:

$$F(x) = \exp\!\bigl[-\exp\!\bigl(-\frac{x - \mu}{\sigma}\bigr)\bigr]$$

$$x_p = \mu - \sigma\,\ln(-\ln p)$$

Appropriate when there is no evidence of a heavy or bounded upper tail.

### 5.2 Generalized Pareto Distribution (POT)

For peaks over a threshold $u$, exceedances $y = x - u$ follow:

$$F(y) = 1 - \Bigl(1 + \xi\,\frac{y}{\tilde\sigma}\Bigr)^{-1/\xi}$$

The threshold $u$ is selected as the empirical quantile at which the mean
residual life plot becomes approximately linear.

**Equivalence:** If block maxima follow $\text{GEV}(\mu, \sigma, \xi)$, then
exceedances above a high threshold follow $\text{GPD}(\tilde\sigma, \xi)$ with
the same shape parameter.

> **Reference:** Davison, A. C., & Smith, R. L. (1990). Models for exceedances
> over high thresholds. *Journal of the Royal Statistical Society B*, 52(3),
> 393–442.
> [doi:10.1111/j.2517-6161.1990.tb01796.x](https://doi.org/10.1111/j.2517-6161.1990.tb01796.x)

### 5.3 Goodness-of-Fit Tests

**Anderson-Darling (AD):**

$$A^2 = -n - \frac{1}{n}\sum_{i=1}^{n}(2i - 1)\bigl[\ln F(x_{(i)}) + \ln(1 - F(x_{(n+1-i)}))\bigr]$$

Emphasises tail behaviour more than the Kolmogorov-Smirnov test.

**Cramér-von Mises (CvM):**

$$W^2 = \sum_{i=1}^{n}\Bigl[F(x_{(i)}) - \frac{2i - 1}{2n}\Bigr]^2 + \frac{1}{12n}$$

**Probability Plot Correlation Coefficient (PPCC):**

$$r = \frac{\sum(x_{(i)} - \bar{x})(m_i - \bar{m})}{\sqrt{\sum(x_{(i)} - \bar{x})^2\sum(m_i - \bar{m})^2}}$$

where $m_i$ are the theoretical quantiles. Values close to 1 indicate good fit.

---

## 6. Copula Analysis

### 6.1 Sklar's Theorem

Any joint distribution $H(x, y)$ can be expressed via a copula $C$:

$$H(x, y) = C\bigl(F_X(x),\, F_Y(y)\bigr)$$

where $F_X, F_Y$ are the marginal CDFs. AquaScope uses **pseudo-observations**
$u_i = R_i / (n + 1)$ (rank-based) to avoid boundary artefacts.

### 6.2 Implemented Families

| Family | CDF $C(u,v)$ | $\tau \to \theta$ | Tail dependence |
|---|---|---|---|
| **Gaussian** | $\Phi_2\bigl(\Phi^{-1}(u), \Phi^{-1}(v); \rho\bigr)$ | $\rho = \sin(\pi\tau/2)$ | None |
| **Clayton** | $\bigl(u^{-\theta} + v^{-\theta} - 1\bigr)^{-1/\theta}$ | $\theta = 2\tau/(1-\tau)$ | Lower: $2^{-1/\theta}$ |
| **Gumbel** | $\exp\!\bigl[-\bigl((-\ln u)^\theta + (-\ln v)^\theta\bigr)^{1/\theta}\bigr]$ | $\theta = 1/(1-\tau)$ | Upper: $2 - 2^{1/\theta}$ |
| **Frank** | $-\frac{1}{\theta}\ln\!\Bigl[1 + \frac{(e^{-\theta u}-1)(e^{-\theta v}-1)}{e^{-\theta}-1}\Bigr]$ | Numerical (Debye function) | None |

**Model selection** uses AIC:

$$\text{AIC} = 2k - 2\ln\hat{L}$$

where $k = 1$ (all families have one parameter) and $\hat{L}$ is the
pseudo-log-likelihood evaluated over the copula density.

> **References:**
> - Nelsen, R. B. (2006). *An Introduction to Copulas.* 2nd ed. Springer.
>   ISBN 978-0387286785.
> - Genest, C., & Favre, A.-C. (2007). Everything you always wanted to know
>   about copula modeling but were afraid to ask. *Journal of Hydrologic
>   Engineering*, 12(4), 347–368.
>   [doi:10.1061/(ASCE)1084-0699(2007)12:4(347)](https://doi.org/10.1061/(ASCE)1084-0699(2007)12:4(347))

### 6.3 When to Use Which Copula

- **Symmetric dependence** (e.g., precipitation at two nearby gauges) → Gaussian
  or Frank
- **Lower tail dependence** (joint low-flow risk, drought co-occurrence) →
  Clayton
- **Upper tail dependence** (joint flood risk) → Gumbel
- **Unsure** → use `fit_copula(x, y, family="auto")` — AquaScope fits all
  four and selects by AIC

---

## 7. Change-Point Detection

### 7.1 PELT (Pruned Exact Linear Time)

Minimises the penalised cost:

$$\sum_{j=0}^{m} \bigl[\mathcal{C}(y_{\tau_j+1:\tau_{j+1}})\bigr] + \beta\,m$$

where $\mathcal{C}$ is the segment cost (e.g., negative Gaussian log-likelihood),
$m$ is the number of changepoints, and $\beta$ is the penalty.
AquaScope uses the BIC penalty $\beta = \ln(n)$ by default.

The pruning rule discards candidate changepoints that provably cannot improve
the optimum, achieving $O(n)$ expected complexity.

> **Reference:** Killick, R., Fearnhead, P., & Eckley, I. A. (2012). Optimal
> detection of changepoints with a linear computational cost. *Journal of the
> American Statistical Association*, 107(500), 1590–1598.
> [doi:10.1080/01621459.2012.737745](https://doi.org/10.1080/01621459.2012.737745)

### 7.2 Pettitt Test

A non-parametric test for a single changepoint. The test statistic:

$$U_T = \max_{1 \leq t < n} \Bigl|\sum_{i=1}^{t}\sum_{j=t+1}^{n} \text{sgn}(X_i - X_j)\Bigr|$$

The approximate p-value is:

$$p \approx 2\exp\!\Bigl(\frac{-6\,U_T^2}{n^3 + n^2}\Bigr)$$

> **Reference:** Pettitt, A. N. (1979). A non-parametric approach to the
> change-point problem. *Journal of the Royal Statistical Society C*, 28(2),
> 126–135.
> [doi:10.2307/2346729](https://doi.org/10.2307/2346729)

### 7.3 CUSUM

The cumulative sum chart detects shifts in the mean:

$$S_t = \max\!\bigl(0,\, S_{t-1} + (x_t - \bar{x}) - k\bigr)$$

where $k$ is a slack parameter (default: $0.5\,\hat\sigma$). A changepoint is
flagged when $S_t$ exceeds threshold $h$ (default: $4\,\hat\sigma$).

---

## 8. Bayesian Uncertainty Quantification

### 8.1 Conjugate Normal-Inverse-Gamma Model

For linear regression $y = X\beta + \varepsilon$, $\varepsilon \sim N(0, \sigma^2)$:

**Prior:**

$$\beta \mid \sigma^2 \sim N(\mu_0,\, \sigma^2 \Lambda_0^{-1}), \qquad \sigma^2 \sim \text{Inv-Gamma}(a_0, b_0)$$

**Posterior (closed-form):**

$$\Lambda_n = \Lambda_0 + X^T X$$
$$\mu_n = \Lambda_n^{-1}(\Lambda_0 \mu_0 + X^T y)$$
$$a_n = a_0 + n/2$$
$$b_n = b_0 + \tfrac{1}{2}\bigl(y^T y + \mu_0^T \Lambda_0 \mu_0 - \mu_n^T \Lambda_n \mu_n\bigr)$$

**Predictive distribution** (Student-t):

$$\tilde{y} \mid x_* \sim t_{2a_n}\!\Bigl(\mu_n^T x_*,\; \frac{b_n}{a_n}\bigl(1 + x_*^T \Lambda_n^{-1} x_*\bigr)\Bigr)$$

> **Reference:** Gelman, A., et al. (2013). *Bayesian Data Analysis.* 3rd ed.
> Chapman & Hall/CRC. ISBN 978-1439840955.

### 8.2 Metropolis-Hastings MCMC

For non-conjugate models, AquaScope uses random-walk Metropolis-Hastings:

1. Propose $\theta^* = \theta^{(t)} + \varepsilon$, $\varepsilon \sim N(0, \Sigma_{\text{prop}})$
2. Compute acceptance ratio:

$$\alpha = \min\!\Bigl(1,\; \frac{p(\theta^*)\,L(y|\theta^*)}{p(\theta^{(t)})\,L(y|\theta^{(t)})}\Bigr)$$

3. Accept with probability $\alpha$; otherwise $\theta^{(t+1)} = \theta^{(t)}$

The proposal covariance $\Sigma_{\text{prop}}$ must be tuned for ~23 % acceptance
rate (high dimension) or ~44 % (1-D).

### 8.3 Convergence Diagnostics

**Gelman-Rubin $\hat{R}$** (split-chain method):

$$\hat{R} = \sqrt{\frac{\hat{V}}{W}}$$

where $W$ is the within-chain variance and $\hat{V}$ is the pooled variance
estimate. Convergence is indicated by $\hat{R} < 1.05$.

**Effective sample size:**

$$n_{\text{eff}} = \frac{mn}{1 + 2\sum_{k=1}^{K} \hat\rho_k}$$

where $\hat\rho_k$ is the estimated autocorrelation at lag $k$.

**DIC (Deviance Information Criterion):**

$$\text{DIC} = \bar{D} + p_D, \qquad p_D = \bar{D} - D(\bar\theta)$$

**WAIC (Widely Applicable Information Criterion):**

$$\text{WAIC} = -2\sum_{i=1}^{n}\ln\!\Bigl(\frac{1}{S}\sum_{s=1}^{S}p(y_i|\theta^{(s)})\Bigr) + 2\,p_{\text{WAIC}}$$

where $p_{\text{WAIC}} = \sum_i \text{Var}_s[\ln p(y_i|\theta^{(s)})]$.

---

## 9. Rating Curves

### 9.1 Power-Law Model

$$Q = a\,(h - h_0)^b$$

where $Q$ is discharge, $h$ is stage, $h_0$ is the effective zero-flow stage,
$a$ is a coefficient, and $b$ is the exponent. Fitted in log-space via linear
regression:

$$\ln Q = \ln a + b\,\ln(h - h_0)$$

### 9.2 Segmented Rating Curves

When channel geometry changes at a breakpoint stage $h_b$, two power-law
segments are fitted with continuity enforced at $h_b$:

$$Q = \begin{cases} a_1(h - h_{0,1})^{b_1} & h \leq h_b \\ a_2(h - h_{0,2})^{b_2} & h > h_b \end{cases}$$

### 9.3 Shift Detection

Temporal shifts in the rating curve (due to scour, deposition, or vegetation
change) are detected by monitoring residuals over time. A significant trend in
residuals triggers a shift warning.

> **Reference:** Rantz, S. E., et al. (1982). *Measurement and Computation of
> Streamflow. Volume 2: Computation of Discharge.* USGS Water-Supply Paper
> 2175.

---

## 10. Transfer Learning for Ungauged Basins

### 10.1 Problem

Many basins lack sufficient streamflow data for calibration. Transfer learning
borrows information from **donor sites** with similar hydrological behaviour.

### 10.2 Donor Selection

AquaScope selects donors by computing the **similarity score** between the
target and candidate sites based on their hydrological signature vectors:

$$d(A, B) = \sqrt{\sum_{k=1}^{K} w_k \Bigl(\frac{s_k^A - s_k^B}{\hat\sigma_k}\Bigr)^2}$$

where $s_k$ are standardised signatures and $w_k$ are optional weights.

**Spatial proximity weighting** uses the Haversine distance:

$$d_{\text{geo}} = 2R\arcsin\!\sqrt{\sin^2\!\frac{\Delta\phi}{2} + \cos\phi_1\cos\phi_2\sin^2\!\frac{\Delta\lambda}{2}}$$

$$w_{\text{spatial}} = \frac{1}{1 + d_{\text{geo}} / d_0}$$

### 10.3 Workflow

1. Compute signatures for all candidate donors and the target
2. Rank donors by similarity score (optionally weighted by spatial proximity)
3. Pool training data from top-$k$ donors
4. Train a model on pooled data
5. Fine-tune on available target data (even if very short record)

> **Reference:** Hrachowitz, M., et al. (2013). A decade of Predictions in
> Ungauged Basins (PUB) — a review. *Hydrological Sciences Journal*, 58(6),
> 1198–1255.
> [doi:10.5194/hess-17-1893-2013](https://doi.org/10.5194/hess-17-1893-2013)

---

## 11. Decision Trees — Choosing the Right Method

### Which Flood Frequency Method?

```
                      ┌─ Yes ─→ EMA (§1.3)
    Zero-flow years? ─┤
                      └─ No
                           ┌─ Yes ─→ LP3 + Bulletin 17C (§1.2)
    US regulatory context? ┤
                           └─ No
                                ┌─ Yes ─→ Non-stationary GEV (§1.6)
    Evidence of trend?         ─┤
                                └─ No
                                     ┌─ < 25 years ─→ L-moments (§1.5)
    Record length?                  ─┤
                                     ├─ 25–50 years ─→ GEV MLE (§1.1)
                                     └─ Multiple sites ─→ Regional (§1.7)
```

### Which Baseflow Method?

```
    Quick estimate, 1 parameter    ─→ Lyne-Hollick (§2.1)
    Aquifer-type–aware, 2 params   ─→ Eckhardt (§2.2)
```

### Which Copula?

```
    Joint flood risk (upper tail)  ─→ Gumbel
    Joint drought risk (lower tail)─→ Clayton
    Symmetric / unsure             ─→ Gaussian or Frank
    Don't know                     ─→ family="auto" (AIC selection)
```

### When to Use Bayesian Methods?

```
    Small sample (n < 30)          ─→ Bayesian with informative prior (§8.1)
    Uncertainty is the main output ─→ Always Bayesian
    Regulatory context             ─→ Frequentist primary, Bayesian as check
    Complex non-linear model       ─→ MH-MCMC (§8.2)
```

### When to Use Transfer Learning?

```
    Target has < 2 years of data   ─→ Transfer from donors (§10)
    Target has > 10 years           ─→ Calibrate directly
    2–10 years                      ─→ Transfer + fine-tune
```

---

## References (Consolidated)

1. Addor, N., et al. (2017). The CAMELS data set. *HESS*, 21, 5293–5313.
   [doi:10.5194/hess-21-5293-2017](https://doi.org/10.5194/hess-21-5293-2017)
2. Cohn, T. A., et al. (1997). *Water Resources Research*, 33(9), 2089–2096.
   [doi:10.1029/96WR03706](https://doi.org/10.1029/96WR03706)
3. Coles, S. (2001). *An Introduction to Statistical Modeling of Extreme Values.* Springer.
   [doi:10.1007/978-1-4471-3675-0](https://doi.org/10.1007/978-1-4471-3675-0)
4. Davison, A. C., & Smith, R. L. (1990). *J. R. Statist. Soc. B*, 52(3), 393–442.
   [doi:10.1111/j.2517-6161.1990.tb01796.x](https://doi.org/10.1111/j.2517-6161.1990.tb01796.x)
5. Eckhardt, K. (2005). *Hydrological Processes*, 19(2), 507–515.
   [doi:10.1016/j.jhydrol.2005.07.017](https://doi.org/10.1016/j.jhydrol.2005.07.017)
6. England, J. F., Jr., et al. (2018). Bulletin 17C. USGS TM 4-B5.
   [doi:10.3133/tm4B5](https://doi.org/10.3133/tm4B5)
7. Gelman, A., et al. (2013). *Bayesian Data Analysis.* 3rd ed. CRC Press.
8. Genest, C., & Favre, A.-C. (2007). *J. Hydrol. Eng.*, 12(4), 347–368.
   [doi:10.1061/(ASCE)1084-0699(2007)12:4(347)](https://doi.org/10.1061/(ASCE)1084-0699(2007)12:4(347))
9. Grubbs, F. E., & Beck, G. (1972). *Technometrics*, 14(4), 847–854.
   [doi:10.1080/00401706.1972.10488981](https://doi.org/10.1080/00401706.1972.10488981)
10. Hosking, J. R. M. (1990). *J. R. Statist. Soc. B*, 52(1), 105–124.
    [doi:10.1111/j.2517-6161.1990.tb01775.x](https://doi.org/10.1111/j.2517-6161.1990.tb01775.x)
11. Hosking, J. R. M., & Wallis, J. R. (1997). *Regional Frequency Analysis.* Cambridge.
12. Hrachowitz, M., et al. (2013). *Hydrol. Sci. J.*, 58(6), 1198–1255.
    [doi:10.5194/hess-17-1893-2013](https://doi.org/10.5194/hess-17-1893-2013)
13. Killick, R., et al. (2012). *JASA*, 107(500), 1590–1598.
    [doi:10.1080/01621459.2012.737745](https://doi.org/10.1080/01621459.2012.737745)
14. McMillan, H. (2020). *WIREs Water*, 7(6), e1481.
    [doi:10.1002/wat2.1481](https://doi.org/10.1002/wat2.1481)
15. Nathan, R. J., & McMahon, T. A. (1990). *Water Resour. Res.*, 26(7), 1465–1473.
    [doi:10.1016/0022-1694(90)90259-2](https://doi.org/10.1016/0022-1694(90)90259-2)
16. Nelsen, R. B. (2006). *An Introduction to Copulas.* 2nd ed. Springer.
17. Pettitt, A. N. (1979). *J. R. Statist. Soc. C*, 28(2), 126–135.
    [doi:10.2307/2346729](https://doi.org/10.2307/2346729)
18. Rantz, S. E., et al. (1982). USGS Water-Supply Paper 2175.
19. Vogel, R. M., & Fennessey, N. M. (1995). *Water Resour. Bull.*, 31(6), 1029–1039.
    [doi:10.1111/j.1752-1688.1995.tb04392.x](https://doi.org/10.1111/j.1752-1688.1995.tb04392.x)
