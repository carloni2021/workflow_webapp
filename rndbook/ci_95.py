from rndbook.rvms import idfStudent
import math
from statistics import mean, stdev, StatisticsError

# --- utility locali ---
def ci95_safe(xs):
    xs = list(xs)
    n = len(xs)
    alpha = 0.05
    if n == 0:
        return float("nan"), (float("nan"), float("nan"))
    m = mean(xs)
    if n == 1:
        return m, (m, m)
    try:
        s = stdev(xs)  # sample stdev (n-1)
    except StatisticsError:
        return m, (m, m)
    se = s / math.sqrt(n-1)
    t_star = idfStudent(n - 1, 1 - alpha / 2)
    return m, (m - t_star * se, m + t_star * se)

def ci95_hw(xs):
    xs = list(xs)
    n = len(xs)
    alpha = 0.05
    if n == 0:
        return float("nan"), (float("nan"), float("nan"))
    m = mean(xs)
    if n == 1:
        return m, (m, m)
    try:
        s = stdev(xs)  # sample stdev (n-1)
    except StatisticsError:
        return m, (m, m)
    se = s / math.sqrt(n-1)
    t_star = idfStudent(n - 1, 1 - alpha / 2)
    return m,t_star*se, n
