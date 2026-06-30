"""Generate an example dataset with a known true treatment effect of 300."""

import random
import csv
import math

random.seed(7)
rows = []
for _ in range(800):
    age = random.randint(22, 62)
    edu = random.randint(10, 20)
    prior = random.uniform(2000, 6000)
    # Treatment depends on confounders (education, prior income)
    logit = -1.0 + 0.1 * (edu - 15) + 0.0003 * (prior - 4000) + random.gauss(0, 0.5)
    p = 1 / (1 + math.exp(-logit))
    treat = 1 if random.random() < p else 0
    # Outcome: true treatment effect is +300
    outcome = 2000 + 80 * edu + 0.4 * prior + treat * 300 + random.gauss(0, 150)
    rows.append({
        "age": age,
        "education_years": edu,
        "prior_income": round(prior, 2),
        "treatment": treat,
        "outcome": round(max(0, outcome), 2),
    })

with open("data/sample_data.csv", "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
    writer.writeheader()
    writer.writerows(rows)

treated = sum(r["treatment"] for r in rows)
print(f"{len(rows)} rows, {treated} treated, true effect = 300")