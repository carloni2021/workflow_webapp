import numpy as np
import matplotlib.pyplot as plt

# 1. Parametri della distribuzione (coerenti con il tuo file ecommerce.py)
n_samples = 10**7  # Si suggerisce 10^7 campioni per la validazione
mean_target = 3.0
p_val = 0.10       # Parametro p usato nel tuo modello

# Calcolo del CV teorico per la H2 Balanced
cv_sq = 1 / (2 * p_val * (1 - p_val)) - 1
cv = np.sqrt(cv_sq)

# 2. Generazione campioni (Logica bilanciata definita in hyperexp.py)
m1 = mean_target / (2.0 * p_val)
m2 = mean_target / (2.0 * (1.0 - p_val))
u = np.random.random(n_samples)
samples = np.where(u < p_val, np.random.exponential(m1, n_samples), np.random.exponential(m2, n_samples))

# 3. Formula della densità teorica f(x) [cite: 530]
def h2_pdf(x, mu, p):
    l1 = 1 / (mu / (2.0 * p))
    l2 = 1 / (mu / (2.0 * (1.0 - p)))
    return p * l1 * np.exp(-l1 * x) + (1-p) * l2 * np.exp(-l2 * x)

# 4. Creazione del grafico di verifica [cite: 548]
plt.figure(figsize=(10, 6))
limit = np.percentile(samples, 98) # Taglio della coda per visibilità
x_range = np.linspace(0, limit, 1000)

# Istogramma continuo con k bin scelti secondo le regole (Sturges, Wand)
plt.hist(samples, bins=600, density=True, range=(0, limit),
         color='skyblue', edgecolor='blue', alpha=0.7,
         label='Densità stimata (bins)')

# Sovrapposizione della densità teorica (linea ROSSA) [cite: 529]
plt.plot(x_range, h2_pdf(x_range, mean_target, p_val),
         'red', lw=2.5, label='PDF teorica (linea)')

# Formattazione assi e intestazione come da esempio [cite: 551, 552, 567]
plt.title(f"Hyp(mean={mean_target} e cv={cv:.2f}) con PDF teorica sovrapposta")
plt.xlabel("x")
plt.ylabel("densità")

plt.legend()
plt.grid(True, alpha=0.2)
plt.tight_layout()
plt.show()