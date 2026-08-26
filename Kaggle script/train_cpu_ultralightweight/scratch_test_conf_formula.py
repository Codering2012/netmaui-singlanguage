import numpy as np

for sim1, sim2 in [(0.88, 0.65), (0.92, 0.70), (0.95, 0.80), (0.75, 0.72)]:
    base_conf = 1.0 / (1.0 + np.exp(-18.0 * (sim1 - 0.76)))
    margin = max(0.0, sim1 - sim2)
    conf = float(np.clip(base_conf + 0.30 * margin, 0.50, 0.99))
    print(f"Sim1: {sim1:.2f}, Sim2: {sim2:.2f}, Margin: {margin:.2f} -> Conf: {conf*100:.1f}%")
