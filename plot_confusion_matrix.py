from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


labels = ["TSS", "NTSS", "NS"]
matrix = np.array([
    [67961, 26406, 25633],
    [20499, 37147, 23597],
    [18623, 3611, 136283],
])
normalized = matrix / matrix.sum(axis=1, keepdims=True)

fig, axes = plt.subplots(1, 2, figsize=(12, 5), constrained_layout=True)
for ax, values, title, fmt in (
    (axes[0], matrix, "Confusion matrix (counts)", "{:.0f}"),
    (axes[1], normalized, "Confusion matrix (row normalized)", "{:.2f}"),
):
    image = ax.imshow(values, cmap="Blues", vmin=0, vmax=1 if values is normalized else None)
    ax.set_title(title)
    ax.set_xlabel("Predicted label")
    ax.set_ylabel("True label")
    ax.set_xticks(range(3), labels)
    ax.set_yticks(range(3), labels)
    threshold = values.max() * 0.55
    for row in range(3):
        for col in range(3):
            text_color = "white" if values[row, col] > threshold else "black"
            ax.text(col, row, fmt.format(values[row, col]), ha="center", va="center", color=text_color)
    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)

output = Path("docs/confusion_matrix_iter_5000.png")
output.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(output, dpi=180)
print(output)
