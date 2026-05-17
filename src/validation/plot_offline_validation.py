import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

# ------------------------------------------------------------------
# 1)  Numbers (per model)
# ------------------------------------------------------------------
models = ["ind_Ryan", "ind_Hridayam", "sync", "unsync"]
rmse   = np.array([1.87, 3.67, 4.90, 5.09])      #  [%]
r2     = np.array([0.908, 0.795, 0.820, 0.849])  #  [–]

# ------------------------------------------------------------------
# 2)  Plot
# ------------------------------------------------------------------
x   = np.arange(len(models))
w   = 0.6

COL_RMSE = "#1962B0"   # deep-blue
COL_R2   = "#B22B34"   # coral-red

fig, ax_rmse = plt.subplots(figsize=(7, 4))

# -- RMSE bars (left axis) -----------------------------------------------------
bars = ax_rmse.bar(x, rmse, width=w, color=COL_RMSE, zorder=2)
ax_rmse.set_ylabel("RMSE [%]")
ax_rmse.set_xticks(x, models, rotation=15)
ax_rmse.tick_params(axis="both", direction="in", top=False, right=False)
ax_rmse.spines["top"].set_visible(False)
ax_rmse.spines["right"].set_visible(False)
ax_rmse.set_ylim(0, rmse.max()*1.15)

# -- R² line (right axis) ------------------------------------------------------
ax_r2 = ax_rmse.twinx()
ax_r2.plot(x, r2, color=COL_R2, marker="o", linewidth=2, zorder=3)
ax_r2.set_ylabel("R²")
ax_r2.tick_params(axis="y", direction="in", colors=COL_R2)
ax_r2.set_ylim(0, 1.0)

# -- Legend (custom handles) ---------------------------------------------------
handles = [
    bars[0],                                   # any bar represents RMSE
    Line2D([], [], color=COL_R2, marker="o",
           linestyle="-", linewidth=2)         # custom line handle
]
fig.legend(handles, ["RMSE", "R²"],
           ncol=2, frameon=False, loc="upper center", bbox_to_anchor=(0.5, 1.05))

fig.suptitle("Offline Validation – single-stride metrics", y=1.06, fontsize=14)
fig.tight_layout()
plt.show()
