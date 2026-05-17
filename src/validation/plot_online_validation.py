import numpy as np
import matplotlib.pyplot as plt

# ------------------------------------------------------------------
# 1)  Hard-code the numbers you reported
#     (add / edit if you have more trials later)
# ------------------------------------------------------------------
models        = ["ind_Ryan", "ind_Hridayam", "sync", "unsync"]

# ------------ RIGHT hip ------------------------------------------------
rmse_R_no = np.array([4.18, 3.16, 4.40, 2.84])   # %
rmse_R_as = np.array([3.09, 2.68, 4.64, 2.79])   # %

r2_R_no   = np.array([0.795, 0.873, 0.830, 0.713])
r2_R_as   = np.array([0.840, 0.865, 0.867, 0.775])

# ------------ LEFT hip -------------------------------------------------
rmse_L_no = np.array([2.88, 2.94, 3.77, 1.73])   # %
rmse_L_as = np.array([2.51, 3.16, 2.62, 3.41])   # %

r2_L_no   = np.array([0.822, 0.771, 0.869, 0.964])
r2_L_as   = np.array([0.813, 0.864, 0.713, 0.777])

rmse_no_assi = (rmse_R_no + rmse_L_no) / 2
rmse_assi = (rmse_R_as + rmse_L_as) / 2

r2_no_assi = (r2_R_no + r2_L_no) / 2
r2_assi = (r2_R_as + r2_L_as) / 2

# ------------------------------------------------------------------
# 2)  Bar-plot helper
# ------------------------------------------------------------------
def double_bar(ax, vals_a, vals_b, ylabel):
    x   = np.arange(len(models))
    w   = 0.35 # bar width  
    gap = 0.05 * w 
    COL_NO  = "#1962B0"   # deep blue
    COL_AS  = "#B22B34"   # coral-red 
    ax.bar(x - (w/2 + gap/2), vals_a, width=w, label="no assi", color=COL_NO)
    ax.bar(x + (w/2 + gap/2), vals_b, width=w, label="with assi", color=COL_AS)
    ax.set_xticks(x, models, rotation=15)
    ax.set_ylabel(ylabel)
    ax.tick_params(axis='both', direction='in', top=False, right=False)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.04), 
              frameon=False, ncol=2)

# ------------------------------------------------------------------
# 3)  Figure layout
# ------------------------------------------------------------------
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4), sharex=True)

double_bar(ax1, rmse_no_assi, rmse_assi, ylabel="RMSE [%]")

double_bar(ax2, r2_no_assi, r2_assi, ylabel="R²")

fig.tight_layout()
plt.show()
