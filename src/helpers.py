import os

def save_pic(plt, name):
  plt.savefig(os.path.join("figures", name), dpi=150, bbox_inches="tight")
