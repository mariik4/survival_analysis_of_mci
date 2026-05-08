import os
from IPython.display import Image

def save_pic(plt, name):
  plt.savefig(os.path.join("figures", name), dpi=150, bbox_inches="tight")

def show_pic(name):
    return Image(filename=os.path.join("figures", name))


def bin_animals(x):
    if x < 15: return 'non-normal'
    else:        return 'normal'

def bin_hrate(x):
    if x >= 60 and x <= 100:  return 'normal'
    else:                     return 'tachycardia_bradycardia'

def define_strata(df):
    df['HRATE_strata']   = df['HRATE'].apply(bin_hrate)
    df['ANIMALS_strata'] = df['ANIMALS'].apply(bin_animals)

    return df.drop(columns=['HRATE', 'ANIMALS'])
