import numpy as n
import numpy as np
import matplotlib.pyplot as plt
from pymsis import msis
import fit_ballistic3 as fb3
def get_msis_density(times_unix, lat_deg, lon_deg, alt_m):
    """
    Evaluate MSIS total mass density for each sample.
    Returns rho_a in kg/m^3.
    """
    times_dt64 = times_unix.astype("datetime64[s]")

    rho_a = np.full(len(times_unix), np.nan)

    for j in range(len(times_unix)):
        data = msis.run(
            np.array([times_dt64[j]]),
            np.array([lat_deg[j]]),
            np.array([lon_deg[j]]),
            np.array([alt_m[j] / 1e3]),   # km
            geomagnetic_activity=-1
        )

        arr = np.asarray(data)

        # pymsis total mass density is typically the first species/output entry.
        # Squeeze to make indexing robust across wrapper return shapes.
        arr = np.squeeze(arr)
        rho_a[j] = arr[0] if np.ndim(arr) > 0 else arr

    return rho_a

import plot_fragments as pf
# $F_1$ are $F_3$, $F_5$, $F_9$, $F_n$, $F_o$, $F_p$, $F_r$, $F_s$, $F_w$, $F_v$, $F_x$, and $F_z$.
# The child fragments of $F_2$ are $F_4$, $F_7$, $F_8$, $F_a$, $F_c$, $F_h$, $F_g$, $F_i$, $F_m$, $F_j$, $F_{\ell}$, $F_k$, $F_t$, $F_u$, $F_e$, and $F_d$. The
#f1_children = ["1","3","5","9","n","o","p","r","s","w","v","x","z"]
f1_children = ["1"]#,"3","5","9","n","o","p","r","s","w","v","x","z"]
f2_children = ["2","4","7","8","a","c","h","g","i","m","j","l","k","t","u","e","d"]

hgt_count, hgt_count_all, fragment_ids, fragment_pos, fragment_pos_err, fragment_geo_pos, fragment_times = pf.get_fragments()
print(fragment_pos[0].shape)
#exit(0)
f1_frags=[]
f2_frags=[]

for i in range(len(fragment_ids)):
    if fragment_ids[i] in f1_children:
        f1_frags.append(i)
    if fragment_ids[i] in f2_children:
        f2_frags.append(i)

if True:
    for i in range(len(f1_frags)):
        plt.plot(fragment_times[f1_frags[i]],fragment_geo_pos[f1_frags[i]][:,2],".",color="C%d"%(i),label=f1_children[i])
    #for i in range(len(f2_frags)):
    #    plt.plot(fragment_times[f2_frags[i]],fragment_geo_pos[f2_frags[i]][:,2],".",color="C1")
    plt.legend()
    plt.show()

import plot_fragments as pf

#f2_children = ["2","4","7","8","a","c","h","g","i","m","j","l","k","t","u","e","d"]

#hgt_count, hgt_count_all, fragment_ids, fragment_pos, fragment_pos_err, fragment_geo_pos, fragment_times = pf.get_fragments()

result = fb3.fit_shared_ballistic_coefficient(
    fragment_ids=fragment_ids,
    fragment_pos=fragment_pos,      # ECEF, shape (n_i, 3) per fragment
    fragment_times=fragment_times,  # unix seconds per fragment
    fit_ids=f1_children,
    B0_guess=5e-3,
    fr1_guess=0.5,
    fr2_guess=0.5,
    verbose=2
)

