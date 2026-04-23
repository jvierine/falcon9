import numpy as np
import matplotlib.pyplot as plt
import plot_fragments as plf
from pymsis import msis
import numpy as n
import jcoord
# Constants
mu = 3.986004418e14   # m^3/s^2
Re = 6371e3           # m
Cd = 2.2              # assumed drag coefficient
min_dur = 20.0        # max segment duration [s]
min_pts = 5           # minimum points per segment


def split_into_time_segments(tv, max_dur):
    """
    Split a time array into consecutive index segments whose duration
    is at most max_dur seconds.
    Returns a list of index arrays.
    """
    segs = []
    n = len(tv)
    i0 = 0

    while i0 < n:
        i1 = i0

        while i1 + 1 < n and (tv[i1 + 1] - tv[i0]) <= max_dur:
            i1 += 1

        seg = np.arange(i0, i1 + 1)
        segs.append(seg)

        if i1 == i0:
            i0 += 1
        else:
            i0 = i1 + 1

    return segs


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


hgt_count, hgt_count_all, fragment_ids, fragment_pos, fragment_pos_err, fragment_geo_pos, fragment_times = plf.get_fragments()

all_results = []

for i in range(len(fragment_ids)):
    tv_full = np.asarray(fragment_times[i], dtype=float)    # unix seconds
    llh_full = np.asarray(fragment_geo_pos[i], dtype=float) # [lat, lon, alt_m]

    lat_full = llh_full[:, 0]
    lon_full = llh_full[:, 1]
    alt_full = llh_full[:, 2]
    ecefs=jcoord.geodetic2ecef(llh_full[:,0],llh_full[:,1],llh_full[:,2])

    a_full = Re + alt_full   # circular orbit assumption

    rho_full = get_msis_density(tv_full, lat_full, lon_full, alt_full)

    tv = tv_full
    lat = lat_full
    lon = lon_full
    alt = alt_full
    a = a_full
    rho_a = rho_full

    step=20

    t0=n.min(tv)
    t1=n.max(tv)
    dur=n.max(tv)-n.min(tv)
    segments=[]
    if dur > step:
        n_step=int(dur/step)
        for si in range(n_step):
            idx=n.where( (tv > (t0+si*step)) & (tv < (t0+si*step+step)) )[0]
            if len(idx)>10:
                segments.append(idx)

#    segments = split_into_time_segments(tv, min_dur)

    frag_results = []

    for k, idx in enumerate(segments):
        if len(idx) < min_pts:
            continue

        tseg = tv[idx]
        aseg = a[idx]
        hseg = alt[idx]
        rhoseg = rho_a[idx]

        seg_dur = tseg[-1] - tseg[0]
        if seg_dur <= 0:
            continue

        # Linear fit to a(t): slope is da/dt [m/s]
        p = np.polyfit(tseg-n.mean(tseg), aseg, 1)
        da_dt = p[0]
        print(p)
        print(da_dt)
        px = np.polyfit(tseg-n.mean(tseg), ecefs[0,idx], 1)
        py = np.polyfit(tseg-n.mean(tseg), ecefs[1,idx], 1)
        pz = np.polyfit(tseg-n.mean(tseg), ecefs[2,idx], 1)

        vx=px[0]
        vy=py[0]
        vz=pz[0]
        if False:
            # ---- 2x2 diagnostic plot ----
            fig, ax = plt.subplots(2, 2, figsize=(10, 8), sharex=True)

            # Semimajor axis
            ax[0,0].plot(tseg, aseg, ".", label="a")
            ax[0,0].plot(tseg, np.polyval(p, tseg), "-", label="fit")
            ax[0,0].set_ylabel("a [m]")
            ax[0,0].set_title(f"a(t), da/dt={da_dt:.3e}")
            ax[0,0].legend()

            # ECEF X
            ax[0,1].plot(tseg, ecefs[0, idx], ".", label="x")
            ax[0,1].plot(tseg, np.polyval(px, tseg), "-")
            ax[0,1].set_ylabel("x [m]")
            ax[0,1].set_title(f"vx={vx:.3e}")

            # ECEF Y
            ax[1,0].plot(tseg, ecefs[1, idx], ".", label="y")
            ax[1,0].plot(tseg, np.polyval(py, tseg), "-")
            ax[1,0].set_ylabel("y [m]")
            ax[1,0].set_xlabel("Time [unix s]")
            ax[1,0].set_title(f"vy={vy:.3e}")

            # ECEF Z
            ax[1,1].plot(tseg, ecefs[2, idx], ".", label="z")
            ax[1,1].plot(tseg, np.polyval(pz, tseg), "-")
            ax[1,1].set_ylabel("z [m]")
            ax[1,1].set_xlabel("Time [unix s]")
            ax[1,1].set_title(f"vz={vz:.3e}")

            plt.suptitle(f"Segment fit diagnostics")
            plt.tight_layout()
            plt.show()

        vg=n.sqrt(vx**2+vy**2+vz**2)

        CdA_over_m=-mu*(da_dt/n.mean(rhoseg)/(vg**3))/2/n.mean(aseg)**2
        print(CdA_over_m)
        # Circular-orbit drag model:
        # da/dt = -rho * (Cd*A/m) * sqrt(mu*a)
        #CdA_over_m = -da_dt / (rho_bar * np.sqrt(mu * a_bar))
        A_over_m = CdA_over_m / Cd

        result = {
            "fragment_id": fragment_ids[i],
            "segment_id": k,
            "t0": tseg[0],
            "t1": tseg[-1],
            "duration_s": seg_dur,
            "npts": len(idx),
            "mean_alt_km": np.mean(hseg) / 1e3,
            "mean_rho": n.mean(rhoseg),
            "da_dt_mps": da_dt,
            "CdA_over_m": CdA_over_m,
            "A_over_m": A_over_m,
            "fit": p,
            "tv": tseg,
            "alt_m": hseg,
            "a_m": aseg,
            "vg":vg,
            "rho_a": rhoseg,
        }

        frag_results.append(result)
        all_results.append(result)

        print(f"fragment {fragment_ids[i]} segment {k}")
        print(f"  t0           = {tseg[0]:.3f}")
        print(f"  t1           = {tseg[-1]:.3f}")
        print(f"  duration     = {seg_dur:.2f} s")
        print(f"  npts         = {len(idx)}")
        print(f"  mean alt     = {np.mean(hseg)/1e3:.3f} km")
        print(f"  mean rho_a   = {n.mean(rhoseg):.3e} kg/m^3")
        print(f"  da/dt        = {da_dt:.3e} m/s")
        print(f"  Cd*A/m       = {CdA_over_m:.3e} m^2/kg")
        print(f"  A/m          = {A_over_m:.3e} m^2/kg  (Cd={Cd})")
        print(f"  vg          = {vg:.3e} m/s")

#        plt.scatter(tseg,aseg,c=n.log10(n.repeat(CdA_over_m,len(tseg))),vmin=-6,vmax=-3)
        plt.scatter(tseg,aseg,c=n.repeat(vg/1e3,len(tseg)),vmin=3,vmax=7.6,s=10)

plt.colorbar()    
plt.show()


exit(0)
import numpy as np
import matplotlib.pyplot as plt
import os
import plot_fragments as plf
import numpy as n
import matplotlib.pyplot as plt
from pymsis import msis

# Constants
mu = 3.986004418e14       # m^3/s^2
Re = 6371e3               # m
Cd = 2.2                  # assumed drag coefficient

hgt_count, hgt_count_all, fragment_ids, fragment_pos, fragment_pos_err, fragment_geo_pos, fragment_times = plf.get_fragments()

min_dur = 20  # s

for i in range(len(fragment_ids)):
    tv = np.asarray(fragment_times[i])         # unix seconds
    llh = np.asarray(fragment_geo_pos[i])      # [lat, lon, h_m] for this fragment

    # Skip short tracks
    if len(tv) < 5 or (tv[-1] - tv[0]) < min_dur:
        continue

    lat_deg = llh[:, 0]
    lon_deg = llh[:, 1]
    alt_m   = llh[:, 2]

    # Circular orbit: a = Re + h
    a = Re + alt_m

    # Convert unix time -> numpy datetime64[s]
    msis_dates = tv.astype("datetime64[s]")

    rho_a = []
    for j in range(len(tv)):
        # msis.run usually expects arrays, so pass 1-element arrays
        data = msis.run(
            np.array([msis_dates[j]]),
            np.array([lat_deg[j]]),
            np.array([lon_deg[j]]),
            np.array([alt_m[j] / 1e3]),   # km
            geomagnetic_activity=-1
        )

        # Adjust this index if your msis package returns a different layout
        rho_a.append(data[0][0])

    rho_a = np.asarray(rho_a)

    # Remove invalid values
    good = np.isfinite(tv) & np.isfinite(a) & np.isfinite(rho_a) & (rho_a > 0)
    tv = tv[good]
    a = a[good]
    rho_a = rho_a[good]

    if len(tv) < 5:
        continue

    # Estimate da/dt from linear fit of a(t)
    # slope in m/s
    p = np.polyfit(tv, a, 1)
    da_dt = p[0]

    # Use representative values over the track
    a_bar = np.mean(a)
    rho_bar = np.mean(rho_a)

    # Fit drag constant K = Cd * A/m from circular-orbit formula:
    # da/dt = -rho * K * sqrt(mu * a)
    K = -da_dt / (rho_bar * np.sqrt(mu * a_bar))

    # Area-to-mass ratio
    A_over_m = K / Cd

    print(f"fragment {fragment_ids[i]}")
    print(f"  duration      = {tv[-1] - tv[0]:.1f} s")
    print(f"  mean altitude = {np.mean(alt_m)/1e3:.2f} km")
    print(f"  mean rho_a    = {rho_bar:.3e} kg/m^3")
    print(f"  da/dt         = {da_dt:.3e} m/s")
    print(f"  Cd*A/m        = {K:.3e} m^2/kg")
    print(f"  A/m           = {A_over_m:.3e} m^2/kg   (Cd={Cd})")

    # Diagnostic plots
    fig, ax = plt.subplots(3, 1, figsize=(8, 8), sharex=True)

    ax[0].plot(tv, alt_m / 1e3, ".-")
    ax[0].set_ylabel("Altitude [km]")

    ax[1].plot(tv, rho_a, ".-")
    ax[1].set_ylabel(r"$\rho_a$ [kg/m$^3$]")

    ax[2].plot(tv, a, ".-", label="a")
    ax[2].plot(tv, np.polyval(p, tv), "-", label="linear fit")
    ax[2].set_ylabel("a [m]")
    ax[2].set_xlabel("Time [unix s]")
    ax[2].legend()

    plt.tight_layout()
    plt.show()

exit(0)    
import os
import plot_fragments as plf
import numpy as n
import matplotlib.pyplot as plt
from pymsis import msis

msis_date0=n.datetime64("2025-02-19T03:30")

msis_dates = n.array([msis_date0])

hgt_count, hgt_count_all, fragment_ids, fragment_pos, fragment_pos_err, fragment_geo_pos, fragment_times = plf.get_fragments()
rlat, rlon, ralt, rsnr, rtime, bragg_enu, rdop = plf.get_radar_detections() 

min_dur=60
for i in range(len(fragment_ids)):
    tv=fragment_times[i]

    llh=fragment_geo_pos[i]
    rho_a=[]
    for j in range(len(tv)):
        data=msis.run(msis_dates, -5, 40.0, llh[i,2]/1e3, geomagnetic_activity=-1)
        rho_a.append(data[0][0])
    rho_a=n.array(rho_a)
    

    plt.plot(tv,llh[:,2],".")
    plt.xlabel("Time (unix)")
    plt.ylabel("Height")
    plt.show()