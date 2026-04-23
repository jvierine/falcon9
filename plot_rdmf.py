import h5py
import numpy as n
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import plot_deco
import plot_fragments as pf

def read_kb():
    h = h5py.File("range_doppler_mf_kb.h5", "r")
    D = h["D"][()]
    N = h["N"][()]
    P = h["P"][()]
    t0 = h["t0"][()]
    h.close()

    tvec = t0/100e3 + n.arange(D.shape[0]) * 1000 * 10e-6
    tvec = n.array(tvec * 1e9, dtype="datetime64[ns]")
    rvec = n.arange(D.shape[1]) * 3.0

    # convert numpy datetime64 to Python datetime for matplotlib
    #t_dt = tvec.astype('datetime64[ns]').astype('O')
    return(P,N,D,tvec,rvec)

import plot_deco
if __name__ == "__main__":

#def get_fragment_info(tx="kborn",rx="hagenow"):

    #station_coords = simone_conf.station_coords
    hgt_count, hgt_count_all, fragment_ids, fragment_pos, fragment_pos_err, fragment_geo_pos, fragment_times = pf.get_fragments()

    fragment_aspects, fragment_dops, fragment_range, fragment_dt = plot_deco.get_fragment_info(tx="kborn",rx="hagenow")

    P,N,D,tvec,rvec=read_kb()

    SN=P/N

    nfloor=n.median(N)
    # poor man compressive sensing
    SNR_est = P/nfloor
    SNR_est[SN<5]=1.2
    plt.figure()
    im = plt.pcolormesh(tvec, rvec, 10.0 * n.log10((SN).T))
    cb = plt.colorbar(im)
    cb.set_label("SNR (dB)")
    plt.xlabel("Time (UTC)")
    plt.ylabel("Bi-static group propagation range (km)")
    ax = plt.gca()
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M:%S'))
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

    if True:
        # First plot: SNR in dB with time & range labels, rotated time ticks HH:MM:SS
        plt.figure()
        im = plt.pcolormesh(tvec, rvec, 10.0 * n.log10((SNR_est).T))
        cb = plt.colorbar(im)
        cb.set_label("SNR (dB)")
        plt.xlabel("Time (UTC)")
        plt.ylabel("Bi-static group propagation range (km)")
        ax = plt.gca()
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M:%S'))
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.show()

    R=n.copy(SN)
    for i in range(SN.shape[0]):
        R[i,:]=rvec/2.0
    rcs=plot_deco.sn_plus_n_over_n_to_rcs(SNR_est+0.2,
                                    R*1e3,
                                    R*1e3,
                                    frequency_hz=32.55e6,
                                    P_tx=500,
                                    G_tx=1,
                                    G_rx=1,
                                    B_rx=100.0,
                                    T_noise=6000)
    rcs[SN<5]=1.0

    plt.figure()
    im = plt.pcolormesh(tvec, rvec, 10.0 * n.log10((rcs).T),cmap="turbo",vmin=1,vmax=60)
    cb = plt.colorbar(im)
    cb.set_label("RCS (dBsm)")
    plt.xlabel("Time (UTC)")
    plt.ylabel("Bi-static group propagation range (km)")
    ax = plt.gca()
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M:%S'))
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

    import numpy as np
    import matplotlib as mpl
    import matplotlib.pyplot as plt
    import matplotlib.dates as mdates

    # Publication-style defaults
    if False:
        mpl.rcParams.update({
            "figure.dpi": 150,
            "savefig.dpi": 600,
            "font.size": 11,
            "axes.labelsize": 12,
            "axes.titlesize": 12,
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
            "legend.fontsize": 10,
            "axes.linewidth": 0.8,
            "xtick.major.width": 0.8,
            "ytick.major.width": 0.8,
            "xtick.minor.width": 0.6,
            "ytick.minor.width": 0.6,
            "xtick.major.size": 4,
            "ytick.major.size": 4,
            "xtick.minor.size": 2,
            "ytick.minor.size": 2,
        })

                # Publication-friendly rc settings
    plt.rcParams.update({
        "figure.dpi": 150,
        "savefig.dpi": 300,
        "font.size": 8,
        "axes.labelsize": 9,
        "axes.titlesize": 9,
        "xtick.labelsize": 7,
        "ytick.labelsize": 7,
        "axes.linewidth": 0.8,
        "xtick.direction": "out",
        "ytick.direction": "out",
    })


    D_plot = D.copy()
    D_plot[SN < 4.0] = np.nan

    cmap = plt.cm.seismic.copy()
    cmap.set_bad("0.5")  # gray for NaN

    fig, axs= plt.subplots(2,1,figsize=(3.5, 4.4), constrained_layout=True)

    ax=axs[0]
    ax.set_title("Kühlungsborn-Hagenow")
    pcm = ax.pcolormesh(
        tvec,
        rvec,
        D_plot.T,
        cmap=cmap,
        vmin=-100,
        vmax=100,
        shading="auto",
        rasterized=True,   # good for vector export if data are dense
    )

    cb = fig.colorbar(pcm, ax=ax, pad=0.02)
    cb.set_label("Doppler shift (Hz)")

    ax.set_xlabel("Time (UTC)")
    ax.set_ylabel("Range (km)")

    ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M:%S"))
    fig.autofmt_xdate(rotation=45)

    # Optional cleanup
    ax.tick_params(direction="out")
    cb.outline.set_linewidth(0.8)

    ax=axs[1]
    for i in range(len(fragment_ids)):
        #print(len(fragment_dt[i]))
        #print(len(fragment_dops[i]))
        print(fragment_dt[i])
        print(fragment_dops[i])
        if fragment_ids[i]=='1' or fragment_ids[i]=='2':    
            ax.plot(fragment_dt[i],fragment_dops[i],".",label=fragment_ids[i])
    ax.set_xlabel("Time (UTC)")
    ax.set_ylabel("Doppler (Hz)")
    ax.legend()
    import datetime as dt

    t0 = dt.datetime.fromisoformat("2025-02-19T03:45:40")
    t1 = dt.datetime.fromisoformat("2025-02-19T03:46:30")

  #  axs[0].set_xlim(t0, t1)
 #   axs[1].set_xlim(t0, t1)
#    axs[0].set_ylim([200,600])
   # fig.savefig("doppler_shift_figure.pdf",format="pdf",bbox_inches="tight",transparent=True)
    plt.show()