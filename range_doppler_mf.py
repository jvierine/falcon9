import h5py
import numpy as n
import matplotlib.pyplot as plt
import glob
import stuffr 

def read_data_and_codes():
    fl0=glob.glob("simone/raw/hagenow/20250219/ch000/*.h5")
    fl0.sort()

    fl1=glob.glob("simone/raw/hagenow/20250219/ch001/*.h5")
    fl1.sort()

    codes_f_kb=glob.glob("simone/raw/hagenow/codes/Kborn/*.h5")
    codes_f_jr=glob.glob("simone/raw/hagenow/codes/Jruh/*.h5")

    z0=n.array([],dtype="complex64")
    z1=n.array([],dtype="complex64")

    t0=None
    for f in fl0:
        h=h5py.File(f,"r")
        z0=n.concatenate((z0,h["rf_data"][()][:,0]))
        #print(stuffr.unix2datestr(h["rf_data_index"][0,0]/1e5))
        if t0 == None:
            t0=h["rf_data_index"][0,0]

        h.close()

    for f in fl1:
        h=h5py.File(f,"r")
        z1=n.concatenate((z1,h["rf_data"][()][:,0]))
        #print(stuffr.unix2datestr(h["rf_data_index"][0,0]/1e5))
        if t0 == None:
            t0=h["rf_data_index"][0,0]

        h.close()

    codes_kb=[]
    codes_jr=[]
    for f in codes_f_kb:
        h=h5py.File(f,"r")
        code=n.array(h["array"][()],dtype="complex64")
        codes_kb.append(code)
    for f in codes_f_jr:
        h=h5py.File(f,"r")
        code=n.array(h["array"][()],dtype="complex64")
        codes_jr.append(code)
        

    return([z0,z1],codes_kb,codes_jr,t0)

def range_dop_mf(zs,codes,n_r=999,ofname="range_doppler_mf.h5",t0=0,fftlen=8192,n_avg=10):
    codelen=len(codes[0])
    step=codelen
    n_codes=len(codes)
    n_z=len(z)
    n_steps=int((len(zs[0])-codelen)/step)
    P=n.zeros([n_steps,n_r],dtype=n.float32)
    D=n.zeros([n_steps,n_r],dtype=n.float32)
    N=n.zeros([n_steps,n_r],dtype=n.float32)
    fvec=n.fft.fftshift(n.fft.fftfreq(fftlen,d=1/100e3))
    prow=n.zeros([n_avg,n_r,fftlen],dtype=n.float32)
    prow[:,:,:]=n.nan
    for t in range(n_steps):
        print("%d/%d"%(t,n_steps))
        if t%n_avg==0:
            prow[t%n_avg,:,:]=0.0
        for r in range(n_r):
            for k in range(n_z):
                for i in range(n_codes):
                    code=codes[i]
                    S=n.fft.fftshift(n.fft.fft(zs[k][(t*step+r):(t*step+codelen+r)]*code,fftlen))
                    prow[t%n_avg,r,:]+=(S*n.conj(S)).real
            prow_this=n.nanmean(prow[:,r,:],axis=0)
            mi=n.argmax(prow_this)
            P[t,r]=prow_this[mi]
            D[t,r]=fvec[mi]
            N[t,r]=n.median(prow_this)
            if P[t,r]/N[t,r] > 10:
                print("peak at t=%d, r=%d, doppler=%f, SNR=%f"%(t,r,D[t,r],P[t,r]/N[t,r]))
        if t%200 == 0:
            ho=h5py.File(ofname,"w")
            ho["P"]=P[0:t,:]
            ho["D"]=D[0:t,:]
            ho["N"]=N[0:t,:]
            ho["t0"]=t0
            ho["n_avg"]=n_avg
            ho.close()

    ho=h5py.File(ofname,"w")
    ho["P"]=P
    ho["D"]=D
    ho["N"]=N
    ho["t0"]=t0
    ho.close()
            


    
    
z,ckb,cjr,t0=read_data_and_codes()
range_dop_mf(z,ckb,t0=t0,ofname="range_doppler_mf_kb.h5")
range_dop_mf(z,cjr,t0=t0,ofname="range_doppler_mf_jr.h5")