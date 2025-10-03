import numpy as n
import h5py
import glob 
import re
import jcoord
import matplotlib.pyplot as plt
fl=glob.glob("fragments/*.h5")
fl.sort()

fragment_ids=[]

for f in fl:
    m=re.match(r"fragments/(\d+)_.*\.h5",f)
    if m:
        fid=int(m.group(1))
        if fid not in fragment_ids:
            fragment_ids.append(fid)

print(fragment_ids)

fragment_pos=[]
fragment_pos_err=[]

fragment_geo_pos=[]
fragment_times=[]
for fid in fragment_ids:
    geo_pos=[]
    pos=[]
    pos_err=[]

    times=[]
    fl=glob.glob("fragments/%d_*.h5"%(fid))
    fl.sort()
    for f in fl:
        print(f)
        h=h5py.File(f,"r")
        tpos=h["pos_est"][()]
        tposerr=h["pos_err"][()]

        pos.append(tpos)
        pos_err.append(tposerr)
        times.append(h["time"][()])
        llh=jcoord.ecef2geodetic(tpos[0],tpos[1],tpos[2])
        geo_pos.append(llh)
        h.close()
    fragment_pos.append(n.array(pos))
    fragment_geo_pos.append(n.array(geo_pos))
    fragment_pos_err.append(n.array(pos_err))

    fragment_times.append(n.array(times))#,"s"))
#fig,(ax1,ax2)=plt.subplots(2,1)
plt.subplot(111)
for fp in range(len(fragment_pos)):
    # convert times to datetime64
    datetime=n.array(fragment_times[fp],dtype="datetime64[s]")
    plt.errorbar(
        datetime,
        fragment_geo_pos[fp][:,2]/1e3,
        yerr=2*fragment_pos_err[fp]/1e3,
        fmt='o',          # circle markers
        linestyle='none', # no line connecting them
        label="%d" % (fragment_ids[fp])
    )
#    plt.errorbar(datetime,fragment_geo_pos[fp][:,2]/1e3,yerr=2*fragment_pos_err[fp]/1e3,label="%d"%(fragment_ids[fp]))
plt.xlabel("unix time (s)")
plt.ylabel("height (m)")
plt.title("Fragment heights over time")
plt.legend()
plt.show()
# plot lat vs long with cartopy on map of europe
import cartopy.crs as ccrs
import cartopy.feature as cfeature  
ax = plt.axes(projection=ccrs.PlateCarree())
ax.set_extent([-20, 40, 30, 70], crs=ccrs.PlateCarree())
ax.add_feature(cfeature.LAND)
ax.add_feature(cfeature.OCEAN)
ax.add_feature(cfeature.COASTLINE)
ax.add_feature(cfeature.BORDERS, linestyle=':') 
ax.add_feature(cfeature.LAKES, alpha=0.5)
ax.add_feature(cfeature.RIVERS)
for fp in range(len(fragment_pos)):
    plt.plot(fragment_geo_pos[fp][:,1],fragment_geo_pos[fp][:,0],".",label="%d"%(fragment_ids[fp]))
plt.xlabel("Longitude")
plt.ylabel("Latitude")
plt.title("Fragment ground tracks")
plt.legend()        
# add gridlines
gl = ax.gridlines(draw_labels=True)
plt.show()

