import cv2
import matplotlib.pyplot as plt
import numpy as n
import scipy.io as sio
import bright_stars


from astropy import units as u
from astropy.coordinates import Angle
from astropy.coordinates import AltAz
from astropy.coordinates import EarthLocation
from astropy.time import Time
from astropy.time import TimeDelta
from astropy.coordinates import SkyCoord

from astropy.coordinates import EarthLocation
import glob
import re
from astropy.time import Time
from astropy.time import TimeDelta

import numpy as np
from scipy.interpolate import RegularGridInterpolator
import jcoord

import matplotlib.pyplot as plt

def fanplot(azs,els,lats,lons):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    for ai in range(len(azs)):
        r0=jcoord.geodetic2ecef(lats[ai],lons[ai],0)
        e0=jcoord.azel_ecef(lats[ai],lons[ai],0,azs[ai],els[ai])
        r01=r0+e0*1000e3
        ax.scatter([r0[0]],[r0[1]],[r0[2]],color="red")
        ax.plot([r0[0],r01[0]],[r0[1],r01[1]],[r0[2],r01[2]])
    plt.show()

def triangulate(azs,els,lats,lons,plot_line_of_sight=False):
    # two camera triangulation
    r0=jcoord.geodetic2ecef(lats[0],lons[0],0)
    r1=jcoord.geodetic2ecef(lats[1],lons[1],0)

    e0=jcoord.azel_ecef(lats[0],lons[0],0,azs[0],els[0])
    e1=jcoord.azel_ecef(lats[1],lons[1],0,azs[1],els[1])

    if plot_line_of_sight:
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')

        r01=r0+e0*1300e3
        r11=r1+e1*1300e3
        ax.scatter([r0[0]],[r0[1]],[r0[2]],color="red")
        ax.scatter([r1[0]],[r1[1]],[r1[2]],color="blue")

        ax.plot([r0[0],r01[0]],[r0[1],r01[1]],[r0[2],r01[2]])
        ax.plot([r1[0],r11[0]],[r1[1],r11[1]],[r1[2],r11[2]])
        plt.show()

    m=n.array([n.dot(r1,e0)-n.dot(r0,e0),n.dot(r1,e1)-n.dot(r0,e1)])
    A=n.zeros([2,2])
    A[0,0]=1
    A[0,1]=-n.dot(e1,e0)
    A[1,0]=n.dot(e0,e1)
    A[1,1]=-1
    xhat=n.linalg.lstsq(A,m)[0]
    pos0=r0+xhat[0]*e0
    pos1=r1+xhat[1]*e1
    pos_est=0.5*(pos0+pos1)
    print(jcoord.ecef2geodetic(pos_est[0],pos_est[1],pos_est[2]))
#    print(jcoord.ecef2geodetic(pos1[0],pos1[1],pos1[2]))

    print("shortest intersection %1.2f m"%(n.linalg.norm(pos0-pos1)))

def file_name_to_datetime(fname,dt=60.0):
    """
    Read date from file name. assume integrated over dt seconds starting at file print time
    """
    res=re.search(".*(....)_(..)_(..)_(..)_(..)_(..)_(...)_(......).mp4.*",fname)
    # '2010-01-01T00:00:00'
    year=res.group(1)
    month=res.group(2)
    day=res.group(3)
    hour=res.group(4)
    mins=res.group(5)
    sec=res.group(6)    
    # add half a file length to be center of file
    dt = TimeDelta(dt/2.0,format="sec")
    t0 = Time("%s-%s-%sT%s:%s:%s"%(year,month,day,hour,mins,sec), format='isot', scale='utc') + dt
    return(t0)

def get_video():
    # Path to your video
    video_path = "2025_02_19_03_44_00_000_012165.mp4"
    t0=file_name_to_datetime(video_path,dt=60.0)
    #print(t0)
    ams216=sio.loadmat("ams216.mat")
    az=180*ams216["az"]/n.pi
    ze=180*ams216["ze"]/n.pi
    el=90-ze
    long=ams216["long_lat"][0,0]
    lat=ams216["long_lat"][0,1]
#    print(long)
 #   print(lat)
  #  print(ams216.keys())
    #azel_to_pixel = build_azel_to_pixel_map(az, ze)
    #print(azel_to_pixel(az[100,100]+0.01,90-ze[100,100]+0.01))
    #exit(0)
    obs=EarthLocation(lon=long,height=0,lat=lat)
    aa_frame = AltAz(obstime=t0, location=obs)
    return(az,el,obs,aa_frame,video_path,lat,long)

def xy_to_azel(az,el,x,y):
    return(az[int(x),int(y)],el[int(x),int(y)])


def get_video2():
    ams95=sio.loadmat("ams95.mat")
    #print(ams95.keys())
    # renamed to standard name...
    video_path = "2025_02_19_03_44_00_000_010954.mp4"#2025_02_19_03_44_00_000_010954__2025_02_19_03_45_00_000_010954.mp4"
    t0=file_name_to_datetime(video_path,dt=60.0)
#    print(t0)
#    ams216=sio.loadmat("ams216.mat")
    az=180*ams95["flipped_az"]/n.pi
    ze=180*ams95["flipped_ze"]/n.pi
    el=90-ze
#            "device_lng": "8.1651",
 #       "device_lat": "53.1529",
  #      "device_alt": "30",
    long=8.1651#ams216["long_lat"][0,0]
    lat=53.1529#ams216["long_lat"][0,1]
  #  print(long)
 #   print(lat)
#    print(ams216.keys())
    #azel_to_pixel = build_azel_to_pixel_map(az, ze)
    #print(azel_to_pixel(az[100,100]+0.01,90-ze[100,100]+0.01))
    #exit(0)
    obs=EarthLocation(lon=long,height=0,lat=lat)
    aa_frame = AltAz(obstime=t0, location=obs)
    return(az,el,obs,aa_frame,video_path,lat,long)

az,el,obs,aa_frame,video_path,lat,long=get_video()
az2,el2,obs2,aa_frame2,video_path2,lat2,long2=get_video2()

# Open video
cap = cv2.VideoCapture(video_path)
cap2 = cv2.VideoCapture(video_path2)

frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
print(f"Total frames: {frame_count}")

# Extract a few frames (e.g. first, middle, last)
frame_indices = n.arange(0,frame_count,25*10)#[0, frame_count // 2, frame_count - 1]
frames = []
bs=bright_stars.bright_stars()
coords={}
coords2={}

aaz=[]
ael=[]
alat=[]
alon=[]

for idx in frame_indices:
    cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
    ret, frame = cap.read()

    cap2.set(cv2.CAP_PROP_POS_FRAMES, idx)
    ret2, frame2 = cap2.read()

    if ret:
        # Convert from BGR (OpenCV default) to RGB for plotting
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_rgb2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2RGB)

        # Convert from BGR (OpenCV default) to RGB for plotting
    #   frame_rgb2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2RGB)
        fig, (ax,ax2) = plt.subplots(2,1)
        #plt.figure(figsize=(12, 6))
#        plt.subplot(211)
        #plt.subplot(1, len(frames), i+1)
        ax.imshow(frame_rgb)
        ax2.imshow(frame_rgb2)

        print("frame %d"%(idx))
        ax.set_title(f"zoom and position cursor, then press 6")#%(lat,long))
        ax2.set_title(f"zoom and position cursor, then press 6")#%(lat,long))

#        ax2.set_title(f"Frame {idx} lat %1.2f lon %1.2f"%(lat2,long2))

        #plt.axis("off")
        plot_stars=False
        if plot_stars: # dont plot stars
            for i in range(200):
                d=bs.get_ra_dec_vmag(i)
                c = SkyCoord(ra=d[0]*u.degree, dec=d[1]*u.degree, frame='icrs')

                altaz=c.transform_to(aa_frame)
                star_az=float(altaz.az/u.deg)
                star_el=float(altaz.alt/u.deg)
                x,y=n.unravel_index(n.argmin((180*n.angle(n.exp(1j*n.pi*star_az/180)*n.exp(-1j*n.pi*az/180))/n.pi)**2 + (star_el-el)**2),el.shape)
                if x > 0 and x < az.shape[0] and y > 0 and y < az.shape[1]:
                    ax.scatter(y,x,s=80, facecolors='none', edgecolors='w',alpha=0.2)

                altaz=c.transform_to(aa_frame2)
                star_az=float(altaz.az/u.deg)
                star_el=float(altaz.alt/u.deg)
                x,y=n.unravel_index(n.argmin((180*n.angle(n.exp(1j*n.pi*star_az/180)*n.exp(-1j*n.pi*az2/180))/n.pi)**2 + (star_el-el2)**2),el.shape)
                if x > 0 and x < az2.shape[0] and y > 0 and y < az2.shape[1]:
                    ax2.scatter(y,x,s=80, facecolors='none', edgecolors='w',alpha=0.2)

 #       plt.subplot(212)
        #plt.subplot(1, len(frames), i+1)
  #      plt.imshow(frame_rgb2)
   #     plt.title(f"Frame {idx}")
        #plt.tight_layout()
     #   plt.axis("off")
        mouse_pos=None
        def onmove(event):
            #nonlocal mouse_pos
            if event.inaxes != ax:
                return
            mouse_pos = (int(event.xdata), int(event.ydata))
        def onkey(event):
            if event.key != "6":
                return 
            if event.inaxes != ax and event.inaxes != ax2:
                return
            if event.inaxes == ax:
                # read cam 1 position
                x, y = int(event.xdata), int(event.ydata)
                taz,tel=xy_to_azel(az,el,y,x)
                coords["%d"%(idx)]=(idx,x, y, taz,tel)
                aaz.append(taz)
                ael.append(tel)
                alat.append(lat)
                alon.append(long)


                print(coords)
    #            xy_to_azel(az,el,x,y)
                ax.plot(x, y, "x", color="red",alpha=0.8)   # mark the click
                fig.canvas.draw()

            if event.inaxes == ax2:
                # read cam 2 position

                x, y = int(event.xdata), int(event.ydata)
                taz,tel=xy_to_azel(az2,el2,y,x)
                coords2["%d"%(idx)]=(idx,x, y, taz,tel)
                print(coords2)
                aaz.append(taz)
                ael.append(tel)
                alat.append(lat2)
                alon.append(long2)

    #            xy_to_azel(az,el,x,y)
                ax2.plot(x, y, "x", color="red",alpha=0.8)   # mark the click
                fig.canvas.draw()


            #if len(coords) >= n:
            #    plt.close(fig)

#        cid = fig.canvas.mpl_connect("button_press_event", onclick)
        fig.canvas.mpl_connect("motion_notify_event", onmove)
        fig.canvas.mpl_connect("key_press_event", onkey)
        plt.show()
  #      print(coords)
 #       print(coords2)
#        fanplot(aaz,ael,alat,alon)
        if True:
            if "%d"%(idx) in coords.keys() and "%d"%(idx) in coords2.keys():
                triangulate([coords["%d"%(idx)][3],coords2["%d"%(idx)][3]],[coords["%d"%(idx)][4],coords2["%d"%(idx)][4]],[lat,lat2],[long,long2])
                # triangulate


#        frames.append((idx, frame_rgb))

cap.release()

