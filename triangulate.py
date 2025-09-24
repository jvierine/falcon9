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


def triangulate(azs,els,lats,lons,plot_line_of_sight=False):
    """
    tbd: support more than two cameras. 
    this is done as follows:
    1) for each unique pair of two cameras
         a) add two rows into m:  m.append(n.dot(r1,e0)-n.dot(r0,e0)) m.append(n.dot(r1,e1)-n.dot(r0,e1))
         b) add two rows into A: A.append([1,-n.dot(e1,e0)]) A.append([n.dot(e0,e1),-1])
    2) solve least squares for xhat
    3) compute pos0, pos1 for each camera
    4) average all pos0, pos1 to get final position estimate
    5) compute shortest intersection distances and report max distance as error estimate
    """
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
    return(pos_est)

def file_name_to_datetime(fname):
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
#    dt = TimeDelta(dt/2.0,format="sec")
    t0 = Time("%s-%s-%sT%s:%s:%s"%(year,month,day,hour,mins,sec), format='isot', scale='utc') #+ dt
    return(t0)

def get_video():
    # Path to your video
    video_path = "2025_02_19_03_44_00_000_012165.mp4"
    cap = cv2.VideoCapture(video_path)
    #   cap2 = cv2.VideoCapture(video_path2)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    dur=frame_count/25.0

    t0=file_name_to_datetime(video_path)
#    print(t0.unix)
    #print(t0)
    ams216=sio.loadmat("ams216.mat")
    az=180*ams216["az"]/n.pi
    ze=180*ams216["ze"]/n.pi
    el=90-ze
    long=ams216["long_lat"][0,0]
    lat=ams216["long_lat"][0,1]

    obs=EarthLocation(lon=long,height=0,lat=lat)
    dt = TimeDelta(dur/2.0,format="sec")
    aa_frame = AltAz(obstime=t0+dt, location=obs)
    return({"az":az,"el":el,"observer_tol":obs,"altaz_frame":aa_frame,"video_path":video_path,"lat":lat,"long":long,"t0":t0.unix,"t1":t0.unix+dur,"frame_count":frame_count,"cap":cap,"fps":25.0,"fragments":{}})

def xy_to_azel(az,el,x,y):
    return(az[int(x),int(y)],el[int(x),int(y)])


def get_video2():
    """
    tbd: make this more generic, so each video has the same type of metadata file. 
    we shouldn't need to make one of these for each video separately!
    """
    ams95=sio.loadmat("ams95.mat")
    # renamed to standard name...
    video_path = "2025_02_19_03_44_00_000_010954.mp4"
    cap = cv2.VideoCapture(video_path)
    # 
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    dur=frame_count/25.0

    t0=file_name_to_datetime(video_path)
    # lens model
    az=180*ams95["flipped_az"]/n.pi
    ze=180*ams95["flipped_ze"]/n.pi
    el=90-ze
    # location
    long=8.1651
    lat=53.1529
    # needed for star plotting
    obs=EarthLocation(lon=long,height=0,lat=lat)
    dt = TimeDelta(dur/2.0,format="sec")
    aa_frame = AltAz(obstime=t0+dt, location=obs)
    return({"az":az,"el":el,"observer_tol":obs,"altaz_frame":aa_frame,"video_path":video_path,"lat":lat,"long":long,"t0":t0.unix,"t1":t0.unix+dur,"frame_count":frame_count,"cap":cap,"fps":25.0,"fragments":{}})

v1=get_video()
v2=get_video2()

# all videos to process
videos = [v1,v2]

# for overplotting stars
bs=bright_stars.bright_stars()

# find smallest t0 in videos
t0_min=n.min([v["t0"] for v in videos])
# find largest t1 in videos
t1_max=n.max([v["t1"] for v in videos])

dt = 1.0 # seconds between frames to sample

# go through all frames with dt spacing
n_frames = int((t1_max-t0_min)/dt)
fragment_ids={}
for fi in range(n_frames):
    # time now in ms since t0_min
    tnow = (t0_min + dt*fi)
    # key for fragments dict
    tnow_key=int(tnow*1000.0)
    # go through all videos and ask user to mark fragments with keys 1-9
    # tbd: figure out how to index more framents if needed...  
    for v in videos:
        idx = int((tnow - v["t0"])*v["fps"])
        # if not in video range, skip
        if idx < 0 or idx >= v["frame_count"]:
            continue

        v["cap"].set(cv2.CAP_PROP_POS_FRAMES, idx)
        #    cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = v["cap"].read()
        #    cap2.set(cv2.CAP_PROP_POS_FRAMES, idx)
        #   ret2, frame2 = cap2.read()
        if ret:
            # Convert from BGR (OpenCV default) to RGB for plotting
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            fig, ax = plt.subplots(1,1)
            ax.imshow(frame_rgb)
            print("frame %d"%(idx))
            ax.set_title(f"zoom and position cursor, then press fragment id number (1-9) ")#%(lat,long))
            plot_stars=False

            if plot_stars: # dont plot stars
                for i in range(100):
                    d=bs.get_ra_dec_vmag(i)
                    c = SkyCoord(ra=d[0]*u.degree, dec=d[1]*u.degree, frame='icrs')

                    altaz=c.transform_to(v["aa_frame"])
                    star_az=float(altaz.az/u.deg)
                    star_el=float(altaz.alt/u.deg)
                    x,y=n.unravel_index(n.argmin((180*n.angle(n.exp(1j*n.pi*star_az/180)*n.exp(-1j*n.pi*v["az"]/180))/n.pi)**2 + (star_el-v["el"])**2),v["el"].shape)
                    if x > 0 and x < v["az"].shape[0] and y > 0 and y < v["az"].shape[1]:
                        ax.scatter(y,x,s=80, facecolors='none', edgecolors='w',alpha=0.2)

            def onkey(event):
                if event.key == "q":
                    return 
                if event.inaxes != ax:
                    return
                if event.inaxes == ax:
                    # read cam 1 position
                    x, y = int(event.xdata), int(event.ydata)
                    taz,tel=xy_to_azel(v["az"],v["el"],y,x)
                    if tnow_key not in v["fragments"].keys():
                        v["fragments"][tnow_key]={}
                    v["fragments"][tnow_key][int(event.key)]={"x":x,"y":y,"az":taz,"el":tel,"fragment_id":int(event.key)}
                    # add record of fragment id
                    if int(event.key) not in fragment_ids.keys():
                        fragment_ids[int(event.key)]=1
                    else:
                        fragment_ids[int(event.key)]+=1

                    print(v["fragments"][tnow_key][int(event.key)])##={"x":x,"y":y,"az":taz,"el":tel,"fragment_id":int(event.key)}

                    ax.plot(x, y, "x", color="red",alpha=0.5)#
                    ax.text(x,y,event.key,color="red",alpha=0.5)
                    
                    fig.canvas.draw()
        fig.canvas.mpl_connect("key_press_event", onkey)
        plt.show()
    if True:
        # triangulate all fragments seen by two cameras
        for fid in fragment_ids.keys():
            print("key %d"%(fid))
            lats=[]
            longs=[]
            azs=[]
            els=[]
            for v in videos:
                if (tnow_key in v["fragments"].keys()) and (fid in v["fragments"][tnow_key].keys()):
                    lats.append(v["lat"])
                    longs.append(v["long"])
                    azs.append(v["fragments"][tnow_key][fid]["az"])
                    els.append(v["fragments"][tnow_key][fid]["el"])    
            if len(lats) == 2: # tbd: support more than two cameras in triangulate()
                print("triangulating fragement id %d"%(fid))
                triangulate(azs,els,lats,longs)


