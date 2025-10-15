import cv2
import matplotlib.pyplot as plt
import numpy as n
import scipy.io as sio
import bright_stars

import numpy as np
from scipy.spatial import KDTree

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
import h5py


import numpy as np
from astropy.coordinates import EarthLocation, AltAz, ITRS, CartesianRepresentation
from astropy.time import Time
import astropy.units as u

# Get default color cycle
prop_cycle = plt.rcParams['axes.prop_cycle']
colors = prop_cycle.by_key()['color']


def wrap_azimuth(az):
    """
    Wrap azimuth value(s) into [0, 360) degrees.
    Works with scalars or numpy arrays.
    """
    return np.mod(az, 360)

def create_kdtree(az,el):
    """
    create k-d tree from (az, el) pairs for fast nearest-neighbor lookup
    1) flatten az,el arrays
    2) build k-d tree from (az, el) pairs
    3) also return flattened x,y arrays for pixel lookup
    4) also return flattened az,el arrays for az,el lookup
    5) return dict with tree, x,y, az,el
    """
    az=wrap_azimuth(az)
    # Build k-d tree from (az, el) pairs
    azf=az.flatten()
    elf=el.flatten()
    points = np.column_stack((azf, elf))
    x=n.arange(az.shape[0])
    y=n.arange(az.shape[1])
    # why indexing="ij"???
    xg,yg=n.meshgrid(x,y ,indexing="ij")
    xg=xg.flatten()
    yg=yg.flatten()
    tree = KDTree(points)
    return({"tree":tree,"x":xg,"y":yg,"az":azf,"el":elf})

def interpolate_pixel(az_query, el_query, kd, az,el,k=4, p=2):
    """
    Map an arbitrary (az, el) to pixel (x, y) using inverse-distance weighting.
    use kd tree to get nlog n search time
    
    Parameters:
        az_query, el_query: query azimuth and elevation
        k: number of nearest neighbors to use
        p: distance power parameter (p=2 means inverse-square weighting)
    """
    tree=kd["tree"]
    x=kd["x"]
    y=kd["y"]
    az_query=wrap_azimuth(az_query)
    dist, idx = tree.query([az_query, el_query], k=k)

    # Handle scalar vs. array case for k=1
    if k == 1:
        return x[idx], y[idx]

    # Compute weights (inverse distance)
    weights = 1 / (dist**p + 1e-12)  # small epsilon avoids divide-by-zero
    weights /= np.sum(weights)
    
    # Interpolate pixel coordinates
    x_interp = np.sum(x[idx] * weights)
    y_interp = np.sum(y[idx] * weights)
    
    return x_interp, y_interp



def line_of_sight(az,el,video1,video2,dl=5e3,L=2000e3):
    """ get pixels x,y corresponding to line of sight vector from camera1 with az,el
    as seen from camera2. 
    """
    n_segments=int(L/dl)
    pos1=jcoord.geodetic2ecef(video1["lat"],video1["long"],video1["h"])
    e0=jcoord.azel_ecef(video1["lat"],video1["long"],video1["h"],az,el)
    # az and el for camera2
    v2x=[]
    v2y=[]
    for i in range(n_segments):
        pos = pos1 + e0*i*dl
        target_llh=jcoord.ecef2geodetic(pos[0], pos[1], pos[2])
#        print(target_llh)
        aer=jcoord.geodetic_to_az_el_r(video2["lat"],video2["long"],video2["h"],target_llh[0],target_llh[1],target_llh[2])
 #       print(aer)
        x,y=interpolate_pixel(aer[0], aer[1], video2["kd"], video2["az"], video2["el"],k=4, p=2)
        v2x.append(x)
        v2y.append(y)

    return(n.array(v2x),n.array(v2y))

    #pos1=jcoord.az(video["lat"],video1["long"],video1["h"])


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
    # TBD: add height!!!
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

    print("shortest intersection %1.2f m distances %1.2f %1.2f km"%(n.linalg.norm(pos0-pos1),xhat[0]/1000.0,xhat[1]/1000.0))
    return(pos_est,n.linalg.norm(pos0-pos1))

def file_name_to_datetime(fname):
    """
    Read date from file name. assume integrated over dt seconds starting at file print time
    """
    res=re.search(".*(....)_(..)_(..)_(..)_(..)_(..)_(...)_(......).*.mp4.*",fname)
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

def get_video(video_path = "2025_02_19_03_44_00_000_012165.mp4",dt=0,calfile="ams216.mat",camera_id="2165",h=0,flip=False):
    # Path to your video
    
    cap = cv2.VideoCapture(video_path)
    #   cap2 = cv2.VideoCapture(video_path2)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    dur=frame_count/25.0

    t0=file_name_to_datetime(video_path)
#    print(t0.unix)
    #print(t0)
    ams216=sio.loadmat(calfile)
    az=ams216["az"]
    el=n.pi/2.0-ams216["ze"]


    if False:
        plt.imshow(180*wrap_azimuth(az)/n.pi)
        plt.colorbar()
        plt.show()

        plt.imshow(180*el/n.pi)
        plt.colorbar()
        plt.show()


    if flip:
        # Björn's calibration has a degneracy, which sometimes allows flipped line of sight!!!
        no=n.cos( -az )*n.cos(el)
        ea=n.sin( az )*n.cos(el)
        up=n.sin(el)
    
        no=-no
        ea=-ea
        up=-up

        el=n.arcsin(up)
        az=-n.arccos(no/n.cos(el))


    az=180*az/n.pi
    el=180*el/n.pi

    if False:
        plt.imshow(wrap_azimuth(az))
        plt.colorbar()
        plt.show()

        plt.imshow(el)
        plt.colorbar()
        plt.show()

#    el=90-ze
    # az,el -> x,y search tree
    # this is needed to speed up az,el -> x,y conversion
    kd=create_kdtree(az,el)

    long=ams216["long_lat"][0,0]
    lat=ams216["long_lat"][0,1]
    obs=EarthLocation(lon=long,height=h,lat=lat)
    #dt = TimeDelta(dur/2.0,format="sec")
    #aa_frame = AltAz(obstime=t0+dt, location=obs)
    return({"camera_id":camera_id,"kd":kd,"az":az,"el":el,"obs":obs,"video_path":video_path,"lat":lat,"long":long,"h":h,"t0":t0.unix+dt,"t1":t0.unix+dur+dt,"frame_count":frame_count,"cap":cap,"fps":25.0,"fragments":{}})

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
    # az,el -> x,y search tree
    kd=create_kdtree(az,el)

    # needed for star plotting
    obs=EarthLocation(lon=long,height=0,lat=lat)
    return({"camera_id":"0954","kd":kd,"az":az,"el":el,"obs":obs,"video_path":video_path,"lat":lat,"long":long,"h":0,"t0":t0.unix,"t1":t0.unix+dur,"frame_count":frame_count,"cap":cap,"fps":25.0,"fragments":{}})

def fragment_positions():
    fl=glob.glob("fragments/*.h5")
    fl.sort()
    fragment_geo_pos={}
    fragment_time={}

    for f in fl:
        h=h5py.File(f,"r")
        fid=re.search(r"fragments/(\d+)_.*\.h5",f).group(1)
        pos_est=h["pos_est"][()]
        if fid not in fragment_geo_pos.keys():
            fragment_geo_pos[fid]=[]
            fragment_time[fid]=[]

#        print(pos_est)
        fragment_geo_pos[fid].append(pos_est)
        fragment_time[fid].append(h["time"][()])

        h.close()
    for fid in fragment_geo_pos.keys():
        fragment_geo_pos[fid]=n.array(fragment_geo_pos[fid])
        fragment_time[fid]=n.array(fragment_time[fid])
    return(fragment_geo_pos,fragment_time)

def fragment_azel(poss,video):

    # Example: observer at latitude, longitude, height
    lat = video["lat"]   # degrees (Tromsø, Norway for example)
    lon = video["long"]   # degrees
    h   = 0     # meters above sea level

    observer = EarthLocation(lat=lat*u.deg, lon=lon*u.deg, height=h*u.m)
    azs=[]
    els=[]
    for pos in poss:
        llh=jcoord.ecef2geodetic(pos[0],pos[1],pos[2])
        az_el_r=jcoord.geodetic_to_az_el_r(lat, lon, h, llh[0], llh[1], llh[2])
        azs.append(az_el_r[0])
        els.append(az_el_r[1])
    return(azs,els)

def plot_star_pos(ax,bs,tnow,v):
    star_ys=[]
    star_xs=[]
    max_stars=100
    for i in range(max_stars):
        # plot bright stars 
        d=bs.get_ra_dec_vmag(i)
        c = SkyCoord(ra=d[0]*u.degree, dec=d[1]*u.degree, frame='icrs')
        aa_frame = AltAz(obstime=Time(tnow, format='unix'), location=v["obs"])
        altaz=c.transform_to(aa_frame)
        star_az=float(altaz.az/u.deg)
        star_el=float(altaz.alt/u.deg)
        
        # this is the slow part...
        # find closest pixel in az,el arrays
        # how to speed this up???
        #x,y=n.unravel_index(n.argmin((180*n.angle(n.exp(1j*n.pi*star_az/180)*n.exp(-1j*n.pi*v["az"]/180))/n.pi)**2 + (star_el-v["el"])**2),v["el"].shape)
        x,y=interpolate_pixel(star_az,star_el,v["kd"],v["az"],v["el"],k=4,p=2)
        if x > 0 and x < v["az"].shape[0] and y > 0 and y < v["az"].shape[1]:
            star_xs.append(x)
            star_ys.append(y)
    ax.scatter(star_ys,star_xs,s=80, facecolors='none', edgecolors='w',alpha=0.2)

def triangulate_dual(v1,v2):
    
    videos=[v1,v2]

    # for overplotting stars
    bs=bright_stars.bright_stars()

    # find smallest t0 in videos
    t0_min=n.min([v["t0"] for v in videos])
    # find largest t1 in videos
    t1_max=n.max([v["t1"] for v in videos])

    dt = 1.0 # seconds between frames to sample

    # load all fragment positions until now
    fp,ft=fragment_positions()

    # go through all frames with dt spacing
    n_frames = int((t1_max-t0_min)/dt)

    fragment_ids={}
    # go through all the frames
    for fi in range(n_frames):
        # time now in ms since t0_min. this is the frame time we are going to try to 
        # triangulate
        tnow = (t0_min + dt*fi)

        # key for fragments dict (use milliseconds to have a unique integer key)
        tnow_key=int(tnow*1000.0)

        # go through all videos and ask user to mark fragments with keys 1-9 
        # tbd: figure out how to index more framents if needed...  
        for v in videos:
            # this is the frame number we need to read from video v
            idx = int((tnow - v["t0"])*v["fps"])

            # if not in video range, skip
            if idx < 0 or idx >= v["frame_count"]:
                continue

            # seek frame number idx
            v["cap"].set(cv2.CAP_PROP_POS_FRAMES, idx)

            # read frame as image array
            ret, frame = v["cap"].read()

            # if we get a frame succesfully 
            if ret:
                # Convert from BGR (OpenCV default) to RGB for plotting
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

                # plot frame
                fig, ax = plt.subplots(1,1,figsize=(8*1.4,6.4*1.1))
                ax.imshow(frame_rgb)
                print("frame %d"%(idx))
                # instructions for usage
                ax.set_title(f"zoom and position cursor, then press fragment id number (1-9) ")#%(lat,long))
                
                # plot stars. useful for checking that star calibration is correct
                plot_stars=True
                if plot_stars:
                    plot_star_pos(ax,bs,tnow,v)

                # show positions of all so far triangulated fragments overlayed on the image
                # to avoid redoing the triangulation and to help identify what fragment is what
                show_fragments=True
                if show_fragments:
                    # reload all fragments from fragments directory
                    fp,ft=fragment_positions()

                    # go through all fragments
                    for frag in fp.keys():
                        poss=fp[frag]
                        ftimes=ft[frag]

                        fazs,fels=fragment_azel(poss,v)
                        frag_xs=[]
                        frag_ys=[]
                        for i in range(len(fazs)):
                            # get pixel position using calibration
                            x,y=interpolate_pixel(fazs[i],fels[i],v["kd"],v["az"],v["el"],k=4,p=2)
                            #if x > 0 and x < v["az"].shape[0] and y > 0 and y < v["az"].shape[1]:
                            frag_xs.append(x)
                            frag_ys.append(y)
                        frag_xs=n.array(frag_xs)
                        frag_ys=n.array(frag_ys)
                        hidx=n.where(ftimes < tnow)[0]
                        fidx=n.where(ftimes >= tnow)[0]
                        # overplot fragments positions on image
                        # red means fragments with timestamp earlier than now
                        ax.plot(frag_ys[hidx],frag_xs[hidx],".",color="red",alpha=0.5)
                        # green means fragments with timestamp later than or equal to now
                        if len(hidx)>0:
                            last_i=n.argmax(ftimes[hidx])
                            ax.text(frag_ys[hidx[last_i]],frag_xs[hidx[last_i]],"%s"%(frag),color="red")
                        ax.plot(frag_ys[fidx],frag_xs[fidx],".",color="green",alpha=0.5)
                        
                # if we are on the second video, and we have 
                # annotated fragments, draw a line from station 1 into space overlayed on camera 2
                # this line should intersect with fragment on video 2. 
                show_line_of_sight=True
                if show_line_of_sight:
                    if v == v2:
                        # do we have a position in camera1 
                        for fid in fragment_ids.keys():
                            if (tnow_key in v1["fragments"].keys()) and (fid in v1["fragments"][tnow_key].keys()):
                                # this is the az and el selected in previous video 
                                taz=v1["fragments"][tnow_key][fid]["az"]
                                tel=v1["fragments"][tnow_key][fid]["el"]
                                # draw a line of sight guide line in video two based on 
                                # annotations in previous video
                                lx,ly=line_of_sight(taz,tel,v1,v2,dl=5e3,L=1200e3)
                                ax.plot(ly,lx,color="white",alpha=0.1)
                            else:
                                print("key",fid,"not in fragments")
                # capture keyboard input
                def onkey(event):
                    # close frame
                    if event.key == "q":
                        return 
                    
                    # do nothing if we haven't pressed key with cursor over image
                    if event.inaxes != ax:
                        return

                    # store image pixel for fragment
                    if event.inaxes == ax:
                        # read cam 1 position
                        x, y = int(event.xdata), int(event.ydata)
                        taz,tel=xy_to_azel(v["az"],v["el"],y,x)
                        # add to the complicated data structure
                        if tnow_key not in v["fragments"].keys():
                            v["fragments"][tnow_key]={}
                        fragment_id=int(event.key)
                        v["fragments"][tnow_key][fragment_id]={"x":x,"y":y,"az":taz,"el":tel,"fragment_id":fragment_id}

                        # add record of fragment id
                        if fragment_id not in fragment_ids.keys():
                            fragment_ids[fragment_id]=1
                        else:
                            fragment_ids[fragment_id]+=1

                        print(v["fragments"][tnow_key][fragment_id])
                        # plot the newly annotated fragment
                        ax.plot(x, y, "x", color="red",alpha=0.5)
                        # text indicating fragment id
                        ax.text(x,y,event.key,color="red",alpha=0.5)
                        
                        fig.canvas.draw()
            fig.canvas.mpl_connect("key_press_event", onkey)
            plt.show()
        
        # do the triangulation part
        # triangulate all fragments seen by two cameras
        for fid in fragment_ids.keys():
            print("key %d"%(fid))
            # list of fragments to triangulate
            lats=[]
            longs=[]
            azs=[]
            els=[]
            cam_ids=[]
            # go through both videos
            for v in videos:
                # does this fragment id contain this frame (tnow_key)
                # if yes, then add to list of fragments to triangulate
                if (tnow_key in v["fragments"].keys()) and (fid in v["fragments"][tnow_key].keys()):
                    cam_ids.append(v["camera_id"])
                    lats.append(v["lat"])
                    longs.append(v["long"])
                    # TBD: add camera height!!!
                    azs.append(v["fragments"][tnow_key][fid]["az"])
                    els.append(v["fragments"][tnow_key][fid]["el"])    
            # if we have exactly two cameras with this fragment position, then triangulate
            # this should always be the case with this script, as we check that both 
            # video 1 and video 2 have this fragment on this timestamp!
            if len(lats) == 2: 
                print("triangulating fragement id %d"%(fid))
                pos_est,error_std=triangulate(azs,els,lats,longs)
            else:
                exit(0)
                print("exiting. we should have never reached this point. debugging needed")

            # save fragment data to hdf5 file with h5py
            # only save if more than two cameras see it
            id_str=""
            if len(cam_ids) > 1:
                for cid in cam_ids:
                    id_str=id_str + "_"+ cid
                # unique identified based on: fragment id, camera id numbers, and timestamp.
                ofname="fragments/%d%s_%s.h5"%(fid,id_str,tnow_key)
                print("saving %s"%(ofname))
                ho=h5py.File(ofname,"w")
                ho["pos_est"]=pos_est
                ho["pos_err"]=error_std
                ho["time"]=tnow
                ho.close()


# 3:44-3:45:23. Fragments: 1,2
v1=get_video2()

# 3:44-3:45 Fragments: 1,2
v0=get_video(video_path = "2025_02_19_03_44_00_000_012165.mp4",calfile="ams216.mat",camera_id="2165")

# 3:45:00 - 3:45:26 
# flipped cal!
# 6 fragments. nice top fragment!
#v6=get_video(video_path="2025_02_19_03_45_00_000_010624.mp4",calfile="624.mat",camera_id="0624",flip=True)

# 3:45:00 - 3:45:32
# 5 fragments. okay top fragment.
#v3=get_video(video_path = "2025_02_19_03_45_00_000_010125.mp4",calfile="ams21_5.mat",camera_id="0215")


# 3:45:30 - 3:46:00 (calibration might be off near horizon)
# 5 fragments. ok top
v5=get_video(video_path="2025_02_19_03_45_00_000_010761.mp4",calfile="ams_761.mat",camera_id="0761")

# 3:45:21-3:45:58 (most likely bad timing?)
# 5 fragments
#v4=get_video(video_path = "2025_02_19_03_45_00_000_010121.mp4",dt=0.5,calfile="ams21_1.mat",camera_id="0211")

# 3:45:00 - 3:46:00. low elevation
# good top fragment!
v2=get_video(video_path = "2025_02_19_03_45_00_000_010095.mp4",calfile="ams016.mat",camera_id="0165")

# 3:46:00 - 3:46:20
#v7=get_video(video_path = "2025_02_19_03_46_00_000_010095.mp4",calfile="ams016.mat",camera_id="0165")

# 3:46:00 - 3:47:00
# 2025_02_19_03_46_00_000_010880.mp4
#v8=get_video(video_path = "2025_02_19_03_46_00_000_010880.mp4",calfile="0881.mat",h=126,camera_id="0881",flip=False)

# 3:46:00 - 3:46:42 (good)
# 2025_02_19_03_46_01_000_010031_ams0221.mp4
#v9=get_video(video_path = "2025_02_19_03_46_01_000_010031.mp4",calfile="0221.mat",camera_id="0221",flip=False)


# 3:46:37 - 3:47:00
#v10=get_video(video_path = "2025_02_19_03_46_01_000_010028.mp4",dt=-0.8,calfile="0228.mat",h=145,camera_id="0228",flip=False)

# 3:46:07 - 3:46:43
# 2025_02_19_03_46_00_000_010096.mp4
#v13=get_video(video_path = "2025_02_19_03_46_00_000_010096.mp4",dt=0,calfile="0166.mat",camera_id="0166",flip=False)


# 3:47:01 - 3:47:30 (until no longer observable)
#v11=get_video(video_path = "2025_02_19_03_47_01_000_010881_ams0882.mp4",calfile="0882.mat",camera_id="0882",flip=False)
# 3:47:00 - 3:47:30 (until not longer observable)
#v12=get_video(video_path = "2025_02_19_03_47_01_000_010028_ams0228.mp4",dt=-0.8,calfile="0228.mat",camera_id="0228",flip=False)



# 3:46:10 - 3:46:50
# 2025_02_19_03_46_00_000_012386.mp4

# 3:46:10 - 3:47:00 (good)
# 2025_02_19_03_46_01_000_010074_ams0352.mp4

triangulate_dual(v0,v1)
#triangulate_dual(v1,v6)
#triangulate_dual(v6,v3)
#triangulate_dual(v6,v5)
#triangulate_dual(v3,v5)
#triangulate_dual(v4,v5)
#triangulate_dual(v5,v2)
#triangulate_dual(v8,v9)
#triangulate_dual(v8,v10)
#triangulate_dual(v11,v12)
#triangulate_dual(v9,v13)
#triangulate_dual(v6,v1)
#triangulate_dual(v3,v5)
#triangulate_dual(v5,v4)
#triangulate_dual(v5,v2)
#triangulate_dual(v8,v9)
#triangulate_dual(v8,v10)
#triangulate_dual(v11,v12)

exit(0)

#triangulate_dual(v2,v5)

#triangulate_dual(v0,v1)
#triangulate_dual(v1,v2)
#triangulate_dual(v1,v3)
#triangulate_dual(v1,v4)
#triangulate_dual(v2,v3)
#triangulate_dual(v2,v4)
# this is the same location. no use triangulating as we don't have two field of views!
#triangulate_dual(v3,v4)



# 3:43:35 3:44:00
# no stars.
#/Users/jvi019/src/falcon9/2025_02_19_03_43_01_000_011011.mp4







# 3:45:10-3:45:28 
#v5=get_video(video_path = "2025_02_19_03_45_01_000_010122.mp4",calfile="uncal.mat",camera_id="0211")
# 3:45:49 - 3:46:01
#v6=get_video(video_path = "2025_02_19_03_45_01_000_012333.mp4",calfile="uncal.mat",camera_id="0211")

#/Users/jvi019/src/falcon9/2025_02_19_03_45_01_000_010122.mp4
def plot_videos(all_videos=[v0,v1,v2,v3,v4,v5]):
    for vi in range(len(all_videos)):
        plt.plot(n.array([all_videos[vi]["t0"],all_videos[vi]["t1"]],"datetime64[s]"),[vi,vi],label="%d - %s"%(vi, all_videos[vi]["camera_id"]))
    plt.legend()
    plt.xlabel("Time")
    plt.show()

