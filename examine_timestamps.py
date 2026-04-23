import cv2
fname="2025_02_19_03_45_00_000_010761.mp4"#2025_02_19_03_45_00_000_010095.mp4"
#fname="2025_02_19_03_46_00_000_012132.mp4"

cap = cv2.VideoCapture(fname)

frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
fps = cap.get(cv2.CAP_PROP_FPS)
print(fps)
dur=frame_count/fps
print(cap.get(cv2.CAP_PROP_POS_MSEC))
for i in range(0,frame_count,10):
    cap.set(cv2.CAP_PROP_POS_FRAMES, i)
    timestamp=cap.get(cv2.CAP_PROP_POS_MSEC)
    print("frame %d time %1.2f ms should be %1.2f s"%(i,timestamp,i/fps))

print(frame_count)
print(dur)