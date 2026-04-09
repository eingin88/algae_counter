import cv2
import numpy as np

IMG_PATH = r"C:"  # 請改成你的合成圖路徑

pts = []

def on_mouse(event, x, y, flags, param):
    global pts
    if event == cv2.EVENT_LBUTTONDOWN:
        pts.append((x, y))
        print(f"Point {len(pts)}: ({x}, {y})")
        if len(pts) == 2:
            (x1, y1), (x2, y2) = pts
            dist = ((x2-x1)**2 + (y2-y1)**2) ** 0.5
            print(f"Pixel distance = {dist:.2f} px")
            pts = []

img = cv2.imread(IMG_PATH, cv2.IMREAD_COLOR)
if img is None:
    raise SystemExit("Cannot read image. Check IMG_PATH.")

cv2.namedWindow("measure", cv2.WINDOW_NORMAL)
cv2.imshow("measure", img)
cv2.setMouseCallback("measure", on_mouse)

print("Click two points to measure pixel distance. Press ESC to exit.")
while True:
    k = cv2.waitKey(20) & 0xFF
    if k == 27:  # ESC
        break

cv2.destroyAllWindows()