import cv2
import numpy as np
import pygetwindow as gw
from mss import mss

sct = mss()

WINDOW_NAME = "Dolphin 2409 | JIT64 DC | Direct3D 11 | HLE | Mario Kart Wii (RMCE01)"

window = gw.getWindowsWithTitle(WINDOW_NAME)[0]
window.activate()

monitor = {"top": window.top, "left":window.left, \
           "width":window.width, "right":window.right}

# Not sure
fourcc = cv2.VideoWriter_fourcc(*"XVID")
out = cv2.VideoWriter("output.avi", fourcc, 12.0, (window.width, window.height))

while True:
    frame = np.array(sct.grab(monitor))
    frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)
    out.write(frame)
    cv2.imshow("Window Capture", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break