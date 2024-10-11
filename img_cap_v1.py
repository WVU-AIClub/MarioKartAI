###
# All this program demostrates is that is possible to capture an window in python

# TO-DOs:
# ~ Someone needs to be able to simulate the keyboard and mouse presses to get to the races or needs to be able get there automatically
# ~ Someone needs to create an Environment that runs the rules of the program
# ~ Someone needs to start simulating and also making sure that AI will have a screen simular to the one it will be playing against


###
import pyautogui # Emulates Keyboard and capture window
import cv2
import numpy as np
import pygetwindow as gw


import DQN_agent_v1 as DQN

# Will need to be more adaptable
WINDOW_NAME = "Dolphin 2409 | JIT64 DC | Direct3D 11 | HLE | Mario Kart Wii (RMCE01)"

window = gw.getWindowsWithTitle(WINDOW_NAME)[0]
window.activate()

# Not sure
fourcc = cv2.VideoWriter_fourcc(*"XVID")
out = cv2.VideoWriter("output.avi", fourcc, 12.0, (window.width, window.height))

# Image Capture Loop
while True:
    img = pyautogui.screenshot(region=(window.left, window.top,\
                                       window.width, window.height))
    frame = np.array(img)
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)


    # Outputs image
    out.write(frame)
    cv2.imshow("Window Capture", frame)

    # Break Loop
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

out.release()
cv2.destroyAllWindows()