import cv2
import numpy as np
import threading
import time
import pygame

# Initialize pygame mixer
pygame.mixer.init()

# Load the alert sound
alert_sound = pygame.mixer.Sound('alert.wav')

# Function to play sound
def play_alert_sound():
    alert_sound.play()

# Initialize the webcam
cap = cv2.VideoCapture(0)

# Initialize the background subtractor
fgbg = cv2.createBackgroundSubtractorMOG2(history=500, varThreshold=25, detectShadows=True)

if not cap.isOpened():
    print("Error: Could not open video capture.")
    exit()

# Initialize the last sound play timestamp
last_sound_time = time.time()

while True:
    # Capture frame-by-frame
    ret, frame = cap.read()
    
    if not ret:
        print("Error: Could not read frame.")
        break

    # Apply Gaussian blur to the frame
    blurred_frame = cv2.GaussianBlur(frame, (5, 5), 0)
    
    # Apply the background subtractor to get the foreground mask
    fgmask = fgbg.apply(blurred_frame)
    
    # Threshold the mask to binarize it
    _, fgmask = cv2.threshold(fgmask, 25, 255, cv2.THRESH_BINARY)
    
    # Apply morphological operations to remove noise
    fgmask = cv2.erode(fgmask, None, iterations=2)
    fgmask = cv2.dilate(fgmask, None, iterations=2)

    # Find contours in the mask
    contours, _ = cv2.findContours(fgmask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Filter out small contours
    motion_detected = False
    for contour in contours:
        if cv2.contourArea(contour) < 500:
            continue
        motion_detected = True
        cv2.drawContours(frame, [contour], -1, (0, 255, 0), 2)
        
    # If motion is detected and 2 seconds have passed since the last sound, play the alert sound
    current_time = time.time()
    if motion_detected and (current_time - last_sound_time) >= 2:
        threading.Thread(target=play_alert_sound).start()
        last_sound_time = current_time

    # Display the resulting frame
    cv2.imshow('Motion Detection', frame)
    
    # Break the loop on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything is done, release the capture
cap.release()
cv2.destroyAllWindows()
