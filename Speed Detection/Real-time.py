import cv2
import time
import numpy as np

# Set up the video capture (use 0 for the default camera)
cap = cv2.VideoCapture(0)

# Set the width and height of the frame
frame_width = int(cap.get(3))
frame_height = int(cap.get(4))

# Set the start time
start_time = time.time()

# Set the initial position of the vehicle
initial_position = None

while cap.isOpened():
    # Capture frame-by-frame
    ret, frame = cap.read()
    if not ret:
        break

    # Convert the frame to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Apply GaussianBlur to reduce high frequency noise
    blur = cv2.GaussianBlur(gray, (5, 5), 0)

    # Detect edges in the image using Canny edge detection
    edges = cv2.Canny(blur, 50, 150)

    # Find contours in the binary image
    contours, _ = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    # Find the contour with the largest area (the vehicle)
    max_area = 0
    max_contour = None
    for contour in contours:
        area = cv2.contourArea(contour)
        if area > max_area:
            max_area = area
            max_contour = contour

    # Draw a bounding box around the vehicle
    if max_contour is not None:
        x, y, w, h = cv2.boundingRect(max_contour)
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

        # Calculate the position of the vehicle
        position = (x + w // 2, y + h // 2)

        # If this is the first frame, set the initial position of the vehicle
        if initial_position is None:
            initial_position = position

        # Calculate the distance traveled by the vehicle
        distance = np.sqrt((position[0] - initial_position[0]) ** 2 + (position[1] - initial_position[1]) ** 2)

        # Calculate the time elapsed since the start of the video
        elapsed_time = time.time() - start_time

        # Calculate the speed of the vehicle (pixels/second)
        speed = distance / elapsed_time

        # Display the speed on the frame
        cv2.putText(frame, 'Speed: {:.1f} pixels/second'.format(speed), (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)

    # Display the resulting frame
    cv2.imshow('Frame', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release everything if job is finished
cap.release()
cv2.destroyAllWindows()
