import cv2
import numpy as np
from collections import deque
import matplotlib.pyplot as plt


PIXEL_TO_METER = 0.0022     # calibrated value
MIN_CONTOUR_AREA = 300
MAX_CONTOUR_AREA = 5000
SMOOTHING_WINDOW = 5        # for height smoothing


height_history = deque(maxlen=SMOOTHING_WINDOW)
bounce_peaks = []
prev_height = None
descending = False

#detect ground y-coordinate
def ground_determination(frame):
    print("Click on the floor and note the Y value shown.")
    #cv2.imshow("Select Ground Reference", frame)
    plt.imshow(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    plt.title("Click on floor to get y axis value..")
    plt.show()
    y = int(input("Enter the ground Y-coordinate: "))
    return y

#detect and process ball in frame
def process_frame(frame, fgmask, ground_y):
    global prev_height, descending

    contours, _ = cv2.findContours(
        fgmask, mode=cv2.RETR_EXTERNAL, method=cv2.CHAIN_APPROX_SIMPLE
    )

    if not contours:
        return frame

    contours = sorted(contours, key=cv2.contourArea, reverse=True)
    cnt = contours[0]

    area = cv2.contourArea(cnt)
    if area < MIN_CONTOUR_AREA or area > MAX_CONTOUR_AREA:
        return frame

    (x, y), radius = cv2.minEnclosingCircle(cnt)
    center = (int(x), int(y))
    radius = int(radius)

    # Bottom of the ball
    bottom_y = int(y + radius)

    # Height calculation
    height_pixels = ground_y - bottom_y
    height_meters = height_pixels * PIXEL_TO_METER

    # Ignore invalid values
    if height_meters < 0:
        return frame

    # height smoothing
   
    height_history.append(height_meters)
    smooth_height = np.mean(height_history)

    
    # BOUNCE PEAK DETECTION
    
    if prev_height is not None:
        if smooth_height < prev_height:
            descending = True
        elif descending and smooth_height > prev_height:
            bounce_peaks.append(prev_height)
            descending = False

    prev_height = smooth_height

    
    # Drawing circle and line
   
    cv2.circle(frame, center, radius,(255, 0, 0), 2)
    cv2.line(frame, (0, ground_y), (frame.shape[1], ground_y), (0, 255, 255), 2)

    cv2.putText(
        frame,
        f"Height: {smooth_height:.3f} m",
        (30, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 255, 0),
        2,
    )

    if bounce_peaks:
        cv2.putText(
            frame,
            f"Max Bounce: {max(bounce_peaks):.3f} m",
            (30, 80),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 0, 255),
            2,
        )

    return frame


def main():
    cap = cv2.VideoCapture(0)

    # KNN Background Subtractor
    bg_sub = cv2.createBackgroundSubtractorKNN(
        history=500,
        dist2Threshold=400,
        detectShadows=False
    )

    
    ret, frame = cap.read()
    if ret:
         bg_sub.apply(frame)

    ret, frame = cap.read()
    if  ret:
        ground_y = ground_determination(frame)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        fgmask = bg_sub.apply(gray)

        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        fgmask = cv2.morphologyEx(fgmask, cv2.MORPH_OPEN, kernel)

        frame = process_frame(frame, fgmask, ground_y)

        cv2.imshow("Bouncing Ball Height Measurement", frame)

        if cv2.waitKey(1) & 0xFF == 27:  # ESC
            break

    cap.release()
    cv2.destroyAllWindows()

    #Showing log outputs
   
    if bounce_peaks:
        print("Bounce Heights (m):", bounce_peaks)
        print("Maximum Bounce Height (m):", max(bounce_peaks))
    else:
        print("No bounces detected.")
if __name__ == "__main__":
    main()
