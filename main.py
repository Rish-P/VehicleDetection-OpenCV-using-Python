import cv2

cap = cv2.VideoCapture('assets/highway.mp4')


_, frame1 = cap.read()
_, frame2 = cap.read()

while(cap.isOpened()):
    # finding the difference between 2 frames,
    # if object would have moved then diff>0, and we can perform computations on it
    diff = cv2.absdiff(frame1, frame2)

    gray_diff = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)

    # get rid of the noise using blurring technique
    blur_diff = cv2.GaussianBlur(gray_diff, (3, 3), 0)

    _, thresh = cv2.threshold(blur_diff, 20, 255, cv2.THRESH_BINARY)
    dilation = cv2.dilate(thresh, (3, 3), iterations = 5)
    contours, _ = cv2.findContours(dilation, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    for ctr in contours:
        (x, y, w, h) = cv2.boundingRect(ctr)

        width = x + w + 5
        height = y + h + 5
        if cv2.contourArea(ctr) < 600:
            cv2.rectangle(frame1, (x-5, y-5), (width, height), (0, 255, 0), 2)
        else:
            continue

    cv2.imshow('FINAL', frame1)

    # copy next frame to present frame, and read the frame ahead
    frame1 = frame2
    _, frame2 = cap.read()

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
