import cv2

# Load the face and smile CascadeClassifiers
faceCascade = cv2.CascadeClassifier("D:\smile-selfie-capture-project\dataset\haarcascade_frontalface_default.xml")
smileCascade = cv2.CascadeClassifier("D:\smile-selfie-capture-project\dataset\haarcascade_smile.xml")

# Start the video capture
video = cv2.VideoCapture(0)

# Set a counter to track the number of images saved
cnt = 0

# Loop through the frames of the video
while True:
    # Read the next frame from the video
    success, img = video.read()

    # Convert the frame to grayscale
    grayImg = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Detect faces in the grayscale image
    faces = faceCascade.detectMultiScale(grayImg, scaleFactor=1.1, minNeighbors=4)

    # Loop through each detected face
    for (x, y, w, h) in faces:
        # Draw a rectangle around the face
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 0, 0), 3)

        # Crop the region of interest (ROI) containing the face
        faceROI = grayImg[y:y+h, x:x+w]

        # Detect smiles in the ROI
        smiles = smileCascade.detectMultiScale(faceROI, scaleFactor=1.8, minNeighbors=15)

        # If a smile is detected, draw a rectangle around it and save the image
        for (sx, sy, sw, sh) in smiles:
            cv2.rectangle(img, (x + sx, y + sy), (x + sx + sw, y + sy + sh), (100, 100, 100), 5)
            path = f"path/to/image{cnt}.jpg"
            cv2.imwrite(path, img)
            print(f"Image {cnt} saved.")
            cnt += 1

    # Display the video feed with the detected faces and smiles
    cv2.imshow("Live Video", img)

    # If the user presses the 'q' key, exit the loop
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# Release the video capture and close all windows
video.release()
cv2.destroyAllWindows()
