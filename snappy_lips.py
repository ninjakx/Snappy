#classes and subclasses to import
import cv2
import numpy as np


def blend_transparent(face_img, overlay_t_img):
    # Split out the transparency mask from the colour info
    overlay_img = overlay_t_img[:,:,:3] # Grab the BRG planes
    overlay_mask = overlay_t_img[:,:,3:]  # And the alpha plane

    # Again calculate the inverse mask
    background_mask = 255 - overlay_mask

    # Turn the masks into three channel, so we can use them as weights
    overlay_mask = cv2.cvtColor(overlay_mask, cv2.COLOR_GRAY2BGR)
    background_mask = cv2.cvtColor(background_mask, cv2.COLOR_GRAY2BGR)

    # Create a masked out face image, and masked out overlay
    # We convert the images to floating point in range 0.0 - 1.0
    face_part = (face_img * (1 / 255.0)) * (background_mask * (1 / 255.0))
    overlay_part = (overlay_img * (1 / 255.0)) * (overlay_mask * (1 / 255.0))

    # And finally just add them together, and rescale it back to an 8bit integer image    
    return np.uint8(cv2.addWeighted(face_part, 255.0, overlay_part, 255.0, 0.0))


 
# build our cv2 Cascade Classifiers
face_cascade = cv2.CascadeClassifier("haarcascade_file/haarcascade_frontalface_default.xml")
smile_cascade = cv2.CascadeClassifier("haarcascade_file/haarcascade_smile.xml")

imgMustache = cv2.imread('images/lips2.png',-1)

cap = cv2.VideoCapture(0)
 
while True:

    ret, frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
 
    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(30, 30),
        flags=cv2.CASCADE_SCALE_IMAGE
    )

    for (x, y, w, h) in faces:

        #face = cv2.rectangle(frame,(x,y),(x+w,y+h),(255,0,0),2)
 
        roi_gray = gray[y:y+h, x:x+w]
        roi_color = frame[y:y+h, x:x+w]

        smile = smile_cascade.detectMultiScale(
            roi_gray,
            scaleFactor= 1.7,
            minNeighbors=22,
            minSize=(25, 25)
            
            )
 
        for (nx,ny,nw,nh) in smile:

            cv2.rectangle(roi_color,(nx,ny),(nx+nw,ny+nh),(255,0,0),2)
 
            
            mustacheWidth =  0.9 * nw
            mustacheHeight = nh//0.6
 
            # Center the mustache on the bottom of the nose
            x1 = nx - int(mustacheWidth/2)
            x2 = nx + nw + int(mustacheWidth/2)
            y1 = ny + nh - int(mustacheHeight)
            y2 = ny + nh + int(mustacheHeight)
 
            # Check for clipping
            if x1 < 0:
                x1 = 0
            if y1 < 0:
                y1 = 0
            if x2 > w:
                x2 = w
            if y2 > h:
                y2 = h
 
            # Re-calculate the width and height of the mustache image
            mustacheWidth = x2 - x1
            mustacheHeight = y2 - y1
        
            mustache = cv2.resize(imgMustache, (int(mustacheWidth),int(mustacheHeight)))
            #print(mustache.shape)

            roi = roi_color[y1:y2, x1:x2,:]
            #print(roi.shape)
            roi_bg = blend_transparent(roi, mustache)
            roi_color[y1:y2, x1:x2] = roi_bg

            break
 
    # Display the resulting frame
    cv2.imshow('Video', frame)
 
    # press any key to exit
    # NOTE;  x86 systems may need to remove: " 0xFF == ord('q')"
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
 

cap.release()
cv2.destroyAllWindows()


