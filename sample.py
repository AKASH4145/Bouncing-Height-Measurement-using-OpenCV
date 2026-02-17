import cv2
from matplotlib import pyplot as plt


def ground_determination(image):
     #cv2.imshow("Click on floor to get y axis value..",image)
     plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
     plt.title("Click on floor to get y axis value..")
     plt.show()
     result=int(input("Enter the Y-coord value:"))
     return result;



def calc_height(image,y_coord):
    ground_y=y_coord
    pixel_to_meter_ratio=0.0022 #calibrated value
    gray=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
    blurred_image=cv2.GaussianBlur(gray,(5,5),0)
    _,threshold=cv2.threshold(blurred_image,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    contours,_=cv2.findContours(threshold,mode=cv2.RETR_EXTERNAL,method=cv2.CHAIN_APPROX_SIMPLE)
    sorted_contour=sorted(contours,key=cv2.contourArea,reverse=True)
    if sorted_contour:
     detected_object = sorted_contour[0]
    else:
      return

    if cv2.contourArea(detected_object) > 300:
        (x,y),radius=cv2.minEnclosingCircle(detected_object)
        center=(int(x),int(y))
        radius=int(radius)
        cv2.circle(image,center,radius,(255,0,0),2)
        object_height_pixels=ground_y - int(y+radius)
        object_height_meters=object_height_pixels * pixel_to_meter_ratio
    #(int(x-radius),int(y-radius-10))    
    cv2.putText(image,"Height: {:.2f} meters".format(object_height_meters),(400,444),cv2.FONT_HERSHEY_SIMPLEX,0.6,(0,127,255),2)
    cv2.drawContours(image,detected_object,-1,(0,255,0),3)
    cv2.imshow("contour",image)

if __name__=='__main__':
    cap=cv2.VideoCapture(0)
    y_coord = None
    
    # Run ground detection once
    ret, frame = cap.read()
    if ret:
        y_coord = ground_determination(frame)
    
    # Run calc_height continuously if y_coord is not None
    if y_coord != 0:
        while True:
            ret, frame = cap.read()
            if ret:
                calc_height(frame,y_coord)
            else:
                print("Video Capture error!")    
            if cv2.waitKey(1) == 13:
                break
    else:
        print("Failed to detect ground!")
        
cap.release()
cv2.destroyAllWindows()