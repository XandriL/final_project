import cv2 
import os 
  
# Read the video from specified path
folder = os.getcwd()
mp4_file_name = 'WalkByShop1cor.mpg'
file = os.path.join(folder,'video_clips',mp4_file_name)
print(file)
cam = cv2.VideoCapture(file) 


try: 
      
    # creating a folder named data 
    if not os.path.exists(os.path.join(folder,'images')): 
        os.makedirs(os.path.join(folder,'images')) 
  
# if not created then raise error 
except OSError: 
    print ('Error: Creating directory of data') 
  
# frame 
currentframe = 0
  
while(True): 
      
    # reading from frame 
    ret,frame = cam.read() 
  
    if ret: 
        # if video is still left continue creating images
        if(currentframe%10 ==0):
            name = os.path.join(folder,'images',mp4_file_name+'_frame')+str(currentframe) + '.jpg'
            print ('Creating...' + name) 
      
            # writing the extracted images 
            cv2.imwrite(name, frame) 
      
        # increasing counter so that it will 
        # show how many frames are created 
        currentframe += 1
    else: 
        break
  
# Release all space and windows once done 
cam.release() 
cv2.destroyAllWindows() 
