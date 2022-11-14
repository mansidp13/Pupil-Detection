Getting started with Pupil detection code
A step-by-step guide to get the code working.
1.	Save the video file of .mkv, .mp4, and .h264 to anywhere on the computer. 
2.	Write the following lines in the command prompt to install the modules used in the python file:
•	OpenCV: pip instOpenCVencv-python
•	Glob: pip install glob2
•	NumPy: pip install NumPy
•	PIL: pip install Pillow
•	Matplotlib: -m pip install -U matplotlib
•	Math: pip install python-math
•	Multiprocessing: pip install multiprocess
•	Statistics: pip install statistics
3.	Linesine #18, 19, and 21 are the initial values that be changed by the user.
4.	W is the window size for which the code gives the moving average. That line optional can be uncommented if the user wants to use that.
5.	S is the initialization value that works as the counter. There is no need to change that.
6.	‘filename’ is the variable that is used to save the CSV file in that name. it can be changed according to the user’s need.
7.	From line #24, the user-defined functions start. The codes can be changed if needed with the help of the comments.
8.	Line # 64 uses the frames per second (fps) numbers to get the time stamp of that frame that will later be used as one of the values of the tuple. 
9.	Line #67 to line # 84 create the mask around the given coordinates of the given size. This is commented out for the videos taken at the clinics as the pupil images are zoomed in and there is no need for a mask.
10.	IMPORTANT: Line #87 is where the image is blurred as it is easier for the camera to recognize the accurate pupil size. That number is always an odd number. It will blur the image by the number of times.
11.	IMPORTANT: Line #90 is where the image correction happens. The values of alpha and beta decide what the after image looks like for the function to detect the circle. The value of alpha will brighten the image by that factor. The values are also always an odd fraction. For the clinical camera, the values are tested okay but can be changed here.
12.	Line #92 is where the circle is detected. Please do not change any numbers except for the min and max radius. The numbers are in pixels.
13.	LinesLine #114 to #116 are commented out but the cane is uncommented out to see the recognized circle and its center point in the image. IMPORTANT: Please do not uncomment if testing the whole video as you will have to close every image tab to go ahead.
14.	Run the program. 
15.	The first line will ask you to insert the path to the video file. Remember to add the filename.type as well in the path.
16.	The second input variable is for the baseline time in seconds. Enter the time in seconds.
17.	The third line will ask you to enter the path name where the converted image frame should be stored. Enter the path and add the new folder’s name to it. It will look like this: C:\Users\user\Desktop\folder, the code will create a directory named “folder” on the desktop.
18.	Wait till the code run. It will take around 5 to 10 minutes for 5000 images. When the detecting is done, the code will print “Done detecting” and give the graph will radius and normalized radius.
19.	If there is any error, you might need to go back and check the variables like alpha, min, and max radius. Uncomment the image showing lines to #116 for only a couple of images by changing the number in line #147 to “for n in range(1, count,1000)” which will test every 1000th image in the folder and show the detected circle. 
20.	Check out the CSV file with the given name in the same directory where the code is saved. 
