

def detect_smile():
	import cv2
	import numpy as np

	print("whats up")

	# Face Classifier
	face_detector = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
	smile_detector = cv2.CascadeClassifier('haarcascade_smile.xml')
	eye_detector = cv2.CascadeClassifier('haarcascade_eye.xml')

	# Grab Webcam feed
	webcam = cv2.VideoCapture(0)

	

	# Show the current frame
	while True:

		# Read the Current frame from the webcam video stream
		successful_frame_read, frame = webcam.read() # read() reads a single frame

		# If there's an error, abort
		if not successful_frame_read:
			break

		# Change to grayscale
		frame_grayscale = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)

		# Detect faces first
		faces = face_detector.detectMultiScale(frame_grayscale)
		#print(faces)

		# Run face detection within each of those faces
		for(x, y, w, h) in faces:

			# Draw a rectangle around the face
			cv2.rectangle(frame, (x, y), (x+w, y+h) , (100, 200, 50), 4)

			# Get the sub frame (using numpy N-dimensional array slicing)
			the_face = frame[y:y+h , x:x+w]

			# Change to grayscale
			face_grayscale = cv2.cvtColor(the_face, cv2. COLOR_BGR2GRAY)

			smiles = smile_detector.detectMultiScale(face_grayscale, scaleFactor = 1.7, minNeighbors = 20) 

			eyes = eye_detector.detectMultiScale(face_grayscale,scaleFactor = 1.7, minNeighbors = 20)

			# Find all smiles in the face
			# for (x_, y_, w_, h_) in smiles:

			# # 	# Draw a rectangle around the smile
			#  	cv2.rectangle(the_face, (x_, y_), (x_+w_, y_+h_) , (50, 50, 200), 4)

			 # Find all smiles in the face
			# for (x_, y_, w_, h_) in eyes:

			# # 	# Draw a rectangle around the smile
			#  	cv2.rectangle(the_face, (x_, y_), (x_+w_, y_+h_) , (255, 255, 255), 4)

			# Label this face as smiling
			if len(smiles) > 0:
				cv2.putText(frame,'smiling', (x, y+h+40), fontScale=3,fontFace=cv2.FONT_HERSHEY_PLAIN, color=(255,255,255))

		# cv2.namedWindow('frame', cv2.WINDOW_FREERATIO)
		# cv2.setWindowProperty('frame', cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

		# show the current frame
		cv2.imshow('Smile Detector', frame)

		# display
		cv2.waitKey(1)

	#Cleanup
	webcam.release()
	cv2.destroyAllWindows()

	print("Code Completed")


detect_smile()
