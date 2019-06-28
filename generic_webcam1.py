import time, cv2, os, Queue, threading
import sys
global_image_queue = Queue.Queue(5)
webcam = None

#camera 1
class Webcam: #USB Class
	def __init__(self, cam_name="Webcam", cam_id=0, cam_width=None, cam_height=None, cam_brightness=None, cam_focus=None, cam_zoom=None): #name, camera_id(0), width, height, brightness(0 to 1), focus(None=auto or 0=infinity to 255=proximity), zoom (100=no zoom to 500=max zoom)
		try:
			print "> Init", cam_name, "... ",
			self.cam_name = cam_name
			self.cam_id = cam_id
			if not cam_width:
				self.cam_width = 300
			else:
				self.cam_width = cam_width
			if not cam_height:
				self.cam_height = 300
			else:
				self.cam_height = cam_height
			if not cam_brightness:
				self.cam_brightness = 0.5
			else:
				self.cam_brightness = cam_brightness

			self.cam_focus = cam_focus
			self.cam_zoom = cam_zoom


			self.cam = cv2.VideoCapture(self.cam_id)
			self.cam.set(3, self.cam_width)			# width
			self.cam.set(4, self.cam_height)		# height
			self.cam.set(10, self.cam_brightness)	# brightness

			os.system('v4l2-ctl -d 0 -c exposure_auto=3')
			os.system('v4l2-ctl -d 0 -c exposure_auto_priority=1')

			os.system('v4l2-ctl -d 0 -c pan_absolute=800')
			os.system('v4l2-ctl -d 0 -c tilt_absolute=-10800')
			
			
			if not self.cam_focus:
				os.system('v4l2-ctl -d 0 -c focus_auto=1')
			else:
				os.system('v4l2-ctl -d 0 -c focus_auto=0')
				os.system('v4l2-ctl -d 0 -c focus_absolute='+str(int(self.cam_focus)))

			if self.cam_zoom:
				os.system('v4l2-ctl -d 0 -c zoom_absolute='+str(int(self.cam_zoom)))
			


			time.sleep(5)
			ret, img = self.cam.read()
			time.sleep(0.1)
			ret, img = self.cam.read()
			time.sleep(0.1)
			ret, img = self.cam.read()
			time.sleep(0.1)
			ret, img = self.cam.read()
			time.sleep(0.1)
			ret, img = self.cam.read()
			time.sleep(0.1)
			print ">", self.cam_name,"ready"
		except Exception as e:
			print "\n> FAILED TO INIT CAMERA - ", e


	def capture(self):
		global global_image_queue
		self.cam.grab()
		self.cam.grab()
		self.cam.grab()
		self.cam.grab()
		ret, img = self.cam.read()
		global_image_queue.put(img)
		print "Capture image"
		return
	def Release(self):
		self.cam.release()


def AsyncCapture():

	"""
	global webcam
	print "Trigger Webcam"
	time.sleep(5)
	webcam.capture()
	"""

	global webcam
	try:
		while 1:
			raw_input("Press Enter to capture")
			time.sleep(0.2)
			webcam.capture()

	except:
		pass
	


def Process_Frame(image_frame, index):
	index+=1
	print "Save image", index
	if len(sys.argv) > 1:
	   cv2.imwrite('rotation/' + str(sys.argv[1]) + ".jpg", image_frame)
	else:
	   cv2.imwrite('rotation/' + "image"+str(index)+".jpg", image_frame)
	return index


if __name__=="__main__":
	index = 100
	webcam = Webcam("Webcam", 0, 640, 720, 0.5,70, 2) #name, camera_id(0), width, height, brightness(0 to 1), focus(None=auto or 0=infinity to 255=proximity), zoom (1=no zoom to 5=max zoom)
	t1 = threading.Thread(target=AsyncCapture)
	t1.start()
	try:
		while True:
			if (not global_image_queue.empty()):
				image_frame = global_image_queue.get()
				index = Process_Frame(image_frame, index)
				time.sleep(0.001)
				#break
			else:
				time.sleep(0.001)
	except KeyboardInterrupt:
		print "> Keyboard Interrupt"
	except Exception as e:
		print "> Exception in main ", e
	finally:
		webcam.Release()
		t1.join()
		print "Quit"