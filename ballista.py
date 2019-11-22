import cv2
import os
import math
import time
import wiringpi
import numpy as np

MAIN_WINDOW = 'Main Window'
PRESCALE = 0.3
AZ_MOTOR = 1
ALT_MOTOR = 2

AZ_PWM_PIN = 13
ALT_PWM_PIN = 12

PWM_TIME_LO = 50
PWM_TIME_HI = 250

K = np.array([
	[600, 0, PRESCALE*640/2],
	[0, 600, PRESCALE*480/2],
	[0, 0, 1]
], dtype=np.float32)
dist_coeffs = np.zeros(4)

face_width = 0.16
face_height = 0.175
rect_model = np.array([
	[0, 0, 0],
	[face_width, 0, 0],
	[0, face_height, 0],
	[face_width, face_height, 0]
], dtype=np.float32)


queue = []


def fire():
	""" Fire! """
	print('FIRE!')


def set_angle(motor, deg):
	""" Set the angle of a motor, 0-180 degrees. """
	deg = min(max(0, deg), 180)  # Clip angle if OOB
	scalar = (PWM_TIME_HI - PWM_TIME_LO) / 180
	time = int(scalar * deg + PWM_TIME_LO)  # Scale to time units
	wiringpi.pwmWrite(motor, time)


def get_alt_angles(x, y):
	""" Compute the altitude angles to hit a target (x, y). """
	v0 = 4.5596
	g = 9.80665
	try:
		theta_1 = math.atan((v0**2 + (v0**4-g*(g*x**2 + 2*y*v0**2))**0.5)/(g*x))
		theta_2 = math.atan((v0**2 - (v0**4-g*(g*x**2 + 2*y*v0**2))**0.5)/(g*x))
		return theta_1, theta_2
	except:
		return None, None


def get_pose(rect_corners):
	""" Compute the pose of the detected rectangle. """
	ret, rvec, tvec = cv2.solvePnP(rect_model, rect_corners, K, dist_coeffs)
	return tvec[0], tvec[1], tvec[2]


def update(frame, face_cascade):
	frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
	frame_gray = cv2.resize(frame_gray, None, fx=PRESCALE, fy=PRESCALE)
	frame_gray = cv2.equalizeHist(frame_gray)
	
	faces = face_cascade.detectMultiScale(frame_gray)
	for face_data in faces:
		x, y, w, h = tuple(map(lambda v: int(v/PRESCALE), face_data))
		if not 0.9 < w/h < 1.1:  # Filter non-squares
			continue
			
		cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 3)  # Draw
		
		rect_corners = np.array([
			[x, y],
			[x+w, y],
			[x, y+h],
			[x+w, y+h]
		], dtype=np.float32)
		
		x, y, z = get_pose(rect_corners)
		az = np.pi/2 - np.arctan2(x, z)[0]
		az *= 180/np.pi
		set_angle(AZ_PWM_PIN, az)
		
		alt1, alt2 = get_alt_angles(z, y)
		alt1 *= 180/np.pi
		alt2 *= 180/np.pi
		alt = alt2
		set_angle(ALT_PWM_PIN, alt + np.pi/2)
		
		global queue
		comp = az*alt
		queue.insert(0, comp)
		queue = queue[:10]
		
		if len(queue) == 10:
			if abs(comp - sum(queue)/len(queue)) < 10:
				fire()
		
		print('AZ: {0:.1f} deg, ALT1: {1:.1f} deg, (ALT2: {2:.1f} deg)'.format(az, alt1, alt2))
			
	cv2.imshow(MAIN_WINDOW, frame)


def setup_pwm():
	""" Configure PWM settings for controlling the servo motors. """
	wiringpi.wiringPiSetupGpio()
	
	wiringpi.pinMode(AZ_PWM_PIN, wiringpi.GPIO.PWM_OUTPUT)
	wiringpi.pinMode(ALT_PWM_PIN, wiringpi.GPIO.PWM_OUTPUT)
	
	wiringpi.pwmSetMode(wiringpi.GPIO.PWM_MODE_MS)
	
	wiringpi.pwmSetClock(192)
	wiringpi.pwmSetRange(2000)


def main():
	setup_pwm()
	
	cap = cv2.VideoCapture(0)
	cv2.namedWindow(MAIN_WINDOW)
	
	face_cascade = cv2.CascadeClassifier()
	cv2_base_dir = os.path.dirname(os.path.abspath(cv2.__file__))
	haar_model = os.path.join(cv2_base_dir, 'data/haarcascade_frontalface_alt.xml')
	if not face_cascade.load(haar_model):
		print('Error loading face cascade from {}'.format(haar_model))
		exit(1)
	
	while cv2.waitKey(30) != 27:
		ret, frame = cap.read()
		if not ret:
			break
		update(frame, face_cascade)
	cv2.destroyAllWindows()


if __name__ == '__main__':
	main()
