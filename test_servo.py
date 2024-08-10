import RPi.GPIO as GPIO
import time

# Set up GPIO
GPIO.setmode(GPIO.BCM)
GPIO.setup(24, GPIO.OUT)

# Set up PWM on the GPIO pin
pwm = GPIO.PWM(24, 50)  # GPIO 17 at 50Hz

# Start PWM with a duty cycle of 0 (servo in neutral position)
pwm.start(0)

try:
    while True:
        # # Rotate to 0 degrees
        pwm.ChangeDutyCycle(2)  # 0 degrees
        time.sleep(1)
        
        # Rotate to 90 degrees
        pwm.ChangeDutyCycle(7)  # 90 degrees
        # pwm.ChangeDutyCycle(8)  # 90 degrees
        time.sleep(1)
        
        # # Rotate to 180 degrees
        pwm.ChangeDutyCycle(12)  # 180 degrees
        # pwm.ChangeDutyCycle(10)  # 180 degrees
        time.sleep(1)
        
        # Rotate to 90 degrees
        pwm.ChangeDutyCycle(7)  # 90 degrees
        # pwm.ChangeDutyCycle(8)  # 90 degrees
        time.sleep(1)

except KeyboardInterrupt:
    pass

# Stop PWM and clean up
pwm.stop()
GPIO.cleanup()
