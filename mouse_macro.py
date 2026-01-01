import pyautogui
import time

interval = 30  # Move every 5 minutes (300 seconds)
movement_size = 50  # Pixels to move

print("Mouse mover started. Press Ctrl + C to stop.")

try:
    while True:
        pyautogui.move(movement_size, 0, duration=0.2)   # Right
        pyautogui.move(0, movement_size, duration=0.2)   # Down
        pyautogui.move(-movement_size, 0, duration=0.2)  # Left
        pyautogui.move(0, -movement_size, duration=0.2)  # Up (back to original)
        
        time.sleep(interval)  # Wait before repeating
except KeyboardInterrupt:
    print("\nMouse mover stopped.")
