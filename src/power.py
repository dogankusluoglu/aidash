import logging
import os
import signal
import subprocess
from gpiozero import Button

class PowerManager:
    def __init__(self, ignition_pin, shutdown_callback):
        self.ignition_pin = ignition_pin
        self.shutdown_callback = shutdown_callback
        self.ignition_sensor = None
        self._setup_logger()

    def _setup_logger(self):
        self.logger = logging.getLogger("PowerManager")
        self.logger.setLevel(logging.INFO)

    def start_monitoring(self):
        self.logger.info(f"Monitoring ignition on GPIO {self.ignition_pin}...")
        try:
            # Assume pull_up=True, and active_low=True (ignition pulls to GND when off)
            # Adjust based on actual hardware circuit (e.g. voltage divider)
            self.ignition_sensor = Button(self.ignition_pin, pull_up=True)
            self.ignition_sensor.when_released = self._on_ignition_off
        except Exception as e:
            self.logger.error(f"Failed to initialize GPIO monitoring: {e}")

    def _on_ignition_off(self):
        self.logger.info("Ignition OFF detected! Triggering safe shutdown...")
        self.shutdown_callback()
        
        # In a real dashcam, you might want to wait a bit or check if it stays off
        self.logger.info("System will shut down in 30 seconds.")
        # subprocess.run(["sudo", "shutdown", "-h", "+1"]) # Example shutdown command

if __name__ == "__main__":
    # Test
    def dummy_callback():
        print("Shutdown callback triggered!")

    pm = PowerManager(17, dummy_callback)
    pm.start_monitoring()
    print("Power monitoring started. Press Ctrl+C to stop.")
    import time
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        pass

