# Copyright 2025 DeepMind Technologies Limited
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
#modified by Logan
"""Keyboard controller class using Tkinter for input capture."""

import threading
import time
import tkinter as tk
import numpy as np


class KeyboardController:
    """Keyboard controller class that reads keyboard input."""

    def __init__(
        self,
        vel_scale_x=0.4,
        vel_scale_y=0.4,
        vel_scale_rot=1.0,
    ):
        self._vel_scale_x = vel_scale_x
        self._vel_scale_y = vel_scale_y
        self._vel_scale_rot = vel_scale_rot

        self.vx = 0.0
        self.vy = 0.0
        self.wz = 0.0
        self._keys_pressed = set()
        self.is_running = True

        self.tk_thread = threading.Thread(target=self._run_tkinter, daemon=True)
        self.tk_thread.start()

    def _key_press(self, event):
        self._keys_pressed.add(event.keysym.lower())
        self._update_command()

    def _key_release(self, event):
        self._keys_pressed.discard(event.keysym.lower())
        self._update_command()

    def _update_command(self):
        target_vx = 0.0
        target_vy = 0.0
        target_wz = 0.0

        if 'w' in self._keys_pressed:
            target_vx = self._vel_scale_x
        if 's' in self._keys_pressed:
            target_vx = -self._vel_scale_x
        if 'a' in self._keys_pressed:
            target_vy = self._vel_scale_y  # Strafe left
        if 'd' in self._keys_pressed:
            target_vy = -self._vel_scale_y # Strafe right
        if 'q' in self._keys_pressed:
            target_wz = self._vel_scale_rot  # Rotate left
        if 'e' in self._keys_pressed:
            target_wz = -self._vel_scale_rot # Rotate right

        # Simple interpolation or direct assignment could be used here.
        # Let's use direct assignment for simplicity now.
        self.vx = target_vx
        self.vy = target_vy
        self.wz = target_wz


    def _run_tkinter(self):
        self.root = tk.Tk()
        self.root.title("Keyboard Input")
        self.root.geometry("200x100")

        label_text = ("Focus this window\n"
                      "to use keyboard controls\n"
                      "(W/A/S/D for move, Q/E for rotate)")
        label = tk.Label(self.root, text=label_text)
        label.pack(pady=20)

        self.root.bind("<KeyPress>", self._key_press)
        self.root.bind("<KeyRelease>", self._key_release)

        # Handle window close event
        self.root.protocol("WM_DELETE_WINDOW", self.stop)

        # Start the Tkinter event loop
        # We run this in a loop to allow checking self.is_running
        while self.is_running:
            try:
                self.root.update_idletasks()
                self.root.update()
                time.sleep(0.01) # Prevent busy-waiting
            except tk.TclError: # Handle window close exception
                self.is_running = False
                break

        # Explicitly destroy the window if the loop exits
        try:
            if self.root.winfo_exists():
                self.root.destroy()
        except tk.TclError:
            pass # Window already destroyed


    def get_command(self):
        return np.array([self.vx, self.vy, self.wz])

    def stop(self):
        print("Stopping keyboard controller...")
        self.is_running = False
        # No need to join tk_thread immediately, allow the loop to exit naturally
        # Ensure Tkinter resources are released if the window is still open
        try:
            if hasattr(self, 'root') and self.root.winfo_exists():
                self.root.quit() # Quit the mainloop
        except tk.TclError:
             pass # Ignore if root is already destroyed


if __name__ == "__main__":
    keyboard_controller = KeyboardController()
    print("Keyboard controller started. Press keys (W/A/S/D, Q/E) in the Tkinter window.")
    print("Close the Tkinter window or press Ctrl+C in the terminal to stop.")
    try:
        while keyboard_controller.is_running:
            command = keyboard_controller.get_command()
            # Only print if command is not zero to avoid spamming
            if np.any(command != 0):
                 print(f"Command: {command}")
            time.sleep(0.1)
    except KeyboardInterrupt:
        print("\nCtrl+C detected. Stopping.")
    finally:
        keyboard_controller.stop()
        print("Keyboard controller stopped.") 