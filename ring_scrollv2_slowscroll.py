# ring_scrollv2
import asyncio
import csv
import os
import queue
import threading
from datetime import datetime

import matplotlib.pyplot as plt
import matplotlib.animation as animation
from bleak import BleakClient
import numpy as np
import pandas as pd
import pyautogui
import time
from collections import deque


# Old ring parsing logic constants
MAIN_SERVICE_UUID = "de5bf728-d711-4e47-af26-65e3012a5dc7"
MAIN_NOTIFY_CHARACTERISTIC_UUID = "de5bf729-d711-4e47-af26-65e3012a5dc7"
RXTX_SERVICE_UUID = "6e40fff0-b5a3-f393-e0a9-e50e24dcca9e"
RXTX_NOTIFY_CHARACTERISTIC_UUID = "6e400003-b5a3-f393-e0a9-e50e24dcca9e"
# Global definitions
RXTX_WRITE_CHARACTERISTIC_UUID = "6e400002-b5a3-f393-e0a9-e50e24dcca9e"
RXTX_NOTIFY_CHARACTERISTIC_UUID = "6e400003-b5a3-f393-e0a9-e50e24dcca9e"

DATA_FOLDER = "raw_data"
os.makedirs(DATA_FOLDER, exist_ok=True)
timestamp_now = datetime.now().strftime("%Y%m%d_%H%M%S")
filename = os.path.join(DATA_FOLDER, f"ring_data_{timestamp_now}.csv")
csv_file = open(filename, "w", newline="")
csv_writer = csv.writer(csv_file)
csv_writer.writerow(["timestamp", "accX", "accY", "accZ"])

acc_queue = queue.Queue()

# Rolling window size for smoothing
WINDOW_SIZE = 10
angle_window = deque([], maxlen=WINDOW_SIZE)


def get_median_angle(raw_angle):
    angle_window.append(raw_angle)
    sorted_window = sorted(angle_window)
    mid_index = len(sorted_window) // 2
    return sorted_window[mid_index]

def get_mean_angle(raw_angle):
    angle_window.append(raw_angle)
    return np.mean(angle_window)

# Dead zone for small movements (radians)
DEAD_ZONE = 0.5

# Scale factor for how many "scroll steps" per radian of tilt
SCROLL_SCALE = 1

# Debounce: minimum time (in seconds) between scrolls
DEBOUNCE_INTERVAL = 1
last_scroll_time = time.time() 


def convertrawtoG(raw):
    return (raw / 2048.0) * 1


df = pd.DataFrame(columns=["timestamp",
                           "raw_angle",         # direct atan2(accY, accX)
                           "filtered_angle",    # exponential moving average
                           "angle_cumulative",  # cumulative rotation over time
                           "accX", "accY", "accZ"])


def angle_diff(current_angle, previous_angle):
    """
    Return the signed difference between two angles (in radians),
    taking into account the wrap-around from -pi to pi.
    """
    return np.arctan2(
        np.sin(current_angle - previous_angle),
        np.cos(current_angle - previous_angle)
    )

def angle_wrap(delta):
    """
    Wraps angle differences into the range [-π, π].
    This prevents sudden jumps when passing the -π/π boundary.
    """
    delta = (delta + np.pi) % (2 * np.pi) - np.pi
    return delta

def short_wrap_angle_diff(diff):
    """
    Ensures a difference of angles is within [-π, π).
    Prevents 2π jumps from crossing the boundary.
    """
    if diff > np.pi:
        diff -= 2 * np.pi
    elif diff < -np.pi:
        diff += 2 * np.pi
    return diff

def circular_mean(angles):
    """Compute the circular mean of a list of angles in radians."""
    s = 0.0
    c = 0.0
    for a in angles:
        s += np.sin(a)
        c += np.cos(a)
    s /= len(angles)
    c /= len(angles)
    return np.arctan2(s, c)

def circular_distance(a, b):
    # Returns the absolute circular distance between angles a and b
    diff = np.abs(a - b)
    return np.minimum(diff, 2*np.pi - diff)

def circular_median(angles, resolution=300):
    """
    Brute-force approach to approximate the circular median.
    Angles in radians, on [-pi, pi) or [0, 2*pi).
    resolution = number of steps in [0, 2*pi).
    """
    # Normalize angles to [0, 2*pi)
    angles = np.mod(angles, 2*np.pi)

    candidates = np.linspace(0, 2*np.pi, resolution, endpoint=False)
    best_m = None
    best_sum = float('inf')
    for c in candidates:
        total_dist = np.sum(circular_distance(angles, c))
        if total_dist < best_sum:
            best_sum = total_dist
            best_m = c
    
    # Convert best_m back to [-pi, pi)
    best_m = (best_m + np.pi) % (2*np.pi) - np.pi
    return best_m


# For filtering
previous_filtered_angle = 0.0
first_run = True# For tracking cumulative angle
angle_cumulative = 0.0
prev_angle_cumulative = 0.0

current_time = time.time()
previous_timestamp = time.time()

alpha=0.9

async def handle_notification(sender, data):
    """Replicates the original ring parse logic for A1 03 accelerometer subtype."""
    # print("handle_notification triggered with:", data.hex())
    global last_scroll_time  # tell Python this refers to the global variable
    global angle_cumulative, prev_angle_cumulative
    global previous_filtered_angle, first_run
    global previous_timestamp
    global df

    accX_raw = 0
    accY_raw = 0
    accZ_raw = 0

    if len(data) < 8:
        return  # ignore short packets

    if data[0] == 0xA1:
        subtype = data[1]
        if subtype == 0x03:
            # Extract accX, accY, accZ from the correct offsets
            accX_raw = ((data[6] << 4) | (data[7] & 0xF))
            if data[6] & 0x8:
                accX_raw -= (1 << 11)

            accY_raw = ((data[2] << 4) | (data[3] & 0xF))
            if data[2] & 0x8:
                accY_raw -= (1 << 11)

            accZ_raw = ((data[4] << 4) | (data[5] & 0xF))
            if data[4] & 0x8:
                accZ_raw -= (1 << 11)

    if accX_raw == 0 and accY_raw == 0 and accZ_raw == 0:
        return
    # Convert raw values to G-force
    accX = convertrawtoG(accX_raw)
    accY = convertrawtoG(accY_raw)
    accZ = convertrawtoG(accZ_raw)

    # Calculate the angle of the ring
    raw_angle = np.arctan2(accY, accX)

    # if raw G force is less than 1, ignore
    raw_total_Gforce = np.sqrt(accX**2 + accY**2 + accZ**2)
    if raw_total_Gforce > 5:
        return

    # get previous (WINDOW_SIZE) raw angles
    raw_angles_window = df["raw_angle"].values[-WINDOW_SIZE:]
    # filtered_angle = circular_mean(raw_angles_window)
    filtered_angle = circular_median(raw_angles_window)


    # add to dataframe
    new_row = pd.DataFrame([{
        "timestamp": current_time,
        "raw_angle": raw_angle,
        "filtered_angle": filtered_angle,
        "angle_cumulative": 0,
        "accX": accX, 
        "accY": accY, 
        "accZ": accZ
    }])

    df = pd.concat([df, new_row], ignore_index=True)


    # if the difference between the current angle and the angle from 3 frames ago is greater than the dead zone
    previous_filtered_angle = df["filtered_angle"].values[-7]
    if abs(filtered_angle - previous_filtered_angle) > DEAD_ZONE and abs(filtered_angle - previous_filtered_angle) < 2:
        # scroll the screen
        if time.time() - last_scroll_time > DEBOUNCE_INTERVAL:
            # calculate the difference in angle
            angle_diff = short_wrap_angle_diff(filtered_angle - previous_filtered_angle)
            # calculate the number of pixels to scroll
            scroll_amount = int(SCROLL_SCALE * angle_diff)
            if scroll_amount != 0:
                # scroll the screen
                print(f"Scrolling by {scroll_amount} pixels")
                pyautogui.scroll(1)
                # update the last scroll time
                last_scroll_time = time.time()


    # send to queue, set other values to 0
    acc_queue.put((accX, accY, accZ, filtered_angle, 0, 0, 0, raw_angle))









# Commands
def create_command(hex_string):
    bytes_array = [int(hex_string[i:i+2], 16) for i in range(0, len(hex_string), 2)]
    while len(bytes_array) < 15:
        bytes_array.append(0)
    checksum = sum(bytes_array) & 0xFF
    bytes_array.append(checksum)
    return bytes(bytes_array)

async def ble_main(device_address: str, duration: int):
    async with BleakClient(device_address) as client:
        if not client.is_connected:
            print("Failed to connect to the device.")
            return

        print(f"Connected to {device_address}.")

        # 1) Start notifications 
        #    (subscribe to the characteristic that sends accelerometer / raw data)
        await client.start_notify(RXTX_NOTIFY_CHARACTERISTIC_UUID, handle_notification)

        # 2) Send the enable raw sensor command to the *write* characteristic
        #    so that the ring begins streaming data.
        ENABLE_RAW_SENSOR_CMD = create_command("a104")  # or whatever your ring needs
        try:
            await client.write_gatt_char(RXTX_WRITE_CHARACTERISTIC_UUID, ENABLE_RAW_SENSOR_CMD)
            print("Sent ENABLE_RAW_SENSOR_CMD successfully.")
        except Exception as e:
            print(f"Failed to send ENABLE_RAW_SENSOR_CMD: {e}")
            return

        # 3) Optionally sleep a bit to ensure notifications and sensor are ready
        await asyncio.sleep(1.0)

        # 4) Keep the BLE event loop active for the desired duration
        try:
            await asyncio.sleep(duration)
        finally:
            # 5) Clean up: send disable command, close CSV, etc.
            DISABLE_RAW_SENSOR_CMD = create_command("a102")  # if your ring needs it
            await client.write_gatt_char(RXTX_WRITE_CHARACTERISTIC_UUID, DISABLE_RAW_SENSOR_CMD)

            csv_file.close()
            print(f"Data saved to {filename}")

def run_ble_in_background(device_address, duration):
    def runner():
        asyncio.run(ble_main(device_address, duration))
    t = threading.Thread(target=runner, daemon=True)
    t.start()
    return t

def animate_ring():
    # ================ 1) CREATE FIGURE and SIX SUBPLOTS ==================
    fig, ((ax1, ax2), (ax3, ax4), (ax5, ax6)) = plt.subplots(3, 2, figsize=(12, 12), sharex=True)
    fig.tight_layout()

    # ------------------ Subplot 1: rawScrollPos vs Time -------------
    ax1.set_ylim([-np.pi*1.1, np.pi*1.1])
    ax1.set_xlabel("Time (frames)")
    ax1.set_ylabel("rawScrollPos (radians)")
    ax1.set_title("Real-Time Accelerometer + Ring Orientation")

    times = []
    raw_scroll_vals = []

    (line_scroll,) = ax1.plot([], [], 'r-', label="rawScrollPos")
    ax1.legend()

    # ------------------ Subplot 2: accX, accY, accZ vs Time ---------
    ax2.set_xlabel("Time (frames)")
    ax2.set_ylabel("Acceleration")

    accX_vals = []
    accY_vals = []
    accZ_vals = []

    (line_accX,) = ax2.plot([], [], 'b-', label="accX")
    (line_accY,) = ax2.plot([], [], 'g-', label="accY")
    (line_accZ,) = ax2.plot([], [], 'k-', label="accZ")
    ax2.legend()

    # ------------------ Subplot 3: raw_total_Gforce vs Time ---------
    ax3.set_xlabel("Time (frames)")
    ax3.set_ylabel("raw_total_Gforce")

    raw_total_Gforce_vals = []

    (line_raw_total_Gforce,) = ax3.plot([], [], 'm-', label="raw_total_Gforce")
    ax3.legend()

    # ------------------ Subplot 4: angular_velocity vs Time ---------
    ax4.set_xlabel("Time (frames)")
    ax4.set_ylabel("angular_velocity")

    angular_velocity_vals = []

    (line_angular_velocity,) = ax4.plot([], [], 'c-', label="angular_velocity")
    ax4.legend()

    # ------------------ Subplot 5: raw_angle vs Time ---------
    ax5.set_xlabel("Time (frames)")
    ax5.set_ylabel("raw_angle")

    raw_angle_vals = []

    (line_raw_angle,) = ax5.plot([], [], 'y-', label="raw_angle")
    ax5.legend()

    # ------------------ Subplot 6: filtered_angle vs Time ---------
    ax6.set_xlabel("Time (frames)")
    ax6.set_ylabel("filtered_angle")

    filtered_angle_vals = []

    (line_filtered_angle,) = ax6.plot([], [], 'k-', label="filtered_angle")
    ax6.legend()

    # ================ 3) DEFINE INIT FUNCTION ==================
    def init():
        """Initialize the plot with no data."""
        line_scroll.set_data([], [])
        line_accX.set_data([], [])
        line_accY.set_data([], [])
        line_accZ.set_data([], [])
        line_raw_total_Gforce.set_data([], [])
        line_angular_velocity.set_data([], [])
        line_raw_angle.set_data([], [])
        line_filtered_angle.set_data([], [])
        return (line_scroll, line_accX, line_accY, line_accZ, 
                line_raw_total_Gforce, line_angular_velocity, 
                line_raw_angle, line_filtered_angle)

    # ================ 4) DEFINE UPDATE FUNCTION ==================
    def update(frame):
        """
        1. If there's new data in the queue, retrieve it.
        2. Unpack accX, accY, accZ, angle_cumulative, raw_total_Gforce, angular_velocity, raw_angle, filtered_angle.
        3. Append to lists and update lines.
        4. Dynamically adjust axes if desired.
        """
        if not acc_queue.empty():
            # print(acc_queue.get_nowait())
            accX, accY, accZ, filtered_angle, angle_cumulative,angular_velocity,raw_total_Gforce,raw_angle = acc_queue.get_nowait()

            # Update lists
            times.append(frame)
            raw_scroll_vals.append(angle_cumulative)
            accX_vals.append(accX)
            accY_vals.append(accY)
            accZ_vals.append(accZ)
            raw_total_Gforce_vals.append(raw_total_Gforce)
            angular_velocity_vals.append(angular_velocity)
            raw_angle_vals.append(raw_angle)
            filtered_angle_vals.append(filtered_angle)

            # Update line data on ax1
            line_scroll.set_data(times, raw_scroll_vals)

            # Update lines on ax2
            line_accX.set_data(times, accX_vals)
            line_accY.set_data(times, accY_vals)
            line_accZ.set_data(times, accZ_vals)

            # Update lines on ax3
            line_raw_total_Gforce.set_data(times, raw_total_Gforce_vals)

            # Update lines on ax4
            line_angular_velocity.set_data(times, angular_velocity_vals)

            # Update lines on ax5
            line_raw_angle.set_data(times, raw_angle_vals)

            # Update lines on ax6
            line_filtered_angle.set_data(times, filtered_angle_vals)

            # Optionally rescale x-axis dynamically
            ax1.set_xlim(0, max(10, frame + 1))
            ax2.set_xlim(0, max(10, frame + 1))
            ax3.set_xlim(0, max(10, frame + 1))
            ax4.set_xlim(0, max(10, frame + 1))
            ax5.set_xlim(0, max(10, frame + 1))
            ax6.set_xlim(0, max(10, frame + 1))

            # Optionally rescale y-limits for ax1
            y_min, y_max = min(raw_scroll_vals), max(raw_scroll_vals)
            ax1.set_ylim(y_min - 0.1, y_max + 0.1)

            # Optionally rescale y-limits for ax2
            all_acc_vals = accX_vals + accY_vals + accZ_vals
            min_acc, max_acc = min(all_acc_vals), max(all_acc_vals)
            ax2.set_ylim(min_acc - 0.1, max_acc + 0.1)

            # Optionally rescale y-limits for ax3
            min_raw_total_Gforce, max_raw_total_Gforce = min(raw_total_Gforce_vals), max(raw_total_Gforce_vals)
            ax3.set_ylim(min_raw_total_Gforce - 0.1, max_raw_total_Gforce + 0.1)

            # Optionally rescale y-limits for ax4
            min_angular_velocity, max_angular_velocity = min(angular_velocity_vals), max(angular_velocity_vals)
            ax4.set_ylim(min_angular_velocity - 0.1, max_angular_velocity + 0.1)

            # Optionally rescale y-limits for ax5
            min_raw_angle, max_raw_angle = min(raw_angle_vals), max(raw_angle_vals)
            # ax5.set_ylim(min_raw_angle - 0.1, max_raw_angle + 0.1)
            ax5.set_ylim(-np.pi, np.pi)

            # Optionally rescale y-limits for ax6
            min_filtered_angle, max_filtered_angle = min(filtered_angle_vals), max(filtered_angle_vals)
            # ax6.set_ylim(min_filtered_angle - 0.1, max_filtered_angle + 0.1)
            ax6.set_ylim(-np.pi, np.pi)

        return (line_scroll, line_accX, line_accY, line_accZ, 
                line_raw_total_Gforce, line_angular_velocity, 
                line_raw_angle, line_filtered_angle)

    # ================ 5) CREATE ANIMATION ==================
    ani = animation.FuncAnimation(
        fig,
        func=update,
        init_func=init,
        frames=200000,   # or choose a suitable limit
        interval=10,     # ms between updates
        blit=False
    )

    # ================ 6) SHOW PLOT ==================
    plt.show()

def main():
    device_address = "9442FF70-584D-628B-C317-2D0FF502F76D"  # Replace with your ring's address
    duration = 200

    # Start BLE in background
    ble_thread = run_ble_in_background(device_address, duration)
    # Animate in main thread
    animate_ring()

    # Once the user closes the plot, we can optionally check if BLE is still running
    ble_thread.join(timeout=0)

    print(df["filtered_angle"].values)
    print("Exiting main.")

if __name__ == "__main__":
    main()