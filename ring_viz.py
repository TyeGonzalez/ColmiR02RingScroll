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

def convertrawtoG(raw):
    return (raw / 2048.0) * 4

# rotateX = Math.atan2(Ax, Math.sqrt(Ay * Ay + Az * Az));


async def handle_notification(sender, data):
    """Replicates the original ring parse logic for A1 03 accelerometer subtype."""
    # print("handle_notification triggered with:", data.hex())

    if len(data) < 8:
        return  # ignore short packets

    if data[0] == 0xA1:
        subtype = data[1]
        if subtype == 0x03:
            # Extract accX, accY, accZ from the correct offsets
            accX = ((data[6] << 4) | (data[7] & 0xF))
            if data[6] & 0x8:
                accX -= (1 << 11)

            accY = ((data[2] << 4) | (data[3] & 0xF))
            if data[2] & 0x8:
                accY -= (1 << 11)

            accZ = ((data[4] << 4) | (data[5] & 0xF))
            if data[4] & 0x8:
                accZ -= (1 << 11)

            print(f"AccX: {accX}, AccY: {accY}, AccZ: {accZ}")
            # CSV
            timestamp = datetime.now().isoformat()
            csv_writer.writerow([timestamp, accX, accY, accZ])
            # For live animation
            acc_queue.put((accX, accY, accZ))

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

def rotation_matrix(roll, pitch, yaw):
    """
    roll  = rotation about X-axis
    pitch = rotation about Y-axis
    yaw   = rotation about Z-axis
    """
    Rx = np.array([[1,           0,            0],
                   [0, np.cos(roll), -np.sin(roll)],
                   [0, np.sin(roll),  np.cos(roll)]])
    
    Ry = np.array([[ np.cos(pitch), 0, np.sin(pitch)],
                   [0,              1,              0],
                   [-np.sin(pitch), 0, np.cos(pitch)]])
    
    Rz = np.array([[np.cos(yaw), -np.sin(yaw), 0],
                   [np.sin(yaw),  np.cos(yaw), 0],
                   [         0,           0,   1]])
    
    # Common choice: Rz * Ry * Rx
    return Rz @ Ry @ Rx


def animate_ring():
    # ================ 1) Create FIGURE and AXES ==================
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection='3d')
    # ax.set_xlim([-2048, 4000])
    # ax.set_ylim([-2048, 4000])
    # ax.set_zlim([-2048, 4000])
    ax.set_xlim([-4, 4])
    ax.set_ylim([-4, 4])
    ax.set_zlim([-4, 4])
    ax.set_xlabel("AccX")
    ax.set_ylabel("AccY")
    ax.set_zlabel("AccZ")
    ax.set_title("Real-Time Accelerometer + Ring Orientation")

    # ================ 2) DEFINE RING GEOMETRY ==================
    R_inner = 1.0
    R_outer = 2.0
    height = 1.0

    # Resolution for mesh
    n_theta = 30
    n_z = 30

    # Create a grid of theta and z
    theta_vals = np.linspace(0, 2 * np.pi, n_theta)
    z_vals = np.linspace(0, height, n_z)
    Theta, Z = np.meshgrid(theta_vals, z_vals)

    # Outer cylinder (simple example ignoring R_inner)
    X_outer = R_outer * np.cos(Theta)
    Y_outer = R_outer * np.sin(Theta)
    Z_outer = Z

    # Flatten ring geometry into (N, 3) for easy rotation
    points_outer = np.vstack([
        X_outer.ravel(),
        Y_outer.ravel(),
        Z_outer.ravel()
    ]).T  # shape (N, 3)

    # We'll store references so we can update them
    ring_surf = [None]  # use list so we can modify inside update
    acc_point = [None]  # likewise for the scatter/point

    # ================ 3) DEFINE INIT FUNCTION ==================
    def init():
        """Initialize the plot with some default orientation."""
        # 3a) Default: no rotation
        # Flatten and rotate by 0,0,0
        R = rotation_matrix(0, 0, 0)
        rotated_outer = points_outer @ R.T
        # Reshape back
        X_rot = rotated_outer[:, 0].reshape(n_z, n_theta)
        Y_rot = rotated_outer[:, 1].reshape(n_z, n_theta)
        Z_rot = rotated_outer[:, 2].reshape(n_z, n_theta)

        # 3b) Create the surface object
        # We store the created Poly3DCollection in ring_surf[0]
        ring_surf[0] = ax.plot_surface(
            X_rot, Y_rot, Z_rot,
            color='blue', alpha=0.5, edgecolor='none'
        )

        # 3c) Create a scatter (point) to show accelerometer reading
        #   We'll start it at (0,0,0).
        acc_point[0] = ax.plot(
            [0], [0], [0], 'o', color='red', markersize=6
        )[0]  # [0] to get the actual line object

        return ring_surf[0], acc_point[0]

    # ================ 4) DEFINE UPDATE FUNCTION ==================
    def update(frame):
        """Update ring orientation + accelerometer point each frame."""
        # 4a) If new data available, pull from queue:
        if not acc_queue.empty():
            accX, accY, accZ = acc_queue.get_nowait()
        # else:
        #     accX, accY, accZ = 0, 0, 0

        # 4b) Convert raw accel to 'g' units
        Ax = convertrawtoG(accX)
        Ay = convertrawtoG(accY)
        Az = convertrawtoG(accZ)

        # 4c) Compute rotation angles from accelerometer
        rotateX = np.arctan2(Ax, np.sqrt(Ay * Ay + Az * Az))
        rotateY = np.arctan2(Ay, np.sqrt(Ax * Ax + Az * Az))
        rotateZ = np.arctan2(Az, np.sqrt(Ax * Ax + Ay * Ay))

        # 4d) Build rotation matrix & rotate the ring geometry
        R = rotation_matrix(rotateX, rotateY, rotateZ)
        rotated_outer = points_outer @ R.T

        X_rot = rotated_outer[:, 0].reshape(n_z, n_theta)
        Y_rot = rotated_outer[:, 1].reshape(n_z, n_theta)
        Z_rot = rotated_outer[:, 2].reshape(n_z, n_theta)

        # 4e) Remove the old surface and plot the new one
        ring_surf[0].remove()  # remove old surface from axes
        ring_surf[0] = ax.plot_surface(
            X_rot, Y_rot, Z_rot,
            color='blue', alpha=0.5, edgecolor='none'
        )

        # 4f) Update the accelerometer point
        acc_point[0].set_data([accX], [accY])
        acc_point[0].set_3d_properties([accZ])

        return ring_surf[0], acc_point[0]

    # ================ 5) CREATE ANIMATION ==================
    ani = animation.FuncAnimation(
        fig,               # figure to animate
        func=update,       # update function
        init_func=init,    # init function
        frames=None,       # or some integer if you want a fixed number
        interval=50,       # ms between updates
        blit=False         # blit=True often causes issues with 3D
    )

    # ================ 6) SHOW PLOT ==================
    plt.show()

def main():
    device_address = "9442FF70-584D-628B-C317-2D0FF502F76D"  # Replace with your ring's address
    duration = 60

    # Start BLE in background
    ble_thread = run_ble_in_background(device_address, duration)
    # Animate in main thread
    animate_ring()

    # Once the user closes the plot, we can optionally check if BLE is still running
    ble_thread.join(timeout=0)
    print("Exiting main.")

if __name__ == "__main__":
    main()