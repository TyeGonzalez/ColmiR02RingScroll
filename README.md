# ColmiR02RingScroll

This repository provides multiple Python applications to work with the Colmi R02 Smart Ring. The code is inspired by includes real-time data acquisition, visualization, and scrolling capabilities based on accelerometer measurements from the ring. Inspiration taken from [this repository](https://github.com/edgeimpulse/example-data-collection-colmi-r02) (python implementation) and [this repository](https://github.com/atc1441/ATC_RF03_Ring) (custom firmware for faster data acquisition).

## Project Structure
- **ring_scrollv2.py**  
  Uses BLE notifications to capture accelerometer data and translates tilt into scrolling actions.  Automatically shows a live visualization of current input data.
  See the implementation in [ring_scrollv2.py](ring_scrollv2.py).

- **ring_scrollv2_slowscroll.py**  
  Alternative implementation of the scrolling application with a slower scroll rate.

- **ring_viz.py**  
  Connects to the ring via BLE, logs accelerometer data to a CSV, and provides a 3D live visualization of the ring's orientation.

## Notes
- **BLE Device Address**
The BLE device address is hard-coded in the examples (e.g., "9442FF70-584D-628B-C317-2D0FF502F76D"). Update this to match your device's address if needed.

- **Data Storage**
CSV files are saved in the raw_data/ directory. This directory is automatically created if it does not exist.

- **Customization**
Adjust settings such as SCROLL_SCALE, DEAD_ZONE, and DEBOUNCE_INTERVAL in ring_scrollv2.py to suit your hardware and desired responsiveness.