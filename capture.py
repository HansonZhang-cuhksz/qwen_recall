import pyautogui
import datetime
import time
import os

from utils import memory_dir
from image_diff import get_similarity

import shared

period = 15
sig = False
timestamps = []

file_count = 0

def take_screenshot(save_path=None):
    global memory_dir
    if save_path is None:
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        save_path = memory_dir / "images" / f"{timestamp}.png"
    screenshot = pyautogui.screenshot()
    screenshot.save(save_path)
    return timestamp

def capture_task(image_diff_driver):
    global file_count, memory_dir, period, sig, timestamps
    
    last_timestamp = None
    while True:
        start_time = time.time()

        timestamp = take_screenshot()
        if last_timestamp:
            last_pth = memory_dir / "images" / f"{last_timestamp}.png"
            this_pth = memory_dir / "images" / f"{timestamp}.png"
            similarity = get_similarity(last_pth, this_pth, image_diff_driver)
            if similarity > 0.99:
                # print("Passing on similar image:", similarity)
                os.remove(this_pth)
                continue

        timestamps.append(timestamp)
        shared.capture_to_decode_1 = timestamp

        if file_count > 100:
            os.remove(memory_dir / "images" / f"{timestamps[0]}.png")
            timestamps.pop(0)
        else:
            file_count += 1

        last_timestamp = timestamp

        period = shared.capture_to_decode_2 + 1 if shared.capture_to_decode_2 else period
        print(f"{time.time() - start_time} / {period}")
        time.sleep(max(0, period - (time.time() - start_time)))