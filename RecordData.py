import torch 
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import classification_report
import AudioUtils as au
import config
import sounddevice as sd
from collections import deque
import time
import threading
import os

class_name = input("Title the audio class you are trying to record:\n")
file_counter = 0

new_directory = os.path.join("data", class_name)

try:
    os.makedirs(new_directory, exist_ok=True)
    print(f"Directory '{new_directory}' created or already exists.")
except OSError as e:
    print(f"Error creating directory: {e}")

for item in os.listdir(new_directory):
    item_path = os.path.join(new_directory, item)
    if os.path.isfile(item_path):
        file_counter += 1

print("Available audio devices:")
print(sd.query_devices())

# Setup the Audio Buffer and Stream 
audio_buffer = deque(maxlen=config.SAMPLE_LENGTH)
buffer_lock = threading.Lock()
shutdown_event = threading.Event()

def audio_callback(indata, frames, time, status):
    """This function is called by sounddevice for each new audio chunk."""
    if shutdown_event.is_set():
        return
    if status:
        print(status)
    with buffer_lock:
        audio_buffer.extend(indata[:, 0])

# Main Inference Loop 
try:
    print("\nStarting data collection... Press Ctrl+C to stop.")
    # Create and start the microphone stream
    stream = sd.InputStream(
        device=config.DEVICE_ID,
        channels=1,
        samplerate=config.SAMPLE_RATE,
        callback=audio_callback
    )

    stream.start()

    while not shutdown_event.is_set():
        # Wait until the buffer has enough data for a full clip
        clip = None

        # fetch the data using the buffer lock
        with buffer_lock:
            if len(audio_buffer) == config.SAMPLE_LENGTH:
                clip = torch.tensor(list(audio_buffer))
                audio_buffer.clear()

        # process the data separately so the stream doesn't wait
        if clip is not None:
            file_path = os.path.join(new_directory, f"{class_name}_example_{file_counter}.pt")
            torch.save(clip, file_path)
            print(f"saved clip: {file_path}")
            file_counter += 1
        else:
            # Wait for more audio data to fill the buffer
            print(f"Buffering... {len(audio_buffer)}/{config.SAMPLE_LENGTH}", end='\r')
            time.sleep(0.1)

except KeyboardInterrupt:
    print("\nStopping inference.")    
except Exception as e:
    print(f"An error occurred: {e}")

finally:
    print("stopping")
    shutdown_event.set()
    print("more stopping")
    if stream:
        print("more more stopping")
        stream.stop()
        stream.close()
    print("stopped")