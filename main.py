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
import traceback
import os

# load and organize data
feat, label = au.load_data("data/")
feat_train, feat_test, label_train, label_test = train_test_split(
    feat, label, test_size=0.2, random_state=42, stratify=label
)

# scale data for SVM
scaler = StandardScaler()
feat_train_scaled = scaler.fit_transform(feat_train)
feat_test_scaled = scaler.transform(feat_test)

# train model
myModel = SVC(kernel='rbf', C=10, gamma='scale', probability=True)
myModel.fit(feat_train_scaled, label_train)

# evaluate model
predictions = myModel.predict(feat_test_scaled)
print(classification_report(label_test, predictions))

print("Available audio devices:")
print(sd.query_devices())

# Setup the Audio Buffer and Stream 
audio_buffer = deque(maxlen=config.SAMPLE_LENGTH)
buffer_lock = threading.Lock()

def inference_worker():
    try: 
        def audio_callback(indata, frames, time, status):
            """This function is called by sounddevice for each new audio chunk."""
            if status:
                print(status)
            with buffer_lock:
                audio_buffer.extend(indata[:, 0])

        # Main Inference Loop 
        # Create and start the microphone stream
        stream = sd.InputStream(
            device=config.DEVICE_ID,
            channels=1,
            samplerate=config.SAMPLE_RATE,
            callback=audio_callback
        )

        stream.start()

        while True:
            # Wait until the buffer has enough data for a full clip
            clip = None

            # fetch the data using the buffer lock
            with buffer_lock:
                if len(audio_buffer) == config.SAMPLE_LENGTH:
                    clip = torch.tensor(list(audio_buffer))

            # process the data separately so the stream doesn't wait
            if clip is not None:
                # Process the clip
                data = au.AudioUtil.MFCC((clip.unsqueeze(0), config.SAMPLE_RATE))
                flat_data = torch.flatten(data)
                scaled_data = scaler.transform(flat_data.unsqueeze(0).numpy())
                
                output = myModel.predict(scaled_data)
                
                # Print the prediction
                print(au.idx_to_class[output.item()])
                
                # Wait for the stride duration before processing the next clip
                time.sleep(config.INFERENCE_STRIDE_SECONDS)
            else:
                # Wait for more audio data to fill the buffer
                print(f"Buffering... {len(audio_buffer)}/{config.SAMPLE_LENGTH}", end='\r')
                time.sleep(0.1)

    except Exception as e:
        print(f"Error in recording worker: {e}", flush=True)  
        print(traceback.format_exc())

    finally:
        if stream:
            stream.stop()
            stream.close()
        print("Recording worker finished.", flush=True)

# Create the worker thread
worker = threading.Thread(target=inference_worker)
# Set it as a daemon thread
worker.daemon = True
# Start it
worker.start()

print("Inference started. Interrupt the kernel to stop.", flush=True)

try:
    # This loop does nothing but keep the main script alive.
    # When you interrupt the kernel, this is the loop that will break.
    while True:
        time.sleep(1)
except KeyboardInterrupt:
    # Since the worker is a daemon, we don't need to do any complex cleanup.
    # The program will simply exit, and the worker will be terminated.
    print("\nInterrupt received. Exiting program.", flush=True)
    os._exit(0)