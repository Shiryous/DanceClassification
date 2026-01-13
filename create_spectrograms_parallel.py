import os
import time

# Parallelization imports
import threading
import queue

import torchaudio
import torchaudio.transforms as T
import numpy as np
from PIL import Image # Pillow library for direct image saving

## Parallelization console
GLOBAL_QUEUE = queue.Queue()
NUM_WORKERS = 16

## Folder Structure
DATASET_FOLDER = 'data/BallroomData'
AUDIO_FOLDER_PATH = f'{DATASET_FOLDER}/audio_files'
SPECTROGRAM_FOLDER_PATH = f'{DATASET_FOLDER}/spectrograms'
ALL_BALLROOM_FILES = f'{DATASET_FOLDER}/allBallroomFiles'

def main():
    start = time.time()
    try:
        with open(ALL_BALLROOM_FILES, 'r') as file:
            audio_paths = [line.strip() for line in file]
    except FileNotFoundError:
        print(f"Error: The file '{ALL_BALLROOM_FILES}' was not found.")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")

    for audio_path in audio_paths:
        spectrogram_path = audio_path.replace(AUDIO_FOLDER_PATH, SPECTROGRAM_FOLDER_PATH).replace('.wav', '.png')
        GLOBAL_QUEUE.put([audio_path, spectrogram_path])
    
    workers = list()
    for i in range(NUM_WORKERS):
        t = threading.Thread(target=worker, name=f"Worker {i+1}")
        t.daemon = True
        t.start()
        workers.append(t)

    for _ in range(NUM_WORKERS):
        GLOBAL_QUEUE.put(None)

    # Wait for the workers
    for t in workers:
        t.join()
    
    print("The images contain only the raw, normalized spectrogram data.")
    print(f'The execution to convert to spectrograms took {(time.time() - start)} seconds')
    print("Program finished")

def process_task(payload):

    thread_name = threading.current_thread().name
    start = time.time()

    # Here is going to be the code to create the spectrograms
    create_raw_mel_spectrogram_image(payload[0], payload[1])

    work_time = time.time() - start
    print(f"[{thread_name}] Completed Task {payload}. (Duration: {work_time:.2f}s)")

def worker():
    thread_name = threading.current_thread().name
    print(f"--- Worker {thread_name} started.")

    while True:
        try:
            payload = GLOBAL_QUEUE.get(block=True)

            if payload is None:
                print(f"[ {thread_name} ] received exit signal.")
                break

            # Execute the function with the task input
            process_task(payload)
            GLOBAL_QUEUE.task_done()

        except GLOBAL_QUEUE.Empty:
            # This branch is usually hit only if a timeout is set
            print(f"[{thread_name}] Queue empty, retrying...")
            continue
        except Exception as e:
            print(f"[{thread_name}] An error occurred: {e}")
            break

def create_raw_mel_spectrogram_image(audio_path, image_path, n_fft=1024, hop_length=512, n_mels=128, target_sr=22050):
    """
    Loads an audio file, generates its Mel spectrogram, standardizes it, 
    and saves the raw spectrogram data as a grayscale image for neural network training.
    
    Args:
        audio_path (str): Path to the input WAV file.
        image_path (str): Path to save the output image (e.g., 'mel_spectrogram.png').
        n_fft (int):
        hop_length (int):
        n_mels (int): 
        target_sr (int): The sample rate to resample the audio to.
    """
    print(f"Loading and processing audio from: {audio_path}")
    
    # 1. Load and Resample the Audio
    try:
        waveform, sr = torchaudio.load(audio_path)
    except Exception as e:
        print(f"Error loading audio file: {e}")
        return

    # Convert to mono (if necessary)
    if waveform.dim() > 1 and waveform.size(0) > 1:
        waveform = waveform.mean(dim=0, keepdim=True)
    
    # Resample to the target sample rate
    if sr != target_sr:
        resampler = T.Resample(orig_freq=sr, new_freq=target_sr)
        waveform = resampler(waveform)
        sr = target_sr
    
    # 2. Define and Apply the Mel Spectrogram Transform
    mel_spectrogram_transform = T.MelSpectrogram(
        sample_rate=sr,
        n_fft=n_fft,
        hop_length=hop_length,
        n_mels=n_mels
    )
    
    mel_spectrogram = mel_spectrogram_transform(waveform)
    
    # 3. Convert Power/Amplitude to Decibels (Log Scale)
    db_transform = T.AmplitudeToDB(top_db=80.0) # top_db helps with dynamic range normalization
    mel_spectrogram_db = db_transform(mel_spectrogram)
    
    # 4. Standardize the Spectrogram Data
    # For neural networks, the data needs to be normalized or standardized. 
    # A common approach is to scale the dB values to 0-1 or 0-255.
    
    # Convert tensor to numpy array and remove the channel dimension
    spectrogram_np = mel_spectrogram_db.squeeze(0).numpy()
    
    # Min-Max Normalization: Scale values to the range [0, 1]
    # This is crucial for consistent input to the neural network
    min_val = spectrogram_np.min()
    max_val = spectrogram_np.max()
    
    if max_val > min_val:
        normalized_spectrogram = (spectrogram_np - min_val) / (max_val - min_val)
    else:
        # Handle case where all values are the same
        normalized_spectrogram = np.zeros_like(spectrogram_np)
        
    # Scale to 0-255 range and convert to unsigned 8-bit integer (standard image format)
    image_data_255 = (normalized_spectrogram * 255).astype(np.uint8)
    
    # 5. Save as Image using PIL (Pillow)
    # Using 'L' mode saves it as a single-channel (grayscale) image
    
    image_data = Image.fromarray(image_data_255, mode='L')
    
    # Resize: This step is optional but often required to fix the input size for ML models
    # Example: Resize to 256x256 (Height x Width)
    # image_data = image_data.resize((256, 256)) 
        # 2. Check if the directory is specified (not just a filename) and doesn't exist
    image_directory = os.path.dirname(image_path)
    if image_directory and not os.path.exists(image_directory):
        print(f"Directory '{image_directory}' does not exist. Creating it now...")
        
        # os.makedirs creates all necessary intermediate directories
        # exist_ok=True prevents an error if the directory already exists (or if another
        # thread/process creates it between the check and the create call).
        try:
            os.makedirs(image_directory, exist_ok=True)
            print("Directory created successfully.")
        except Exception as e:
            print(f"Error creating directory: {e}")
            return # Stop execution if folder creation fails

    # 3. Save the image
    try:
        image_data.save(image_path)   
        print(f"Successfully generated Mel spectrogram (shape: {image_data_255.shape}) and saved to: {image_path}")
    except Exception as e:
        print(f"Error saving image: {e}")

if __name__ == "__main__":
    main()