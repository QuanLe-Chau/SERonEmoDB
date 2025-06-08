import pandas as pd
import numpy as np
import librosa
import os
import seaborn as sns
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.patches as patches
import soundfile as sf

def draw_class_distribution(dataframe, title, source, figsize = (12,4), color_palette = 'crest'):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
    colors = sns.color_palette(color_palette)

    class_count = dataframe['label'].value_counts().to_dict()

    ax1.bar(class_count.keys(), class_count.values(), color=colors)
    ax1.tick_params(axis='x', labelrotation=45)
    ax1.grid(True)  
    # ax1.text(0.5, -0.4, 'Count Plot of the Distribution', 
    #         transform=ax1.transAxes, ha='center', fontsize=10)


    ax2.pie(class_count.values(), labels=class_count.keys(), autopct='%1.1f%%', colors=colors)
    ax2.grid(True)
    # ax2.text(0.5, -0.4, 'Pie Chart of the Distribution', 
    #         transform=ax2.transAxes, ha='center', fontsize=10)

    fig.suptitle(title, fontweight='bold',  ha='center')
    fig.text(0.5, 0.9, source, ha='center', fontsize=10)

    plt.tight_layout(rect=[0.2, 0, 0.85, 0.95])
    plt.show()

def draw_duration_distribution(dataframe, source, title = 'Audio Duration Distribution per Emotion Class', figsize = (12.5, 6), color_palette = 'crest'):

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize= figsize)
    colors = sns.color_palette(color_palette)

    sns.histplot(dataframe['duration'], bins=40, kde=True, ax=ax1, edgecolor='black', color=colors[5])
    ax1.set_xlabel('Duration')
    ax1.grid(True)  

    sns.boxplot(data=dataframe, x='duration', y='label', hue='label', palette=color_palette, legend=False, vert=False, ax=ax2)
    ax2.set_xlabel("Duration (seconds)")
    ax2.set_ylabel("Emotion")
    ax2.grid(True, linestyle='--', alpha=0.6)

    fig.suptitle(title, fontweight='bold',  ha='center')
    fig.text(0.5, 0.92, source, ha='center', fontsize=10)

    plt.tight_layout(rect=[0.2, 0, 0.78, 0.95])
    plt.show()

    plt.tight_layout()
    plt.show()

def draw_single_waveform(path = None, y = None):

    if path != None:
        y, sr = librosa.load(path, sr=None)
    else: 
        sr = 16000

    plt.figure(figsize=(12, 3))
    librosa.display.waveshow(y, sr=sr)
    plt.title("Waveform")
    plt.xlabel("Time (s)")
    plt.ylabel("Amplitude")
    plt.grid(True)
    plt.show()

def draw_single_spectrogram(path=None, y = None):

    if path != None:
        y, sr = librosa.load(path, sr=None)
    else:
        sr = 16000 
        
    D= librosa.amplitude_to_db(np.abs(librosa.stft(y)), ref=np.max)

    plt.figure(figsize=(10, 4))
    librosa.display.specshow(D, sr=sr, x_axis='time', y_axis='log')
    plt.title('Spectrogram')
    plt.colorbar(format='%+2.0f dB')
    plt.show()

def draw_peak_amplitude_distribution(df):

    # Compute peak amplitude for each audio signal
    peak_amplitudes = df['wave'].apply(lambda x: np.max(np.abs(x)))

    # Plot histogram
    plt.figure(figsize=(7, 4))
    plt.hist(peak_amplitudes, bins=50, color='skyblue', edgecolor='black')
    plt.title("Peak Amplitude Distribution Across Dataset")
    plt.xlabel("Peak Amplitude")
    plt.ylabel("Number of Samples")
    plt.grid(True)
    plt.show()

def load_dataset_emoDB(dataset_folder_path = "EmoDB\wav", add_path = True, add_wave = False):

    emotion_map_emoDB = {
    'W': 'angry',      # Wut
    'L': 'bored',    # Langeweile
    'E': 'disgust',    # Ekel
    'A': 'fearful',       # Angst
    'F': 'happy',  # Freude
    'T': 'sad',    # Traurigkeit
    'N': 'neutral'     # Neutral
    }

    audio_data = []

    # Loop through all WAV files
    for file in os.listdir(dataset_folder_path):
        if file.endswith(".wav"):
            path = os.path.join(dataset_folder_path, file)
            try:
                sample = dict()
                y, sr = librosa.load(path, sr=None)
                duration = librosa.get_duration(y=y, sr=sr)
                emotion_code = file[5]  # 6th character in filename
                gender_code = file[0:2]
                # gender = 'male' if gender_code in ['03', '10', '11', '12', '15'] else 'female'
                emotion = emotion_map_emoDB.get(emotion_code, "unknown")

                sample['label'] = emotion
                if add_wave:
                    sample['wave'] = y
                sample['sample_rate'] = sr
                sample['duration'] = duration
                sample['channel'] = 'mono' if y.ndim == 1 else 'multi-dimension'
                if add_path == True:
                    sample['path'] = path

            except Exception as e:
                print(f"Error processing {file}: {e}")

            audio_data.append(sample)

    # Create DataFrame
    df = pd.DataFrame(audio_data)
    return df

def load_dataset_RAVDESS(dataset_folder_path = "RAVDESS", add_path = True, add_wave = False):
    emotion_map_RAVDESS = {
    "01": 'neutral', 
    "02": 'calm', 
    "03": 'happy',
    "04": 'sad',
    "05": 'angry', 
    "06": "fearful", 
    "07": 'disgust', 
    "08": 'surprised'   
    }

    # Initialize data storage
    audio_data = []
        
    for folder in os.listdir(dataset_folder_path):
        sub_folder = os.path.join(dataset_folder_path, folder)
        for file in os.listdir(sub_folder):
            file_path = os.path.join(sub_folder, file)

            try:
                sample = dict()
                y, sr = librosa.load(file_path, sr=None)
                duration = librosa.get_duration(y=y, sr=sr)
                emotion_code = file[6:8]  
                gender_code = file[18:20]
                # gender = 'female' if (int(gender_code) % 2 == 0) else 'male'
                emotion = emotion_map_RAVDESS.get(emotion_code, "unknown")
                sample['label'] = emotion
                if add_wave:
                    sample['wave'] = y
                sample['sample_rate'] = sr
                sample['duration'] = duration
                sample['channel'] = 'mono' if y.ndim == 1 else 'multi-dimension'
                if add_path == True:
                    sample['path'] = file_path

            except Exception as e:
                print(f"Error processing {file}: {e}")
            
            audio_data.append(sample)

    # Create DataFrame
    df = pd.DataFrame(audio_data)
    return df

def load_dataset(dataset_folder_path = '', add_path = True):

    audio_data = []
        
    for folder in os.listdir(dataset_folder_path):
        sub_folder = os.path.join(dataset_folder_path, folder)
        for file in os.listdir(sub_folder):
            file_path = os.path.join(sub_folder, file)
            try:
                sample = dict()
                y, sr = librosa.load(file_path, sr=None)
                duration = librosa.get_duration(y=y, sr=sr)
                sample['label'] = folder
                sample['wave'] = y
                sample['sample_rate'] = sr
                sample['duration'] = duration
                sample['channel'] = 'mono' if y.ndim == 1 else 'multi-dimension'
                if add_path == True:
                    sample['path'] = file_path
                    
            except Exception as e:
                print(f"Error processing {file}: {e}")
            
            audio_data.append(sample)

    # Create DataFrame
    df = pd.DataFrame(audio_data)
    return df

def copy_folders(dataframe, output_base="preprocessed_data"):
    """
    Save preprocessed audio data (waveform, duration, etc.) into structured folders by label.
    Each file will be re-written from waveform to ensure consistency with preprocessing.
    """

    os.makedirs(output_base, exist_ok=True)  # Ensure base directory exists

    output_paths = []

    for label in dataframe['label'].unique():
        label_df = dataframe[dataframe['label'] == label].reset_index(drop=True)
        label_folder = os.path.join(output_base, label)
        os.makedirs(label_folder, exist_ok=True)

        for i, row in label_df.iterrows():
            new_filename = f"{label}_{i+1:03d}.wav"
            output_path = os.path.join(label_folder, new_filename)
            
            # Write waveform to file to preserve preprocessing (not copying original file)
            y = row['wave']
            sr = row.get('sr', 16000)  # Default to 16000 if sample rate not present
            sf.write(output_path, y, sr)

            output_paths.append(output_path)

    # Assign new paths to the original DataFrame
    dataframe = dataframe.copy()
    dataframe['output_path'] = output_paths

    print(f"All preprocessed files saved in: {output_base}/")
    return dataframe
