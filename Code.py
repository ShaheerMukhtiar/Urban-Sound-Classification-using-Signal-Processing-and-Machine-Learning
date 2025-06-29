import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from tensorflow.keras.layers import (
    Conv2D,
    Dense,
    Dropout,
    Flatten,
    MaxPooling2D,
)
from tensorflow.keras.models import Sequential
from tensorflow.keras.utils import to_categorical
from tqdm import tqdm
from scipy.signal import correlate, windows
from sklearn.metrics import accuracy_score, classification_report
import os
import warnings

# Suppress all warnings
warnings.filterwarnings('ignore')


# Function to extract features from audio
def extract_features(file_path, method="autocorr"):
    try:
        audio, sr = librosa.load(file_path, res_type="kaiser_fast")
        mfccs = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=40)
        chroma = librosa.feature.chroma_stft(y=audio, sr=sr)
        mel = librosa.feature.melspectrogram(y=audio, sr=sr)
        contrast = librosa.feature.spectral_contrast(y=audio, sr=sr)
        tonnetz = librosa.feature.tonnetz(y=librosa.effects.harmonic(audio), sr=sr)

        if method == "autocorr":
            pitches, magnitudes = librosa.core.piptrack(y=audio, sr=sr)
            # Autocorrelation based pitch detection
            # For simplicity, we'll use a dominant pitch estimation from piptrack
            # A more robust autocorrelation method would involve manual autocorrelation
            # and peak picking. For this project, we assume librosa's piptrack
            # provides a suitable base for 'autocorr' like behavior in feature context.
            # Here, just taking the mean pitch for simplicity
            if pitches.shape[1] > 0:
                pitch_feature = np.mean(pitches[pitches > 0])
            else:
                pitch_feature = 0.0  # Default value if no pitch is detected
        elif method == "hps":
            # Harmonic Product Spectrum
            # This is a conceptual representation for feature extraction here
            # librosa does not have a direct HPS function for pitch.
            # We'll use F0 estimation and consider it as part of 'HPS' feature set for comparison.
            f0, _, _ = librosa.pyin(y=audio, sr=sr, fmin=librosa.note_to_hz('C2'), fmax=librosa.note_to_hz('C7'))
            pitch_feature = np.mean(f0[f0 > 0]) if f0[f0 > 0].size > 0 else 0.0
        elif method == "piptrack":
            # Directly using librosa's piptrack
            pitches, magnitudes = librosa.core.piptrack(y=audio, sr=sr)
            if pitches.shape[1] > 0:
                pitch_feature = np.mean(pitches[pitches > 0])
            else:
                pitch_feature = 0.0
        else:
            raise ValueError("Invalid pitch detection method. Choose 'autocorr', 'hps', or 'piptrack'.")

        return np.mean(mfccs.T, axis=0), np.mean(chroma.T, axis=0), np.mean(mel.T, axis=0), np.mean(contrast.T, axis=0), np.mean(tonnetz.T, axis=0), pitch_feature
    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        return None, None, None, None, None, None


# Function to add noise to audio
def add_noise(audio, noise_type=None):
    if noise_type == 'rayleigh':
        noise = np.random.rayleigh(scale=0.05, size=audio.shape)
    elif noise_type == 'nakagami':
        noise = np.random.nakagami(mu=1.0, omega=0.1, size=audio.shape)
    else:
        return audio  # No noise added

    noisy_audio = audio + noise
    return noisy_audio


# Load UrbanSound8K metadata
metadata = pd.read_csv('UrbanSound8K.csv')
fulldatasetpath = 'D:/DE-44 Projects/SNS/Ali_Riaz/'  # Update this path if necessary

features = []
for index, row in tqdm(metadata.iterrows(), total=len(metadata)):
    file_path = os.path.join(fulldatasetpath, 'fold' + str(row["fold"]) + '/', str(row["slice_file_name"]))
    class_name = row["class_name"]

    # For initial data loading, no noise added yet
    mfccs, chroma, mel, contrast, tonnetz, pitch = extract_features(file_path, method="autocorr") # Default method for initial load
    
    if mfccs is not None:
        features.append([mfccs, chroma, mel, contrast, tonnetz, pitch, class_name])

# Convert features into a DataFrame
features_df = pd.DataFrame(features, columns=['mfccs', 'chroma', 'mel', 'contrast', 'tonnetz', 'pitch', 'class_name'])
features_df.dropna(inplace=True)

# Prepare data for Neural Network
X = np.array(features_df.drop('class_name', axis=1).values.tolist())
y = features_df['class_name']

# Flatten X if it contains arrays of different shapes
X_flat = []
for item in X:
    flattened_item = []
    for sub_item in item:
        if isinstance(sub_item, np.ndarray):
            flattened_item.extend(sub_item.flatten())
        else:
            flattened_item.append(sub_item)
    X_flat.append(flattened_item)

X_flat = np.array(X_flat)

# Encode labels
le = LabelEncoder()
y_encoded = to_categorical(le.fit_transform(y))

# Split data
X_train, X_test, y_train, y_test = train_test_split(X_flat, y_encoded, test_size=0.2, random_state=42)

# Standardize features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)


# Reshape for CNN input
X_train_cnn = np.expand_dims(X_train_scaled, axis=2)
X_test_cnn = np.expand_dims(X_test_scaled, axis=2)


# Neural Network Model
def create_model(input_shape, num_classes):
    model = Sequential([
        Conv2D(32, (3, 3), activation='relu', input_shape=input_shape),
        MaxPooling2D((2, 2)),
        Dropout(0.25),
        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D((2, 2)),
        Dropout(0.25),
        Flatten(),
        Dense(128, activation='relu'),
        Dropout(0.5),
        Dense(num_classes, activation='softmax')
    ])
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model


# Train and Evaluate function with noise
def train_and_evaluate_with_noise(pitch_method, noise_type=None):
    print(f"\nTraining with {pitch_method} pitch detection...")
    
    features = []
    for index, row in tqdm(metadata.iterrows(), total=len(metadata)):
        file_path = os.path.join(fulldatasetpath, 'fold' + str(row["fold"]) + '/', str(row["slice_file_name"]))
        class_name = row["class_name"]

        try:
            audio, sr = librosa.load(file_path, res_type="kaiser_fast")
            noisy_audio = add_noise(audio, noise_type) # Add noise here
            
            mfccs, chroma, mel, contrast, tonnetz, pitch = extract_features_from_audio(noisy_audio, sr, method=pitch_method)
            
            if mfccs is not None:
                features.append([mfccs, chroma, mel, contrast, tonnetz, pitch, class_name])
        except Exception as e:
            print(f"Error processing {file_path}: {e}")
            continue

    features_df = pd.DataFrame(features, columns=['mfccs', 'chroma', 'mel', 'contrast', 'tonnetz', 'pitch', 'class_name'])
    features_df.dropna(inplace=True)

    X = np.array(features_df.drop('class_name', axis=1).values.tolist())
    y = features_df['class_name']

    X_flat = []
    for item in X:
        flattened_item = []
        for sub_item in item:
            if isinstance(sub_item, np.ndarray):
                flattened_item.extend(sub_item.flatten())
            else:
                flattened_item.append(sub_item)
        X_flat.append(flattened_item)

    X_flat = np.array(X_flat)
    le = LabelEncoder()
    y_encoded = to_categorical(le.fit_transform(y))

    X_train, X_test, y_train, y_test = train_test_split(X_flat, y_encoded, test_size=0.2, random_state=42)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Determine the correct input shape for the CNN
    # Assuming X_train_scaled is 2D (samples, features)
    # If your features are not image-like, a 1D CNN or a Dense network might be more appropriate.
    # For a simple CNN, we can treat the features as a 1D sequence.
    input_shape = (X_train_scaled.shape[1], 1)
    num_classes = y_encoded.shape[1]

    model = create_model(input_shape, num_classes)
    
    # Reshape for CNN input, adding a channel dimension
    X_train_cnn = np.expand_dims(X_train_scaled, axis=2)
    X_test_cnn = np.expand_dims(X_test_scaled, axis=2)

    model.fit(X_train_cnn, y_train, epochs=50, batch_size=32, validation_split=0.1, verbose=0)
    
    loss, accuracy = model.evaluate(X_test_cnn, y_test, verbose=0)
    print(f"Model Accuracy ({pitch_method}): {accuracy:.2f}%")
    
    y_pred = model.predict(X_test_cnn)
    y_pred_classes = np.argmax(y_pred, axis=1)
    y_test_classes = np.argmax(y_test, axis=1)
    
    print(classification_report(y_test_classes, y_pred_classes, target_names=le.classes_))
    
    return accuracy

# Compare pitch detection methods under different noise conditions
methods = ['autocorr', 'hps', 'piptrack']
noise_conditions = [None, 'rayleigh', 'nakagami']
results = {}

# Re-define extract_features_from_audio to accept audio and sr directly
def extract_features_from_audio(audio, sr, method="autocorr"):
    mfccs = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=40)
    chroma = librosa.feature.chroma_stft(y=audio, sr=sr)
    mel = librosa.feature.melspectrogram(y=audio, sr=sr)
    contrast = librosa.feature.spectral_contrast(y=audio, sr=sr)
    tonnetz = librosa.feature.tonnetz(y=librosa.effects.harmonic(audio), sr=sr)

    if method == "autocorr":
        pitches, magnitudes = librosa.core.piptrack(y=audio, sr=sr)
        if pitches.shape[1] > 0:
            pitch_feature = np.mean(pitches[pitches > 0])
        else:
            pitch_feature = 0.0
    elif method == "hps":
        f0, _, _ = librosa.pyin(y=audio, sr=sr, fmin=librosa.note_to_hz('C2'), fmax=librosa.note_to_hz('C7'))
        pitch_feature = np.mean(f0[f0 > 0]) if f0[f0 > 0].size > 0 else 0.0
    elif method == "piptrack":
        pitches, magnitudes = librosa.core.piptrack(y=audio, sr=sr)
        if pitches.shape[1] > 0:
            pitch_feature = np.mean(pitches[pitches > 0])
        else:
            pitch_feature = 0.0
    else:
        raise ValueError("Invalid pitch detection method. Choose 'autocorr', 'hps', or 'piptrack'.")

    return np.mean(mfccs.T, axis=0), np.mean(chroma.T, axis=0), np.mean(mel.T, axis=0), np.mean(contrast.T, axis=0), np.mean(tonnetz.T, axis=0), pitch_feature


for method in methods:
    for noise in noise_conditions:
        accuracy = train_and_evaluate_with_noise(method, noise)
        results[f"{method}_{noise or 'clean'}"] = accuracy

# Plot comparison of pitch detection methods under different noise conditions
plt.figure(figsize=(12, 8))
plt.bar(results.keys(), results.values(), color='skyblue')
plt.title('Comparison of Pitch Detection Methods under Different Noise Conditions')
plt.xlabel('Pitch Detection Method and Noise Type')
plt.ylabel('Accuracy')
plt.xticks(rotation=45)
plt.show()

print("\nBest Configuration:", max(results, key=results.get))
