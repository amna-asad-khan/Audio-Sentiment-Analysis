!pip install pandas numpy seaborn matplotlib librosa scipy joblib scikit-learn tensorflow soundfile scikeras

import warnings
warnings.filterwarnings('ignore')

# ======================
# Configuration
# ======================
TEST_MODE = False  # Set to False for full dataset
SUBSET_SIZE = 1000  # Samples for testing
USE_GPU = False  # Set to True if using TensorFlow-GPU
PARALLEL_PROCESSING = False

# ======================
# Imports
# ======================
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import librosa
import librosa.display
import os
import time
from scipy import signal
from joblib import Parallel, delayed

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
from sklearn.decomposition import PCA

if USE_GPU:
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import Dense, Dropout
    from tensorflow.keras.optimizers import Adam
    from scikeras.wrappers import KerasClassifier
else:
    from sklearn.neural_network import MLPClassifier


path = 'Crema-D/AudioWAV/'
audio_path = []
audio_emotion = []

for audio in os.listdir(path):
    if audio.endswith('.wav'):
        parts = audio.split('_')
        emotion_code = parts[2]
        emotion_map = {
            'ANG': 'angry', 'DIS': 'disgust', 'FEA': 'fear',
            'HAP': 'happy', 'NEU': 'neutral', 'SAD': 'sad'
        }
        if emotion_code in emotion_map:
            audio_path.append(os.path.join(path, audio))
            audio_emotion.append(emotion_map[emotion_code])

dataset = pd.DataFrame({'Path': audio_path, 'Emotions': audio_emotion})
# print(dataset.head())
# Reduce dataset for testing
if TEST_MODE:
    dataset = dataset.sample(n=min(SUBSET_SIZE, len(dataset)), random_state=42)
    


# Visualizing the emotion distribution
plt.figure(figsize=(8,6), dpi=100)
sns.countplot(x='Emotions', data=dataset, palette='mako')
plt.title("Emotion Distribution")
plt.xlabel('Emotions')
plt.ylabel('Count')
plt.xticks(rotation=45)
plt.show()

# ======================
# Feature Selection Menu
# ======================

def feature_selection_menu():
    print("\nFeature Selection Menu:")
    print("1. Zero Crossing Rate (zcr)")
    print("2. MFCC (mfcc)")
    print("3. Melspectrogram (mel)")
    print("4. Chroma Features (chroma)")
    print("5. Spectral Features (spectral)")
    print("6. FFT Statistics (fft)")
    print("Enter numbers separated by commas (e.g., 1,2,3):")
    
    selected = input().strip().split(',')
    feature_map = {
        '1': 'zcr', '2': 'mfcc', '3': 'mel',
        '4': 'chroma', '5': 'spectral', '6': 'fft'
    }
    return [feature_map[num] for num in selected if num in feature_map]

selected_features = feature_selection_menu()
print("Selected Features:", selected_features)

# ======================
# Audio Processing & Feature Extraction
# ======================

def apply_lowpass_filter(audio, sr, cutoff=4000, order=5):
    """Apply Butterworth low-pass filter"""
    nyq = 0.5 * sr
    normal_cutoff = cutoff / nyq
    b, a = signal.butter(order, normal_cutoff, btype='low', analog=False)
    return signal.filtfilt(b, a, audio)



def extract_features(file_path):
    """Extract selected audio features with error handling for invalid or empty files."""
    try:
        # Check if the file exists and is not empty
        if not os.path.isfile(file_path) or os.path.getsize(file_path) == 0:
            print(f"Skipping file: {file_path} (empty or invalid)")
            return None
        
        # Load audio file
        audio, sr = librosa.load(file_path, sr=None)
        
        # Apply low-pass filter (if applicable)
        filtered_audio = apply_lowpass_filter(audio, sr)
        
        features = {}
        
        # Time-domain features
        if 'zcr' in selected_features:
            features['zcr'] = np.mean(librosa.feature.zero_crossing_rate(filtered_audio))
        
        # Frequency-domain features
        if 'mfcc' in selected_features:
            mfcc = librosa.feature.mfcc(y=filtered_audio, sr=sr, n_mfcc=13)
            features['mfcc_mean'] = mfcc.mean(axis=1)
            features['mfcc_var'] = mfcc.var(axis=1)
        
        if 'mel' in selected_features:
            mel = librosa.feature.melspectrogram(y=filtered_audio, sr=sr)
            features['mel_mean'] = mel.mean()
            features['mel_var'] = mel.var()
        
        if 'chroma' in selected_features:
            chroma = librosa.feature.chroma_stft(y=filtered_audio, sr=sr)
            features['chroma_mean'] = chroma.mean()
            features['chroma_var'] = chroma.var()
        
        if 'spectral' in selected_features:
            features['spectral_centroid'] = librosa.feature.spectral_centroid(y=filtered_audio, sr=sr).mean()
            features['spectral_bandwidth'] = librosa.feature.spectral_bandwidth(y=filtered_audio, sr=sr).mean()
            features['spectral_contrast'] = librosa.feature.spectral_contrast(y=filtered_audio, sr=sr).mean()
        
        if 'fft' in selected_features:
            fft = np.abs(np.fft.fft(filtered_audio))
            features['fft_mean'] = fft.mean()
            features['fft_std'] = fft.std()
        
        # Flatten features
        feature_vector = np.concatenate([
            np.array([v for v in features.values() if isinstance(v, float)]),
            *[v for v in features.values() if isinstance(v, np.ndarray)]
        ])
        
        return feature_vector
    
    except Exception as e:
        # Handle unexpected errors
        print(f"Error processing file {file_path}: {e}")
        return None

# ======================
# Parallel Feature Extraction
# ======================

print("\nExtracting features...")
start_time = time.time()

if PARALLEL_PROCESSING:
    X = Parallel(n_jobs=-1)(delayed(extract_features)(path) for path in dataset.Path)
else:
    X = [extract_features(path) for path in dataset.Path]

print(f"Feature extraction completed in {time.time()-start_time:.2f} seconds")

X = np.array(X)
Y = dataset.Emotions.values
X = np.nan_to_num(X)  # Handle NaN values


# ======================
# Neural Network Configuration (GPU)
# ======================

if USE_GPU:
    def create_model(input_dim):
        model = Sequential([
            Dense(128, activation='relu', input_dim=input_dim),
            Dropout(0.3),
            Dense(64, activation='relu'),
            Dropout(0.3),
            Dense(len(np.unique(Y)), activation='softmax')
        ])
        model.compile(optimizer=Adam(learning_rate=0.001),
                      loss='sparse_categorical_crossentropy',
                      metrics=['accuracy'])
        return model

    mlp_model = KerasClassifier(build_fn=lambda: create_model(X.shape[1]), 
                               epochs=20, 
                               batch_size=128, 
                               verbose=0)
else:
    mlp_model = MLPClassifier(
    hidden_layer_sizes=(128, 64),
    activation='relu',
    solver='adam',
    early_stopping=True,
    max_iter=500,
    random_state=42)


# ======================
# Data Preparation
# ======================

le = LabelEncoder()
Y_encoded = le.fit_transform(Y)

X_train, X_test, y_train, y_test = train_test_split(
    X, Y_encoded, test_size=0.2, stratify=Y_encoded, random_state=42
)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# ======================
# Classifier Evaluation (Updated)
# ======================

classifiers = {
    "KNN": KNeighborsClassifier(n_neighbors=5),
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "Naive Bayes": GaussianNB(),
    "SVM": SVC(kernel='rbf'),
    "MLP": mlp_model
}

results = []

# Create figure for all confusion matrices
plt.figure(figsize=(20, 15))
plt.suptitle("Confusion Matrices for All Classifiers", y=1.02, fontsize=16)

# Create subplots grid (adjust rows/columns based on number of classifiers)
n_classifiers = len(classifiers)
rows = 2  # You can adjust these
cols = 3  # based on your needs
subplot_idx = 1

for name, clf in classifiers.items():
    # Training and evaluation
    print(f"\nTraining {name}...")
    start_time = time.time()
    
    if USE_GPU and name == "MLP":
        clf.fit(X_train_scaled, y_train, validation_split=0.1)
    else:
        clf.fit(X_train_scaled, y_train)
    
    train_time = time.time() - start_time
    test_pred = clf.predict(X_test_scaled)
    
    # Store results
    results.append({
        'Classifier': name,
        'Test Accuracy': accuracy_score(y_test, test_pred),
        'Test F1': f1_score(y_test, test_pred, average='weighted'),
        'Training Time': f"{train_time:.2f}s"
    })
    
    # Plot confusion matrix for current classifier
    plt.subplot(rows, cols, subplot_idx)
    cm = confusion_matrix(y_test, test_pred)
    sns.heatmap(cm, annot=True, fmt='d', 
                xticklabels=le.classes_, 
                yticklabels=le.classes_,
                cmap='Blues')
    plt.title(f'{name}\nAccuracy: {accuracy_score(y_test, test_pred):.2%}')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    
    subplot_idx += 1

plt.tight_layout()
plt.show()

# ======================
# Results Visualization
# ======================

results_df = pd.DataFrame(results)
print("\nFinal Results:")
print(results_df.sort_values(by='Test Accuracy', ascending=False))
