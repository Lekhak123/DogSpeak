import os
import pandas as pd
import numpy as np
import librosa
import opensmile
from tqdm import tqdm
from pathlib import Path

class AudioFeatureExtractor:
    def __init__(self, dataset_path, output_base_dir):
        self.dataset_path = dataset_path
        self.output_base_dir = output_base_dir

        # Initialize OpenSMILE feature extractors
        self.smile_gemaps = opensmile.Smile(
            feature_set=opensmile.FeatureSet.GeMAPS,
            feature_level=opensmile.FeatureLevel.Functionals,
        )
        self.smile_egemaps = opensmile.Smile(
            feature_set=opensmile.FeatureSet.eGeMAPS,
            feature_level=opensmile.FeatureLevel.Functionals,
        )
        self.gemaps_feature_names = self.smile_gemaps.feature_names
        self.egemaps_feature_names = self.gemaps_feature_names

        # Create output directories for each feature
        self.feature_dirs = self.create_feature_dirs()

        # Track processed files
        self.processed_files = self.load_processed_files()

    def create_feature_dirs(self):
        """Create subdirectories for each feature type"""
        features = ['mfcc', 'filterbank', 'gemaps', 'egemaps']
        feature_dirs = {}
        for feature in features:
            train_dir = os.path.join(self.output_base_dir, feature, 'train')
            test_dir = os.path.join(self.output_base_dir, feature, 'test')
            os.makedirs(train_dir, exist_ok=True)
            os.makedirs(test_dir, exist_ok=True)
            feature_dirs[feature] = {'train': train_dir, 'test': test_dir}
        return feature_dirs

    def load_processed_files(self):
        """Load list of already processed files from tracking CSVs"""
        processed = set()
        for feature in self.feature_dirs:
            for split in ['train', 'test']:
                csv_path = os.path.join(self.feature_dirs[feature][split], 'results.csv')
                if os.path.exists(csv_path):
                    df = pd.read_csv(csv_path)
                    processed.update(df['file_name'].tolist())
        return processed

    def pad_audio(self, audio, desired_length):
        """Pad audio signal to the desired length"""
        if len(audio) < desired_length:
            pad_width = desired_length - len(audio)
            audio = np.pad(audio, (0, pad_width), mode='constant')
        return audio

    def extract_mfcc(self, audio, sr):
        """Extract MFCC features"""
        n_fft = 2048
        audio = self.pad_audio(audio, n_fft)  # Pad audio if it is too small
        mfccs = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=40, n_fft=n_fft)
        mfcc_mean = np.mean(mfccs, axis=1)
        mfcc_std = np.std(mfccs, axis=1)
        return np.concatenate([mfcc_mean, mfcc_std])

    def extract_filterbank(self, audio, sr):
        """Extract Mel filterbank features"""
        n_fft = 2048
        audio = self.pad_audio(audio, n_fft)  # Pad audio if it is too small
        mel_spec = librosa.feature.melspectrogram(y=audio, sr=sr, n_mels=128, n_fft=n_fft)
        filterbank_mean = np.mean(mel_spec, axis=1)
        filterbank_std = np.std(mel_spec, axis=1)
        return np.concatenate([filterbank_mean, filterbank_std])

    def process_audio_files(self, batch_size=200):
        """Process audio files and extract features"""
        all_files = [f for f in Path(self.dataset_path).rglob("*.wav")]
        remaining_files = [f for f in all_files if os.path.splitext(f.name)[0] not in self.processed_files]

        print(f"Total files: {len(all_files)}")
        print(f"Already processed: {len(self.processed_files)}")
        print(f"Remaining to process: {len(remaining_files)}")

        results = {
            'mfcc': {'train': [], 'test': []},
            'filterbank': {'train': [], 'test': []},
            'gemaps': {'train': [], 'test': []},
            'egemaps': {'train': [], 'test': []},
        }

        counter = 0
        with tqdm(total=len(remaining_files), desc="Processing Audio Files") as pbar:
            for file_path in remaining_files:
                file_name = os.path.splitext(file_path.name)[0]

                # Extract metadata from directory structure
                path_parts = file_path.parts
                if len(path_parts) < 4:
                    print(f"Skipping {file_path}, invalid directory structure.")
                    continue
                breed = path_parts[-3]
                split = path_parts[-2]

                try:
                    audio, sr = librosa.load(str(file_path), sr=16000)

                    # Extract features
                    gemaps_features = self.smile_gemaps.process_signal(audio, sr).values[0]
                    egemaps_features = self.smile_egemaps.process_signal(audio, sr).values[0]

                    features = {
                        'mfcc': self.extract_mfcc(audio, sr),
                        'filterbank': self.extract_filterbank(audio, sr),
                        'gemaps': gemaps_features,
                        'egemaps': egemaps_features,
                    }

                    base_result = {
                        'file_name': file_name,
                        'path': str(file_path),
                        'split': split,
                        'breed': breed
                    }

                    for feature_type, feature_data in features.items():
                        result = base_result.copy()
                        if feature_data is None:
                            print(f"Warning: {feature_type} features for {file_name} returned None, skipping.")
                            continue

                        if feature_type in ['gemaps', 'egemaps']:
                            # Use GeMAPS feature names for both GeMAPS and eGeMAPS
                            feature_names = self.gemaps_feature_names
                            for name, value in zip(feature_names, feature_data):
                                result[f'{feature_type}_{name}'] = value
                        elif isinstance(feature_data, dict):
                            for key, value in feature_data.items():
                                result[f'{feature_type}_{key.replace(" ", "_")}'] = value
                        elif isinstance(feature_data, (list, np.ndarray)):
                            for i, value in enumerate(feature_data):
                                result[f'{feature_type}_{i}'] = value
                        else:
                            result[f'{feature_type}'] = feature_data

                        results[feature_type][split].append(result)

                    counter += 1
                    pbar.update(1)

                    if counter >= batch_size:
                        self.save_results(results)
                        counter = 0

                except Exception as e:
                    print(f"Error processing {file_name}: {e}")

        self.save_results(results)
        print(f"Processing complete. Results saved in {self.output_base_dir}")

    def save_results(self, results):
        """Save results to CSV files"""
        for feature, split_data in results.items():
            for split, split_results in split_data.items():
                if split_results:
                    output_csv = os.path.join(self.feature_dirs[feature][split], 'results.csv')
                    df = pd.DataFrame(split_results)
                    df.to_csv(output_csv, mode='a', index=False, header=not os.path.exists(output_csv))
                    results[feature][split] = []

if __name__ == "__main__":
    # Include the path to the dataset (dog barks)
    """
    The dataset path should be created in the format: path_to_dataset/[breed]/[split]
    split: train or test
    breed: options included in the metadata json file
    path_to_dataset: the path to the base folder of the dataset
    """
    DATASET_PATH = "path_to_dataset"
    # Output folder for extracted features.
    OUTPUT_BASE_DIR = "./classification/acoustic/dataset"
    OUTPUT_BASE_DIR = "./classification/acoustic/dataset"
    os.makedirs(OUTPUT_BASE_DIR, exist_ok=True)
    extractor = AudioFeatureExtractor(DATASET_PATH, OUTPUT_BASE_DIR)
    extractor.process_audio_files(batch_size=100)