import os
import pandas as pd
import numpy as np
import librosa
import torch
import torch.nn.functional as F
import fairseq
import logging
from tqdm import tqdm
from pathlib import Path

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class HubertAudioFeatureExtractor:
    def __init__(self, dataset_path, output_base_dir, ckpt_path, layer, max_chunk=160000000):
        self.dataset_path = dataset_path
        self.output_base_dir = output_base_dir
        
        # Create output directory for HuBERT features
        self.feature_dir = os.path.join(output_base_dir, 'dataset')
        os.makedirs(self.feature_dir, exist_ok=True)
        
        # Initialize HuBERT model
        model, cfg, task = fairseq.checkpoint_utils.load_model_ensemble_and_task([ckpt_path])
        self.model = model[0].eval().cuda(1)
        self.task = task
        self.layer = layer
        self.max_chunk = max_chunk
        
        # Track processed files
        self.processed_files = self.load_processed_files()
        
    def load_processed_files(self):
        """Load list of already processed files from tracking CSVs"""
        processed = set()
        overall_csv = os.path.join(self.feature_dir, 'overall_results.csv')
        if os.path.exists(overall_csv):
            df = pd.read_csv(overall_csv)
            processed.update(df['file_name'].tolist())
        return processed
    
    def read_audio(self, path):
        """Read and preprocess audio file"""
        wav, _ = librosa.load(path, sr=self.task.cfg.sample_rate)
        if wav.ndim == 2:
            wav = wav.mean(-1)
        assert wav.ndim == 1, wav.ndim
        return wav
    
    def extract_hubert_features(self, audio_path):
        """Extract HuBERT features from audio file"""
        try:
            # Read audio
            x = self.read_audio(audio_path)
            
            with torch.no_grad():
                x = torch.from_numpy(x).float().cuda(1)
                if self.task.cfg.normalize:
                    x = F.layer_norm(x, x.shape)
                x = x.view(1, -1)
                
                feat = []
                for chunk_start in range(0, x.size(1), self.max_chunk):
                    x_chunk = x[:, chunk_start:chunk_start + self.max_chunk]
                    if x_chunk.size(1) < 2:
                        continue
                        
                    try:
                        feat_chunk, _ = self.model.extract_features(
                            source=x_chunk,
                            padding_mask=None,
                            mask=False,
                            output_layer=self.layer,
                        )
                        feat.append(feat_chunk)
                    except RuntimeError as e:
                        if "Kernel size can't be greater than actual input size" in str(e):
                            logger.warning(f"Skipping chunk due to runtime error: {str(e)}")
                            continue
                        else:
                            raise e
                
                if not feat:
                    return None
                    
                # Concatenate all chunks and compute mean across time dimension
                features = torch.cat(feat, 1).squeeze(0)
                return features.mean(dim=0).cpu().numpy()
                
        except Exception as e:
            logger.error(f"Error processing {audio_path}: {e}")
            return None
    
    def process_audio_files(self, batch_size=200):
        """Process audio files and extract HuBERT features"""
        all_files = [f for f in Path(self.dataset_path).rglob("*.wav")]
        remaining_files = [f for f in all_files if os.path.splitext(f.name)[0] not in self.processed_files]
        
        print(f"Total files: {len(all_files)}")
        print(f"Already processed: {len(self.processed_files)}")
        print(f"Remaining to process: {len(remaining_files)}")
        
        # Initialize results
        results = {
            'train': [],
            'test': []
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

                # Extract features
                features = self.extract_hubert_features(str(file_path))
                
                if features is not None:
                    # Create result dictionary
                    result = {
                        'file_name': file_name,
                        'path': str(file_path),
                        'split': split,
                        'breed': breed
                    }
                    
                    # Add features
                    for i, value in enumerate(features):
                        result[f'hubert_{i}'] = value
                    
                    results[split].append(result)
                    
                    counter += 1
                    pbar.update(1)
                    
                    # Save every batch_size files
                    if counter >= batch_size:
                        self.save_results(results)
                        counter = 0
                        
        # Final save
        self.save_results(results)
        print(f"Processing complete. Results saved in {self.feature_dir}")

    def save_results(self, results):
        """Save results to CSV files"""
        # Save train results
        if results['train']:
            train_csv_path = os.path.join(self.feature_dir, 'train_results.csv')
            df = pd.DataFrame(results['train'])
            df.to_csv(train_csv_path, mode='a', index=False, header=not os.path.exists(train_csv_path))

        # Save test results
        if results['test']:
            test_csv_path = os.path.join(self.feature_dir, 'test_results.csv')
            df = pd.DataFrame(results['test'])
            df.to_csv(test_csv_path, mode='a', index=False, header=not os.path.exists(test_csv_path))

        # Clear processed results
        results['train'] = []
        results['test'] = []

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
    OUTPUT_BASE_DIR = "./classification/hubert"
    os.makedirs(OUTPUT_BASE_DIR, exist_ok=True)

    # Add the checkpoint path for pretrained Hubert
    CKPT_PATH = "path_to_hubert_checkpoint/checkpoint_105_130000.pt"

    # hubert config
    LAYER = 11
    MAX_CHUNK = 160000000
    
    # Initialize and run feature extraction
    extractor = HubertAudioFeatureExtractor(
        DATASET_PATH, 
        OUTPUT_BASE_DIR,
        CKPT_PATH,
        LAYER,
        MAX_CHUNK
    )
    extractor.process_audio_files(batch_size=200)
