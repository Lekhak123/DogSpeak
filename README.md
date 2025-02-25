
# **DogSpeak**  

## **Overview**  
This repository contains a **sample dataset** and code for analyzing dog barks using acoustic features and evaluating the results.  

## **Dataset**  
1. **Sample Data:**  
   - We’ve included a **small randomly sampled subset** of our dataset (`sampled dog barks.zip`), containing **5,000 dog barks**.  
   - The **full dataset** will be released after our paper is accepted.  

2. **Metadata:**  
   - `DogSpeak_Metadata.json` provides metadata for each audio clip.  
   - It includes:  
     - `clip_name`: The name of the WAV file containing the bark.  
     - `dog_id`: A unique identifier (e.g., `dog[index]`) for each individual dog.  
     - `sex`: The dog's gender (`male` or `female`).  
     - `breed`: The breed of the dog.  

## **Code & Feature Extraction**  
The `code/` folder contains scripts for feature extraction and classification:  

### **Feature Extraction**  
- `extract_acoustic_features.py` – Extracts acoustic features such as **MFCCs, GeMAPS,** and more.  
- `extract_hubert_embeddings.py` – Extracts **HuBERT embeddings** for deep learning tasks.  

> 📌 **Note:** Before running these scripts, ensure that the dataset path and other configurations are correctly set (see the comments in the code). You’ll also need to create the appropriate subfolder structure for each task using the metadata file. The dataset structure is included in the `code/` folder for reference.  

### **Classification**  
- `classification/classify_acoustic_data.py` – Performs classification using extracted acoustic features.  
- `classification/classify_hubert_embeddings.py` – Classifies data using HuBERT embeddings.  

> 📌 **Note:** You’ll need to download the **pretrained HuBERT checkpoint** before running the classification scripts. For details on how to do this, refer to the paper we’ve cited.  

## **Getting Started**  
To reproduce our results:  
1. Extract the dataset and metadata.  
2. Run the feature extraction scripts.  
3. Train the classification models using the extracted features.  

For more details, check the comments in each script.  
