# 🎧Urban-Sound-Classification-using-Signal-Processing-and-Machine-Learning

This project implements an audio classification system capable of categorizing urban sound signals into **speech**, **music**, and **noise** using classical **signal processing techniques** and a **Random Forest Classifier**. It uses the [UrbanSound8K](https://urbansounddataset.weebly.com/urbansound8k.html) dataset and evaluates model robustness under various noise conditions.

---

## 📌 Project Highlights

- ✅ **Pitch Detection Methods:** Autocorrelation, Harmonic Product Spectrum (HPS), Librosa's Piptrack  
- ✅ **Feature Extraction:** MFCCs, Zero Crossing Rate, RMS Energy, Pitch  
- ✅ **Classification Algorithm:** Random Forest  
- ✅ **Noise Testing:** Added Rayleigh and Nakagami noise for robustness analysis  
- ✅ **Accuracy:** Up to **89.98%** under clean conditions using HPS

---

## 🧠 Objectives

- Classify audio into speech, music, and noise
- Compare different pitch detection methods
- Analyze performance under noisy environments

---

## 📊 Features Extracted

- **MFCCs (Mel-Frequency Cepstral Coefficients):** Capture timbral features
- **ZCR (Zero Crossing Rate):** Measures signal complexity
- **RMS Energy:** Indicates loudness and power
- **Pitch:** Captured through multiple estimation techniques

---

## 🛠️ Technology Stack

- **Language:** Python  
- **Libraries:** Librosa, NumPy, Scikit-learn, Matplotlib  
- **Dataset:** [UrbanSound8K](https://urbansounddataset.weebly.com/urbansound8k.html)

---

## 📈 Performance Summary

| Condition        | Autocorrelation | HPS    | Piptrack |
|------------------|------------------|--------|----------|
| **Clean Audio**  | 89.70%           | 89.98% | 89.52%   |
| **Rayleigh Noise** | 85.90%         | 86.20% | 84.88%   |
| **Nakagami Noise** | 85.12%         | 85.98% | 84.55%   |

**Best Performer:** Harmonic Product Spectrum (HPS) under all conditions.

---

