# ECG Data Iris Spectrogram Visualization UI

This project provides a user interface for visualizing ECG data as iris spectrograms using Streamlit in Python. It utilizes algorithms for ECG signal processing and spectrogram generation to facilitate the analysis of ECG data.

## Features

- Load and visualize ECG data from CSV files.
- Preprocess ECG data with Pan-Tompkins algorithm.
- Generate and visualize Power Spectral Density (PSD) and Spectrogram (with and without log transform).
- Interpolate k-space data for improved visualization.
- 3D surface plot for k-space data visualization.

## Getting Started

### Prerequisites

Before running this project, make sure you have installed:

- Python 3.10.13
- Streamlit
- Matplotlib
- WFDB
- Scipy
- Numpy
- Pandas

### Data Structure
The application expects a CSV file containing 'ID' and 'Label' columns. The corresponding ECG recordings should be present in the training2017 directory as .mat and .hea files.

## Acknowledgements
This work makes use of algorithms and processing methods from the following repositories:
Pan-Tompkins algorithm by Pramod07Ch: GitHub Repository
Spectrogram function by awerdich: GitHub Repository

## Authors
- **Dong woo Lee** - *Initial work* - [Hepsdata](https://github.com/hepsdata)
- OpenAI's ChatGPT - Code and Documentation Assistance

## License
This project is licensed under the MIT License
