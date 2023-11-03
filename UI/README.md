# ECG Data Iris Spectrogram Visualization UI

This project provides a user interface for visualizing ECG data as iris spectrograms using Streamlit in Python. It utilizes algorithms for ECG signal processing and spectrogram generation to facilitate the analysis of ECG data.

## Features

- Load and visualize ECG data from CSV files.
- Preprocess ECG data with Pan-Tompkins algorithm.
- Generate and visualize Power Spectral Density (PSD) and Spectrogram (with and without log transform).
- Interpolate k-space data for improved visualization.
- 3D surface plot for k-space data visualization.

## Dataset

The ECG data used for visualization in this project is taken from the PhysioNet 2017 Challenge dataset, which can be accessed [here](https://physionet.org/content/challenge-2017/1.0.0/).

### Data Structure
The application expects a CSV file containing 'ID' and 'Label' columns. The corresponding ECG recordings should be present in the training2017 directory as .mat and .hea files.

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

## Acknowledgements
This work makes use of algorithms and processing methods from the following repositories:
Pan-Tompkins algorithm by Pramod07Ch: GitHub Repository
Spectrogram function by awerdich: GitHub Repository

## Interactive Parameters

The Streamlit UI allows users to interact with several parameters for customizing the visualization of the ECG data. Below is a description of each parameter:

- **Choose a Label for Visualization**: This dropdown allows users to select a specific label associated with the ECG data. The dataset contains various labels which represent different classifications of ECG signals.

- **Choose an ID for Visualization**: Based on the selected label, this dropdown lists all the IDs associated with that label. Users can choose one to visualize the corresponding ECG data.

- **Enable ECG Preprocessing**: This checkbox, when checked, enables the preprocessing of the ECG data using the Pan-Tompkins algorithm. This helps in enhancing the quality of the signal for better visualization and analysis.

- **Column Zero Padding**: This checkbox enables zero-padding on the columns of the spectrogram. Zero-padding can improve the resolution of the frequency domain representation of the signal.

- **Row Zero Padding**: This checkbox applies zero-padding to the rows of the data, which can be useful in maintaining a consistent data format and potentially enhancing the visualization of the time domain representation.

- **Choose an Interpolation Method**: This dropdown allows users to select the interpolation method used in plotting the spectrogram. The available options are 'linear', 'cubic', and 'nearest', which affect the smoothness and fidelity of the plotted spectrogram.

- **Select Scatter Point Size**: A slider that adjusts the size of the scatter points in the k-space scatter plot visualization. Smaller points may be useful for dense datasets, while larger points can be better for highlighting individual data points.

- **Select Grid Resolution**: This slider adjusts the resolution of the grid used in the interpolated k-space data plot. A higher value results in a finer grid, which can reveal more detail in the frequency domain at the cost of increased computational load.

These parameters allow for a highly customizable visualization experience, enabling users to explore the ECG data in various ways to suit their specific research or clinical needs.




## Authors
- **Dong woo Lee** - *Initial work* - [Hepsdata](https://github.com/hepsdata)
- OpenAI's ChatGPT - Code and Documentation Assistance

## License
This project is licensed under the MIT License
