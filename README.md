# CerebroSeg

## Cerebral Tumor Analysis and Segmentation Web Application

Welcome to CerebroSeg, a state-of-the-art web application designed for cerebral tumor analysis and segmentation. Our application leverages the power of deep learning to provide precise and efficient tumor detection and segmentation, aiding in medical diagnosis and research.

## Paper Implemented

This application is based on the following research paper:
- **Title:** [Attention U-Net: Learning Where to Look for the Pancreas](https://paperswithcode.com/paper/attention-u-net-learning-where-to-look-for)
- **Authors:** Oktay, O., Schlemper, J., Le Folgoc, L., Lee, M., Heinrich, M. P., Misawa, K., Mori, K., McDonagh, S. G., Hammerla, N. Y., Kainz, B., Glocker, B., and Rueckert, D.
- **Abstract:** The paper introduces Attention U-Net, an advanced model that integrates attention mechanisms into the U-Net architecture to enhance performance in medical image segmentation tasks. The attention mechanism enables the model to focus on relevant regions, improving segmentation accuracy and efficiency.

## Features

- **Advanced Segmentation:** Utilizes the Attention U-Net model to accurately segment cerebral tumors from MRI images.
- **User-Friendly Interface:** Easy-to-use web interface for uploading images, viewing segmentation results, and analyzing tumor characteristics.
- **Real-Time Processing:** Quick image processing and segmentation, providing immediate results.
- **Visualization Tools:** Interactive tools to visualize and analyze segmented regions.
- **Secure Data Handling:** Ensures the privacy and security of medical data.

## Installation

1. **Clone the Repository**
    ```bash
    git clone https://github.com/4insu/CerebroSeg.git
    ```

2. **Create and Activate Virtual Environment**
    ```bash
    conda create -n <env_name> python=3.9 -y
    conda activate <env_name>
    ```

3. **Install Dependencies**
    ```bash
    pip install -r requirements.txt
    ```

## Usage

1. **Start the Application**
    ```bash
    streamlit run app.py
    ```

2. **Access the Web Interface**
    Open the web browser and go to `http://localhost:5000`

3. **Upload and Analyze Images**
    - Upload MRI images using the interface.
    - View and analyze the segmentation results.

## License

This project is licensed under the MIT License. See the [LICENSE](https://github.com/4insu/CerebroSeg/blob/main/LICENSE) file for details.