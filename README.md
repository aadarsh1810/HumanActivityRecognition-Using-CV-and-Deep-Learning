# HumanActivityRecognition-Using-CV-and-Deep-Learning

## About

In this project, we developed a robust system capable of accurately recognizing human activities from video data in real-time. The system leverages the Kinetics dataset, a large-scale dataset containing human action videos across a wide range of categories, and employs a pretrained ResNet-34 model for activity classification.

## Prerequisites

- Python 3.6+
- OpenCV
- PyTorch
- NumPy

## Installation

1. Clone the repository:
    ```bash
    git clone https://github.com/aadarsh1810/HumanActivityRecognition-Using-CV-and-Deep-Learning.git
    cd HumanActivityRecognition-Using-CV-and-Deep-Learning
    ```

## Usage

1. To recognize human activities in a video, place the video file in the repository directory and run the following command:
    ```bash
    python human_activity_reco.py --model resnet-34_kinetics.onnx --classes action_recognition_kinetics.txt --input <video file>
    ```

2. To recognize human activities using a deque data structure, run:
    ```bash
    python human_activity_reco_deque.py --model resnet-34_kinetics.onnx --classes action_recognition_kinetics.txt --input <video file>
    ```

## Files

- `human_activity_reco.py`: Our human activity recognition script which samples N frames at a time to make an activity classification prediction.
- `human_activity_reco_deque.py`:A similar human activity recognition script that implements a rolling average queue. This script is slower to run.
- `action_recognition_kinetics.txt`: Contains labels for the Kinetics dataset.
- `sample3.mp4` and `sample4.mp4`: Sample videos for testing.

## Model

We used a pretrained ResNet-34 model, which is fine-tuned on the Kinetics dataset to classify human activities. The model is capable of recognizing various activities such as running, jumping, and dancing.

## Results

The model demonstrated high accuracy in recognizing activities from the Kinetics dataset and was able to process video data in real-time.

## Contributing

Contributions are welcome. Please fork the repository and submit pull requests for any improvements.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
