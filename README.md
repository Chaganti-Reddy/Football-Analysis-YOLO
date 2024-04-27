<h2 align="center">Football Analysis using YOLO and CNNsn</h2>

<div align= "center"><img src="output_videos/screenshot.png" width="550" height="350"/>
  <h5>This project analyzes Football players in a video to measure their speed, ball control and more.</h5>
</div>

<div align="center">
    <a href="https://www.python.org/"><img src="https://img.shields.io/badge/python-v3.8-blue?style=flat-square"/></a>
    <img src="https://img.shields.io/github/issues/Chaganti-Reddy/Football-Analysis-YOLO?tyle=flat-square"/>
    <img src="https://img.shields.io/github/stars/Chaganti-Reddy/Football-Analysis-YOLO?style=flat-square"/>
    <img src="https://img.shields.io/github/forks/Chaganti-Reddy/Football-Analysis-YOLO?style=flat-square"/>
    <a href="https://github.com/Chaganti-Reddy/Football-Analysis-YOLO/issues"><img src="https://img.shields.io/github/issues/Chaganti-Reddy/Football-Analysis-YOLO?style=flat-square"/></a>
</div>

## :innocent: Introduction

The goal of this project is to detect and track players, referees, and footballs in a video using YOLO, one of the best AI object detection models available. We will also train the model to improve its performance. Additionally, we will assign players to teams based on the colors of their t-shirts using Kmeans for pixel segmentation and clustering. With this information, we can measure a team's ball acquisition percentage in a match. We will also use optical flow to measure camera movement between frames, enabling us to accurately measure a player's movement. Furthermore, we will implement perspective transformation to represent the scene's depth and perspective, allowing us to measure a player's movement in meters rather than pixels. Finally, we will calculate a player's speed and the distance covered. This project will be useful for football coaches, analysts, and fans who want to analyze a player's performance in a match. This project is done with the help of Abtullatarek's video on YOLO and Foot ball analysis.

## Table of Contents

- [:star: Features](#star-features)
- [:warning: Frameworks and Libraries](#warning-frameworks-and-libraries)
- [:file_folder: Datasets](#file_folder-datasets)
- [📂 Directory Tree](#-directory-tree)
- [:bulb: Models Used](#bulb-models-used)
- [🚀&nbsp; Installation & Running](#nbsp-installation--running)
- [:raising_hand: Citation](#raising_hand-citation)
- [:beginner: Future Goals](#beginner-future-goals)
- [:eyes: License](#eyes-license)

## :star: Features

- **Player Detection**: Detects players in the video using YOLO Models.
- **Foot Ball Detection**: Detects Football in the video continuously.
- **Ball COntrol**: Measures the ball control of the players.
- **Speed Measurement**: Measures the speed of the players.
- **Distance Measurement**: Measures the distance covered by the players.

## :warning: Frameworks and Libraries

- **[YOLO](https://github.com/ultralytics/ultralytics)** - YOLO is a state-of-the-art, real-time object detection system. It is a deep learning algorithm that can detect objects in real-time. YOLO is a clever neural network for doing object detection in real-time. YOLO stands for 'You Only Look Once'.
- **[PyTorch](https://pytorch.org/)** - PyTorch is an open source machine learning library based on the Torch library. It is used for applications such as natural language processing. It is primarily developed by Facebook's AI Research lab.
- **[OpenCV](https://opencv.org/)** - OpenCV is a library of programming functions mainly aimed at real-time computer vision. Originally developed by Intel, it was later supported by Willow Garage then Itseez. The library is cross-platform and free for use under the open-source BSD license.
- **[CNNs](https://en.wikipedia.org/wiki/Convolutional_neural_network)** - In deep learning, a convolutional neural network (CNN, or ConvNet) is a class of deep neural networks, most commonly applied to analyzing visual imagery.
- **[KMeans](https://en.wikipedia.org/wiki/K-means_clustering)** - K-means clustering is a method of vector quantization, originally from signal processing, that aims to partition n observations into k clusters in which each observation belongs to the cluster with the nearest mean.
- **[Optical Flow](https://en.wikipedia.org/wiki/Optical_flow)** - Optical flow or optic flow is the pattern of apparent motion of objects, surfaces, and edges in a visual scene caused by the relative motion between an observer and a scene.
- **[Perspective Transformation](https://en.wikipedia.org/wiki/3D_projection)** - Perspective transformation is a way to represent 3D objects in a 2D plane. It is used to represent the depth and perspective of a scene.

## :file_folder: Datasets

The DataSet used for this project is collected from [Roboflow](https://public.roboflow.com/).

## 📂 Directory Tree

```bash
.
├── camera_movement_estimator
├── development_and_analysis
├── input_videos
├── models
├── output_videos
├── player_ball_assigner
├── speed_and_distance_estimator
├── stubs
├── team_assigner
├── trackers
├── training
├── utils
└── view_transformer
```

## :bulb: Models Used

- YOLO: AI object detection model
- Kmeans: Pixel segmentation and clustering to detect t-shirt color
- Optical Flow: Measure camera movement
- Perspective Transformation: Represent scene depth and perspective
- Speed and distance calculation per player
- And a trained model -- [Trained Yolo v5](https://drive.google.com/file/d/1DC2kCygbBWUKheQ_9cFziCsYVSRw6axK/view?usp=sharing)

## 🚀&nbsp; Installation & Running

1. Clone the repository and navigate to the directory

```bash
git clone https://github.com/Chaganti-Reddy/Football-Analysis-YOLO.git && cd Football-Analysis-YOLO
```

2. Change the path of video file to be analyzed in the [main.py](main.py) file.

3. Install these requirements.

- python3.8
- ultralytics
- pytroch
- pandas
- numpy
- opencv

4. [Sample input video](https://drive.google.com/file/d/1t6agoqggZKx6thamUuPAIdN_1zR9v9S_/view?usp=sharing) - Use this video to test the application.

5. Then run the following command to run the application.

```bash
python main.py
```

## :clap: And it's done!

Feel free to mail me for any doubts/query
:email: [Mail to Me :smile:](chagantivenkataramireddy1@gmail.com)

---

## :raising_hand: Citation

You are allowed to cite any part of the code or our dataset. You can use it in your Research Work or Project. Remember to provide credit to the Maintainer Chaganti Venkatarami Reddy by mentioning a link to this repository and her GitHub Profile.

Follow this format:

- Author's name - Name
- Date of publication or update in parentheses.
- Title or description of document.
- URL.

## :eyes: License

MIT © [Chaganti Reddy](https://github.com/Chaganti-Reddy/Football-Analysis-YOLO/blob/main/LICENSE)
