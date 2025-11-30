# Giant freshwater prawn keypoint detection > small_data_Set
https://universe.roboflow.com/prawns/giant-freshwater-prawn-keypoint-detection-umvh3

Provided by a Roboflow user
License: BY-NC-SA 4.0

### 🧠 Project Overview

This project focuses on the **automated detection of keypoints of giant freshwater prawns (*Macrobrachium rosenbergii*)** to support length calculation a step toward **biomass estimation in aquaculture**. The images were taken from **mobile platforms** used in the study. Using **pose estimation**, we locate anatomical landmarks and calculate the **Euclidean distance between specific keypoints** to estimate **carapace length and total length**, a critical indicator for biomass modeling.

Our approach integrates:

* Pose estimation (Yolov11-Pose)
* Ground truth annotations using expert-labeled prawn images
* Euclidean distance computation of total and carapce length
* trigonometric calculations to covnert the pixel length to mm

Ultimately, this system contributes to **precision aquaculture** by enabling **non-invasive, scalable, and automated length measurements** for biomass prediction.


### 🦐 Class Descriptions

the following keypoints are annotated:

* `eye_center`: Central point between the eyes
* `carapace_edge`: Point marking the rear end of the carapace
* `rostrum_tip` and `tail_tip` for total length estimation)



### 📅 Current Status & Timeline


* | Data Collection: videos captured using a GoPro on a mobile platform across multiple pond types  | ✅ Completed    |
* | Annotation: Keypoints manually labeled using Roboflow and validated by aquaculture experts      | ✅ Completed    |
* | Model Training: Pose estimation models trained and compared for accuracy across pond types | ✅ Completed    |
* | Validation: Compared predictions to manual ImageJ measurements using MAPE, MAE | ✅ Completed    |
* | paper for publish:    | 🔄 In Progress |                                        

---

### 🔗 Resources


* 📑 Related paper: *in progress* 
* 🧠 Models weights: https://drive.google.com/drive/folders/1ioh-IC-7wVVUydyJIQqEKEEHEaK9d7gP?usp=drive_link

---

### 🤝 Contribution & Labeling Guidelines

github repo for the project: in progess
