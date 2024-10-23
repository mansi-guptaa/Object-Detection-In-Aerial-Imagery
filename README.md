# Object Detection in Aerial Imagery using YOLOv8


This project aims to train an object detection model to identify "Houses" and "Tennis Courts" in aerial images using the YOLOv8 framework. The solution leverages a curated dataset to develop a robust model capable of accurately detecting these objects in unseen aerial imagery.

## Problem Statement

The objective is to develop an object detection model that can locate and classify specific objects—namely "Houses" and "Tennis Courts"—within aerial images. The task involves:

1. **Dataset Preparation:** Utilizing a dataset from Roboflow that contains aerial images annotated with bounding boxes around "Houses" and "Tennis Courts."
2. **Model Training:** Training a YOLOv8 model on this dataset to detect and classify these objects in new, unseen aerial images.
3. **Model Evaluation and Prediction:** Evaluating the performance of the trained model and using it to make predictions on new images.

## Project Steps

The project involves the following steps:

### 1. Setup

- Install necessary libraries such as `ultralytics` and `roboflow`.
- Authenticate with Roboflow to access the dataset.

```
pip install ultralytics roboflow
````
### 2. Data Preparation
- Download the Dataset: Use Roboflow to download the dataset containing aerial images annotated with "Houses" and "Tennis Courts."
- Rename Label Folders: Rename the labelTxt folders to labels for compatibility with the YOLOv8 framework.
- Normalize Coordinates: Ensure the bounding box coordinates are normalized as required by YOLO.

3. Model Training
- Load YOLOv8 Model: Load the YOLOv8 model architecture.
- Train the Model: Specify parameters such as the number of epochs, image size, and batch size to fine-tune the model. The training process will involve optimizing the model to detect "Houses" and "Tennis Courts."

4. Model Evaluation
- Evaluate Model Performance: After training, evaluate the model using metrics such as precision, recall, F1-score, and confusion matrix. This helps in understanding how well the model is performing on the validation set.

5. Prediction
- Load Trained Model Weights: Load the best model weights obtained from the training process.
- Predict on New Images: Use the model to predict objects in new, unseen aerial images.

## Conclusion

The project successfully demonstrates the use of the YOLOv8 framework to detect "Houses" and "Tennis Courts" in aerial imagery. This application can be beneficial in various domains such as urban planning, real estate analysis, and aerial surveillance, where accurate detection and classification of objects from aerial views are critical.

## Future Work

- **Model Deployment:** Deploy the trained model on a cloud platform (e.g., AWS, GCP) for real-time object detection to make the solution more accessible and scalable.
- **Performance Optimization:** Experiment with different YOLO versions and architectures to further improve model accuracy and processing speed, ensuring better detection performance.
- **Expand Dataset:** Incorporate a more diverse dataset with varied aerial images and additional object classes to generalize the model and enhance its robustness.

## References

- [Roboflow](https://roboflow.com) - A platform for managing computer vision datasets, used to prepare the aerial imagery dataset for this project.
- [YOLOv8 by Ultralytics](https://github.com/ultralytics/ultralytics) - The latest version of the YOLO (You Only Look Once) object detection framework, used to train and evaluate the model.
