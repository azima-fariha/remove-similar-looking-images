# remove-similar-looking-images<br />
This repository contains a program designed to preprocess a dataset of parking garage images captured by surveillance cameras. The program aims to clean, standardize, and remove similar-looking images from the dataset to ensure its suitability for training deep learning models.

## Q1. What did you learn after looking at our dataset? <br />
The dataset comprises parking garage images captured by surveillance cameras, totaling 1080 images.
## Q2. How does your program work? <br />
### Step 1:
The program examines the available file extensions within the dataset. It helps us to know what kind of dataset we are dealing with.

### Step 2:
It determines the total number of images present in the dataset.

### Step 3:
The program identifies and eliminates any none or empty image files, if they exist, from the dataset.

* **None type image** - When loading or accessing an image that doesn't exist or cannot be read properly, the image variable may be assigned the value "None" to indicate the absence of valid image data. <br />
* **Empty image** - An "empty" image refers to an image object that exists but contains no meaningful data or has dimensions of zero.

### Step 4:
After the removal process, it reevaluates the number of images remaining in the dataset.

### Step 5:
Next, the program resizes the dataset to a standardized 224 x 224 dimensions. This step is crucial for ensuring consistent data and meets the requirements of deep learning models. Many neural networks expect square-shaped input images, and 224x224 is a commonly accepted size for various deep learning models. This choice of size is made considering the model the dataset will be trained on; it may vary based on the specific model being used.

### Step 6:
The program aims to detect and remove similar-looking images from the dataset. It accomplishes this through the following steps:

* The program starts by comparing the first image with all the other images in the dataset to check for similarity. If any similar-looking images are found, they are added to a list for deletion at the end.

* After analyzing the first image, the program proceeds to compare the second image with all the remaining images, repeating the similarity check process.

* This process continues for each subsequent image in the dataset, comparing it with the remaining images to identify and compile a list of similar-looking images for removal.

### Step 7:
Upon completion of the similarity-looking images removal process, the program displays the number of remaining images in the dataset.

### Q3. What values did you decide to use for input parameters and how did you find these values?<br />

For my `remove_similar_images` function, I opted to utilize the following input parameters:

1. **folder_path** - This parameter allows me to provide the path to the dataset.
2. **min_contour_area** - The input parameter `min_contour_area` plays a crucial role in calculating the similarity score within the `compare_frames_change_detection` function. Therefore, setting an appropriate minimum contour area is of utmost importance. It's essential to strike a balance, avoiding setting `min_contour_area` too small, as it may include image noises, or too large, leading to the exclusion of significant contours.

To identify a suitable `min_contour_area` for the dataset, I conducted an analysis of contour areas from various images. In one of the experiments, the contour areas were as follows:

contour area: 32.0
contour area: 32.0
contour area: 61.0
contour area: 61.0
contour area: 30.0
contour area: 30.0
contour area: 20.0
contour area: 20.0
contour area: 7381.0
contour area: 16.0
contour area: 16.0
contour area: 104.5
contour area: 769.5

From the example above, it was evident that any contour with an area less than 80-90 could be considered an outlier. This insight helped me experiment with multiple images to determine an appropriate `min_contour_area` for optimal performance.
Ultimately, I set `min_contour_area` to 95, as it was determined to strike the right balance and effectively handle contour areas while processing the images.

3. **similarity_threshold** - In the `compare_frames_change_detection` function, we obtain a similarity score for two images. To determine if the images are considered similar, we establish a `similarity_threshold`, which indicates the score below which the images are deemed similar. To achieve this, we pass the `similarity_threshols` as an input parameter to the `remove_similar_images` function.
To identify an appropriate threshold for the dataset, I experimented with various thresholds. For instance, setting `similarity_threshold` to 1500 and `min_contour_area` to 300 resulted in the following two images being predicted as similar.

image image 

However, upon inspection, I observed that the images had a score of 553.0, but there was a significant difference between them. The left image contained a car that was absent in the right image. This difference is crucial, depending on the specific use case. Since our `similarity_threshold` is 1500 and the score is 553, the model erroneously assumed the images to be similar. To rectify this, it became necessary to reduce the `similarity_threshold`.<br />
To address this, I reduced both `similarity_threshold` to 500 and `min_contour_area` to 100. As a result, the score was 406.0, and the two images below were concluded to be similar. However, as evident in the right image, there are more cars present that do not exist in the left image, making them dissimilar.<br />

 image image 

This iterative process involved experimenting with various images to determine suitable values for both the `similarity_threshold` and `min_contour_area`, ultimately ensuring accurate results for the dataset.<br />
After analyzing carefully, I eventually set the `similarity_threshold` to 150, aiming for more accurate results and better handling of image similarities.

## Q4. What would you suggest to implement to improve data collection of unique cases in the future?
Regular Data Cleanup: Perform regular data cleanup by running algorithms to detect and remove duplicated or highly similar images or data points. This helps maintain a clean and relevant dataset.
## Q5. Any other comments about your solution?
The effectiveness of the solution heavily relies on the specific use case and the overall size of the dataset. When training a deep neural network, having a large number of images is advantageous. However, it is crucial to consider the potential risks involved in removing similar images, especially if important images may be mistakenly eliminated.

To address these concerns, it becomes essential to carefully determine the appropriate values for `similarity_threshold` and `min_contour_area`. This thoughtful selection ensures that the removal process strikes the right balance between eliminating redundant data and retaining valuable images for accurate and reliable model training.


