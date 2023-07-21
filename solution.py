import cv2
import imutils
import os
from PIL import Image

def check_file_extension(folder_path):
    image_files = [f for f in os.listdir(folder_path)]
    file_extensions = set()
    for image in image_files:
        split_tup = os.path.splitext(image)
        file_extensions.add(split_tup[1])
    print(f"Image file extensions found in the folder: {file_extensions}")


def count_image(folder_path):
     image_files = [f for f in os.listdir(folder_path) if f.endswith('.png')]
     image_count = len(image_files)
     
     return image_count


def delete_empty_image(folder_path):
    print("Checking for none/empty images.")
    image_files = [f for f in os.listdir(folder_path) if f.endswith('.png')]
    empty_image_count = 0
    none_image_count = 0
    for image_file in image_files:
            image_path = os.path.join(folder_path, image_file)
            image = cv2.imread(image_path)

            if image is None:
                os.remove(image_path)
                none_image_count += 1
                continue

            if image.size == 0:
                os.remove(image_path)
                empty_image_count += 1

    print(f"Deleted {none_image_count} None image.")
    print(f"Deleted {empty_image_count} Empty image.")


def resize_images(folder_path, output_size):
    print("Image resizing to 224x224 started.")
    image_files = [f for f in os.listdir(folder_path) if f.endswith('.png')]

    for image_file in image_files:
        image_path = os.path.join(folder_path, image_file)
        image = Image.open(image_path)
        resized_image = image.resize(output_size, Image.Resampling.LANCZOS)
        resized_image.save(image_path)

    print("Image resizing completed.")    


def draw_color_mask(img, borders, color=(0, 0, 0)):
    h = img.shape[0]
    w = img.shape[1]

    x_min = int(borders[0] * w / 100)
    x_max = w - int(borders[2] * w / 100)
    y_min = int(borders[1] * h / 100)
    y_max = h - int(borders[3] * h / 100)

    img = cv2.rectangle(img, (0, 0), (x_min, h), color, -1)
    img = cv2.rectangle(img, (0, 0), (w, y_min), color, -1)
    img = cv2.rectangle(img, (x_max, 0), (w, h), color, -1)
    img = cv2.rectangle(img, (0, y_max), (w, h), color, -1)

    return img


def preprocess_image_change_detection(img, gaussian_blur_radius_list=None, black_mask=(5, 10, 5, 0)):
    gray = img.copy()
    gray = cv2.cvtColor(gray, cv2.COLOR_BGR2GRAY)
    if gaussian_blur_radius_list is not None:
        for radius in gaussian_blur_radius_list:
            gray = cv2.GaussianBlur(gray, (radius, radius), 0)

    gray = draw_color_mask(gray, black_mask)

    return gray


def compare_frames_change_detection(prev_frame, next_frame, min_contour_area):
    frame_delta = cv2.absdiff(prev_frame, next_frame)
    thresh = cv2.threshold(frame_delta, 45, 255, cv2.THRESH_BINARY)[1]

    thresh = cv2.dilate(thresh, None, iterations=2)
    cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL,
                            cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)

    score = 0
    res_cnts = []
    for c in cnts:
        if cv2.contourArea(c) < min_contour_area:
            continue

        res_cnts.append(c)
        score += cv2.contourArea(c)

    return score, res_cnts, thresh


def remove_similar_images(folder_path, similarity_threshold, min_contour_area):
    print("Similar-looking image removal started.")
    image_files = [f for f in os.listdir(folder_path) if f.endswith('.png')]
    image_files = sorted(image_files)

    i = 0
    while i < (len(image_files) - 1) :
        image_path = os.path.join(folder_path, image_files[i])
        original_image = cv2.imread(image_path)

        delete_image_list = []
        j = i + 1
        while j < len(image_files):
            compare_image_file = image_files[j]
            compare_image_path = os.path.join(folder_path, compare_image_file)
            compare_image = cv2.imread(compare_image_path)

            original_preprocessed = preprocess_image_change_detection(original_image)
            compare_preprocessed = preprocess_image_change_detection(compare_image)

            score,_,_= compare_frames_change_detection(original_preprocessed, compare_preprocessed, min_contour_area)
            
            if score < similarity_threshold:
                print(f"Found {compare_image_file} similar to {image_path} with a score of {score}.")
                delete_image_list.append(compare_image_file)
                
            j += 1

        for img in delete_image_list:
           os.remove(os.path.join(folder_path, img))
           image_files.remove(img)
        
        i += 1


if __name__ == '__main__':

    folder_path = "dataset"
    check_file_extension(folder_path)
    image_count = count_image(folder_path)
    print(f"Total number of images are: {image_count}")
    delete_empty_image(folder_path)
    image_count = count_image(folder_path)
    print(f"After deleting none/empty images total number of images are: {image_count}")
    output_size = (224, 224)
    resize_images(folder_path, output_size)
    similarity_threshold = 150  
    min_contour_area = 95  
    remove_similar_images(folder_path, similarity_threshold, min_contour_area)
    image_count = count_image(folder_path)
    print(f"After deleting similar-looking images total number of images are: {image_count}")
