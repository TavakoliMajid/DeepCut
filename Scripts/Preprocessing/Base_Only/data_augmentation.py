import cv2
import numpy as np
import random
import os

def load_images_from_folder(folder, size=(224, 224)):
    images = []
    filenames = os.listdir(folder)
    for filename in filenames:
        img_path = os.path.join(folder, filename)
        img = cv2.imread(img_path, cv2.IMREAD_COLOR)
        if img is not None:
            img = cv2.resize(img, size)
            images.append(img)
    return images

def apply_gaussian_blur(img):
    ksize = random.choice([3, 5, 7])
    return cv2.GaussianBlur(img, (ksize, ksize), 0)

def apply_motion_blur(img):
    ksize = random.choice([3, 5, 7])
    kernel = np.zeros((ksize, ksize))
    kernel[int((ksize - 1)/2), :] = np.ones(ksize)
    kernel /= ksize
    return cv2.filter2D(img, -1, kernel)

def adjust_brightness_contrast(img):
    alpha = random.uniform(0.8, 1.2)  # Contrast control
    beta = random.randint(-30, 30)  # Brightness control
    return cv2.convertScaleAbs(img, alpha=alpha, beta=beta)

def adjust_hue_saturation(img):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    hue_shift = random.randint(-10, 10)
    sat_shift = random.randint(-20, 20)
    hsv[:, :, 0] = np.clip(hsv[:, :, 0] + hue_shift, 0, 179)
    hsv[:, :, 1] = np.clip(hsv[:, :, 1] + sat_shift, 0, 255)
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

def apply_rotation(img):
    angle = random.uniform(-10, 10)
    h, w = img.shape[:2]
    M = cv2.getRotationMatrix2D((w/2, h/2), angle, 1)
    return cv2.warpAffine(img, M, (w, h))

def apply_translation(img):
    h, w = img.shape[:2]
    tx, ty = random.randint(-10, 10), random.randint(-10, 10)
    M = np.float32([[1, 0, tx], [0, 1, ty]])
    return cv2.warpAffine(img, M, (w, h))

def apply_cropping(img):
    h, w = img.shape[:2]
    crop_size = random.uniform(0.8, 1.0)
    new_h, new_w = int(h * crop_size), int(w * crop_size)
    y, x = random.randint(0, h - new_h), random.randint(0, w - new_w)
    return img[y:y+new_h, x:x+new_w]

def apply_gaussian_noise(img):
    noise = np.random.normal(0, 10, img.shape).astype(np.uint8)
    return cv2.add(img, noise)

def generate_augmented_images(input_folder, output_folder, num_images=5000):
    os.makedirs(output_folder, exist_ok=True)
    images = load_images_from_folder(input_folder)
    
    if len(images) == 0:
        print("No images found in the folder.")
        return
    
    for i in range(num_images):
        img = random.choice(images).copy()
        
        if random.random() < 0.3:
            img = apply_gaussian_blur(img)
        if random.random() < 0.3:
            img = apply_motion_blur(img)
        if random.random() < 0.3:
            img = adjust_brightness_contrast(img)
        if random.random() < 0.3:
            img = adjust_hue_saturation(img)
        if random.random() < 0.3:
            img = apply_rotation(img)
        if random.random() < 0.3:
            img = apply_translation(img)
        if random.random() < 0.3:
            img = apply_cropping(img)
        if random.random() < 0.3:
            img = apply_gaussian_noise(img)
        
        filename = f"augmented_{i}.jpg"
        cv2.imwrite(os.path.join(output_folder, filename), img)
    
    print(f"Generated {num_images} augmented images in '{output_folder}'")


input_folder = "path_to_your_images"
output_folder = "path_to_save_augmented_images"
generate_augmented_images(input_folder, output_folder, num_images=5000)
