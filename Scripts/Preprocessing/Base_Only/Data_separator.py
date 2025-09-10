import os
import shutil
import random

def split_data(source_dir, train_dir, validation_dir, test_dir, train_ratio=0.7, validation_ratio=0.15, test_ratio=0.15):
    # Ensure the ratios sum to 1
    assert abs((train_ratio + validation_ratio + test_ratio) - 1.0) < 0.001, "Ratios must sum to 1"

    # Create the target directories if they don't exist
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(validation_dir, exist_ok=True)
    os.makedirs(test_dir, exist_ok=True)

    # Get a list of all files in the source directory
    files = [f for f in os.listdir(source_dir) if os.path.isfile(os.path.join(source_dir, f))]
    random.shuffle(files)  # Shuffle the files to ensure random distribution

    # Calculate the number of files for each set
    num_files = len(files)
    num_train = int(num_files * train_ratio)
    num_validation = int(num_files * validation_ratio)
    num_test = num_files - num_train - num_validation

    # Split the files into the three sets
    train_files = files[:num_train]
    validation_files = files[num_train:num_train + num_validation]
    test_files = files[num_train + num_validation:]

    # Move the files to their respective directories
    for file in train_files:
        shutil.move(os.path.join(source_dir, file), os.path.join(train_dir, file))
    for file in validation_files:
        shutil.move(os.path.join(source_dir, file), os.path.join(validation_dir, file))
    for file in test_files:
        shutil.move(os.path.join(source_dir, file), os.path.join(test_dir, file))

    print(f"Data split and moved successfully:\n"
          f"Training: {num_train} files\n"
          f"Validation: {num_validation} files\n"
          f"Test: {num_test} files")

# Example usage
source_directory = "D:\\Users\\Majid\\RES_NET\\New_Data\\BadCut"
train_directory = "D:\\Users\\Majid\\RES_NET\\New_Data\\Train"
validation_directory = "D:\\Users\\Majid\\RES_NET\\New_Data\\Validation"
test_directory = "D:\\Users\\Majid\\RES_NET\\New_Data\\Test"

split_data(source_directory, train_directory, validation_directory, test_directory)