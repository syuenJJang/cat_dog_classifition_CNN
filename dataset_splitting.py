import os
import shutil


def split_dataset():
    # 폴더 만들기
    folders = [
        "./train_data/dog",
        "./train_data/cat",
        "./validate_data/dog",
        "./validate_data/cat",
        "./test_data/dog",
        "./test_data/cat",
    ]

    for folder in folders:
        if not os.path.exists(folder):
            os.makedirs(folder)

    cat_image_files = []
    dog_image_files = []

    for file in os.listdir("cat_final"):
        if file.lower().endswith("jpg"):
            cat_image_files.append(file)

    for file in os.listdir("dog_final"):
        if file.lower().endswith("jpg"):
            dog_image_files.append(file)

    # 강아지 이미지 파일 개수를 기준으로 분할 지점 계산
    total_dogs = len(dog_image_files)
    train_dog_end = int(total_dogs * 0.8)
    val_dog_end = int(total_dogs * 0.9)

    print(
        f"강아지 이미지 분배: Train {train_dog_end}, Validate {val_dog_end - train_dog_end}, Test {total_dogs - val_dog_end}"
    )

    dog_train_data = dog_image_files[0:train_dog_end]
    for file in dog_train_data:
        shutil.copy(f"./dog_final/{file}", "./train_data/dog")

    dog_validate_data = dog_image_files[train_dog_end:val_dog_end]
    for file in dog_validate_data:
        shutil.copy(f"./dog_final/{file}", "./validate_data/dog")

    dog_test_data = dog_image_files[val_dog_end:]
    for file in dog_test_data:
        shutil.copy(f"./dog_final/{file}", "./test_data/dog")

    # 고양이 이미지 파일 개수를 기준으로 분할 지점 계산
    total_cats = len(cat_image_files)
    train_cat_end = int(total_cats * 0.8)
    val_cat_end = int(total_cats * 0.9)  # 80% train + 10% validate

    print(
        f"고양이 이미지 분배: Train {train_cat_end}, Validate {val_cat_end - train_cat_end}, Test {total_cats - val_cat_end}"
    )

    cat_train_data = cat_image_files[0:train_cat_end]
    for file in cat_train_data:
        shutil.copy(f"./cat_final/{file}", "./train_data/cat")

    cat_validate_data = cat_image_files[train_cat_end:val_cat_end]
    for file in cat_validate_data:
        shutil.copy(f"./cat_final/{file}", "./validate_data/cat")

    cat_test_data = cat_image_files[val_cat_end:]
    for file in cat_test_data:
        shutil.copy(f"./cat_final/{file}", "./test_data/cat")


split_dataset()
