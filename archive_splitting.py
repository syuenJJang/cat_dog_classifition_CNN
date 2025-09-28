import os
import shutil


def archive_splitting():

    folders = ["./cat", "./dog"]

    for i in folders:
        if not os.path.exists(i):
            os.makedirs(i)

    cat_image = []
    dog_image = []

    for file in os.listdir("./archive/cat_dog"):
        if "cat" in file.lower():
            cat_image.append(file)
        elif "dog" in file.lower():
            dog_image.append(file)

    for file in cat_image:
        shutil.move(f"./archive/cat_dog/{file}", "./cat")

    for file in dog_image:
        shutil.move(f"./archive/cat_dog/{file}", "./dog")

    print(f"고양이 이미지 {len(cat_image)}개 이동 완료")
    print(f"강아지 이미지 {len(dog_image)}개 이동 완료")


archive_splitting()
