import torchvision.transforms as tf
import os
from PIL import Image


def preprocess_dataset_pytorch():

    transform = tf.Compose([tf.Resize(224), tf.CenterCrop(224), tf.ToTensor()])
    to_pil = tf.ToPILImage()

    folders = ["./cat_processed", "./dog_processed"]
    for i in folders:
        os.makedirs(i, exist_ok=True)

    data_folders = ["./cat", "./dog"]

    for i in data_folders:
        print(f"\n처리 중: {i}")

        for file in os.listdir(i):
            try:
                input_path = os.path.join(i, file)
                img = Image.open(input_path)
                original_size = img.size

                # 변환
                tensor_img = transform(img)
                print(f"Tensor shape: {tensor_img.shape}")

                # PIL 이미지로 변환
                processed_img = to_pil(tensor_img)

                # 저장 경로 결정
                if "cat" in i:
                    output_path = os.path.join("./cat_processed", file)
                elif "dog" in i:
                    output_path = os.path.join("./dog_processed", file)

                processed_img.save(output_path, qulity=95)
                print(f"✅ {file}: {original_size} → {processed_img.size}")

            except Exception as e:
                print(f"❌ {file} 실패: {e}")


preprocess_dataset_pytorch()
print(f"\n전처리 완료!")
