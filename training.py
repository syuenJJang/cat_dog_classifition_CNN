import torch
from torchvision import datasets
import torchvision.transforms as tf
from torch.utils.data import DataLoader
import torch.nn as nn

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

train_transform = tf.Compose(
    [
        tf.ToTensor(),
        tf.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        tf.RandomHorizontalFlip(p=0.5),
    ]
)

val_test_transform = tf.Compose(
    [
        tf.ToTensor(),
        tf.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)

train_dataset = datasets.ImageFolder("./train_data/", transform=train_transform)
val_dataset = datasets.ImageFolder("./validate_data/", transform=val_test_transform)
test_dataset = datasets.ImageFolder("./test_data/", transform=val_test_transform)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=True)


print(f"Classes: {train_dataset.classes}")
print(f"Class to index: {train_dataset.class_to_idx}")
print(f"Train samples: {len(train_dataset)}")
print(f"Val samples: {len(val_dataset)}")
print(f"Test samples: {len(test_dataset)}")


class BinaryCNN(nn.Module):
    def __init__(self, num_classes=2):
        super(BinaryCNN, self).__init__()

        # 1 Layer: 224*224 -> 112*112
        self.conv1 = nn.Conv2d(
            in_channels=3, out_channels=64, kernel_size=3, padding="same"
        )
        self.bn1 = nn.BatchNorm2d(64)
        self.relu1 = nn.ReLU()
        self.mp1 = nn.MaxPool2d(2, 2)

        # 2 Layer: 112*112 -> 56*56
        self.conv2 = nn.Conv2d(
            in_channels=64, out_channels=128, kernel_size=3, padding="same"
        )
        self.bn2 = nn.BatchNorm2d(128)
        self.relu2 = nn.ReLU()
        self.mp2 = nn.MaxPool2d(2, 2)

        # 3 Layer: 56*56 -> 28*28
        self.conv3 = nn.Conv2d(
            in_channels=128, out_channels=256, kernel_size=3, padding="same"
        )
        self.bn3 = nn.BatchNorm2d(256)
        self.relu3 = nn.ReLU()
        self.mp3 = nn.MaxPool2d(2, 2)

        # 4 Layer: 28*28 -> 14*14
        self.conv4 = nn.Conv2d(
            in_channels=256, out_channels=512, kernel_size=3, padding="same"
        )
        self.bn4 = nn.BatchNorm2d(512)
        self.relu4 = nn.ReLU()
        self.mp4 = nn.MaxPool2d(2, 2)

        # Global Average Pooling + Classifier
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.dropout = nn.Dropout(0.5)
        self.fc = nn.Linear(512, num_classes)

    def forward(self, x):

        # Block 1
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu1(x)
        x = self.mp1(x)

        # Block 2
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu2(x)
        x = self.mp2(x)

        # Block 3
        x = self.conv3(x)
        x = self.bn3(x)
        x = self.relu3(x)
        x = self.mp3(x)

        # Block 4
        x = self.conv4(x)
        x = self.bn4(x)
        x = self.relu4(x)
        x = self.mp4(x)

        # Classifier
        x = self.global_pool(x)
        x = x.view(x.size(0), -1)
        x = self.dropout(x)
        x = self.fc(x)

        return x


# 모델 초기화
model = BinaryCNN(num_classes=2)

# 손실함수와 옵티마이저
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

model = model.to(device)


def train_epoch(model, train_loader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    Accuracy = 0
    total = 0

    for i, (images, labels) in enumerate(train_loader):
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss = running_loss + loss.item()
        _, predicted = outputs.max(1)
        total = total + labels.size(0)
        Accuracy = Accuracy + predicted.eq(labels).sum().item()

        if (i + 1) % 100 == 0:
            print(f"Batch: [{i+1}/{len(train_loader)})], Loss: {loss.item():.4f}")

    return running_loss / (len(train_loader)), 100.0 * Accuracy / total


def validate(model, val_loader, criterion, device):
    model.eval()
    running_loss = 0.0
    Accuracy = 0
    total = 0

    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)

        running_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        Accuracy += predicted.eq(labels).sum().item()

    return running_loss / len(val_loader), 100.0 * Accuracy / total


epochs = 120
best_train_acc = 0.0

"""
모델 훈련 코드
"""
# for epoch in range(epochs):
#     train_loss, train_acc = train_epoch(
#         model, train_loader, criterion, optimizer, device
#     )
#     val_loss, val_acc = validate(model, val_loader, criterion, device)

#     print(f"Epoch [{epoch+1}/{epochs}]")
#     print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")
#     print(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")

#     # ▼▼▼ 훈련 정확도 기준 모델 저장 로직 ▼▼▼
#     if train_acc > best_train_acc:
#         best_train_acc = train_acc
#         torch.save(model.state_dict(), "best_model.pth")
#         print(f"Model saved based on best train accuracy: {train_acc:.2f}%")

#     print("-" * 50)


###################################################################

"""
테스트 샘플 분석 및 결과
"""

# def log_all_predictions(model, loader, device, idx_to_class, show_details=False):
#     """
#     전체 데이터셋의 모든 샘플에 대한 예측 결과를 분석하는 함수
#     """
#     print("\n--- 전체 샘플 예측 결과 분석 ---")
#     model.eval()
#     total_samples = 0
#     correct_samples = 0
#     class_stats = {}

#     with torch.no_grad():
#         for batch_idx, (images, labels) in enumerate(loader):
#             images, labels = images.to(device), labels.to(device)
#             outputs = model(images)
#             _, predicted = outputs.max(1)

#             # 배치별 처리
#             for j in range(len(images)):
#                 true_class = idx_to_class[labels[j].item()]
#                 pred_class = idx_to_class[predicted[j].item()]
#                 is_correct = true_class == pred_class

#                 # 처음 몇 개만 출력 (선택사항)
#                 if show_details and total_samples < 20:
#                     correct_mark = "✓" if is_correct else "✗"
#                     print(f"{correct_mark} 실제: {true_class:<5} | 예측: {pred_class}")

#                 # 통계 수집
#                 if is_correct:
#                     correct_samples += 1
#                 total_samples += 1

#                 # 클래스별 통계
#                 if true_class not in class_stats:
#                     class_stats[true_class] = {"correct": 0, "total": 0}
#                 class_stats[true_class]["total"] += 1
#                 if is_correct:
#                     class_stats[true_class]["correct"] += 1

#             # 진행상황 출력 (옵션)
#             if (batch_idx + 1) % 10 == 0:
#                 print(f"처리 중... {total_samples}개 완료")

#     # 전체 결과 출력
#     overall_accuracy = (
#         (correct_samples / total_samples) * 100 if total_samples > 0 else 0
#     )

#     print(f"\n--- 전체 샘플 분석 결과 ---")
#     print(f"총 샘플 수: {total_samples}개")
#     print(f"정답 샘플: {correct_samples}개")
#     print(f"오답 샘플: {total_samples - correct_samples}개")
#     print(f"**전체 샘플 정확도: {overall_accuracy:.2f}%**")

#     print(f"\n--- 클래스별 상세 성능 ---")
#     for class_name, stats in class_stats.items():
#         class_acc = (
#             (stats["correct"] / stats["total"]) * 100 if stats["total"] > 0 else 0
#         )
#         print(f"{class_name}: {stats['correct']}/{stats['total']} = {class_acc:.1f}%")

#     return {
#         "total_samples": total_samples,
#         "correct_samples": correct_samples,
#         "accuracy": overall_accuracy,
#         "class_stats": class_stats,
#     }


# # --- 최종 모델 성능 평가 (수정된 버전) ---
# print("\n--- Final Model Evaluation ---")

# final_model = BinaryCNN(num_classes=2)
# try:
#     final_model.load_state_dict(torch.load("best_model.pth"))
#     final_model.to(device)

#     # 1. 기존 validate 함수로 전체 성능 평가
#     test_loss, test_acc = validate(final_model, test_loader, criterion, device)

#     print(f"\n--- 모델 성능 비교 ---")
#     print(f"validate() 함수 결과: {test_acc:.2f}%")

#     # 2. 전체 샘플 개별 분석
#     idx_to_class = {v: k for k, v in train_dataset.class_to_idx.items()}
#     results = log_all_predictions(
#         final_model, test_loader, device, idx_to_class, show_details=True
#     )

#     print(f"개별 분석 결과: {results['accuracy']:.2f}%")
#     print(f"차이: {abs(test_acc - results['accuracy']):.2f}%")

#     # 3. 혼동 행렬 스타일 요약
#     print(f"\n--- 요약 ---")
#     print(f"전체 테스트 데이터: {results['total_samples']}개")
#     print(f"정확히 예측: {results['correct_samples']}개")
#     print(f"잘못 예측: {results['total_samples'] - results['correct_samples']}개")
#     print(f"최종 정확도: {results['accuracy']:.2f}%")

# except FileNotFoundError:
#     print("\n오류: 저장된 'best_model_v5.pth' 파일을 찾을 수 없습니다.")

###################################################################

"""
외부 개 고양이 이미지 테스트
"""

from PIL import Image
import torch.nn.functional as F


def predict_single_image(model, image_path, device, transform, class_names):
    """
    하나의 이미지 파일을 받아 모델의 예측 결과를 출력하는 함수
    """
    # 1. 모델을 평가 모드로 설정
    model.eval()

    # 2. 이미지 불러오기 및 전처리
    try:
        image = Image.open(image_path).convert("RGB")
    except FileNotFoundError:
        print(f"\n오류: '{image_path}' 파일을 찾을 수 없습니다. 경로를 확인해주세요.")
        return

    # 전처리 및 배치 차원 추가 (모델은 항상 배치 단위로 입력을 받음)
    image_tensor = transform(image).unsqueeze(0).to(device)

    # 3. 예측 수행 (그래디언트 계산 비활성화)
    with torch.no_grad():
        outputs = model(image_tensor)
        # 소프트맥스를 적용해 확률로 변환
        probabilities = F.softmax(outputs, dim=1)
        # 가장 높은 확률을 가진 클래스의 인덱스와 확률값 가져오기
        confidence, predicted_idx = torch.max(probabilities, 1)

    # 4. 결과 해석 및 출력
    predicted_class = class_names[predicted_idx.item()]
    confidence_percent = confidence.item() * 100

    print("\n--- 단일 이미지 예측 결과 ---")
    print(f"입력 이미지: {image_path}")
    print(f"모델 예측: '{predicted_class}'")
    print(f"신뢰도: {confidence_percent:.2f}%")
    print("-----------------------------")


# --- 최종적으로 저장된 모델로 단일 이미지 예측 실행 ---

# 예측하고 싶은 이미지 파일 경로를 여기에 입력하세요.
# (예: 'my_cat_photo.jpg', 'C:/Users/User/Pictures/dog.png')
IMAGE_TO_PREDICT = "./cat.jpg"

print(f"\n\n이제 '{IMAGE_TO_PREDICT}' 파일로 최종 예측을 시작합니다...")

# 예측을 위한 모델 객체 생성 및 가중치 로드
prediction_model = BinaryCNN(num_classes=2)
try:
    # 훈련 과정에서 저장된 가장 좋은 모델을 불러옵니다.
    # 저장 로직에서 사용한 파일 이름 ('best_model.pth', 'best_balanced_model.pth' 등)과 일치해야 합니다.
    prediction_model.load_state_dict(torch.load("best_model.pth", map_location=device))
    prediction_model.to(device)

    # 클래스 이름 정의 (훈련 데이터셋 순서와 동일하게)
    class_names = ["cat", "dog"]

    # 이미지 전처리 정의 (훈련 시 val_test_transform과 동일)
    # 어떤 크기의 이미지가 들어와도 224x224로 맞춤
    image_transform = tf.Compose(
        [
            tf.Resize((224, 224)),
            tf.ToTensor(),
            tf.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    # 예측 함수 호출
    predict_single_image(
        prediction_model, IMAGE_TO_PREDICT, device, image_transform, class_names
    )

except FileNotFoundError:
    print(f"\n오류: 모델 파일 'best_model.pth'을 찾을 수 없습니다.")
    print("모델 훈련을 먼저 실행하여 모델 파일을 생성해야 합니다.")
