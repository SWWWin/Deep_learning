import sys
import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from simple_convnet import SimpleConvNet
from common.trainer import Trainer

# 이미지 및 마스크 경로 설정
images_path = "/Users/suminsim/Desktop/python/my_study/G1020/Images_Square"
masks_path = "/Users/suminsim/Desktop/python/my_study/G1020/Masks_Square"

class GlaucomaData:
    def __init__(self, output_size=(256, 256)):
        self.output_size = output_size

    def load_image(self, image_path):
        img = cv2.imread(image_path)
        img = cv2.resize(img, self.output_size)
        return img

    def load_mask(self, mask_path):
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        mask = cv2.resize(mask, self.output_size)
        return mask

    def create_od_oc_masks(self, mask):
        od_mask = (mask == 1).astype(np.float32)  # OD
        oc_mask = (mask == 2).astype(np.float32)  # OC
        return od_mask, oc_mask

def load_dataset(images_path, masks_path):
    images = []
    masks = []
    for filename in os.listdir(images_path):
        if filename.endswith(".jpg") or filename.endswith(".png"):
            image_path = os.path.join(images_path, filename)
            mask_path = os.path.join(masks_path, filename[:-3] + "png")
            if os.path.exists(mask_path):
                images.append(image_path)
                masks.append(mask_path)
    return images, masks

def split_dataset(images, masks, split_ratio=0.8):
    split_index = int(len(images) * split_ratio)
    train_images = images[:split_index]
    test_images = images[split_index:]
    train_masks = masks[:split_index]
    test_masks = masks[split_index:]
    return train_images, test_images, train_masks, test_masks

# 데이터셋 로드
images, masks = load_dataset(images_path, masks_path)
train_images, test_images, train_masks, test_masks = split_dataset(images, masks)

# 모델 학습을 위한 데이터 준비
glaucoma_data = GlaucomaData()

# 학습 데이터 준비
x_train = np.array([glaucoma_data.load_image(img) for img in train_images])
t_train = np.array([glaucoma_data.load_mask(mask) for mask in train_masks])

# OD와 OC 마스크 생성
t_train_od = np.array([glaucoma_data.create_od_oc_masks(mask)[0] for mask in t_train])  # OD 마스크
t_train_oc = np.array([glaucoma_data.create_od_oc_masks(mask)[1] for mask in t_train])  # OC 마스크

# OD와 OC 마스크를 원-핫 인코딩 형태로 변환
t_train_one_hot = np.zeros((len(t_train), 3, *t_train_od.shape[1:]))  # (batch_size, 3, height, width) 형태로 초기화
t_train_one_hot[:, 0] = 1 - t_train_od  # 배경
t_train_one_hot[:, 1] = t_train_od  # OD
t_train_one_hot[:, 2] = t_train_oc  # OC

# 테스트 데이터 준비
x_test = np.array([glaucoma_data.load_image(img) for img in test_images])
t_test = np.array([glaucoma_data.load_mask(mask) for mask in test_masks])

# OD와 OC 테스트 마스크 생성
t_test_od = np.array([glaucoma_data.create_od_oc_masks(mask)[0] for mask in t_test])  # OD 마스크
t_test_oc = np.array([glaucoma_data.create_od_oc_masks(mask)[1] for mask in t_test])  # OC 마스크

# OD와 OC 테스트 마스크를 원-핫 인코딩 형태로 변환
t_test_one_hot = np.zeros((len(t_test), 3, *t_test_od.shape[1:]))  # (batch_size, 3, height, width) 형태로 초기화
t_test_one_hot[:, 0] = 1 - t_test_od  # 배경
t_test_one_hot[:, 1] = t_test_od  # OD
t_test_one_hot[:, 2] = t_test_oc  # OC

# x_train의 shape 변환
x_train = x_train.transpose(0, 3, 1, 2)  # (816, 3, 256, 256) 형태로 변경
x_test = x_test.transpose(0, 3, 1, 2)  # (N, 3, 256, 256) 형태로 변경

# 형태 출력
print("x_train shape:", x_train.shape)
print("t_train shape:", t_train_one_hot.shape)
print("x_test shape:", x_test.shape)
print("t_test shape:", t_test_one_hot.shape)

# 모델 설정
max_epochs = 20
network = SimpleConvNet(input_dim=(3, 256, 256),  # 입력 채널 수에 따라 수정
                        conv_param={'filter_num': 30, 'filter_size': 5, 'pad': 0, 'stride': 1},
                        hidden_size=100, output_size=3, weight_init_std=0.01)

trainer = Trainer(network, x_train, t_train_one_hot, x_test, t_test_one_hot,
                  epochs=max_epochs, mini_batch_size=100,
                  optimizer='Adam', optimizer_param={'lr': 0.001},
                  evaluate_sample_num_per_epoch=1000)
trainer.train()

# 학습 코드 끝난 후 추가
y_pred = network.predict(x_test)
print("예측 예시:", y_pred[0])  # 첫 번째 예측 출력

# 매개변수 보존
network.save_params("params.pkl")
print("Saved Network Parameters!")

# 그래프 그리기
markers = {'train': 'o', 'test': 's'}
x = np.arange(max_epochs)
plt.plot(x, trainer.train_acc_list, marker='o', label='train', markevery=2)
plt.plot(x, trainer.test_acc_list, marker='s', label='test', markevery=2)
plt.xlabel("epochs")
plt.ylabel("accuracy")
plt.ylim(0, 1.0)
plt.legend(loc='lower right')
plt.show()
