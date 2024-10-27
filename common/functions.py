# coding: utf-8
import numpy as np


def identity_function(x):
    return x


def step_function(x):
    return np.array(x > 0, dtype=np.int)


def sigmoid(x):
    return 1 / (1 + np.exp(-x))    


def sigmoid_grad(x):
    return (1.0 - sigmoid(x)) * sigmoid(x)
    

def relu(x):
    return np.maximum(0, x)


def relu_grad(x):
    grad = np.zeros(x)
    grad[x>=0] = 1
    return grad
    

def softmax(x):
    if x.ndim == 2:
        x = x.T
        x = x - np.max(x, axis=0)
        y = np.exp(x) / np.sum(np.exp(x), axis=0)
        return y.T 

    x = x - np.max(x) # 오버플로 대책
    return np.exp(x) / np.sum(np.exp(x))


def mean_squared_error(y, t):
    return 0.5 * np.sum((y-t)**2)

def cross_entropy_error(y, t):
    # y가 1D 배열일 경우
    if y.ndim == 1:
        batch_size = y.shape[0]
        return -np.sum(np.log(y[t] + 1e-7)) / batch_size

    # y가 2D 배열일 경우
    if y.ndim == 2:
        batch_size = y.shape[0]
        
        # t가 4D 배열인 경우 처리
        if t.ndim == 4:  # t의 shape: (batch_size, channels, height, width)
            # t를 2D 형태로 변환 (batch_size, height * width)
            t = t.reshape(batch_size, -1)  # (100, 3 * 256 * 256)
            t = np.argmax(t, axis=1)  # 클래스 인덱스를 선택 (0, 1, 2)

            # y의 shape를 (batch_size, channels)로 변환
            y = y.reshape(batch_size, -1, y.shape[-1])  # (100, 256 * 256, 3)
            y = y.reshape(-1, y.shape[-1])  # (100 * 256 * 256, 3)

    # y와 t의 형태 출력 (디버깅)
    print("y shape:", y.shape)
    print("t shape:", t.shape)

    batch_size = y.shape[0]

    print("y:", y)
    print("t:", t)
    print(-np.sum(np.log(y[np.arange(batch_size), t] + 1e-7)) / batch_size)
    # 크로스 엔트로피 손실 계산
    return -np.sum(np.log(y[np.arange(batch_size), t] + 1e-7)) / batch_size


def softmax_loss(X, t):
    y = softmax(X)
    return cross_entropy_error(y, t)

