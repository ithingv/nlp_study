import torch
import torch.nn as nn

class ReviewClassifier(nn.Module):
    """퍼셉트론 기반 분류기"""
    def __init__(self, num_features):
        """
        매개변수:
            num_features (int): 입력 특성 벡터의 크기
        """
        super(ReviewClassifier, self).__init__()
        self.fc1 = nn.Linear(in_features=num_features, out_features=1)

    def forward(self, x_in, apply_sigmoid=False):
        """
        분류기의 순전파 계산

        매개변수:
            x_in (torch.Tensor) : 입력 데이터 텐서
                x_in.shape는 (batch, num_features) 이다.
            apply_sigmoid (bool) : 시그모이드 활성화 함수를 위한 플래그
                크로스 엔트로피 손실을 사용하려면 False로 지정
        반환:
            결과 텐서: tensor.shape는 (batch, )이다.
        """
        y_out = self.fc1(x_in).squeeze()
        if apply_sigmoid:
            y_out = torch.sigmoid(y_out)
        
        return y_out

