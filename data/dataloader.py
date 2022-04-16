# 데이터 셋에서 미니배치 생성
from torch.utils.data import DataLoader

def generate_batches(dataset, batch_size, shuffle=True, drop_last = True, device='cpu'):
    """
    파이토치 DataLoader 를 wrapping 하는 제너레이터 함수
    각 텐서를 지정된 장치로 이동한다.
    """
    dataloader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=shuffle, drop_last=drop_last)

    for data_dict in dataloader:
        out_data_dict = {}
        for name, _ in data_dict.items():
            out_data_dict[name] = data_dict[name].to(device)
        yield out_data_dict
        