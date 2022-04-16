from torch.utils.data import Dataset
import pandas as pd

class ReviewDataset(Dataset):
    def __init__(self, review_df, vectorizer):
        """
        매개변수:
            review_df (pandas.DataFrame): 데이터
            vectorizer (ReviewVectorizer): ReviewVectorizer 객체
        """
        self.review_df = review_df
        self._vectorizer = vectorizer
        # 훈련 데이터
        self.train_df = self.review_df[self.review_df.split == 'train']
        self.train_size = len(self.train_df)
        # 검증 데이터
        self.val_df = self.review_df[self.review_df.split == 'val']
        self.val_size = len(self.val_df)
        # 테스트 데이터
        self.test_df = self.review_df[self.review_df.split == 'test']
        self.test_size = len(self.test_df)

        self._lookup_dict = {
            'train': (self.train_df, self.train_size), 
            'val': (self.val_df, self.val_size),
            'test': (self.test_df, self.test_size)
        }

        self.set_split('train')

    @classmethod
    def load_dataset_and_make_vectorizer(cls, review_csv):
        """
        데이터셋을 로드하고 새로운 ReviewVectorizer 객체를 생성한다.
        매개변수:
            review_csv (str): 데이터셋의 위치
        반환:
            ReviewDataset의 인스턴스
        """
        review_df = pd.read_csv(review_csv)
        return cls(review_df, ReviewVectorizer.from_dataframe(review_df))

    def get_vectorizer(self):
        """
        ReviewVectorzer 객체를 반환
        """
        return self._vectorizer

    def set_split(self, split='train'):
        """
        데이터 프레임에 있는 열을 사용해 분할 세트를 선택한다.

        매개변수:
            split (str): "train", "val", "test" 중 하나
        """
        self._target_split = split
        self._target_df, self._target_size = self._lookup_dict[split]
    
    def __len__(self):
        return self._target_size

    def __getitem__(self, index):
        """
        파이토치 데이터셋의 주요 진입 메서드

        매개변수:
            index (int): 데이터 포인트의 인덱스
        반환:
            데이터 포인트의 특성(x_data)과 레이블(y_target)으로 이루어진 딕셔너리
        """
        row = self._target_df.iloc[index]

        review_vector = self._vectorizer.vectorize(row.review)

        rating_index = self._vectorizer.rating_vocab.lookup_token(row.rating)