from data.vocabulary import Vocabulary
from collections import Counter
import string
import numpy as np

class ReviewVectorizer(object):
    """어휘 사전을 생성하고 관리한다."""
    def __init__(self, review_vocab, rating_vocab):
        """
        매개변수:
            review_vocab (Vocabulary): 단어를 정수에 매핑하는 Vocabulary
            rating_vocab (Vocabulary): 클래스 레이블을 정수에 매핑하는 Vocabulary
        """
        self.review_vocab = review_vocab
        self.rating_vocab = rating_vocab

    def vectorize(self, review):
        """
        리뷰에 대한 원-핫 벡터를 만든다

        매개변수:
            review (str): 리뷰
        반환:
            one-hot (np.ndarray): 원-핫 벡터
        """
        one_hot = np.zeros(len(self.review_vocab), dtype=np.float32)
        
        for token in review.split(" "):
            if token not in string.punctuation:
                one_hot[self.review_vocab.lookup_token(token)] = 1
        
        return one_hot

    @classmethod
    def from_dataframe(cls, review_df, cutoff=25):
        """
        데이터셋 데이터프레임에서 Vectorizer 객체를 만든다.

        매개변수:
            review_df (pandas Dataframe): 리뷰 데이터셋
            cutoff (int): 빈도 기반 필터링 설정
        반환:
            ReviewVectorizer 객체
        """
        review_vocab = Vocabulary(add_unk=True)
        rating_vocab = Vocabulary(add_unk=False)

        # 점수를 추가
        for rating in sorted(set(review_df.rating)):
            rating_vocab.add_token(rating)

        
        # count > cutoff인 단어를 추가
        word_counts = Counter()
        for review in review_df.review:
            for word in review.split(" "):
                if word not in string.puntuation:
                    word_counts[word] += 1

        for word, count in word_counts.items():
            if count > cutoff:
                review_vocab.add_token(word)
        
        return cls(review_vocab, rating_vocab)

    @classmethod
    def from_serializable(cls, contents):
        """
        직렬화된 딕셔너리에서 ReviewVectorizer 객체를 만든다.
        매개변수:
            contents (dict): 직렬화된 딕셔너리
        반환:
            ReviewVectorizer 클래스 객체
        """
        review_vocab = Vocabulary.from_serializable(contents['review_vocab'])
        rating_vocab = Vocabulary.from_serializable(contents['rating_vocab'])

        return cls(review_vocab=review_vocab, rating_vocab=rating_vocab)

    def to_serializable(self):
        """
        캐싱을 위해 직렬화된 딕셔너리를 만든다.

        반환:
            contents (dict): 직렬화된 딕셔너리
        """
        return {
            "review_vocab": self.review_vocab.to_serializable(),
            "rating_vocab": self.rating_vocab.to_serializable()
        }
        