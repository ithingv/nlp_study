from collections import defaultdict
import numpy as np
import pandas as pd
import re

from argparse import Namespace

args = Namespace(
    raw_train_dataset_csv="yelp/raw_train.csv",
    raw_test_dataset_csv="yelp/raw_test.csv",
    proportion_subset_of_train=0.1,
    train_proportion=0.7,
    val_proportion=0.15,
    test_proportion=0.15,
    output_munged_csv="yelp/reviews_with_splits_lite.csv",
    seed=1337
)

train_reviews = pd.read_csv(args.raw_train_dataset_csv, header=None, names=['rating', 'review'])

# 리뷰 클래스 비율이 동일하도록 만든다.

by_rating = defaultdict(list)
for _, row in train_reviews.iterrows():
    by_rating[row.rating].append(row.to_dict())

review_subset = []

for _, item_list in sorted(by_rating.items()):

    n_total = len(item_list)
    n_subset = int(args.proportion_subset_of_train * n_total)
    review_subset.extend(item_list[:n_subset])

review_subset = pd.DataFrame(review_subset)

# 훈련, 검증, 테스트를 만들기 위해 별점 기준으로 나눈다.

by_rating = defaultdict(list)
for _, row in review_subset.iterrows():
    by_rating[row.rating].append(row.to_dict())

# 분할 데이터를 만듭니다.
final_list = []
np.random.seed(args.seed)

for _, item_list in sorted(by_rating.items()):
    
    np.random.shuffle(item_list)
    
    n_total = len(item_list)
    n_train = int(args.train_proportion * n_total)
    n_val = int(args.val_proportion * n_total)
    n_test = int(args.test_proportion * n_total)
    
    # 데이터 포인터에 분할 속성을 추가합니다
    for item in item_list[:n_train]:
        item['split'] = 'train'
    
    for item in item_list[n_train:n_train+n_val]:
        item['split'] = 'val'
        
    for item in item_list[n_train+n_val:n_train+n_val+n_test]:
        item['split'] = 'test'

    # 최종 리스트에 추가합니다
    final_list.extend(item_list)

# 분할 데이터를 데이터 프레임으로 만듭니다
final_reviews = pd.DataFrame(final_list)


# 리뷰를 전처리합니다
def preprocess_text(text):
    text = text.lower()
    text = re.sub(r"([.,!?])", r" \1 ", text)
    text = re.sub(r"[^a-zA-Z.,!?]+", r" ", text)
    return text
    
final_reviews.review = final_reviews.review.apply(preprocess_text)

final_reviews['rating'] = final_reviews.rating.apply({1: 'negative', 2: 'positive'}.get)
final_reviews.to_csv(args.output_munged_csv, index=False)