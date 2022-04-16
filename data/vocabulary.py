class Vocabulary(object):
    """
    매핑을 위해 텍스트를 저치하고 어휘 사전을 만드는 클래스
    """

    def __init__(self, token_to_idx=None, add_unk=True, unk_token="<UNK>"):
        """
        매개변수:
            token_to_idx (dict): 기존 토큰-인덱스 매핑 딕셔너리
            add_unk (bool): UNK 토큰을 추가할지 지정하는 플래그
            unk_token (str): Vocabulary에 추가할 UNK 토큰
        """
        if token_to_idx is None:
            token_to_idx = {}
        
        self._token_to_idx = token_to_idx
        self._idx_to_token = { idx : token
            for token, idx
            in self._token_to_idx.items()
        }

        self._add_unk = add_unk
        self._unk_token = unk_token

        self.unk_index = -1
        if add_unk:
            self.unk_idx = self.add_token(unk_token)

    def to_serializable(self):
        """
        직렬화할 수 있는 딕셔너리를 반환
        """
        return {
            "token_to_idx": self._token_to_idx,
            "add_unk": self._add_unk,
            "unk_token": self._unk_token
        }
    
    @classmethod
    def from_serializable(cls, contents):
        """
        직렬화된 딕셔너리에서 Vocabulary 객체를 만든다.
        """
        return cls(**contents)


    def add_token(self, token):
        """
        토큰을 기반으로 매핑 딕셔너리를 업데이트 합니다.

        매개변수:
            token (str): Vocabulary에 추가할 토큰
        반환:
            index (int): 토큰에 상응하는 정수
        """
        if token in self._token_to_idx:
            idx = self._token_to_idx[token]
        else:
            idx = len(self._token_to_idx)
            self._token_to_idx[token] = idx
            self._idx_to_token[idx] = token
        return idx

    def lookup_token(self, token):
        """
        토큰에 대응하는 인덱스를 추출한다.
        토큰이 없으면 UNK 인덱스를 반환한다.

        매개변수:
            token (str): 찾을 토큰
        반환:
            index (int): 토큰에 해당하는 인덱스
        노트:
            UNK 토큰을 사용하려면 UNK_INDEX가 0보다 커야한다.
        """
        if self.add_unk:
            return self._token_to_idx.get(token, self.unk_index)
        else:
            return self._token_to_idx[token]

    def lookup_index(self, index):
        """
        인덱스에 해당하는 토큰을 반환
        
        매개변수:
            index (int): 찾을 인덱스
        반환:
            token (str): 인덱스에 해당하는 토큰
        에러:
            KeyError: 인덱스가 Vocabulary에 없을 때 발생
        """
        if index not in self._idx_to_token:
            raise KeyError("Vocabulary에 인덱스 (%d)가 없습니다." % index)
        else:
            return self._idx_to_token[index]

    def __str__(self):
        return "<Vocabulary(size=%d)>" % len(self)
    
    def __len__(self):
        return len(self._token_to_idx)
