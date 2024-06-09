# Processerでの実施事項
# - TokenizerでTokenize
# - 時系列データをdataframe, numpy array, torch tensorの状態からtorch tensor化
# input_ids: , attention_mask: , time_series_values: の形式で返す。

from typing import List, Optional, Union

from pandas import DataFrame
import numpy as np
import torch
import tensorflow as tf
import jax.numpy as jnp

from transformers import ProcessorMixin
from transformers import TensorType
from transformers import BatchFeature
from transformers.tokenization_utils_base import PaddingStrategy, PreTokenizedInput, TextInput, TruncationStrategy


class MistsProcessor(ProcessorMixin):

    # 本来はMoment側のTokenizerもts_tokenizerとして入れたかったが、モデルに組み込まれてしまっている。
    # refers: https://github.com/moment-timeseries-foundation-model/moment/blob/088b253a1138ac7e48a7efc9bf902336c9eec8d9/momentfm/models/moment.py#L105

    # この2パーツが本来はts_tokenizerの領分になる気がする。
    # (normalizer): RevIN()
    # (tokenizer): Patching()
    attributes = ["tokenizer"]
    tokenizer_class = "AutoTokenizer"

    def __init__(self, tokenizer=None):
        super().__init__(tokenizer)


    def __call__(
        self,
        text: Union[TextInput, PreTokenizedInput, List[TextInput], List[PreTokenizedInput]] = None,
        time_series: Union[DataFrame, np.ndarray, torch.Tensor, List[DataFrame], List[np.ndarray], List[torch.Tensor]] = None,
        padding: Union[bool, str, PaddingStrategy] = False,
        truncation: Union[bool, str, TruncationStrategy] = None,
        max_length=None,
        return_tensors: Optional[Union[str, TensorType]] = TensorType.PYTORCH,
    ) -> BatchFeature:
        if time_series is not None:
            time_series_values = self._convert_time_series(time_series, return_tensors)
        else:
            time_series_values = None
        text_inputs = self.tokenizer(
            text, return_tensors=return_tensors, padding=padding, truncation=truncation, max_length=max_length
        )

        return BatchFeature(data={**text_inputs, "time_series_values": time_series_values})


    def _convert_time_series(self, time_series, return_tensors):
        # DataFrame, np.ndarray, または torch.Tensor を torch.Tensor に変換
        if isinstance(time_series, DataFrame):
            time_series_tensor = torch.tensor(time_series.values)
        elif isinstance(time_series, np.ndarray):
            time_series_tensor = torch.tensor(time_series)
        elif isinstance(time_series, torch.Tensor):
            time_series_tensor = time_series
        elif isinstance(time_series, list):
            # リスト内の各要素を torch.Tensor に変換し、最終的には1つのTensorに結合
            time_series_tensor = torch.stack([torch.tensor(ts.values) if isinstance(ts, DataFrame) else torch.tensor(ts) if isinstance(ts, np.ndarray) else ts for ts in time_series])
        else:
            raise ValueError("Unsupported time_series type")

        # return_tensorsの指定に応じてデータ形式を変換
        if return_tensors == 'pt':
            return time_series_tensor
        elif return_tensors == 'np':
            return time_series_tensor.numpy()
        elif return_tensors == 'tf':
            return tf.convert_to_tensor(time_series_tensor.numpy())
        elif return_tensors == 'jax':
            return jnp.array(time_series_tensor.numpy())    
        else:
            raise ValueError("Unsupported return_tensors type")
