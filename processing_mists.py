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
    attributes = ["time_series_processor", "tokenizer"]
    time_series_processor_class = "ProcessorMixin"
    tokenizer_class = "AutoTokenizer"

    def __init__(self, time_series_processor=None, tokenizer=None):
        super().__init__(time_series_processor, tokenizer)


    def __call__(
        self,
        text: Union[TextInput, PreTokenizedInput, List[TextInput], List[PreTokenizedInput]] = None,
        time_series: Union[DataFrame, np.ndarray, torch.Tensor, List[DataFrame], List[np.ndarray], List[torch.Tensor]] = None,
        padding: Union[bool, str, PaddingStrategy] = False,
        truncation: Union[bool, str, TruncationStrategy] = None,
        max_length=None,
        return_tensors: Optional[Union[str, TensorType]] = TensorType.PYTORCH,
        torch_dtype: Optional[Union[str, torch.dtype]] = torch.float,
    ) -> BatchFeature:
        if time_series is not None:
            time_series_values = self.time_series_processor(time_series, return_tensors, torch_dtype)
        else:
            time_series_values = None
        text_inputs = self.tokenizer(
            text, return_tensors=return_tensors, padding=padding, truncation=truncation, max_length=max_length
        )

        return BatchFeature(data={**text_inputs, "time_series_values": time_series_values})
