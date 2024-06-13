# FeatureExtractorでの実施事項
# - 時系列データをdataframe, numpy array, torch tensorの状態からtorch tensor化
# - input validation

from typing import List, Optional, Union

from pandas import DataFrame
import numpy as np
import torch
import tensorflow as tf
import jax.numpy as jnp

from transformers import FeatureExtractionMixin
from transformers import TensorType
from transformers import BatchFeature
from transformers.utils import logging

logger = logging.get_logger(__name__)


class MomentFeatureExtractor(FeatureExtractionMixin):

    # TODO: 本来はMoment側のTokenizerもts_tokenizerとして入れたかったが、モデルに組み込まれてしまっている。
    # refers: https://github.com/moment-timeseries-foundation-model/moment/blob/088b253a1138ac7e48a7efc9bf902336c9eec8d9/momentfm/models/moment.py#L105

    model_input_names = ["time_series_values", "input_mask"]

    def __init__(self, **kwargs):
        super().__init__(**kwargs)


    def __call__(
        self,
        time_series: Union[DataFrame, np.ndarray, torch.Tensor, List[DataFrame], List[np.ndarray], List[torch.Tensor]] = None,
        return_tensors: Optional[Union[str, TensorType]] = TensorType.PYTORCH,
        torch_dtype: Optional[Union[str, torch.dtype]] = torch.float,
    ) -> BatchFeature:
        if time_series is not None:
            time_series_values = self._convert_time_series(time_series, return_tensors, torch_dtype)
        else:
            time_series_values = None

        return BatchFeature(data=time_series_values)


    def _convert_time_series(self, time_series, return_tensors, torch_dtype):
        # DataFrame, np.ndarray, または torch.Tensor を torch.Tensor に変換
        if isinstance(time_series, list):
            # リスト内の各要素を torch.Tensor に変換し、最終的には1つのTensorに結合
            time_series_tensor = torch.stack([self._convert_to_tensor(ts, torch_dtype) for ts in time_series])
        else:
            time_series_tensor = self._convert_to_tensor(time_series, torch_dtype)

        # 次元数の確認
        if time_series_tensor.dim() > 3:
            raise ValueError("time_series_tensor must not have more than 3 dimensions")
        elif time_series_tensor.dim() == 2:
            time_series_tensor = time_series_tensor.unsqueeze(0)
        elif time_series_tensor.dim() == 1:
            time_series_tensor = time_series_tensor.unsqueeze(0).unsqueeze(0)

        # 形式の出力
        batch_size, n_channels, d_model = time_series_tensor.shape
        logger.info(f"Batch size: {batch_size}, Number of channels: {n_channels}, Dimension of model: {d_model}")

        # seq_lenを最大値512までに絞り込み
        if time_series_tensor.shape[2] > 512:
            time_series_tensor = time_series_tensor[:, :, :512]
            logger.info("Sequence length has been truncated to 512.")

        # return_tensorsの指定に応じてデータ形式を変換
        if return_tensors == 'pt' or return_tensors == TensorType.PYTORCH:
            return time_series_tensor
        elif return_tensors == 'np' or return_tensors == TensorType.NUMPY:
            return time_series_tensor.numpy()
        elif return_tensors == 'tf' or return_tensors == TensorType.TENSORFLOW:
            return tf.convert_to_tensor(time_series_tensor.numpy())
        elif return_tensors == 'jax' or return_tensors == TensorType.JAX:
            return jnp.array(time_series_tensor.numpy())
        else:
            raise ValueError("Unsupported return_tensors type")
        
    def _convert_to_tensor(self, time_series, torch_dtype):
        if isinstance(time_series, DataFrame):
            time_series_tensor = torch.tensor(time_series.values, dtype=torch_dtype).t()
        elif isinstance(time_series, np.ndarray):
            time_series_tensor = torch.tensor(time_series, dtype=torch_dtype)
        elif isinstance(time_series, torch.Tensor):
            time_series_tensor = time_series.to(torch_dtype)

        return time_series_tensor
