# FeatureExtractorでの実施事項
# - 時系列データをdataframe, numpy array, torch tensorの状態からtorch tensor化
# - input validation

from typing import List, Optional, Union, Literal, Tuple

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

    model_input_names = ["time_series_values", "time_series_input_mask"]

    def __init__(self, **kwargs):
        super().__init__(**kwargs)


    """
    padding ( bool、strまたはPaddingStrategy、オプション、デフォルトはFalse): 
        paddingをアクティブ化および制御します。次の値を受け入れます:
         - True or 'longest': バッチ内の最長シーケンスにパディングします (シーケンスが 1 つだけの場合はパディングしません)。
         - 'max_length': 引数で指定された最大長までパディングします。max_length引数が指定されていない場合は、モデルで許容される最大入力長までパディングします。
         - False or 'do_not_pad'(デフォルト): パディングなし (つまり、異なる長さのシーケンスを含むバッチを出力できます)。
    """
    def __call__(
        self,
        time_series: Union[DataFrame, np.ndarray, torch.Tensor, List[DataFrame], List[np.ndarray], List[torch.Tensor]] = None,
        return_tensors: Optional[Union[str, TensorType]] = TensorType.PYTORCH,
        torch_dtype: Optional[Union[str, torch.dtype]] = torch.float,
        padding: Union[bool, str] = False,  # 追加
        max_length: Union[int, None] = None,   # 追加
    ) -> BatchFeature:
        if time_series is not None:
            time_series_values, input_mask = self._convert_time_series(time_series, return_tensors, torch_dtype, padding, max_length)
        else:
            time_series_values = None
            input_mask = None

        return BatchFeature(data={"time_series_values": time_series_values, "time_series_input_mask": input_mask})


    def _convert_time_series(self, time_series, return_tensors, torch_dtype, padding, max_length):
        # DataFrame, np.ndarray, または torch.Tensor を torch.Tensor に変換
        if isinstance(time_series, list):
            # リスト内の各要素を torch.Tensor に変換し、最終的には1つのTensorに結合
            time_series_list = [self._convert_to_tensor(ts, torch_dtype) for ts in time_series]
            # 次元数の確認
            time_series_list = [self._convert_tensor_dim(ts, dim=2) for ts in time_series_list]
            # trancate, padding
            time_series_tensor, input_mask = self._pad_time_series(time_series_list, padding, max_length)
        else:
            time_series_tensor = self._convert_to_tensor(time_series, torch_dtype)
            # 次元数の確認
            time_series_tensor = self._convert_tensor_dim(time_series_tensor, dim=3)
            # trancate, padding
            time_series_tensor, input_mask = self._pad_time_series(time_series_tensor, padding, max_length)

        # 形式の出力
        batch_size, n_channels, d_model = time_series_tensor.shape
        logger.info(f"Batch size: {batch_size}, Number of channels: {n_channels}, Dimension of model: {d_model}")

        # seq_lenを最大値512までに切り詰め
        if time_series_tensor.shape[2] > 512:
            time_series_tensor = time_series_tensor[:, :, :512]
            logger.info("Sequence length has been truncated to 512.")

        # return_tensorsの指定に応じてデータ形式を変換
        if return_tensors == 'pt' or return_tensors == TensorType.PYTORCH:
            return time_series_tensor, input_mask
        elif return_tensors == 'np' or return_tensors == TensorType.NUMPY:
            return time_series_tensor.numpy(), input_mask
        elif return_tensors == 'tf' or return_tensors == TensorType.TENSORFLOW:
            return tf.convert_to_tensor(time_series_tensor.numpy()), input_mask
        elif return_tensors == 'jax' or return_tensors == TensorType.JAX:
            return jnp.array(time_series_tensor.numpy()), input_mask
        else:
            raise ValueError("Unsupported return_tensors type")
        
    def _convert_to_tensor(self, time_series, torch_dtype):
        if isinstance(time_series, DataFrame):
            time_series_tensor = torch.tensor(time_series.values, dtype=torch_dtype).t()
        elif isinstance(time_series, np.ndarray) or isinstance(time_series, list):
            time_series_tensor = torch.tensor(time_series, dtype=torch_dtype)
        elif isinstance(time_series, torch.Tensor):
            time_series_tensor = time_series.to(torch_dtype)

        return time_series_tensor
    
    def _convert_tensor_dim(self, time_series, dim=3):
        if time_series.dim() > dim:
            raise ValueError("time_series must not have more than 3 dimensions")
        
        while time_series.dim() < dim:
            time_series = time_series.unsqueeze(0)
        
        return time_series
    

    def _pad_time_series(
        self,
        time_series_values: Union[torch.Tensor, List[torch.Tensor]],
        padding: Union[bool, Literal['longest', 'max_length', 'do_not_pad']] = 'do_not_pad',
        max_length: Union[int, None] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        時系列データにパディングを適用し、対応するinput_maskを生成する関数。

        Args:
            time_series_values (Union[torch.Tensor, List[torch.Tensor]]): 
                パディングする時系列データ。
                3次元テンソル (batch_size, n_channels, seq_len) または
                2次元テンソル (n_channels, seq_len) のリストを想定。
            padding (Union[bool, Literal['longest', 'max_length', 'do_not_pad']], optional): 
                パディングの種類。デフォルトは 'do_not_pad'。
                - True または 'longest': バッチ内の最長シーケンスにパディング
                - 'max_length': 指定された最大長までパディング
                - False または 'do_not_pad': パディングなし（最短シーケンスに合わせて切り捨て）
            max_length (Union[int, None], optional): 
                'max_length' パディング時の最大長。
                指定がない場合は512を使用。デフォルトは None。

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: 
                - パディングされた時系列データ。形状は (batch_size, n_channels, padded_seq_len)。
                - input_mask。形状は (batch_size, padded_seq_len)。
                1はデータが存在する部分、0はパディングされた部分を示す。

        Raises:
            ValueError: サポートされていない入力形状、無効なパディングオプション、
                        不適切なmax_length、またはチャンネル数の不一致の場合。
        """
        # max_lengthの検証
        if max_length is not None:
            if not isinstance(max_length, int) or max_length <= 0:
                raise ValueError("max_length は正の整数である必要があります。")

        if isinstance(time_series_values, list):
            if not all(isinstance(ts, torch.Tensor) and ts.dim() == 2 for ts in time_series_values):
                raise ValueError("リストの各要素は2次元のtorch.Tensorである必要があります。")
            
            batch_size = len(time_series_values)
            n_channels = time_series_values[0].shape[0]
            seq_lens = [ts.shape[1] for ts in time_series_values]

            # チャンネル数の一貫性チェック
            if not all(ts.shape[0] == n_channels for ts in time_series_values):
                raise ValueError("全ての時系列データは同じチャンネル数を持つ必要があります。")

        elif isinstance(time_series_values, torch.Tensor):
            if time_series_values.dim() == 3:
                batch_size, n_channels, seq_len = time_series_values.shape
                seq_lens = [seq_len] * batch_size
                time_series_values = [time_series_values[i] for i in range(batch_size)]
            elif time_series_values.dim() == 2:
                n_channels, seq_len = time_series_values.shape
                batch_size = 1
                seq_lens = [seq_len]
                time_series_values = [time_series_values]
            else:
                raise ValueError("テンソルは2次元または3次元である必要があります。")
        else:
            raise ValueError("入力は torch.Tensor または torch.Tensor のリストである必要があります。")

        if padding == True or padding == 'longest':
            target_len = max(seq_lens)
        elif padding == 'max_length':
            target_len = max_length if max_length is not None else 512
        elif padding == False or padding == 'do_not_pad':
            target_len = min(seq_lens)
        else:
            raise ValueError("無効なパディングオプションです。")

        # デバイスの一貫性を保証
        device = time_series_values[0].device

        padded_values = torch.zeros((batch_size, n_channels, target_len), dtype=time_series_values[0].dtype, device=device)
        input_mask = torch.zeros((batch_size, target_len), dtype=time_series_values[0].dtype, device=device)
        
        for i in range(batch_size):
            seq = time_series_values[i]
            length = min(seq.shape[1], target_len)
            padded_values[i, :, :length] = seq[:, :length]
            input_mask[i, :length] = True

        return padded_values, input_mask