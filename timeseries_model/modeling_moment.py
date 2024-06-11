# Auton LabによるMomentライブラリをTransformers向けに書き換えたものです。
# Embeddingに特化したアーキテクチャとなっています。
# refers: https://github.com/moment-timeseries-foundation-model/moment

from dataclasses import dataclass
from typing import List, Optional, Tuple, Union

import math
import numpy.typing as npt
import torch
from torch import nn

from transformers import PreTrainedModel
from transformers import T5Config, T5Model
from transformers.utils import logging

from .configuration_moment import MomentConfig

logger = logging.get_logger(__name__)

@dataclass
class TimeseriesOutputs:
    # forecast: npt.NDArray = None
    # anomaly_scores: npt.NDArray = None
    logits: npt.NDArray = None
    labels: int = None
    input_mask: npt.NDArray = None
    pretrain_mask: npt.NDArray = None
    # reconstruction: npt.NDArray = None
    embeddings: npt.NDArray = None
    metadata: dict = None
    illegal_output: bool = False
    hidden_states: npt.NDArray = None  # For Mists model


# refers: https://github.com/moment-timeseries-foundation-model/moment/blob/088b253a1138ac7e48a7efc9bf902336c9eec8d9/momentfm/utils/masking.py#L6C1-L6C2
class Masking:
    def __init__(
        self, mask_ratio: float = 0.3, patch_len: int = 8, stride: Optional[int] = None
    ):
        """
        Indices with 0 mask are hidden, and with 1 are observed.
        """
        self.mask_ratio = mask_ratio
        self.patch_len = patch_len
        self.stride = patch_len if stride is None else stride

    @staticmethod
    def convert_seq_to_patch_view(
        mask: torch.Tensor, patch_len: int = 8, stride: Optional[int] = None
    ):
        """
        Input:
            mask : torch.Tensor of shape [batch_size x seq_len]
        Output
            mask : torch.Tensor of shape [batch_size x n_patches]
        """
        stride = patch_len if stride is None else stride
        mask = mask.unfold(dimension=-1, size=patch_len, step=stride)
        # mask : [batch_size x n_patches x patch_len]
        return (mask.sum(dim=-1) == patch_len).long()

    @staticmethod
    def convert_patch_to_seq_view(
        mask: torch.Tensor,
        patch_len: int = 8,
    ):
        """
        Input:
            mask : torch.Tensor of shape [batch_size x n_patches]
        Output:
            mask : torch.Tensor of shape [batch_size x seq_len]
        """
        return mask.repeat_interleave(patch_len, dim=-1)

    def generate_mask(self, x: torch.Tensor, input_mask: Optional[torch.Tensor] = None):
        """
        Input:
            x : torch.Tensor of shape
            [batch_size x n_channels x n_patches x patch_len] or
            [batch_size x n_channels x seq_len]
            input_mask: torch.Tensor of shape [batch_size x seq_len] or
            [batch_size x n_patches]
        Output:
            mask : torch.Tensor of shape [batch_size x seq_len]
        """
        if x.ndim == 4:
            return self._mask_patch_view(x, input_mask=input_mask)
        elif x.ndim == 3:
            return self._mask_seq_view(x, input_mask=input_mask)

    def _mask_patch_view(self, x, input_mask=None):
        """
        Input:
            x : torch.Tensor of shape
            [batch_size x n_channels x n_patches x patch_len]
            input_mask: torch.Tensor of shape [batch_size x seq_len]
        Output:
            mask : torch.Tensor of shape [batch_size x n_patches]
        """
        input_mask = self.convert_seq_to_patch_view(
            input_mask, self.patch_len, self.stride
        )
        n_observed_patches = input_mask.sum(dim=-1, keepdim=True)  # batch_size x 1

        batch_size, _, n_patches, _ = x.shape
        len_keep = torch.ceil(n_observed_patches * (1 - self.mask_ratio)).long()
        noise = torch.rand(
            batch_size, n_patches, device=x.device
        )  # noise in [0, 1], batch_size x n_channels x n_patches
        noise = torch.where(
            input_mask == 1, noise, torch.ones_like(noise)
        )  # only keep the noise of observed patches

        # Sort noise for each sample
        ids_shuffle = torch.argsort(
            noise, dim=1
        )  # Ascend: small is keep, large is remove
        ids_restore = torch.argsort(
            ids_shuffle, dim=1
        )  # ids_restore: [batch_size x n_patches]

        # Generate the binary mask: 0 is keep, 1 is remove
        mask = torch.zeros(
            [batch_size, n_patches], device=x.device
        )  # mask: [batch_size x n_patches]
        for i in range(batch_size):
            mask[i, : len_keep[i]] = 1

        # Unshuffle to get the binary mask
        mask = torch.gather(mask, dim=1, index=ids_restore)

        return mask.long()

    def _mask_seq_view(self, x, input_mask=None):
        """
        Input:
            x : torch.Tensor of shape
            [batch_size x n_channels x seq_len]
            input_mask: torch.Tensor of shape [batch_size x seq_len]
        Output:
            mask : torch.Tensor of shape [batch_size x seq_len]
        """
        x = x.unfold(dimension=-1, size=self.patch_len, step=self.stride)
        mask = self._mask_patch_view(x, input_mask=input_mask)
        return self.convert_patch_to_seq_view(mask, self.patch_len).long()


# refers: https://github.com/moment-timeseries-foundation-model/moment/blob/088b253a1138ac7e48a7efc9bf902336c9eec8d9/momentfm/models/layers/revin.py#L5
def nanvar(tensor, dim=None, keepdim=False):
    tensor_mean = tensor.nanmean(dim=dim, keepdim=True)
    output = (tensor - tensor_mean).square().nanmean(dim=dim, keepdim=keepdim)
    return output

# refers: https://github.com/moment-timeseries-foundation-model/moment/blob/088b253a1138ac7e48a7efc9bf902336c9eec8d9/momentfm/models/layers/revin.py#L11
def nanstd(tensor, dim=None, keepdim=False):
    output = nanvar(tensor, dim=dim, keepdim=keepdim)
    output = output.sqrt()
    return output

# refers: https://github.com/moment-timeseries-foundation-model/moment/blob/088b253a1138ac7e48a7efc9bf902336c9eec8d9/momentfm/models/layers/revin.py#L17
class RevIN(nn.Module):
    def __init__(self, num_features: int, eps: float = 1e-5, affine: bool = False):
        """
        :param num_features: the number of features or channels
        :param eps: a value added for numerical stability
        :param affine: if True, RevIN has learnable affine parameters
        """
        super(RevIN, self).__init__()
        self.num_features = num_features
        self.eps = eps
        self.affine = affine

        if self.affine:
            self._init_params()

    def forward(self, x: torch.Tensor, mode: str = "norm", mask: torch.Tensor = None):
        """
        :param x: input tensor of shape (batch_size, n_channels, seq_len)
        :param mode: 'norm' or 'denorm'
        :param mask: input mask of shape (batch_size, seq_len)
        :return: RevIN transformed tensor
        """
        if mode == "norm":
            self._get_statistics(x, mask=mask)
            x = self._normalize(x)
        elif mode == "denorm":
            x = self._denormalize(x)
        else:
            raise NotImplementedError
        return x

    def _init_params(self):
        # initialize RevIN params: (C,)
        self.affine_weight = nn.Parameter(torch.ones(1, self.num_features, 1))
        self.affine_bias = nn.Parameter(torch.zeros(1, self.num_features, 1))

    def _get_statistics(self, x, mask=None):
        """
        x    : batch_size x n_channels x seq_len
        mask : batch_size x seq_len
        """
        if mask is None:
            mask = torch.ones((x.shape[0], x.shape[-1]))
        n_channels = x.shape[1]
        mask = mask.unsqueeze(1).repeat(1, n_channels, 1).bool()
        # Set masked positions to NaN, and unmasked positions are taken from x
        masked_x = torch.where(mask, x, torch.nan)
        self.mean = torch.nanmean(masked_x, dim=-1, keepdim=True).detach()
        self.stdev = nanstd(masked_x, dim=-1, keepdim=True).detach() + self.eps
        # self.stdev = torch.sqrt(
        #     torch.var(masked_x, dim=-1, keepdim=True) + self.eps).get_data().detach()
        # NOTE: By default not bessel correction

    def _normalize(self, x):
        x = x - self.mean
        x = x / self.stdev

        if self.affine:
            x = x * self.affine_weight
            x = x + self.affine_bias
        return x

    def _denormalize(self, x):
        if self.affine:
            x = x - self.affine_bias
            x = x / (self.affine_weight + self.eps * self.eps)
        x = x * self.stdev
        x = x + self.mean
        return x


# refers: https://github.com/moment-timeseries-foundation-model/moment/blob/088b253a1138ac7e48a7efc9bf902336c9eec8d9/momentfm/models/layers/embed.py#L10
class PositionalEmbedding(nn.Module):
    def __init__(self, d_model, max_len=5000, model_name="MOMENT"):
        super(PositionalEmbedding, self).__init__()
        self.model_name = model_name

        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, d_model).float()
        pe.require_grad = False

        position = torch.arange(0, max_len).float().unsqueeze(1)
        div_term = (
            torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model)
        ).exp()

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0)
        self.register_buffer("pe", pe)

    def forward(self, x):
        if (
            self.model_name == "MOMENT"
            or self.model_name == "TimesNet"
            or self.model_name == "GPT4TS"
        ):
            return self.pe[:, : x.size(2)]
        else:
            return self.pe[:, : x.size(1)]
        

# refers: https://github.com/moment-timeseries-foundation-model/moment/blob/088b253a1138ac7e48a7efc9bf902336c9eec8d9/momentfm/models/layers/embed.py#L181
class PatchEmbedding(nn.Module):
    def __init__(
        self,
        d_model: int = 768,
        seq_len: int = 512,
        patch_len: int = 8,
        stride: int = 8,
        dropout: int = 0.1,
        add_positional_embedding: bool = False,
        value_embedding_bias: bool = False,
        orth_gain: float = 1.41,
    ):
        super(PatchEmbedding, self).__init__()
        self.patch_len = patch_len
        self.seq_len = seq_len
        self.stride = stride
        self.d_model = d_model
        self.add_positional_embedding = add_positional_embedding

        self.value_embedding = nn.Linear(patch_len, d_model, bias=value_embedding_bias)
        self.mask_embedding = nn.Parameter(torch.zeros(d_model))

        if orth_gain is not None:
            torch.nn.init.orthogonal_(self.value_embedding.weight, gain=orth_gain)
            if value_embedding_bias:
                self.value_embedding.bias.data.zero_()
            # torch.nn.init.orthogonal_(self.mask_embedding, gain=orth_gain) # Fails

        # Positional embedding
        if self.add_positional_embedding:
            self.position_embedding = PositionalEmbedding(d_model)

        # Residual dropout
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
        mask = Masking.convert_seq_to_patch_view(
            mask, patch_len=self.patch_len
        ).unsqueeze(-1)
        # mask : [batch_size x n_patches x 1]
        n_channels = x.shape[1]
        mask = (
            mask.repeat_interleave(self.d_model, dim=-1)
            .unsqueeze(1)
            .repeat(1, n_channels, 1, 1)
        )
        # mask : [batch_size x n_channels x n_patches x d_model]

        # Input encoding
        x = mask * self.value_embedding(x) + (1 - mask) * self.mask_embedding
        if self.add_positional_embedding:
            x = x + self.position_embedding(x)

        return self.dropout(x)


# refers: https://github.com/moment-timeseries-foundation-model/moment/blob/088b253a1138ac7e48a7efc9bf902336c9eec8d9/momentfm/models/layers/embed.py#L237C1-L251C17
class Patching(nn.Module):
    def __init__(self, patch_len: int, stride: int):
        super().__init__()
        self.patch_len = patch_len
        self.stride = stride
        if self.stride != self.patch_len:
            logger.warning(
                "Stride and patch length are not equal. "
                "This may lead to unexpected behavior."
            )

    def forward(self, x):
        x = x.unfold(dimension=-1, size=self.patch_len, step=self.stride)
        # x : [batch_size x n_channels x num_patch x patch_len]
        return x


class MomentPreTrainedModel(PreTrainedModel):
    config_class = MomentConfig

    base_model_prefix = "model"
    supports_gradient_checkpointing = True
    
    # 本来のT5の_init_weightsはもっと詳細だが、事前学習の予定はないためここでは簡単にしている。
    # refers: https://github.com/huggingface/transformers/blob/517df566f572d90e6301df87870f651f0d1b1110/src/transformers/models/t5/modeling_t5.py#L810
    def _init_weights(self, module):
        std = self.config.t5_config.initializer_factor
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()


class MomentEmbeddingModel(MomentPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.config = config
        self.seq_len = config.seq_len
        self.patch_len = config.patch_len

        # TODO: normalizer, tokenizerはProcessor側に配置するべきか？
        # 現状の考え: 特にMomentから切り離す用途もない。
        # 　　　　　　Processor側では入力の512timestepsへの切り取り等、
        # 　　　　　　input validationとTensorへの切り替えを行うで良さそう。
        self.normalizer = RevIN(
            num_features=1, affine=config.getattr("revin_affine", False)
        )
        self.tokenizer = Patching(
            patch_len=config.patch_len, stride=config.patch_stride_len
        )
        # モデル構成
        self.patch_embedding = PatchEmbedding(
            d_model=config.d_model,
            seq_len=config.seq_len,
            patch_len=config.patch_len,
            stride=config.patch_stride_len,
            dropout=config.getattr("dropout", 0.1),
            add_positional_embedding=config.getattr("add_positional_embedding", True),
            value_embedding_bias=config.getattr("value_embedding_bias", False),
            orth_gain=config.getattr("orth_gain", 1.41),
        )
        self.mask_generator = Masking(mask_ratio=config.getattr("mask_ratio", 0.0))
        self.encoder = self._get_t5_encoder(config.t5_config, config.enable_gradient_checkpointing)
        self.head = nn.Identity()

        # Frozen parameters
        self.freeze_embedder = config.getattr("freeze_embedder", True)
        self.freeze_encoder = config.getattr("freeze_encoder", True)
        self.freeze_head = config.getattr("freeze_head", False)

        if self.freeze_embedder:
            self.patch_embedding = freeze_parameters(self.patch_embedding)
        if self.freeze_encoder:
            self.encoder = freeze_parameters(self.encoder)
        if self.freeze_head:
            self.head = freeze_parameters(self.head)

    def _get_t5_encoder(self, config: T5Config, enable_gradient_checkpointing: bool) -> nn.Module:
        # random initialize
        # Momentでは(言語で)事前学習済みのモデルを取得することもできるようになっている
        # refers: https://github.com/moment-timeseries-foundation-model/moment/blob/088b253a1138ac7e48a7efc9bf902336c9eec8d9/momentfm/models/moment.py#L205
        t5_model = T5Model(config)
        t5_model_encoder = t5_model.get_encoder()

        if enable_gradient_checkpointing:
            t5_model_encoder.gradient_checkpointing_enable()
            logger.info("Enabling gradient checkpointing.")

        return t5_model_encoder
    
    def embed(
        self,
        x_enc: torch.Tensor,
        input_mask: torch.Tensor = None,
        reduction: str = "mean",
        **kwargs,
    ) -> TimeseriesOutputs:
        batch_size, n_channels, seq_len = x_enc.shape

        if input_mask is None:
            input_mask = torch.ones((batch_size, seq_len)).to(x_enc.device)

        x_enc = self.normalizer(x=x_enc, mask=input_mask, mode="norm")
        x_enc = torch.nan_to_num(x_enc, nan=0, posinf=0, neginf=0)

        input_mask_patch_view = Masking.convert_seq_to_patch_view(
            input_mask, self.patch_len
        )

        x_enc = self.tokenizer(x=x_enc)
        enc_in = self.patch_embedding(x_enc, mask=input_mask)

        n_patches = enc_in.shape[2]
        enc_in = enc_in.reshape(
            (batch_size * n_channels, n_patches, self.config.d_model)
        )

        patch_view_mask = Masking.convert_seq_to_patch_view(input_mask, self.patch_len)
        attention_mask = patch_view_mask.repeat_interleave(n_channels, dim=0)
        outputs = self.encoder(inputs_embeds=enc_in, attention_mask=attention_mask)
        enc_out = outputs.last_hidden_state

        enc_out = enc_out.reshape((-1, n_channels, n_patches, self.config.d_model))
        # [batch_size x n_channels x n_patches x d_model]

        # For Mists model
        # [batch_size, n_channels x n_patches, d_model]
        hidden_states = enc_out.reshape(batch_size, n_channels * n_patches, self.config.d_model)

        if reduction == "mean":
            enc_out = enc_out.mean(dim=1, keepdim=False)  # Mean across channels
            # [batch_size x n_patches x d_model]
            input_mask_patch_view = input_mask_patch_view.unsqueeze(-1).repeat(
                1, 1, self.config.d_model
            )
            enc_out = (input_mask_patch_view * enc_out).sum(
                dim=1
            ) / input_mask_patch_view.sum(dim=1)
        else:
            raise NotImplementedError(f"Reduction method {reduction} not implemented.")

        return TimeseriesOutputs(
            embeddings=enc_out, input_mask=input_mask, metadata=reduction, hidden_states=hidden_states
        )
    
    def forward(
        self,
        x_enc: torch.Tensor,
        mask: torch.Tensor = None,
        input_mask: torch.Tensor = None,
        **kwargs,
    ) -> TimeseriesOutputs:
        if input_mask is None:
            input_mask = torch.ones_like(x_enc[:, 0, :])

        return self.embed(x_enc=x_enc, input_mask=input_mask, **kwargs)


# refers: https://github.com/moment-timeseries-foundation-model/moment/blob/088b253a1138ac7e48a7efc9bf902336c9eec8d9/momentfm/models/moment.py#L601
def freeze_parameters(model):
    """
    Freeze parameters of the model
    """
    # Freeze the parameters
    for name, param in model.named_parameters():
        param.requires_grad = False

    return model