"""Moment model configuration"""

from transformers import PretrainedConfig
from transformers import logging


DEFAULT_T5_CONFIG = {
    # "_name_or_path": "google/flan-t5-large",
    # "architectures": [
    #     "T5ForConditionalGeneration"
    # ],
    "classifier_dropout": 0.0,
    "d_ff": 2816,
    "d_kv": 64,
    "d_model": 1024,
    "decoder_start_token_id": 0,
    "dense_act_fn": "gelu_new",
    "dropout_rate": 0.1,
    "eos_token_id": 1,
    "feed_forward_proj": "gated-gelu",
    "initializer_factor": 1.0,
    "is_encoder_decoder": False,
    "is_gated_act": True,
    "layer_norm_epsilon": 1e-06,
    # "model_type": "t5",
    "n_positions": 512,
    "num_decoder_layers": 24,
    "num_heads": 16,
    "num_layers": 24,
    "output_past": True,
    "pad_token_id": 0,
    "relative_attention_max_distance": 128,
    "relative_attention_num_buckets": 32,
    "tie_word_embeddings": False,
    # "transformers_version": "4.33.3",
    "use_cache": False,
    "vocab_size": 32128
}


class MomentConfig(PretrainedConfig):
    model_type = "moment"

    def __init__(
        self,
        t5_config: dict = DEFAULT_T5_CONFIG,
        d_model: int = None,
        seq_len: int = 512,
        patch_len: int = 16,
        patch_stride_len: int = 16,
        dropout: float = 0.1,
        revin_num_features: int = 1,
        revin_eps: float = 1e-5,
        revin_affine: bool = True,
        add_positional_embedding: bool = True,
        value_embedding_bias: bool = False,
        orth_gain: float = 1.41,
        mask_ratio: float = 0.15,
        freeze_embedder: bool = True,
        freeze_encoder: bool = True,
        freeze_head: bool = False,
        enable_gradient_checkpointing: bool = True,
        randomly_initialize_backbone: bool = False,
        **kwargs
    ):
        self.t5_config = self._init_t5_config(t5_config)
        self.d_model = d_model
        self.seq_len = seq_len
        self.patch_len = patch_len
        self.patch_stride_len = patch_stride_len
        self.dropout = dropout
        self.revin_num_features = revin_num_features
        self.revin_eps = revin_eps
        self.revin_affine = revin_affine
        self.add_positional_embedding = add_positional_embedding
        self.value_embedding_bias = value_embedding_bias
        self.orth_gain = orth_gain
        self.mask_ratio = mask_ratio
        self.freeze_embedder = freeze_embedder
        self.freeze_encoder = freeze_encoder
        self.freeze_head = freeze_head
        self.enable_gradient_checkpointing = enable_gradient_checkpointing
        self.randomly_initialize_backbone = randomly_initialize_backbone

        self._validation_config()

        super().__init__(**kwargs)

    def _init_t5_config(self, config: dict):
        if config is None:
            return DEFAULT_T5_CONFIG
        else:
            # 与えられたconfigでDEFAULT_T5_CONFIGを更新
            updated_config = DEFAULT_T5_CONFIG.copy()
            updated_config.update(config)
            return updated_config
        
    def _validation_config(self):
        """
        Validate configuration.
        """
        if self.d_model is None:
            self.d_model = self.t5_config["d_model"]
