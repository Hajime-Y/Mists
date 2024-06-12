import warnings

from transformers import PretrainedConfig
from transformers import CONFIG_MAPPING

from timeseries_model.configuration_moment import MomentConfig

class MistsConfig(PretrainedConfig):
    model_type = "mists"

    def __init__(
        self, 
        time_series_config=None,
        text_config=None,
        # ignore_index=-100,
        time_series_token_index=32000,
        projector_hidden_act="gelu",  # projector用
        # time_series_feature_select_strategy="default",  # TODO: modelのforward用(画像モデルのhidden_stateからEmbeddingをどう取得するか)。将来的に対応。
        # time_series_feature_layer=-2,  # modelのforward用  # TODO: modelのforward用(画像モデルのhidden_stateからEmbeddingをどう取得するか)。将来的に対応。
        time_series_hidden_size=1024,  # projector用
        **kwargs,
    ):
        
        # self.ignore_index = ignore_index
        self.time_series_token_index = time_series_token_index
        self.projector_hidden_act = projector_hidden_act
        self.time_series_hidden_size = time_series_hidden_size

        # 将来的に、MomentモデルがTransformersに登録されることを想定して追加する
        # そのため、CONFIG_MAPPINGは機能しない。
        if isinstance(time_series_config, dict):
            time_series_config["model_type"] = (
                time_series_config["model_type"] if "model_type" in time_series_config else "moment"
            )
            # time_series_config = CONFIG_MAPPING[time_series_config["model_type"]](**time_series_config)
            time_series_config = MomentConfig(**time_series_config)
        elif time_series_config is None:
            time_series_config = MomentConfig()

        self.time_series_config = time_series_config

        if isinstance(text_config, dict):
            text_config["model_type"] = text_config["model_type"] if "model_type" in text_config else "mistral"
            text_config = CONFIG_MAPPING[text_config["model_type"]](**text_config)
        elif text_config is None:
            text_config = CONFIG_MAPPING["mistral"]()

        self.text_config = text_config

        super().__init__(**kwargs)


    def to_dict(self):
        output = super().to_dict()
        return output



        

