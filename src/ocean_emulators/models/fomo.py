from ocean_emulators.config import EncoderConfig, SamudraConfig
from ocean_emulators.constants import Grid
from ocean_emulators.models.base import BaseModel


class FOMO(BaseModel):
    def __init__(
        self,
        encoder_config: EncoderConfig,
        samudra_config: SamudraConfig,
        n_channels: int,
        hist: int,
        wet: Grid,
        area_weights: Grid,
        static_data,
    ):
        super().__init__(
            ch_width=[n_channels]
            + samudra_config.ch_width,  # Only the first and last channel matter.
            n_out=samudra_config.n_out,
            wet=wet,
            hist=hist,
            pred_residuals=samudra_config.pred_residuals,
            last_kernel_size=samudra_config.last_kernel_size,
            pad=samudra_config.pad,
            static_data=static_data,
        )
