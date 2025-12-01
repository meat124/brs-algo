"""
RGB-based WBVIMA Policy for BRS (replacing PointNet with ResNet18).

This policy uses ResNet18 for RGB image encoding instead of PointNet for point cloud encoding.
The overall architecture (Transformer + Diffusion head) remains the same.
"""

from typing import Optional, Union, List

import torch
import torch.nn as nn
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler
from diffusers.schedulers.scheduling_ddim import DDIMScheduler
from einops import rearrange

from brs_algo.learning.nn.common import MLP
from brs_algo.learning.nn.gpt.gpt import GPT
from brs_algo.learning.nn.features import ResNet18Encoder, ObsTokenizer
from brs_algo.learning.policy.base import BaseDiffusionPolicy
from brs_algo.learning.nn.diffusion import WholeBodyUNetDiffusionHead
from brs_algo.optim import default_optimizer_groups, check_optimizer_groups
from brs_algo.lightning.lightning import rank_zero_info


class WBVIMAPolicyRGB(BaseDiffusionPolicy):
    """
    WBVIMA Policy with RGB image input (using ResNet18).
    
    Replaces PointNet encoder with ResNet18 for visual observation encoding.
    """
    is_sequence_policy = True

    def __init__(
        self,
        *,
        prop_dim: int,
        prop_keys: List[str],
        prop_mlp_hidden_depth: int,
        prop_mlp_hidden_dim: int,
        # ====== ResNet18 Encoder ======
        resnet_pretrained: bool = True,
        resnet_freeze_backbone: bool = False,
        num_camera_views: int = 1,
        num_latest_obs: int,
        use_modality_type_tokens: bool,
        # ====== Transformer ======
        xf_n_embd: int,
        xf_n_layer: int,
        xf_n_head: int,
        xf_dropout_rate: float,
        xf_use_geglu: bool,
        # ====== Action Decoding ======
        learnable_action_readout_token: bool,
        action_dim: int,
        action_prediction_horizon: int,
        diffusion_step_embed_dim: int,
        unet_down_dims: List[int],
        unet_kernel_size: int,
        unet_n_groups: int,
        unet_cond_predict_scale: bool,
        action_keys: List[str],
        action_key_dims: dict[str, int],
        # ====== Diffusion ======
        noise_scheduler: Union[DDPMScheduler, DDIMScheduler],
        noise_scheduler_step_kwargs: Optional[dict] = None,
        num_denoise_steps_per_inference: int,
    ):
        super().__init__()

        self._prop_keys = prop_keys
        self.obs_tokenizer = ObsTokenizer(
            {
                "proprioception": MLP(
                    prop_dim,
                    hidden_dim=prop_mlp_hidden_dim,
                    output_dim=xf_n_embd,
                    hidden_depth=prop_mlp_hidden_depth,
                    add_output_activation=True,
                ),
                "rgb": ResNet18Encoder(
                    output_dim=xf_n_embd,
                    pretrained=resnet_pretrained,
                    freeze_backbone=resnet_freeze_backbone,
                    use_global_pool=True,
                    num_views=num_camera_views,
                ),
            },
            use_modality_type_tokens=use_modality_type_tokens,
            token_dim=xf_n_embd,
            token_concat_order=["proprioception", "rgb"],
            strict=True,
        )
        self.num_latest_obs = num_latest_obs
        if learnable_action_readout_token:
            self.action_readout_token = nn.Parameter(
                torch.zeros(
                    xf_n_embd,
                )
            )
        else:
            self.action_readout_token = torch.zeros(xf_n_embd)
        self.transformer = GPT(
            n_embd=xf_n_embd,
            n_layer=xf_n_layer,
            n_head=xf_n_head,
            dropout=xf_dropout_rate,
            use_geglu=xf_use_geglu,
        )
        self.action_decoder = WholeBodyUNetDiffusionHead(
            whole_body_decoding_order=["mobile_base", "torso", "arms"],
            action_dim_per_part={"mobile_base": 3, "torso": 1, "arms": 12},
            obs_dim=xf_n_embd,
            action_horizon=action_prediction_horizon,
            diffusion_step_embed_dim=diffusion_step_embed_dim,
            noise_scheduler=noise_scheduler,
            noise_scheduler_step_kwargs=noise_scheduler_step_kwargs,
            inference_denoise_steps=num_denoise_steps_per_inference,
            unet_down_dims=unet_down_dims,
            unet_kernel_size=unet_kernel_size,
            unet_n_groups=unet_n_groups,
            unet_cond_predict_scale=unet_cond_predict_scale,
        )
        self.action_dim = action_dim
        assert sum(action_key_dims.values()) == action_dim
        assert set(action_keys) == set(action_key_dims.keys())
        self._action_keys = action_keys
        self._action_key_dims = action_key_dims

    def forward(
        self,
        obs: dict[str, torch.Tensor],
    ):
        """
        Forward pass through encoder and transformer.
        
        Args:
            obs: dict with keys:
                - "rgb": (B, L, C, H, W) or (B, L, V, C, H, W) for multi-view
                - proprioception keys: (B, L, D)
        
        Returns:
            transformer output: (B, L * n_tokens_per_step, E)
        """
        # construct prop obs
        prop_obs = []
        for prop_key in self._prop_keys:
            if "/" in prop_key:
                group, key = prop_key.split("/")
                prop_obs.append(obs[group][key])
            else:
                prop_obs.append(obs[prop_key])
        prop_obs = torch.cat(prop_obs, dim=-1)  # (B, L, Prop_dim)

        obs_tokens = self.obs_tokenizer(
            {
                "proprioception": prop_obs,
                "rgb": {"rgb": obs["rgb"]},  # ResNet18Encoder expects dict with "rgb" key
            }
        )  # (B, L, E), where L is interleaved modalities tokens
        B, _, E = obs_tokens.shape
        action_readout_tokens = self.action_readout_token.view(1, 1, -1).expand(
            B, self.num_latest_obs, -1
        )

        n_tokens_per_step = self.obs_tokenizer.num_tokens_per_step + 1
        n_total_tokens = self.num_latest_obs * n_tokens_per_step
        tokens_in = torch.zeros(
            (B, n_total_tokens, E),
            device=obs_tokens.device,
            dtype=obs_tokens.dtype,
        )
        # insert obs tokens
        for j in range(self.obs_tokenizer.num_tokens_per_step):
            tokens_in[:, j::n_tokens_per_step] = obs_tokens[
                :, j :: self.obs_tokenizer.num_tokens_per_step
            ]
        # insert action readout tokens
        tokens_in[:, self.obs_tokenizer.num_tokens_per_step :: n_tokens_per_step] = (
            action_readout_tokens
        )

        # construct attention mask
        mask = torch.ones(B, n_total_tokens, dtype=torch.bool, device=self.device)
        # we mask action readout tokens
        mask[:, self.obs_tokenizer.num_tokens_per_step :: n_tokens_per_step] = False

        # construct position ids, which starts from 0
        # for all obs tokens in the same step, they share the same position id
        position_ids = torch.zeros(
            (B, n_total_tokens), device=self.device, dtype=torch.long
        )
        p_id = 0
        for t in range(self.num_latest_obs):
            obs_st = t * n_tokens_per_step
            obs_end = obs_st + self.obs_tokenizer.num_tokens_per_step
            action_readout_p = obs_st + self.obs_tokenizer.num_tokens_per_step
            position_ids[:, obs_st:obs_end] = p_id
            p_id += 1
            position_ids[:, action_readout_p] = p_id
            p_id += 1

        # run transformer forward
        tokens_in = rearrange(tokens_in, "B T E -> T B E")
        mask = mask.unsqueeze(1)  # (B, 1, T)
        tokens_out = self.transformer(
            tokens_in, custom_mask=mask, batch_first=False, position_ids=position_ids
        )
        assert tokens_out.shape == (n_total_tokens, B, E)
        tokens_out = rearrange(tokens_out, "T B E -> B T E")
        return tokens_out

    def compute_loss(
        self,
        *,
        obs: dict[str, torch.Tensor] | None = None,
        transformer_output: torch.Tensor | None = None,
        gt_action: torch.Tensor,
    ):
        """
        Compute loss.

        Args:
            obs: dict of (B, T, ...), where T = num_latest_obs
            transformer_output: (B, L, E), where L = num_latest_obs * n_tokens_per_step
            gt_action: Ground truth action of size (B, T_obs, T_act, A)
        """
        assert not (
            obs is None and transformer_output is None
        ), "Provide either obs or transformer_output"
        if transformer_output is None:
            transformer_output = self.forward(obs)
        action_readout_tokens = self._get_action_readout_tokens(transformer_output)
        # BigYM 16-dim structure: mobile_base(3) + torso(1) + arms(12)
        mobile_base_action = gt_action[..., :3]
        torso_action = gt_action[..., 3:4]
        arms_action = gt_action[..., 4:]
        loss = self.action_decoder.compute_loss(
            obs=action_readout_tokens,
            gt_action={
                "mobile_base": mobile_base_action,
                "torso": torso_action,
                "arms": arms_action,
            },
        )
        return loss

    @torch.no_grad()
    def inference(
        self,
        *,
        obs: dict[str, torch.Tensor] | None = None,
        transformer_output: torch.Tensor | None = None,
        return_last_timestep_only: bool,
    ):
        """
        Compute prediction.

        Args:
            obs: dict of (B, T, ...), where T = num_latest_obs
            transformer_output: (B, L, E)
            return_last_timestep_only: Whether to return only the last timestep actions.
        """
        assert not (
            obs is None and transformer_output is None
        ), "Provide either obs or transformer_output"
        if transformer_output is None:
            transformer_output = self.forward(obs)
        action_readout_tokens = self._get_action_readout_tokens(transformer_output)
        pred = self.action_decoder.inference(
            obs=action_readout_tokens,
            return_last_timestep_only=return_last_timestep_only,
        )
        return {
            "mobile_base": pred["mobile_base"],
            "torso": pred["torso"],
            "arms": pred["arms"],
        }

    @torch.no_grad()
    def act(
        self,
        obs: dict[str, torch.Tensor],
    ):
        return self.inference(
            obs=obs,
            return_last_timestep_only=True,
        )

    def _get_action_readout_tokens(self, transformer_output: torch.Tensor):
        B, _, E = transformer_output.shape
        n_tokens_per_step = self.obs_tokenizer.num_tokens_per_step + 1
        action_readout_tokens = transformer_output[
            :, self.obs_tokenizer.num_tokens_per_step :: n_tokens_per_step
        ]  # (B, T_obs, E)
        assert action_readout_tokens.shape == (B, self.num_latest_obs, E)
        return action_readout_tokens

    def get_optimizer_groups(self, weight_decay, lr_layer_decay, lr_scale=1.0):
        (
            feature_encoder_pg,
            feature_encoder_pid,
        ) = self.obs_tokenizer.get_optimizer_groups(
            weight_decay=weight_decay,
            lr_layer_decay=lr_layer_decay,
            lr_scale=lr_scale,
        )
        transformer_pg, transformer_pid = self.transformer.get_optimizer_groups(
            weight_decay=weight_decay,
            lr_layer_decay=lr_layer_decay,
            lr_scale=lr_scale,
        )
        action_decoder_pg, action_decoder_pid = (
            self.action_decoder.get_optimizer_groups(
                weight_decay=weight_decay,
                lr_layer_decay=lr_layer_decay,
                lr_scale=lr_scale,
            )
        )
        other_pg, _ = default_optimizer_groups(
            self,
            weight_decay=weight_decay,
            lr_scale=lr_scale,
            no_decay_filter=[
                "action_readout_token",
            ],
            exclude_filter=lambda name, p: id(p)
            in feature_encoder_pid + transformer_pid + action_decoder_pid,
        )
        all_groups = feature_encoder_pg + transformer_pg + action_decoder_pg + other_pg
        _, table_str = check_optimizer_groups(self, all_groups, verbose=True)
        rank_zero_info(table_str)
        return all_groups
