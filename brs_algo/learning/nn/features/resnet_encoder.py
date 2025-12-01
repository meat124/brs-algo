"""
ResNet18-based Image Encoder for BRS Policy (RGB version).

Replaces PointNet for RGB observation encoding.
Output shape: (B, T, E) to be compatible with ObsTokenizer.
"""

import torch
import torch.nn as nn
from torchvision.models import resnet18, ResNet18_Weights


class ResNet18Encoder(nn.Module):
    """
    ResNet18-based encoder for RGB images.
    
    Takes RGB images and outputs feature vectors compatible with ObsTokenizer.
    
    Input: dict with key "rgb" -> (B, T, C, H, W) or (B, T, V, C, H, W) for multi-view
    Output: (B, T, E) where E is output_dim
    
    Args:
        output_dim: Output dimension (should match xf_n_embd, typically 512)
        pretrained: Whether to use pretrained ImageNet weights
        freeze_backbone: Whether to freeze ResNet backbone
        use_global_pool: If True, use global average pooling. 
                        If False, output spatial features as tokens.
    """
    
    # ImageNet normalization constants
    MEAN = torch.tensor([0.485, 0.456, 0.406])
    STD = torch.tensor([0.229, 0.224, 0.225])
    
    def __init__(
        self,
        output_dim: int = 512,
        pretrained: bool = True,
        freeze_backbone: bool = False,
        use_global_pool: bool = True,
        num_views: int = 1,
    ):
        super().__init__()
        
        self.output_dim = output_dim
        self.use_global_pool = use_global_pool
        self.num_views = num_views
        
        # Build ResNet18 backbone
        if pretrained:
            weights = ResNet18_Weights.DEFAULT
            backbone = resnet18(weights=weights)
        else:
            backbone = resnet18(weights=None)
        
        # Remove the final FC layer
        # ResNet18 outputs 512 channels after layer4
        self.backbone = nn.Sequential(
            backbone.conv1,
            backbone.bn1,
            backbone.relu,
            backbone.maxpool,
            backbone.layer1,
            backbone.layer2,
            backbone.layer3,
            backbone.layer4,
        )
        
        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False
        
        # Global average pooling
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        
        # Projector to output_dim (if output_dim != 512)
        resnet_out_dim = 512
        if output_dim != resnet_out_dim:
            self.projector = nn.Linear(resnet_out_dim * num_views, output_dim)
        else:
            if num_views > 1:
                self.projector = nn.Linear(resnet_out_dim * num_views, output_dim)
            else:
                self.projector = None
        
        # Register normalization buffers
        self.register_buffer('mean', self.MEAN.view(1, 1, 3, 1, 1))
        self.register_buffer('std', self.STD.view(1, 1, 3, 1, 1))
    
    def normalize_image(self, x: torch.Tensor) -> torch.Tensor:
        """Normalize image tensor to ImageNet stats.
        
        Args:
            x: Input tensor of shape (B, T, C, H, W) or (B, T, V, C, H, W)
               Values should be in [0, 1] range
        """
        if x.dim() == 5:
            # (B, T, C, H, W)
            return (x - self.mean.squeeze(2)) / self.std.squeeze(2)
        elif x.dim() == 6:
            # (B, T, V, C, H, W)
            return (x - self.mean.unsqueeze(2)) / self.std.unsqueeze(2)
        else:
            raise ValueError(f"Expected 5D or 6D tensor, got {x.dim()}D")
    
    def forward(self, x: dict) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: dict with key "rgb" containing tensor of shape:
               - (B, T, C, H, W) for single-view
               - (B, T, V, C, H, W) for multi-view
               Values should be in [0, 1] range (float) or [0, 255] (uint8)
               
        Returns:
            Tensor of shape (B, T, E) where E is output_dim
        """
        rgb = x["rgb"]
        
        # Handle uint8 images
        if rgb.dtype == torch.uint8:
            rgb = rgb.float() / 255.0
        
        # Check input shape
        if rgb.dim() == 5:
            # Single-view: (B, T, C, H, W)
            B, T, C, H, W = rgb.shape
            multi_view = False
        elif rgb.dim() == 6:
            # Multi-view: (B, T, V, C, H, W)
            B, T, V, C, H, W = rgb.shape
            multi_view = True
            assert V == self.num_views, f"Expected {self.num_views} views, got {V}"
        else:
            raise ValueError(f"Expected 5D or 6D tensor, got {rgb.dim()}D")
        
        # Normalize
        rgb = self.normalize_image(rgb)
        
        if multi_view:
            # Process each view through backbone
            # (B, T, V, C, H, W) -> (B*T*V, C, H, W)
            rgb_flat = rgb.reshape(B * T * V, C, H, W)
            features = self.backbone(rgb_flat)  # (B*T*V, 512, H', W')
            
            if self.use_global_pool:
                features = self.global_pool(features)  # (B*T*V, 512, 1, 1)
                features = features.flatten(1)  # (B*T*V, 512)
                features = features.view(B, T, V, -1)  # (B, T, V, 512)
                features = features.flatten(2)  # (B, T, V*512)
            else:
                raise NotImplementedError("Non-pooled multi-view not yet implemented")
        else:
            # Single-view: (B, T, C, H, W) -> (B*T, C, H, W)
            rgb_flat = rgb.reshape(B * T, C, H, W)
            features = self.backbone(rgb_flat)  # (B*T, 512, H', W')
            
            if self.use_global_pool:
                features = self.global_pool(features)  # (B*T, 512, 1, 1)
                features = features.flatten(1)  # (B*T, 512)
                features = features.view(B, T, -1)  # (B, T, 512)
            else:
                # Output spatial features as tokens
                # (B*T, 512, H', W') -> (B*T, 512, H'*W')
                features = features.flatten(2)  # (B*T, 512, N)
                features = features.permute(0, 2, 1)  # (B*T, N, 512)
                features = features.view(B, T, -1, 512)  # (B, T, N, 512)
                # For now, just mean pool spatial tokens
                features = features.mean(dim=2)  # (B, T, 512)
        
        # Project to output_dim
        if self.projector is not None:
            features = self.projector(features)  # (B, T, output_dim)
        
        return features


class ResNet18TokenEncoder(nn.Module):
    """
    ResNet18 encoder that outputs spatial tokens instead of a single vector.
    
    Useful for attention-based policies that want to attend to different
    spatial locations in the image.
    
    Input: (B, T, C, H, W)
    Output: (B, T*N, E) where N is number of spatial tokens per image
    """
    
    MEAN = torch.tensor([0.485, 0.456, 0.406])
    STD = torch.tensor([0.229, 0.224, 0.225])
    
    def __init__(
        self,
        output_dim: int = 512,
        pretrained: bool = True,
        freeze_backbone: bool = False,
    ):
        super().__init__()
        
        self.output_dim = output_dim
        
        # Build ResNet18 backbone
        if pretrained:
            weights = ResNet18_Weights.DEFAULT
            backbone = resnet18(weights=weights)
        else:
            backbone = resnet18(weights=None)
        
        # Remove the final FC layer and avgpool
        self.backbone = nn.Sequential(
            backbone.conv1,
            backbone.bn1,
            backbone.relu,
            backbone.maxpool,
            backbone.layer1,
            backbone.layer2,
            backbone.layer3,
            backbone.layer4,
        )
        
        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False
        
        # Projector for each spatial token
        resnet_out_dim = 512
        if output_dim != resnet_out_dim:
            self.projector = nn.Linear(resnet_out_dim, output_dim)
        else:
            self.projector = None
        
        # Register normalization buffers
        self.register_buffer('mean', self.MEAN.view(1, 1, 3, 1, 1))
        self.register_buffer('std', self.STD.view(1, 1, 3, 1, 1))
        
        # Will be computed on first forward pass
        self._n_tokens = None
    
    @property
    def n_tokens_per_image(self) -> int:
        """Number of spatial tokens output per image."""
        return self._n_tokens
    
    def normalize_image(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 5:
            return (x - self.mean.squeeze(2)) / self.std.squeeze(2)
        raise ValueError(f"Expected 5D tensor, got {x.dim()}D")
    
    def forward(self, x: dict) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: dict with "rgb" key containing (B, T, C, H, W) tensor
               
        Returns:
            (B, T*N, E) where N is spatial tokens per image
        """
        rgb = x["rgb"]
        
        if rgb.dtype == torch.uint8:
            rgb = rgb.float() / 255.0
        
        B, T, C, H, W = rgb.shape
        
        # Normalize
        rgb = self.normalize_image(rgb)
        
        # (B*T, C, H, W)
        rgb_flat = rgb.reshape(B * T, C, H, W)
        features = self.backbone(rgb_flat)  # (B*T, 512, H', W')
        
        # Flatten spatial dims and transpose
        BT, D, H_out, W_out = features.shape
        N = H_out * W_out
        self._n_tokens = N
        
        features = features.flatten(2)  # (B*T, 512, N)
        features = features.permute(0, 2, 1)  # (B*T, N, 512)
        
        if self.projector is not None:
            features = self.projector(features)  # (B*T, N, output_dim)
        
        # Reshape to (B, T*N, E)
        features = features.view(B, T * N, self.output_dim)
        
        return features
