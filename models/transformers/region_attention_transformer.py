#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Region-Specific Attention Transformer for Parkinson's Disease Detection

This module implements a specialized transformer architecture with
attention mechanisms focused on neuroanatomical regions relevant
to Parkinson's disease.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
from typing import Dict, List, Optional, Tuple, Union


class RegionalEmbedding(nn.Module):
    """
    Specialized embedding layer that incorporates regional spatial information 
    from brain anatomy, with an emphasis on regions related to Parkinson's disease.
    """
    def __init__(
        self,
        input_size: Tuple[int, int, int],
        embed_dim: int,
        patch_size: int = 8,
        dropout: float = 0.1,
        region_weights: Optional[Dict[str, float]] = None
    ):
        """
        Initialize the regional embedding layer.
        
        Args:
            input_size: Size of the input volume (D, H, W)
            embed_dim: Embedding dimension
            patch_size: Size of patches to extract
            dropout: Dropout rate
            region_weights: Dictionary mapping region names to importance weights
        """
        super().__init__()
        
        self.input_size = input_size
        self.patch_size = patch_size
        self.embed_dim = embed_dim
        
        # Calculate number of patches in each dimension
        depth, height, width = input_size
        n_patches_d = depth // patch_size
        n_patches_h = height // patch_size
        n_patches_w = width // patch_size
        self.n_patches = n_patches_d * n_patches_h * n_patches_w
        
        # Linear projection for patch embedding
        self.proj = nn.Conv3d(
            in_channels=1,  # Single channel for MRI
            out_channels=embed_dim,
            kernel_size=patch_size,
            stride=patch_size
        )
        
        # Create a learnable class token
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        
        # Create positional embeddings for 3D patches
        self.pos_embed = nn.Parameter(
            torch.zeros(1, self.n_patches + 1, embed_dim)  # +1 for class token
        )
        
        # Initialize region-specific embeddings
        if region_weights is None:
            # Default weights emphasizing PD-relevant regions
            region_weights = {
                'substantia_nigra': 2.0,
                'striatum': 1.8,
                'globus_pallidus': 1.6,
                'thalamus': 1.4,
                'cortex': 1.0,
                'cerebellum': 0.9,
                'other': 0.7
            }
        
        # Create a regional embedding that highlights important structures
        # This is a simplified version - in practice this would use a brain atlas
        self.region_embed = nn.Parameter(
            self._create_region_embedding(n_patches_d, n_patches_h, n_patches_w, region_weights)
        )
        
        # Dropout
        self.dropout = nn.Dropout(dropout)
        
        # Initialize embeddings
        nn.init.normal_(self.cls_token, std=0.02)
        nn.init.normal_(self.pos_embed, std=0.02)
    
    def _create_region_embedding(self, d_patches, h_patches, w_patches, region_weights):
        """
        Create regional embeddings based on approximate brain anatomy.
        
        This is a simplified representation for demonstration purposes.
        A real implementation would use a proper brain atlas.
        
        Args:
            d_patches: Number of patches in depth dimension
            h_patches: Number of patches in height dimension
            w_patches: Number of patches in width dimension
            region_weights: Importance weights for different regions
            
        Returns:
            Regional embedding tensor of shape (1, n_patches, embed_dim)
        """
        # Create empty embedding
        region_embed = torch.zeros(1, self.n_patches, self.embed_dim)
        
        # Get centers
        center_d = d_patches // 2
        center_h = h_patches // 2
        center_w = w_patches // 2
        
        # Create a distance-based weighting
        for d in range(d_patches):
            for h in range(h_patches):
                for w in range(w_patches):
                    # Calculate index
                    idx = d * (h_patches * w_patches) + h * w_patches + w
                    
                    # Calculate distances from center
                    dist_from_center = np.sqrt((d - center_d)**2 + (h - center_h)**2 + (w - center_w)**2)
                    
                    # Base weight on distance (higher for central structures)
                    weight = 1.0 / (1.0 + dist_from_center)
                    
                    # Apply region-specific weighting (simplified)
                    if 0.4 < d / d_patches < 0.6 and 0.4 < h / h_patches < 0.6 and 0.4 < w / w_patches < 0.6:
                        # Approximate midbrain/substantia nigra
                        region_embed[0, idx] = weight * region_weights['substantia_nigra']
                    elif 0.3 < d / d_patches < 0.7 and 0.3 < h / h_patches < 0.7 and 0.3 < w / w_patches < 0.7:
                        # Approximate basal ganglia regions
                        region_embed[0, idx] = weight * region_weights['striatum']
                    else:
                        # Other brain regions
                        region_embed[0, idx] = weight * region_weights['other']
        
        return region_embed * 0.02  # Scale embedding
    
    def forward(self, x):
        """
        Forward pass of the regional embedding layer.
        
        Args:
            x: Input tensor of shape (B, C, D, H, W)
                B - batch size
                C - channels (1 for MRI)
                D, H, W - volume dimensions
                
        Returns:
            Embedded patches of shape (B, N+1, E)
                N - number of patches
                E - embedding dimension
        """
        B, C, D, H, W = x.shape
        assert C == 1, "Input should be single-channel MRI data"
        
        # Project patches
        x = self.proj(x)  # (B, E, D//P, H//P, W//P)
        
        # Flatten spatial dimensions
        x = x.flatten(2).transpose(1, 2)  # (B, N, E)
        
        # Add region-specific embeddings
        x = x + self.region_embed
        
        # Add positional embeddings and class token
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        
        # Add position embeddings (excluding CLS token position)
        x = x + self.pos_embed
        
        # Apply dropout
        x = self.dropout(x)
        
        return x


class MultiHeadRegionalAttention(nn.Module):
    """
    Multi-head attention mechanism with regional bias for neuroanatomical structures.
    """
    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        dropout: float = 0.1,
        region_attn_bias: bool = True
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        assert self.head_dim * num_heads == embed_dim, "embed_dim must be divisible by num_heads"
        
        # Standard attention components
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)
        
        # Dropout
        self.dropout = nn.Dropout(dropout)
        
        # Region-specific attention components
        self.region_attn_bias = region_attn_bias
        if region_attn_bias:
            # Learnable regional attention bias
            self.region_bias = nn.Parameter(torch.zeros(1, num_heads, 1, 1))
            nn.init.normal_(self.region_bias, std=0.02)
    
    def forward(
        self,
        x: torch.Tensor,
        region_mask: Optional[torch.Tensor] = None,
        attn_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Forward pass for the regional attention mechanism.
        
        Args:
            x: Input of shape (B, N, E)
            region_mask: Optional mask highlighting anatomical regions (B, N)
            attn_mask: Optional general attention mask (B, N, N)
            
        Returns:
            Output tensor of shape (B, N, E)
        """
        B, N, E = x.shape
        
        # Project q, k, v
        q = self.q_proj(x).reshape(B, N, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        k = self.k_proj(x).reshape(B, N, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        v = self.v_proj(x).reshape(B, N, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        
        # Compute attention scores
        attn_scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        
        # Apply region-specific attention bias if enabled
        if self.region_attn_bias and region_mask is not None:
            # Convert region mask to attention bias
            # region_mask should be (B, N) with values indicating region importance
            region_mask = region_mask.unsqueeze(1).unsqueeze(-1)  # (B, 1, N, 1)
            region_bias = self.region_bias * region_mask
            attn_scores = attn_scores + region_bias
        
        # Apply attention mask if provided
        if attn_mask is not None:
            attn_scores = attn_scores.masked_fill(attn_mask, float("-inf"))
        
        # Apply softmax and dropout
        attn_weights = F.softmax(attn_scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        # Apply attention weights to values
        output = torch.matmul(attn_weights, v)
        
        # Reshape output
        output = output.permute(0, 2, 1, 3).reshape(B, N, E)
        output = self.out_proj(output)
        
        return output


class RegionalTransformerEncoderLayer(nn.Module):
    """
    Transformer encoder layer with regional attention mechanism.
    """
    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        dropout: float = 0.1,
        activation: str = "gelu",
        region_attn_bias: bool = True,
        layer_norm_eps: float = 1e-6
    ):
        super().__init__()
        
        # Layer normalization
        self.norm1 = nn.LayerNorm(embed_dim, eps=layer_norm_eps)
        self.norm2 = nn.LayerNorm(embed_dim, eps=layer_norm_eps)
        
        # Regional attention
        self.attn = MultiHeadRegionalAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=dropout,
            region_attn_bias=region_attn_bias
        )
        
        # MLP
        mlp_hidden_dim = int(embed_dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, mlp_hidden_dim),
            nn.GELU() if activation == "gelu" else nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_hidden_dim, embed_dim),
            nn.Dropout(dropout)
        )
    
    def forward(
        self,
        x: torch.Tensor,
        region_mask: Optional[torch.Tensor] = None,
        attn_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Forward pass for the transformer encoder layer.
        """
        # Apply attention with residual connection
        x = x + self.attn(self.norm1(x), region_mask, attn_mask)
        
        # Apply MLP with residual connection
        x = x + self.mlp(self.norm2(x))
        
        return x


class RegionAttentionTransformer(nn.Module):
    """
    Complete region-specific attention transformer for Parkinson's disease detection.
    """
    def __init__(
        self,
        input_size: Tuple[int, int, int] = (128, 128, 128),
        patch_size: int = 8,
        in_channels: int = 1,
        num_classes: int = 2,
        embed_dim: int = 768,
        depth: int = 12,
        num_heads: int = 12,
        mlp_ratio: float = 4.0,
        dropout: float = 0.1,
        attention_dropout: float = 0.1,
        region_weights: Optional[Dict[str, float]] = None,
        use_region_bias: bool = True
    ):
        """
        Initialize the Region Attention Transformer model.
        
        Args:
            input_size: Size of input volume (D, H, W)
            patch_size: Size of patches to extract
            in_channels: Number of input channels (1 for MRI)
            num_classes: Number of output classes
            embed_dim: Dimension of token embeddings
            depth: Number of transformer layers
            num_heads: Number of attention heads
            mlp_ratio: Ratio of mlp hidden dim to embedding dim
            dropout: Dropout rate
            attention_dropout: Dropout rate for attention
            region_weights: Dictionary mapping region names to importance weights
            use_region_bias: Whether to use regional attention bias
        """
        super().__init__()
        
        self.input_size = input_size
        self.patch_size = patch_size
        self.in_channels = in_channels
        self.num_classes = num_classes
        self.embed_dim = embed_dim
        
        # Patch embedding with regional information
        self.embedding = RegionalEmbedding(
            input_size=input_size,
            embed_dim=embed_dim,
            patch_size=patch_size,
            dropout=dropout,
            region_weights=region_weights
        )
        
        # Number of patches
        depth_patches = input_size[0] // patch_size
        height_patches = input_size[1] // patch_size
        width_patches = input_size[2] // patch_size
        num_patches = depth_patches * height_patches * width_patches
        
        # Create regional mask based on anatomical knowledge
        # This would be replaced with an actual atlas-based approach
        self.register_buffer(
            "region_mask", 
            self._create_region_mask(depth_patches, height_patches, width_patches)
        )
        
        # Transformer encoder layers
        encoder_layers = []
        for _ in range(depth):
            encoder_layers.append(
                RegionalTransformerEncoderLayer(
                    embed_dim=embed_dim,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    dropout=dropout,
                    region_attn_bias=use_region_bias
                )
            )
        self.encoder = nn.ModuleList(encoder_layers)
        
        # Layer normalization
        self.norm = nn.LayerNorm(embed_dim)
        
        # Classification head
        self.head = nn.Linear(embed_dim, num_classes)
    
    def _create_region_mask(self, depth_patches, height_patches, width_patches):
        """
        Create a mask tensor highlighting important brain regions.
        
        In a real implementation, this would use a brain atlas.
        Here we use a simplified example with manually defined regions.
        """
        # Total number of patches plus CLS token
        total_patches = depth_patches * height_patches * width_patches + 1
        
        # Initialize mask with ones
        mask = torch.ones(1, total_patches)
        
        # Set CLS token importance
        mask[0, 0] = 1.5  # CLS token gets moderate importance
        
        # Define center coordinates
        center_d = depth_patches // 2
        center_h = height_patches // 2
        center_w = width_patches // 2
        
        # Assign weights to different regions
        patch_idx = 1
        
        for d in range(depth_patches):
            for h in range(height_patches):
                for w in range(width_patches):
                    # Distance from center
                    dist = np.sqrt((d - center_d)**2 + (h - center_h)**2 + (w - center_w)**2)
                    
                    # Give more importance to central regions
                    importance = 2.0 / (1.0 + dist)
                    
                    # Highlight regions relevant to PD
                    if 0.4 < d/depth_patches < 0.6 and 0.4 < h/height_patches < 0.6 and 0.4 < w/width_patches < 0.6:
                        # Higher importance for substantia nigra (midbrain)
                        mask[0, patch_idx] = importance * 2.0
                    elif 0.3 < d/depth_patches < 0.7 and 0.3 < h/height_patches < 0.7 and 0.3 < w/width_patches < 0.7:
                        # Higher importance for basal ganglia
                        mask[0, patch_idx] = importance * 1.5
                    else:
                        mask[0, patch_idx] = importance
                    
                    patch_idx += 1
        
        return mask
    
    def forward(self, x, return_features=False):
        """
        Forward pass of the Region Attention Transformer.
        
        Args:
            x: Input tensor of shape (B, C, D, H, W)
            return_features: Whether to return features before classification
            
        Returns:
            Classification logits and optionally features
        """
        # Extract batch size
        B = x.shape[0]
        
        # Create regional embeddings
        x = self.embedding(x)
        
        # Apply transformer encoder layers
        for layer in self.encoder:
            x = layer(x, self.region_mask.expand(B, -1))
        
        # Apply final normalization
        x = self.norm(x)
        
        # Use CLS token for classification
        cls_feature = x[:, 0]
        
        # Classification head
        logits = self.head(cls_feature)
        
        if return_features:
            return logits, cls_feature
        else:
            return logits


class RegionContrastiveTransformer(RegionAttentionTransformer):
    """
    Extension of the Region Attention Transformer with contrastive learning.
    """
    def __init__(
        self,
        projection_dim: int = 128,
        temperature: float = 0.1,
        **kwargs
    ):
        super().__init__(**kwargs)
        
        # Projection head for contrastive learning
        self.projection = nn.Sequential(
            nn.Linear(self.embed_dim, self.embed_dim),
            nn.ReLU(),
            nn.Linear(self.embed_dim, projection_dim)
        )
        
        # Temperature parameter for contrastive loss
        self.temperature = temperature
        
        # Region-specific projection heads (for key anatomical regions)
        self.num_regions = 3  # Substantia nigra, striatum, others
        self.region_projections = nn.ModuleList([
            nn.Linear(self.embed_dim, projection_dim)
            for _ in range(self.num_regions)
        ])
    
    def forward(
        self, 
        x, 
        return_features=False, 
        return_projections=False,
        return_region_preds=False
    ):
        """
        Forward pass with optional contrastive features and region predictions.
        
        Args:
            x: Input tensor of shape (B, C, D, H, W)
            return_features: Whether to return features
            return_projections: Whether to return projection vectors
            return_region_preds: Whether to return region-specific predictions
            
        Returns:
            Tuple with various outputs based on settings
        """
        # Get base outputs
        if return_features:
            logits, features = super().forward(x, return_features=True)
        else:
            logits = super().forward(x, return_features=False)
            features = None
        
        # Process outputs based on what's requested
        outputs = [logits]
        
        if return_features:
            outputs.append(features)
        
        if return_projections:
            # Create projection vectors for contrastive learning
            projections = self.projection(features)
            projections = F.normalize(projections, dim=1)
            outputs.append(projections)
        
        if return_region_preds:
            # Create region-specific predictions
            # In a real implementation, this would use extracted features
            # from specific brain regions using an atlas
            
            # We use the embedding sequence for demonstration
            B = x.shape[0]
            embeddings = self.embedding(x)
            
            # Use the CLS token and a few key patches as proxies for brain regions
            region_features = [
                embeddings[:, 0],  # CLS token
                # Center patches - approximate midbrain
                embeddings[:, 1 + len(embeddings[0])//2], 
                # Average of other patches
                embeddings[:, 2:].mean(dim=1)
            ]
            
            region_preds = []
            for i, region_proj in enumerate(self.region_projections):
                region_pred = region_proj(region_features[i])
                region_preds.append(region_pred)
            
            outputs.append(region_preds)
        
        # Return tuple of outputs
        if len(outputs) == 1:
            return outputs[0]
        else:
            return tuple(outputs)


class MultitaskLoss(nn.Module):
    """
    Multitask loss combining classification, contrastive, and region-specific losses.
    """
    def __init__(
        self,
        classification_weight: float = 1.0,
        contrastive_weight: float = 0.5,
        region_weight: float = 0.3,
        contrastive_temperature: float = 0.1,
        focal_gamma: float = 2.0,
        class_weights: List[float] = None
    ):
        super().__init__()
        self.classification_weight = classification_weight
        self.contrastive_weight = contrastive_weight
        self.region_weight = region_weight
        self.temperature = contrastive_temperature
        self.focal_gamma = focal_gamma
        
        # Use class weights to address class imbalance
        if class_weights is None:
            # Default weights - slightly favor PD class to fix the imbalance issue
            class_weights = [1.0, 2.0]  # [Control, PD]
        self.class_weights = torch.tensor(class_weights)
        
        # Use weighted cross entropy loss to address class imbalance
        self.classification_loss = lambda logits, targets: F.cross_entropy(
            logits, targets, 
            weight=self.class_weights.to(logits.device) if self.class_weights is not None else None
        )
    
    def focal_loss(self, logits, labels):
        """
        Compute focal loss for better handling of hard examples.
        
        Args:
            logits: Prediction logits
            labels: Ground truth labels
            
        Returns:
            Focal loss value
        """
        # Standard cross entropy
        ce_loss = F.cross_entropy(logits, labels, reduction='none')
        
        # Compute probabilities
        pt = torch.exp(-ce_loss)
        
        # Apply focal weighting
        focal_weight = (1 - pt) ** self.focal_gamma
        
        # Compute weighted loss
        loss = focal_weight * ce_loss
        
        return loss.mean()
    
    def contrastive_loss(self, projections, labels=None):
        """
        Compute contrastive loss (InfoNCE) on projections.
        
        Args:
            projections: Projection vectors
            labels: Optional class labels to define positive pairs
            
        Returns:
            Contrastive loss value
        """
        batch_size = projections.shape[0]
        
        # Normalize projections for cosine similarity
        projections = F.normalize(projections, p=2, dim=1)
        
        # Compute similarity matrix
        similarity = torch.matmul(projections, projections.T) / self.temperature
        
        # Remove diagonal (self-similarity)
        mask = torch.eye(batch_size, device=projections.device)
        mask = 1 - mask
        
        # For numerical stability
        similarity = similarity * mask - 1e9 * (1 - mask)
        
        # If labels are provided, use them to define positive pairs
        if labels is not None:
            # Same class = positive pair
            pos_mask = (labels.unsqueeze(0) == labels.unsqueeze(1)).float() * mask
        else:
            # Otherwise use adjacent examples as positive pairs (simplified approach)
            pos_mask = torch.roll(torch.eye(batch_size), 1, dims=0).to(projections.device) * mask
        
        # Calculate loss
        loss = 0
        
        # For each example, calculate contrastive loss
        for i in range(batch_size):
            if pos_mask[i].sum() > 0:  # If there are positive pairs
                # Get positive and negative similarities
                pos_sim = similarity[i][pos_mask[i] > 0]
                neg_sim = similarity[i][pos_mask[i] <= 0]
                
                if len(neg_sim) > 0:  # If there are negative pairs
                    # For each positive pair
                    for pos in pos_sim:
                        # Calculate probability of picking the positive among all
                        all_sim = torch.cat([pos.unsqueeze(0), neg_sim])
                        logits = all_sim
                        labels = torch.zeros(1, device=logits.device, dtype=torch.long)
                        
                        # Use cross entropy loss (equivalent to InfoNCE)
                        loss += F.cross_entropy(logits.unsqueeze(0), labels)
        
        # Normalize by number of examples
        return loss / batch_size if batch_size > 0 else torch.tensor(0.0, device=projections.device)
    
    def region_loss(self, region_preds, labels):
        """
        Compute region-specific loss.
        
        Args:
            region_preds: List of region-specific predictions
            labels: Class labels
            
        Returns:
            Region loss value
        """
        # Enhanced region loss with label information
        loss = 0
        
        # Group regions by label
        pd_indices = (labels == 1).nonzero(as_tuple=True)[0]
        control_indices = (labels == 0).nonzero(as_tuple=True)[0]
        
        # Calculate consistency within regions for same class
        for i in range(len(region_preds)):
            for j in range(i+1, len(region_preds)):
                # Get predictions for each region
                pred_i = F.softmax(region_preds[i], dim=1)
                pred_j = F.softmax(region_preds[j], dim=1)
                
                # For each class, promote consistency within that class
                if len(pd_indices) > 0:
                    # PD subjects should have consistent region predictions
                    loss += F.kl_div(
                        torch.log(pred_i[pd_indices] + 1e-10), 
                        pred_j[pd_indices], 
                        reduction='batchmean'
                    )
                
                if len(control_indices) > 0:
                    # Control subjects should have consistent region predictions
                    loss += F.kl_div(
                        torch.log(pred_i[control_indices] + 1e-10), 
                        pred_j[control_indices], 
                        reduction='batchmean'
                    )
        
        # Add regularization to ensure regions are discriminative
        for i, region_pred in enumerate(region_preds):
            # Each region should be predictive of the class
            loss += F.cross_entropy(region_pred, labels)
        
        # Calculate total loss
        num_comparisons = len(region_preds) * (len(region_preds) - 1) / 2
        if num_comparisons > 0:
            loss = loss / (num_comparisons + len(region_preds))
        
        return loss
    
    def forward(self, logits, labels, projections=None, region_preds=None):
        """
        Compute multitask loss.
        
        Args:
            logits: Classification logits
            labels: Class labels
            projections: Projection vectors for contrastive loss (optional)
            region_preds: Region-specific predictions (optional)
            
        Returns:
            total_loss, loss_dict
        """
        # Apply a temperature scaling to logits to make predictions less extreme
        scaled_logits = logits / 1.5  # Temperature scaling
        
        # Classification loss with focal loss and class weights
        cls_loss = self.focal_loss(scaled_logits, labels)
        loss_dict = {'classification': cls_loss.item()}
        
        # Total loss starts with classification
        total_loss = self.classification_weight * cls_loss
        
        # Add contrastive loss if projections are provided
        if projections is not None and self.contrastive_weight > 0:
            cont_loss = self.contrastive_loss(projections, labels)
            total_loss = total_loss + self.contrastive_weight * cont_loss
            loss_dict['contrastive'] = cont_loss.item()
        
        # Add region loss if region predictions are provided
        if region_preds is not None and self.region_weight > 0:
            reg_loss = self.region_loss(region_preds, labels)
            total_loss = total_loss + self.region_weight * reg_loss
            loss_dict['region'] = reg_loss.item()
        
        return total_loss, loss_dict


def create_region_attention_transformer(
    input_size=(128, 128, 128),
    patch_size=8,
    embed_dim=768,
    depth=12,
    num_heads=12,
    num_classes=2,
    dropout=0.1,
    use_contrastive=False,
    **kwargs
):
    """
    Factory function to create a Region Attention Transformer model.
    
    Args:
        input_size: Size of input volume (D, H, W)
        patch_size: Size of patches
        embed_dim: Embedding dimension
        depth: Number of transformer layers
        num_heads: Number of attention heads
        num_classes: Number of output classes
        dropout: Dropout rate
        use_contrastive: Whether to use contrastive version
        **kwargs: Additional arguments
        
    Returns:
        Configured model
    """
    if use_contrastive:
        return RegionContrastiveTransformer(
            input_size=input_size,
            patch_size=patch_size,
            embed_dim=embed_dim,
            depth=depth,
            num_heads=num_heads,
            num_classes=num_classes,
            dropout=dropout,
            **kwargs
        )
    else:
        return RegionAttentionTransformer(
            input_size=input_size,
            patch_size=patch_size,
            embed_dim=embed_dim,
            depth=depth,
            num_heads=num_heads,
            num_classes=num_classes,
            dropout=dropout,
            **kwargs
        ) 