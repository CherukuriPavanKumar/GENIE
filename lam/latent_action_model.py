"""
Latent Action Model (LAM)

Infers discrete latent actions between consecutive frames and reconstructs future frames
conditioned on those actions, without requiring any action labels during training.

Encoder: [z_t, z_{t+1}] -> Action Token a_t
Decoder: [z_1, a_1 ... a_{T-1}] -> [z_2 ... z_T]
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from tokenizer.st_transformer import STTransformer
from tokenizer.fsq import FiniteScalarQuantizer


class ActionEncoder(nn.Module):
    def __init__(self, embed_dim=32, num_heads=8, hidden_dim=128, num_blocks=4, action_dim=2, num_bins=4):
        super().__init__()
        # Takes concatenated temporal pair of frames: [B, 2, P, E]
        self.transformer = STTransformer(embed_dim, num_heads, hidden_dim, num_blocks, causal=True)
        # We mean-pool over patches, so output of transformer [B, 2, P, E] -> mean(dim=2) -> [B, 2, E]
        # Then we flatten time to predict action: [B, 2*E]
        self.action_head = nn.Linear(embed_dim * 2, action_dim)
        
    def forward(self, z):
        # z: [B, T, P, E] -> patch embeddings from VideoTokenizer
        B, T, P, E = z.shape
        assert T >= 2, "Need at least 2 frames to infer an action"
        
        # We need to process each adjacent pair (t, t+1) independently.
        # Shape trick: form pairs [B, T-1, 2, P, E]
        z_pairs = torch.stack([z[:, t:t+2] for t in range(T - 1)], dim=1) # [B, T-1, 2, P, E]
        
        # Merge B and T-1 for transformer processing
        z_pairs = z_pairs.view(B * (T - 1), 2, P, E)
        
        # Process pairs
        out = self.transformer(z_pairs) # [B*(T-1), 2, P, E]
        
        # Mean pool over patches
        out = out.mean(dim=2) # [B*(T-1), 2, E]
        
        # Flatten temporal pairs
        out = out.view(B * (T - 1), 2 * E) # [B*(T-1), 2*E]
        
        # Predict action latents
        action_latents = self.action_head(out) # [B*(T-1), action_dim]
        
        action_latents = action_latents.view(B, T - 1, self.action_head.out_features)
        return action_latents


class ActionDecoder(nn.Module):
    def __init__(self, embed_dim=32, num_heads=8, hidden_dim=128, num_blocks=4, action_dim=2):
        super().__init__()
        # Action embeddings
        self.action_embed = nn.Linear(action_dim, embed_dim)
        # STTransformer with FiLM conditioning on actions
        self.transformer = STTransformer(embed_dim, num_heads, hidden_dim, num_blocks, causal=True, condition_dim=embed_dim)
        
        # Learnable mask token for predicting future frames
        self.mask_token = nn.Parameter(torch.randn(1, 1, 1, embed_dim) * 0.02)
        
    def forward(self, z_initial, action_latents, T):
        # z_initial: [B, 1, P, E] -> first frame patch embeddings
        # action_latents: [B, T-1, action_dim] -> FSQ quantized actions
        # T: target total sequence length
        B, _, P, E = z_initial.shape
        
        # 1. Prepare masked frame sequence
        # We start with the first frame, and mask all subsequent frames
        # using the learned mask token for unknown frames.
            
        mask_tokens = self.mask_token.expand(B, T - 1, P, E)
        
        # Concatenate: [z_1, MASK, MASK, ...]
        z_seq = torch.cat([z_initial, mask_tokens], dim=1) # [B, T, P, E]
        
        # 2. Prepare action conditioning
        # Actions are between frames: a_1 is action after frame 1.
        # For frame t, we condition on a_{t-1}. For frame 1, condition is null (zeros).
        # action_latents is [B, T-1, action_dim]
        action_emb = self.action_embed(action_latents) # [B, T-1, E]
        
        null_action = torch.zeros(B, 1, E, device=z_initial.device)
        condition = torch.cat([null_action, action_emb], dim=1) # [B, T, E]
        
        # 3. Predict sequence
        z_pred = self.transformer(z_seq, condition=condition) # [B, T, P, E]
        
        return z_pred


class LatentActionModel(nn.Module):
    def __init__(self, embed_dim=32, num_heads=8, hidden_dim=128, num_blocks=4, action_dim=2, num_bins=4, var_weight=100.0, var_target=0.01, entropy_weight=10.0):
        super().__init__()
        self.var_weight = var_weight
        self.entropy_weight = entropy_weight
        self.var_target = var_target
        self.encoder = ActionEncoder(embed_dim, num_heads, hidden_dim, num_blocks, action_dim, num_bins)
        self.quantizer = FiniteScalarQuantizer(action_dim, num_bins)
        self.decoder = ActionDecoder(embed_dim, num_heads, hidden_dim, num_blocks, action_dim)
        self.action_vocab_size = num_bins ** action_dim
        
    def forward(self, z):
        # z: [B, T, P, E] -> actual patch embeddings
        
        # 1. Encode actions
        # action_latents: [B, T-1, action_dim]
        action_latents = self.encoder(z)
        
        # Quantize
        action_latents_q = self.quantizer(action_latents)
        
        # 2. Decode future frames
        # Only pass the first frame: z[:, :1]
        z_pred = self.decoder(z[:, :1], action_latents_q, T=z.shape[1])
        
        # 3. Compute Reconstruction Loss
        # We only care about reconstructing frames 2 to T (since 1 is given)
        recon_loss = F.mse_loss(z_pred[:, 1:], z[:, 1:])
        
        # 4. Compute Variance Loss to prevent action collapse
        # Penalize only when variance across the batch is below the target variance.
        flattened_actions = action_latents.reshape(-1, action_latents.shape[-1])
        # unbiased=False is typically used in these custom loss objectives to match standard definitions, but default var works too.
        action_variance = torch.var(flattened_actions, dim=0).mean()
        var_loss = self.var_weight * F.relu(self.var_target - action_variance)
        
        # 5. Compute Entropy Loss to penalize dead action codes
        # Unlike variance (which measures continuous spread), entropy directly
        # measures whether all discrete action codes are being used uniformly.
        action_indices = self.quantizer.get_indices_from_latents(action_latents_q)
        counts = torch.zeros(self.action_vocab_size, device=action_indices.device)
        counts.scatter_add_(0, action_indices.reshape(-1),
                            torch.ones_like(action_indices.reshape(-1), dtype=torch.float))
        probs = counts / counts.sum().clamp_min(1)
        probs = probs + 1e-8  # avoid log(0)
        entropy = -(probs * probs.log()).sum()
        max_entropy = torch.tensor(float(self.action_vocab_size), device=action_indices.device).log()
        entropy_loss = self.entropy_weight * (max_entropy - entropy)
        
        total_loss = recon_loss + var_loss + entropy_loss
        
        return total_loss, recon_loss, var_loss, entropy_loss, action_latents_q

    @torch.no_grad()
    def infer_actions(self, z):
        action_latents = self.encoder(z)
        action_latents_q = self.quantizer(action_latents)
        return self.quantizer.get_indices_from_latents(action_latents_q)
