"""
Diffusion Samplers for DiffV2X
Implements DDPM (training) and DDIM (inference) with extensible interface
"""

import torch
import torch.nn as nn
from typing import Optional, Callable, List
from abc import ABC, abstractmethod


class DiffusionSchedule:
    """
    Diffusion noise schedule with various beta schedules.
    """
    def __init__(
        self,
        num_timesteps: int = 1000,
        beta_schedule: str = "linear",
        beta_start: float = 1e-4,
        beta_end: float = 0.02,
    ):
        self.num_timesteps = num_timesteps

        # Generate beta schedule
        if beta_schedule == "linear":
            betas = torch.linspace(beta_start, beta_end, num_timesteps)
        elif beta_schedule == "cosine":
            # Cosine schedule from "Improved Denoising Diffusion Probabilistic Models"
            s = 0.008
            steps = num_timesteps + 1
            x = torch.linspace(0, num_timesteps, steps)
            alphas_cumprod = torch.cos(((x / num_timesteps) + s) / (1 + s) * torch.pi * 0.5) ** 2
            alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
            betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
            betas = torch.clip(betas, 0.0001, 0.9999)
        elif beta_schedule == "quadratic":
            betas = torch.linspace(beta_start**0.5, beta_end**0.5, num_timesteps) ** 2
        else:
            raise ValueError(f"Unknown beta schedule: {beta_schedule}")

        self.betas = betas
        self.alphas = 1.0 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
        self.alphas_cumprod_prev = torch.cat([torch.ones(1), self.alphas_cumprod[:-1]])

        # Calculations for diffusion q(x_t | x_0)
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - self.alphas_cumprod)

        # Calculations for posterior q(x_{t-1} | x_t, x_0)
        self.posterior_variance = (
            self.betas * (1.0 - self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)
        )
        self.posterior_log_variance_clipped = torch.log(
            torch.clamp(self.posterior_variance, min=1e-20)
        )
        self.posterior_mean_coef1 = (
            self.betas * torch.sqrt(self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)
        )
        self.posterior_mean_coef2 = (
            (1.0 - self.alphas_cumprod_prev) * torch.sqrt(self.alphas) / (1.0 - self.alphas_cumprod)
        )

    def to(self, device):
        """Move all tensors to device."""
        self.betas = self.betas.to(device)
        self.alphas = self.alphas.to(device)
        self.alphas_cumprod = self.alphas_cumprod.to(device)
        self.alphas_cumprod_prev = self.alphas_cumprod_prev.to(device)
        self.sqrt_alphas_cumprod = self.sqrt_alphas_cumprod.to(device)
        self.sqrt_one_minus_alphas_cumprod = self.sqrt_one_minus_alphas_cumprod.to(device)
        self.posterior_variance = self.posterior_variance.to(device)
        self.posterior_log_variance_clipped = self.posterior_log_variance_clipped.to(device)
        self.posterior_mean_coef1 = self.posterior_mean_coef1.to(device)
        self.posterior_mean_coef2 = self.posterior_mean_coef2.to(device)
        return self


class BaseSampler(ABC):
    """
    Base class for diffusion samplers.
    Provides extensible interface for different sampling strategies.
    """
    def __init__(self, schedule: DiffusionSchedule):
        self.schedule = schedule

    @abstractmethod
    def sample(
        self,
        model: nn.Module,
        shape: tuple,
        condition: dict,
        num_steps: Optional[int] = None,
        return_trajectory: bool = False,
    ) -> torch.Tensor:
        """
        Sample from the diffusion model.

        Args:
            model: The denoising model
            shape: Shape of the sample [B, C, H, W]
            condition: Dictionary of conditioning information
            num_steps: Number of sampling steps (optional)
            return_trajectory: Whether to return full trajectory

        Returns:
            Sampled tensor or trajectory
        """
        pass

    @abstractmethod
    def training_losses(
        self,
        model: nn.Module,
        x_0: torch.Tensor,
        condition: dict,
    ) -> dict:
        """
        Compute training losses.

        Args:
            model: The denoising model
            x_0: Clean data [B, C, H, W]
            condition: Dictionary of conditioning information

        Returns:
            Dictionary of losses
        """
        pass


class DDPMSampler(BaseSampler):
    """
    DDPM (Denoising Diffusion Probabilistic Models) sampler.
    Used primarily for training with standard diffusion process.
    """
    def __init__(
        self,
        schedule: DiffusionSchedule,
        model_var_type: str = "learned_range",  # "fixed_small", "fixed_large", "learned", "learned_range"
        loss_type: str = "mse",  # "mse", "l1"
    ):
        super().__init__(schedule)
        self.model_var_type = model_var_type
        self.loss_type = loss_type

    def q_sample(self, x_0: torch.Tensor, t: torch.Tensor, noise: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Forward diffusion process: q(x_t | x_0)
        """
        if noise is None:
            noise = torch.randn_like(x_0)

        # Get device from input tensor
        device = x_0.device

        sqrt_alphas_cumprod_t = self.schedule.sqrt_alphas_cumprod[t].to(device)
        sqrt_one_minus_alphas_cumprod_t = self.schedule.sqrt_one_minus_alphas_cumprod[t].to(device)

        # Reshape for broadcasting
        while len(sqrt_alphas_cumprod_t.shape) < len(x_0.shape):
            sqrt_alphas_cumprod_t = sqrt_alphas_cumprod_t.unsqueeze(-1)
            sqrt_one_minus_alphas_cumprod_t = sqrt_one_minus_alphas_cumprod_t.unsqueeze(-1)

        return sqrt_alphas_cumprod_t * x_0 + sqrt_one_minus_alphas_cumprod_t * noise

    def p_sample(
        self,
        model: nn.Module,
        x_t: torch.Tensor,
        t: torch.Tensor,
        condition: dict,
    ) -> torch.Tensor:
        """
        Reverse diffusion process: p(x_{t-1} | x_t)
        """
        device = x_t.device

        # Predict noise
        noise_pred = model(
            x_t,
            t,
            agent_latents=condition.get("agent_latents", []),
            transforms=condition.get("transforms", []),
            ego_bev=condition.get("ego_bev"),
        )

        # Get schedule values
        alpha_t = self.schedule.alphas[t].to(device)
        alpha_cumprod_t = self.schedule.alphas_cumprod[t].to(device)
        beta_t = self.schedule.betas[t].to(device)

        # Reshape for broadcasting
        while len(alpha_t.shape) < len(x_t.shape):
            alpha_t = alpha_t.unsqueeze(-1)
            alpha_cumprod_t = alpha_cumprod_t.unsqueeze(-1)
            beta_t = beta_t.unsqueeze(-1)

        # Predict x_0
        x_0_pred = (x_t - torch.sqrt(1 - alpha_cumprod_t) * noise_pred) / torch.sqrt(alpha_cumprod_t)

        # Get posterior mean
        posterior_mean_coef1 = self.schedule.posterior_mean_coef1[t].to(device)
        posterior_mean_coef2 = self.schedule.posterior_mean_coef2[t].to(device)

        while len(posterior_mean_coef1.shape) < len(x_t.shape):
            posterior_mean_coef1 = posterior_mean_coef1.unsqueeze(-1)
            posterior_mean_coef2 = posterior_mean_coef2.unsqueeze(-1)

        posterior_mean = posterior_mean_coef1 * x_0_pred + posterior_mean_coef2 * x_t

        # Add noise (except for t=0)
        if t[0] > 0:
            posterior_variance = self.schedule.posterior_variance[t].to(device)
            while len(posterior_variance.shape) < len(x_t.shape):
                posterior_variance = posterior_variance.unsqueeze(-1)

            noise = torch.randn_like(x_t)
            return posterior_mean + torch.sqrt(posterior_variance) * noise
        else:
            return posterior_mean

    def sample(
        self,
        model: nn.Module,
        shape: tuple,
        condition: dict,
        num_steps: Optional[int] = None,
        return_trajectory: bool = False,
    ) -> torch.Tensor:
        """
        DDPM sampling (slow but accurate).
        """
        device = next(model.parameters()).device
        num_steps = num_steps or self.schedule.num_timesteps

        # Start from random noise
        x_t = torch.randn(shape, device=device)

        trajectory = [x_t] if return_trajectory else None

        # Reverse process
        for i in reversed(range(num_steps)):
            t = torch.full((shape[0],), i, device=device, dtype=torch.long)
            x_t = self.p_sample(model, x_t, t, condition)

            if return_trajectory:
                trajectory.append(x_t)

        if return_trajectory:
            return torch.stack(trajectory, dim=0)
        else:
            return x_t

    def training_losses(
        self,
        model: nn.Module,
        x_0: torch.Tensor,
        condition: dict,
    ) -> dict:
        """
        Compute training losses for DDPM.
        """
        device = x_0.device
        batch_size = x_0.shape[0]

        # Sample random timesteps
        t = torch.randint(0, self.schedule.num_timesteps, (batch_size,), device=device).long()

        # Sample noise
        noise = torch.randn_like(x_0)

        # Forward diffusion
        x_t = self.q_sample(x_0, t, noise)

        # Predict noise
        noise_pred = model(
            x_t,
            t,
            agent_latents=condition.get("agent_latents", []),
            transforms=condition.get("transforms", []),
            ego_bev=condition.get("ego_bev"),
        )

        alpha_cumprod_t = self.schedule.alphas_cumprod[t].to(device)
        while len(alpha_cumprod_t.shape) < len(x_0.shape):
            alpha_cumprod_t = alpha_cumprod_t.unsqueeze(-1)

        x_0_pred = (
            x_t - torch.sqrt(1 - alpha_cumprod_t) * noise_pred
        ) / torch.sqrt(alpha_cumprod_t).clamp(min=1e-8)

        # Compute loss
        if self.loss_type == "mse":
            loss = torch.nn.functional.mse_loss(noise_pred, noise)
        elif self.loss_type == "l1":
            loss = torch.nn.functional.l1_loss(noise_pred, noise)
        else:
            raise ValueError(f"Unknown loss type: {self.loss_type}")

        return {
            "diffusion_loss": loss,
            "noise_pred": noise_pred,
            "noise_target": noise,
            "x_0_pred": x_0_pred,
            "x_t": x_t,
            "t": t,
        }


class DDIMSampler(BaseSampler):
    """
    DDIM (Denoising Diffusion Implicit Models) sampler.
    Provides deterministic sampling with fewer steps for fast inference.

    Reference: "Denoising Diffusion Implicit Models" (Song et al., 2020)

    Supports both noise prediction and x0 prediction models.
    """
    def __init__(
        self,
        schedule: DiffusionSchedule,
        eta: float = 0.0,  # 0 = deterministic, 1 = stochastic (DDPM)
        prediction_type: str = "noise",  # "noise" or "x0"
    ):
        super().__init__(schedule)
        self.eta = eta
        self.prediction_type = prediction_type

    def ddim_step(
        self,
        model: nn.Module,
        x_t: torch.Tensor,
        t: torch.Tensor,
        t_prev: torch.Tensor,
        condition: dict,
    ) -> torch.Tensor:
        """
        Single DDIM sampling step.
        """
        device = x_t.device

        # Get model prediction
        model_output = model(
            x_t,
            t,
            agent_latents=condition.get("agent_latents", []),
            transforms=condition.get("transforms", []),
            ego_bev=condition.get("ego_bev"),
        )

        # Get alpha values
        alpha_t = self.schedule.alphas_cumprod[t].to(device)
        alpha_prev = self.schedule.alphas_cumprod[t_prev].to(device) if t_prev[0] >= 0 else torch.ones_like(alpha_t)

        # Reshape for broadcasting
        while len(alpha_t.shape) < len(x_t.shape):
            alpha_t = alpha_t.unsqueeze(-1)
            alpha_prev = alpha_prev.unsqueeze(-1)

        # Predict x_0 based on prediction type
        if self.prediction_type == "x0":
            # Model directly predicts x0 (JiT style)
            x_0_pred = model_output
            # Reconstruct noise from x0 prediction for DDIM step
            # noise = (x_t - sqrt(alpha_t) * x0) / sqrt(1 - alpha_t)
            noise_pred = (x_t - torch.sqrt(alpha_t) * x_0_pred) / torch.sqrt(1 - alpha_t).clamp(min=1e-8)
        elif self.prediction_type == "noise":
            # Model predicts noise (standard DDPM)
            noise_pred = model_output
            x_0_pred = (x_t - torch.sqrt(1 - alpha_t) * noise_pred) / torch.sqrt(alpha_t)
        else:
            raise ValueError(f"Unknown prediction_type: {self.prediction_type}")

        # Compute variance
        sigma_t = self.eta * torch.sqrt((1 - alpha_prev) / (1 - alpha_t) * (1 - alpha_t / alpha_prev))

        # Compute direction pointing to x_t
        dir_xt = torch.sqrt(1 - alpha_prev - sigma_t**2) * noise_pred

        # Random noise
        noise = torch.randn_like(x_t) if self.eta > 0 else 0

        # Compute x_{t-1}
        x_prev = torch.sqrt(alpha_prev) * x_0_pred + dir_xt + sigma_t * noise

        return x_prev

    def sample(
        self,
        model: nn.Module,
        shape: tuple,
        condition: dict,
        num_steps: Optional[int] = None,
        return_trajectory: bool = False,
    ) -> torch.Tensor:
        """
        DDIM sampling (fast, deterministic by default).

        Args:
            num_steps: Number of sampling steps (e.g., 50 instead of 1000)
        """
        device = next(model.parameters()).device
        num_steps = num_steps or 50  # Default to 50 steps for DDIM

        # Create subsequence of timesteps
        step_size = self.schedule.num_timesteps // num_steps
        timesteps = list(range(0, self.schedule.num_timesteps, step_size))[:num_steps]
        timesteps = list(reversed(timesteps))

        # Start from random noise
        x_t = torch.randn(shape, device=device)

        trajectory = [x_t] if return_trajectory else None

        # Reverse process
        for i, t_idx in enumerate(timesteps):
            t = torch.full((shape[0],), t_idx, device=device, dtype=torch.long)
            t_prev_idx = timesteps[i + 1] if i + 1 < len(timesteps) else -1
            t_prev = torch.full((shape[0],), t_prev_idx, device=device, dtype=torch.long)

            x_t = self.ddim_step(model, x_t, t, t_prev, condition)

            if return_trajectory:
                trajectory.append(x_t)

        if return_trajectory:
            return torch.stack(trajectory, dim=0)
        else:
            return x_t

    def training_losses(
        self,
        model: nn.Module,
        x_0: torch.Tensor,
        condition: dict,
    ) -> dict:
        """
        DDIM uses same training loss as DDPM.
        """
        device = x_0.device
        batch_size = x_0.shape[0]

        # Sample random timesteps
        t = torch.randint(0, self.schedule.num_timesteps, (batch_size,), device=device).long()

        # Sample noise
        noise = torch.randn_like(x_0)

        # Forward diffusion (same as DDPM)
        sqrt_alphas_cumprod_t = self.schedule.sqrt_alphas_cumprod[t].to(device)
        sqrt_one_minus_alphas_cumprod_t = self.schedule.sqrt_one_minus_alphas_cumprod[t].to(device)

        while len(sqrt_alphas_cumprod_t.shape) < len(x_0.shape):
            sqrt_alphas_cumprod_t = sqrt_alphas_cumprod_t.unsqueeze(-1)
            sqrt_one_minus_alphas_cumprod_t = sqrt_one_minus_alphas_cumprod_t.unsqueeze(-1)

        x_t = sqrt_alphas_cumprod_t * x_0 + sqrt_one_minus_alphas_cumprod_t * noise

        # Predict noise
        noise_pred = model(
            x_t,
            t,
            agent_latents=condition.get("agent_latents", []),
            transforms=condition.get("transforms", []),
            ego_bev=condition.get("ego_bev"),
        )

        alpha_cumprod_t = self.schedule.alphas_cumprod[t].to(device)
        while len(alpha_cumprod_t.shape) < len(x_0.shape):
            alpha_cumprod_t = alpha_cumprod_t.unsqueeze(-1)

        x_0_pred = (
            x_t - torch.sqrt(1 - alpha_cumprod_t) * noise_pred
        ) / torch.sqrt(alpha_cumprod_t).clamp(min=1e-8)

        # MSE loss
        loss = torch.nn.functional.mse_loss(noise_pred, noise)

        return {
            "diffusion_loss": loss,
            "noise_pred": noise_pred,
            "noise_target": noise,
            "x_0_pred": x_0_pred,
            "x_t": x_t,
            "t": t,
        }


def create_sampler(sampler_type: str, schedule: DiffusionSchedule, **kwargs) -> BaseSampler:
    """
    Factory function to create samplers.

    Args:
        sampler_type: "ddpm" or "ddim"
        schedule: DiffusionSchedule instance
        **kwargs: Additional arguments for specific samplers

    Returns:
        BaseSampler instance
    """
    if sampler_type == "ddpm":
        return DDPMSampler(schedule, **kwargs)
    elif sampler_type == "ddim":
        return DDIMSampler(schedule, **kwargs)
    else:
        raise ValueError(f"Unknown sampler type: {sampler_type}")


if __name__ == "__main__":
    # Test the samplers
    from diffusion_model import ConditionalDiffusionUNet

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Create schedule
    schedule = DiffusionSchedule(
        num_timesteps=1000,
        beta_schedule="cosine",
    ).to(device)

    # Create model
    model = ConditionalDiffusionUNet(
        in_channels=64,
        model_channels=128,
        out_channels=64,
    ).to(device)

    # Create samplers
    ddpm_sampler = DDPMSampler(schedule, loss_type="mse")
    ddim_sampler = DDIMSampler(schedule, eta=0.0)

    # Test training
    x_0 = torch.randn(2, 64, 100, 352).to(device)
    condition = {
        # ego_obs removed - no cheating!
        "agent_latents": [torch.randn(2, 16, 13, 44).to(device) for _ in range(3)],
        "transforms": [torch.eye(2, 3).unsqueeze(0).repeat(2, 1, 1).to(device) for _ in range(3)],  # [B, 2, 3] affine
    }

    losses = ddpm_sampler.training_losses(model, x_0, condition)
    print(f"DDPM training loss: {losses['diffusion_loss'].item():.4f}")

    # Test sampling
    with torch.no_grad():
        sample_ddim = ddim_sampler.sample(
            model,
            shape=(2, 64, 100, 352),
            condition=condition,
            num_steps=50,
        )
        print(f"DDIM sample shape: {sample_ddim.shape}")
