from typing import List, Literal

import torch
import torch.nn as nn
import torch.nn.functional as F

class KL_Divergence(nn.Module):
    """
    Computes the Kullback-Leibler divergence between two distributions.
    adapted from: https://github.com/nilsmechtel/PELICAN/blob/main/models/utils/losses.py
    """
    def __init__(self):
        super().__init__()   
    
    def forward(
            self, 
            mu: torch.tensor, 
            log_var: torch.tensor, 
            mode: Literal["train", "val"], 
            mu2: torch.tensor = None,
            log_var2: torch.tensor = None,
            log_prefix: str = "",
            run = None,
            ) -> torch.tensor:
        """
        Compute the KL divergence between two distributions.
        If mu2 and log_var2 are None, Gaussian is used.

        Parameters
        ----------
        mu : torch.Tensor
            Mean of the distribution.
        log_var : torch.Tensor
            Logarithm of the variance of the distribution.
        mode : str
            Mode of the model (train or val). -> this is used for logging
        mu2 : torch.Tensor
            Mean of a second distribution
        log_var2
            Logarithm of the variance of a second distribution.
        log_prefix
            string to be added to the Neptune log path
        run 
            Neptune run for logging (optional)

        Returns
        -------
        torch.Tensor
            KL divergence between the two distributions.
        """
        if mu2 is None and log_var2 is None:
            kl_divergence = torch.mean(-0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp(), dim=1)) 
        else:
            var1 = torch.exp(log_var)
            var2 = torch.exp(log_var2)
            kl_divergence = torch.mean(torch.sum(0.5 * (torch.log(var2 / var1) + (var1 + (mu - mu2) ** 2) / var2 - 1), dim=1))
        
        if run is not None:
            run[log_prefix + "/metrics/" + mode + "/kl_divergence"].log(kl_divergence)
        return kl_divergence


class MSE_Loss(nn.Module):
    """
    Computes the Mean Squared Error (MSE) loss.
    """
    def __init__(self):
        super().__init__()
    
    def forward(
            self, 
            x: torch.tensor, 
            x_hat: torch.tensor, 
            mode: Literal["train", "val"], 
            log_prefix: str = "",
            run=None
            ) -> torch.tensor:
        """
        Compute the Mean Squared Error (MSE) loss.

        Parameters
        ----------
        x : torch.Tensor
            original data.
        x_hat : torch.Tensor
            reconstructed data.
        mode : str
            Mode of the model (train or val). -> this is used for logging
        log_prefix
            string to be added to the Neptune log path
        run 
            Neptune run for logging (optional)

        Returns
        -------
        torch.Tensor
            MSE loss.
        """
        mse_loss = torch.mean(F.mse_loss(x_hat, x, reduction="sum")) # sum over features and average over minibatch
        if run is not None:
            run[log_prefix + "/metrics/" + mode + "/mse_loss"].log(mse_loss)
        return mse_loss


class VAE_Loss(nn.Module):
    """
    Computes the VAE loss consisting of reconstruction loss (MSE) and weighted KL divergence.
    """
    def __init__(self):
        super().__init__()
    
    def forward(
            self, 
            x: torch.tensor, 
            x_hat: torch.tensor, 
            mu: torch.tensor, 
            log_var: torch.tensor, 
            kl_weight: float,
            mode: Literal["train", "val"],
            log_prefix: str = "",
            run = None,
            ) -> torch.tensor:
        """
        Calculates VAE loss as combination of reconstruction loss and weighted Kullback-Leibler loss.

        Parameters
        ----------
        x : torch.Tensor
           original data.
        x_hat : torch.Tensor
            reconstructed data.
        mu : torch.Tensor
            Mean of the distribution.
        log_var : torch.Tensor
            Logarithm of the variance of the distribution.
        kl_weight : float
            Weighting coefficient for the KL divergence.
        mode   
            Mode of the model (train or val). -> this is used for logging
        log_prefix
            string to be added to the Neptune log path
        run
            Neptune run for logging (optional)

        Returns
        -------
        torch.Tensor
            VAE loss.
        """
        rec_loss = MSE_Loss()(x_hat, x, mode=mode, run=run)
        kl_div = KL_Divergence()(mu, log_var, mode=mode, run=run)
        vae_loss = rec_loss + kl_weight * kl_div
        if run is not None:
            run[log_prefix + "/metrics/" + mode + "/vae_loss"].log(vae_loss)
        return vae_loss


class CrossEntropyLoss(nn.Module):
    """
    Computes the Cross Entropy Loss for a (multi-)class classifier.
    """
    def __init__(self):
        super().__init__()
    
    def forward(
            self, 
            logits: torch.tensor, 
            labels: torch.tensor, 
            mode: Literal["train", "val"], 
            log_prefix: str="",
            run = None,
            ) -> torch.tensor:

        loss = F.cross_entropy(logits, labels)
        if run is not None:
            run[log_prefix + "/metrics/" + mode + "/cross_entropy"].log(loss)
        return loss
    


class Gradient_Penalty(nn.Module):
    """
    This class defines the Gradient Penalty for a Wasserstein GAN.
    from https://github.com/nilsmechtel/PELICAN/blob/main/models/utils/losses.py
    """
    def __init__(self):
        super().__init__()
    
    def forward(
            self,
            critic: nn.Module,
            z_input: torch.tensor,
            z_target: torch.tensor,
            z_fake: torch.tensor,
            gp_weight: float,
            device: str,
            flag: torch.tensor = None,
    ):
        """
        Computation of Gradient Penalty

        Parameters
        ----------
        critic
            the Critic for which Gradient Penalty is to be computed
        z_target
            The actual latent space embedding vector
        z_fake
            The generators latent space embedding vector
        device
            device on which the tensors are located
        """   

        # interpolate between the real z and the fake z
        alpha = torch.rand(z_target.size(0), 1).to(device)
        interpolate = (alpha * z_target + (1 - alpha) * z_fake).requires_grad_(True)

        # compute the critics output for the interpolate
        inter_output = critic(z_input, interpolate, flag)

        # compute the average gradient penalty
        gradients = torch.autograd.grad(
            outputs = inter_output,
            inputs = interpolate,
            grad_outputs = torch.ones_like(inter_output),
            create_graph = True,
        )[0]

        # penalize gradients that have a magnitude significantly different from 1
        gradients_L2 = torch.norm(gradients, p=2, dim=-1)[
            ..., None
        ]  # over latent features

        gradient_penalty = (gradients_L2 - 1) ** 2
        return gradient_penalty * gp_weight
    


class ContrastiveLoss(nn.Module):
    """
    This class implements a ContrastiveLoss 
    which encourages the model to learn a representation
    which minimizes the distance between positive pairs
    while maximizing the distance between negative pairs.    
    """

    def __init__(self, temperature=0.1):
        super().__init__()
        self.temperature = temperature

    def forward(
            self, 
            z_src, 
            z_tgt, 
            labels_src,
            labels_tgt: torch.tensor=None,
            mode: Literal["train", "val"]="train", 
            log_prefix: str="",
            run = None,
            ):
        
        if labels_tgt is None:
            labels_tgt = labels_src
            
        # Normalize embeddings
        z_src = F.normalize(z_src, dim=1)
        z_tgt = F.normalize(z_tgt, dim=1)

        # Compute cosine similarity matrix (B, B)
        logits = torch.matmul(z_src, z_tgt.T) / self.temperature

        # Create (B, B) mask of positive pairs
        pos_mask = labels_src.unsqueeze(1) == labels_tgt.unsqueeze(0)  # (B, B)
        neg_mask = ~pos_mask  # (B, B)

        # Mask out invalid (zero-positive) rows
        valid_rows = pos_mask.any(dim=1)

        # Log-sum-exp over all logits per row
        logsumexp_all = torch.logsumexp(logits, dim=1)

        # For numerical stability, mask and compute logsumexp over positives only
        logits_pos = logits.masked_fill(~pos_mask, float('-inf'))
        logsumexp_pos = torch.logsumexp(logits_pos, dim=1)

        # Final contrastive loss (only valid rows)
        loss = (logsumexp_all - logsumexp_pos)[valid_rows].mean()

        if run is not None:
            run[log_prefix + "/metrics/" + mode + "/contrastive_loss"].log(loss)

        return loss
    

class MMDLoss(nn.Module):
    def __init__(self, kernel='rbf', sigma=1.0):
        super(MMDLoss, self).__init__()
        self.kernel = kernel
        self.sigma = sigma

    def compute_kernel(self, x, y):
        """
        Compute RBF kernel between x and y.
        x: [n, d]
        y: [m, d]
        Returns: [n, m] kernel matrix
        """
        x = x.unsqueeze(1)  # [n, 1, d]
        y = y.unsqueeze(0)  # [1, m, d]
        dist = ((x - y) ** 2).sum(2)  # [n, m]
        return torch.exp(-dist / (2 * self.sigma ** 2))

    def forward(
            self, 
            z_a, 
            z_b,
            mode: Literal["train", "val"]="train", 
            log_prefix: str="",
            run = None,
            ):
        """
        Compute MMD loss between latent codes from two batches.
        z_a: [batch_size_a, latent_dim]
        z_b: [batch_size_b, latent_dim]
        """
        K_aa = self.compute_kernel(z_a, z_a)
        K_bb = self.compute_kernel(z_b, z_b)
        K_ab = self.compute_kernel(z_a, z_b)

        #m = z_a.size(0)
        #n = z_b.size(0)

        # Unbiased estimate of MMD (simplified for clarity)
        mmd = K_aa.mean() + K_bb.mean() - 2 * K_ab.mean()

        if run is not None:
            run[log_prefix + "/metrics/" + mode + "/mmd_loss"].log(mmd)

        return mmd
