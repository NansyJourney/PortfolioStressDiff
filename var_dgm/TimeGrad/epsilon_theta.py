import math
import torch
import torch.nn.functional as F
from torch import nn


class DiffusionEmbedding(nn.Module):
    def __init__(self, dim, proj_dim, max_steps=500):
        """
        Initializes the DiffusionEmbedding class.

        Parameters
        ----------
        dim : int
            Dimension of the embedding.
        proj_dim : int
            Dimension of the projected embedding.
        max_steps : int, optional
            Maximum number of diffusion steps, by default 500.
        """
        super().__init__()
        self.register_buffer(
            "embedding", self._build_embedding(dim, max_steps), persistent=False
        )
        self.projection1 = nn.Linear(dim * 2, proj_dim)
        self.projection2 = nn.Linear(proj_dim, proj_dim)

    def forward(self, diffusion_step):
        """
        Forward pass for the DiffusionEmbedding class.

        Parameters
        ----------
        diffusion_step : int
            Current diffusion step.

        Returns
        -------
        torch.Tensor
            Projected embedding for the given diffusion step.
        """
        x = self.embedding[diffusion_step]
        x = self.projection1(x)
        x = F.silu(x)
        x = self.projection2(x)
        x = F.silu(x)
        return x

    def _build_embedding(self, dim, max_steps):
        """
        Builds the diffusion embedding.

        Parameters
        ----------
        dim : int
            Dimension of the embedding.
        max_steps : int
            Maximum number of diffusion steps.

        Returns
        -------
        torch.Tensor
            Embedding tensor.
        """
        steps = torch.arange(max_steps).unsqueeze(1)  # [T,1]
        dims = torch.arange(dim).unsqueeze(0)  # [1,dim]
        table = steps * 10.0 ** (dims * 4.0 / dim)  # [T,dim]
        table = torch.cat([torch.sin(table), torch.cos(table)], dim=1)
        return table


class ResidualBlock(nn.Module):
    def __init__(self, hidden_size, residual_channels, dilation):
        """
        Initializes the ResidualBlock class.

        Parameters
        ----------
        hidden_size : int
            Size of the hidden layer.
        residual_channels : int
            Number of residual channels.
        dilation : int
            Dilation rate for the dilated convolution.
        """
        super().__init__()
        self.dilated_conv = nn.Conv1d(
            residual_channels,
            2 * residual_channels,
            3,
            padding=dilation,
            dilation=dilation,
            padding_mode="circular",
        )
        self.diffusion_projection = nn.Linear(hidden_size, residual_channels)
        self.conditioner_projection = nn.Conv1d(
            1, 2 * residual_channels, 1, padding=2, padding_mode="circular"
        )
        self.output_projection = nn.Conv1d(residual_channels, 2 * residual_channels, 1)

        nn.init.kaiming_normal_(self.conditioner_projection.weight)
        nn.init.kaiming_normal_(self.output_projection.weight)

    def forward(self, x, conditioner, diffusion_step):
        """
        Forward pass for the ResidualBlock class.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor.
        conditioner : torch.Tensor
            Conditioning tensor.
        diffusion_step : torch.Tensor
            Tensor representing the current diffusion step.

        Returns
        -------
        tuple
            Tuple containing the residual and skip connection tensors.
        """
        diffusion_step = self.diffusion_projection(diffusion_step).unsqueeze(-1)
        conditioner = self.conditioner_projection(conditioner)

        y = x + diffusion_step
        y = self.dilated_conv(y) + conditioner

        gate, filter = torch.chunk(y, 2, dim=1)
        y = torch.sigmoid(gate) * torch.tanh(filter)

        y = self.output_projection(y)
        y = F.leaky_relu(y, 0.4)
        residual, skip = torch.chunk(y, 2, dim=1)
        return (x + residual) / math.sqrt(2.0), skip


class CondUpsampler(nn.Module):
    def __init__(self, cond_dim, target_dim):
        """
        Initializes the CondUpsampler class.

        Parameters
        ----------
        cond_dim : int
            Dimension of the conditioning input.
        target_dim : int
            Target dimension for the upsampled output.
        """
        super().__init__()
        self.linear1 = nn.Linear(cond_dim, target_dim // 2)
        self.linear2 = nn.Linear(target_dim // 2, target_dim)

    def forward(self, x):
        """
        Forward pass for the CondUpsampler class.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor.

        Returns
        -------
        torch.Tensor
            Upsampled output tensor.
        """
        x = self.linear1(x)
        x = F.leaky_relu(x, 0.4)
        x = self.linear2(x)
        x = F.leaky_relu(x, 0.4)
        return x


class EpsilonTheta(nn.Module):
    def __init__(
        self,
        target_dim,
        cond_dim,
        time_emb_dim=16,
        residual_layers=8,
        residual_channels=8,
        dilation_cycle_length=2,
        residual_hidden=64,
    ):
        """
        Initializes the EpsilonTheta class.

        Parameters
        ----------
        target_dim : int
            Target dimension for the output.
        cond_dim : int
            Dimension of the conditioning input.
        time_emb_dim : int, optional
            Dimension of the time embedding, by default 16.
        residual_layers : int, optional
            Number of residual layers, by default 8.
        residual_channels : int, optional
            Number of residual channels, by default 8.
        dilation_cycle_length : int, optional
            Length of the dilation cycle, by default 2.
        residual_hidden : int, optional
            Size of the hidden layer in the residual blocks, by default 64.
        """
        super().__init__()
        self.flag_1_dim = True if target_dim == 1 else False
        self.target_dim = target_dim if target_dim > 1 else 2

        self.input_projection = nn.Conv1d(
            1, residual_channels, 1, padding=2, padding_mode="circular"
        )
        self.diffusion_embedding = DiffusionEmbedding(
            time_emb_dim, proj_dim=residual_hidden
        )
        self.cond_upsampler = CondUpsampler(
            target_dim=self.target_dim, cond_dim=cond_dim
        )
        self.residual_layers = nn.ModuleList(
            [
                ResidualBlock(
                    residual_channels=residual_channels,
                    dilation=2 ** (i % dilation_cycle_length),
                    hidden_size=residual_hidden,
                )
                for i in range(residual_layers)
            ]
        )
        self.skip_projection = nn.Conv1d(residual_channels, residual_channels, 3)
        self.output_projection = nn.Conv1d(residual_channels, 1, 3)

        nn.init.kaiming_normal_(self.input_projection.weight)
        nn.init.kaiming_normal_(self.skip_projection.weight)
        nn.init.zeros_(self.output_projection.weight)

    def forward(self, inputs, time, cond):
        """
        Forward pass for the EpsilonTheta class.

        Parameters
        ----------
        inputs : torch.Tensor
            Input tensor.
        time : torch.Tensor
            Tensor representing the diffusion step.
        cond : torch.Tensor
            Conditioning tensor.

        Returns
        -------
        torch.Tensor
            Output tensor.
        """
        if self.flag_1_dim:
            # 1-dimensional time series: add another time series with zeros
            inputs = torch.cat(
                (inputs, torch.zeros(inputs.shape, device=inputs.device)), dim=2
            )

        x = self.input_projection(inputs)
        x = F.leaky_relu(x, 0.4)

        diffusion_step = self.diffusion_embedding(time.long())
        cond_up = self.cond_upsampler(cond)
        skip = []
        for layer in self.residual_layers:
            x, skip_connection = layer(x, cond_up, diffusion_step)
            skip.append(skip_connection)

        x = torch.sum(torch.stack(skip), dim=0) / math.sqrt(len(self.residual_layers))
        x = self.skip_projection(x)
        x = F.leaky_relu(x, 0.4)
        x = self.output_projection(x)
        if self.flag_1_dim:
            x = x[:, :, [0]]
        return x
