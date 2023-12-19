import torch.nn as nn


class SharedMLP(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size=1,
        stride=1,
        transpose=False,
        padding_mode="zeros",
        bn=False,
        activation_fn=None,
    ):
        super(SharedMLP, self).__init__()

        conv_fn = nn.ConvTranspose2d if transpose else nn.Conv2d

        self.conv = conv_fn(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding_mode=padding_mode,
        )
        self.batch_norm = (
            nn.BatchNorm2d(out_channels, eps=1e-6, momentum=0.99) if bn else None
        )
        self.activation_fn = activation_fn

    def forward(self, input):
        r"""
        Forward pass of the network

        Parameters
        ----------
        input: torch.Tensor, shape (B, d_in, N, K)

        Returns
        -------
        torch.Tensor, shape (B, d_out, N, K)
        """
        x = self.conv(input)
        if self.batch_norm:
            x = self.batch_norm(x)
        if self.activation_fn:
            x = self.activation_fn(x)
        return x