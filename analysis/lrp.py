import copy

import torch
from captum.attr import LRP


def generate_lrp_map(policy_model, input_tensor):
    """
    Generate an LRP relevance map for the convolutional block of a policy network.

    Args:
        policy_model (torch.nn.Module): The policy network, expected to have a `conv` attribute.
        input_tensor (torch.Tensor): Input tensor of shape [1, C, H, W].

    Returns:
        np.ndarray: Relevance map with the same spatial dimensions as the conv output.
    """
    policy_model.eval()

    # Extract and deep-copy the conv module to avoid moving original weights
    conv_module = policy_model.conv
    conv_copy = copy.deepcopy(conv_module).cpu()
    conv_copy.eval()

    # Wrapper to flatten conv output
    class ConvFlattenWrapper(torch.nn.Module):
        def __init__(self, conv):
            super().__init__()
            self.conv = conv

        def forward(self, x):
            out = self.conv(x)
            return out.view(out.size(0), -1)

    # Instantiate the CPU-only model
    model = ConvFlattenWrapper(conv_copy).cpu()
    model.eval()

    # Prepare input on CPU with gradient tracking
    input_tensor = input_tensor.cpu().detach().requires_grad_(True)

    # Determine the target index (max conv activation)
    with torch.no_grad():
        output = model(input_tensor)
        target = output.argmax(dim=1).item()

    # Compute LRP attributions
    lrp = LRP(model)
    relevance = lrp.attribute(input_tensor, target=target)

    # Squeeze out batch dimension and return as numpy
    return relevance.squeeze(0).detach().numpy()
