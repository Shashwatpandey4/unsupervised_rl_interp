import cv2
import torch
import torch.nn.functional as F


class GradCAM:
    def __init__(self, model, target_layer_name):
        self.model = model
        self.model.eval()

        self.target_layer = dict([*self.model.named_modules()])[target_layer_name]
        self.gradients = None
        self.activations = None

        self._register_hooks()

    def _register_hooks(self):
        def forward_hook(module, input, output):
            self.activations = output.detach()

        def backward_hook(module, grad_input, grad_output):
            self.gradients = grad_output[0].detach()

        self.target_layer.register_forward_hook(forward_hook)
        self.target_layer.register_backward_hook(backward_hook)

    def generate_cam(self, input_tensor, class_idx=None):
        output = self.model(input_tensor)

        # ✅ Only apply if output is a tuple
        if isinstance(output, tuple):
            output = output[0]  # use logits or Q-values only

        if class_idx is None:
            class_idx = output.argmax(dim=1).item()

        one_hot = torch.zeros_like(output)
        one_hot[0, class_idx] = 1
        self.model.zero_grad()
        output.backward(gradient=one_hot, retain_graph=True)

        weights = self.gradients.mean(dim=(2, 3), keepdim=True)
        cam = (weights * self.activations).sum(dim=1).squeeze()

        cam = F.relu(cam)
        cam -= cam.min()
        cam /= cam.max() + 1e-6
        cam_np = cam.cpu().numpy()

        cam_np = cv2.resize(cam_np, (input_tensor.shape[3], input_tensor.shape[2]))
        return cam_np
