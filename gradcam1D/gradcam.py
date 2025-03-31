import torch
import torch.nn.functional as F

class GradCAM1D:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        self.hooks = []  # List to store hook handles
        self.hook_layers()

    def hook_layers(self):
        def forward_hook(module, input, output):
            self.activations = output.detach()

        def backward_hook(module, grad_in, grad_out):
            self.gradients = grad_out[0].detach()

        # Attach hooks and store the handles
        self.hooks.append(self.target_layer.register_forward_hook(forward_hook))
        self.hooks.append(self.target_layer.register_backward_hook(backward_hook))

    def remove_hooks(self):
        # Remove all hooks in the list
        for hook in self.hooks:
            hook.remove()
        self.hooks = []  # Clear the list of hooks

    def compute_cam(self, input_tensor, target_category=None):
        output = self.model(input_tensor)
        if target_category is None:
            target_category = torch.argmax(output)
        
        self.model.zero_grad()
        one_hot_output = torch.zeros_like(output)
        one_hot_output[0, target_category] = 1
        output.backward(gradient=one_hot_output, retain_graph=True)

        guided_gradients = self.gradients
        alpha = guided_gradients.mean(dim=2, keepdim=True)
        cam = F.relu(torch.sum(alpha * self.activations, dim=1)).squeeze(0)
        
        return cam