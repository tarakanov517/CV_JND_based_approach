import torch
import torch.nn as nn
import numpy as np

class JNDBPDAFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x_rgb, converter, jnd_model, layer_type, target_width_cm):
        device = x_rgb.device
        x_np = x_rgb.detach().cpu().numpy().transpose(0, 2, 3, 1)

        if x_np.max() <= 1.0:
            x_np = x_np * 255.0
        x_np = np.clip(x_np, 0.0, 255.0)

        batch_out = []
        for img in x_np:
            xyzL_image = converter.RGB_to_xyzL(img)
            L_physical = xyzL_image[..., 3]

            jnd_model.find_La_with_background(L_patch=L_physical, target_width_cm=target_width_cm)
            jnd_model.build_level_boundaries()
            k_map = jnd_model.L_to_k(L_physical)

            k_map_expanded = np.expand_dims(k_map, axis=-1)
            xyzk_matrix = np.concatenate([xyzL_image[..., :3], k_map_expanded], axis=-1).astype(np.float32)

            if layer_type == 'kkk':
                jnd_img = np.repeat(xyzk_matrix[..., 3:4], repeats=3, axis=-1)
            elif layer_type == 'xzk':
                jnd_img = xyzk_matrix[..., [0, 2, 3]]
            else:
                raise ValueError(f"Неизвестный layer_type: {layer_type}")

            batch_out.append(jnd_img)

        # (B, C, H, W)
        out_tensor = torch.from_numpy(np.array(batch_out)).permute(0, 3, 1, 2).float().to(device)
        return out_tensor

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output, None, None, None, None

class BPDAModelWrapper(nn.Module):
    def __init__(self, classifier_model, converter, jnd_model, layer_type='xzk', target_width_cm=None):
        super().__init__()
        self.model = classifier_model
        self.converter = converter
        self.jnd_model = jnd_model
        self.layer_type = layer_type
        self.target_width_cm = target_width_cm

    def forward(self, x_rgb):
        if self.layer_type == 'rgb':
            return self.model(x_rgb)

        x_jnd = JNDBPDAFunction.apply(
            x_rgb, 
            self.converter, 
            self.jnd_model, 
            self.layer_type, 
            self.target_width_cm
        )
        return self.model(x_jnd)