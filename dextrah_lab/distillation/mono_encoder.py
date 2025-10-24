from PIL import Image
import math
import numpy as np

import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from torch.nn import functional as F

from gfm.resnet_bifpn import ResNetBiFPN
import time

CKPT_PATH = "/media/sharedaccess/Manthan_RSL_SSD/clariden_models/vitl14_data_all_distill_multi_resnet18_34_50_625k_webd_bs_2048/resnet18_ema_checkpoint.pth"

def conv_output_size(h_w, kernel_size=1, stride=1, pad=0, dilation=1):
    """
    Utility function to compute the output size of a convolution layer.
    
    h_w: Tuple[int, int] - height and width of the input
    kernel_size: int or Tuple[int, int] - size of the convolution kernel
    stride: int or Tuple[int, int] - stride of the convolution
    pad: int or Tuple[int, int] - padding
    dilation: int or Tuple[int, int] - dilation rate
    """
    if isinstance(kernel_size, tuple):
        kernel_h, kernel_w = kernel_size
    else:
        kernel_h, kernel_w = kernel_size, kernel_size
    
    if isinstance(stride, tuple):
        stride_h, stride_w = stride
    else:
        stride_h, stride_w = stride, stride
    
    if isinstance(pad, tuple):
        pad_h, pad_w = pad
    else:
        pad_h, pad_w = pad, pad
    
    h = (h_w[0] + 2 * pad_h - dilation * (kernel_h - 1) - 1) // stride_h + 1
    w = (h_w[1] + 2 * pad_w - dilation * (kernel_w - 1) - 1) // stride_w + 1
    return h, w


class CustomCNN(nn.Module):
    def __init__(self, input_height, input_width, device):
        super().__init__()
        self.device = device
        num_channel = 3
        
        # Initial input dimensions
        h, w = input_height, input_width
        
        # Layer 1
        h, w = conv_output_size((h, w), kernel_size=6, stride=2)
        layer1_norm_shape = [16, h, w]
        
        # Layer 2
        h, w = conv_output_size((h, w), kernel_size=4, stride=2)
        layer2_norm_shape = [32, h, w]
        
        # Layer 3
        h, w = conv_output_size((h, w), kernel_size=4, stride=2)
        layer3_norm_shape = [64, h, w]
        
        # Layer 4
        h, w = conv_output_size((h, w), kernel_size=4, stride=2)
        layer4_norm_shape = [128, h, w]
        
        # CNN definition
        self.cnn = nn.Sequential(
            nn.Conv2d(num_channel, 16, kernel_size=6, stride=2, padding=0),
            nn.ReLU(),
            nn.LayerNorm(layer1_norm_shape),  # Dynamically calculated layer norm
            nn.Conv2d(16, 32, kernel_size=4, stride=2, padding=0),
            nn.ReLU(),
            nn.LayerNorm(layer2_norm_shape),  # Dynamically calculated layer norm
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=0),
            nn.ReLU(),
            nn.LayerNorm(layer3_norm_shape),  # Dynamically calculated layer norm
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=0),
            nn.ReLU(),
            nn.LayerNorm(layer4_norm_shape),  # Dynamically calculated layer norm
        )
        

    def forward(self, x, train_encoder=True):
        cnn_x = self.cnn(x)
        return cnn_x


# def get_standard_transform():
#     transform = [transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]) ]
#     transform = transforms.Compose(transform)
#     return transform

def get_standard_transform(device):
    # Pre-create the mean and std tensors on the target device with bf16 dtype
    mean = torch.tensor([0.485, 0.456, 0.406], device=device, dtype=torch.bfloat16)
    std = torch.tensor([0.229, 0.224, 0.225], device=device, dtype=torch.bfloat16)
    
    # Create a lambda transform that explicitly casts to bf16 and normalizes
    transform = [
        transforms.Lambda(lambda x: (x.to(dtype=torch.bfloat16) - mean[None, :, None, None]) / std[None, :, None, None])
    ]
    transform = transforms.Compose(transform)
    return transform




class ResnetEncoder(nn.Module):
    def __init__(self, input_height, input_width, device="cuda", train_resnet=True, **kwargs):
        super().__init__()
        self.device = device

        self.train_resnet = train_resnet

        device = "cuda:0"
        self.resnet18 = torchvision.models.resnet18(
            weights=torchvision.models.ResNet18_Weights.IMAGENET1K_V1
        ).to(torch.bfloat16)
        # remove last 2 layers of resnet18
        self.resnet18.fc = nn.Identity()
        self.resnet18.avgpool = nn.Identity()

        if train_resnet:
            self.resnet18.train().to(device)
        else:
            self.resnet18.eval().to(device)

        self.transform = get_standard_transform(self.device)

        # Linear layers
        self.linear = nn.Sequential(
            nn.Linear(40960, 16384)
        )


    def forward(self, x, train_encoder=True):
        x = x.to(torch.bfloat16)

        if train_encoder:
            x = self.transform(x)
            resnet_out = self.resnet18(x)
        else:
            with torch.no_grad():
                x = self.transform(x)
                resnet_out = self.resnet18(x)
        out = self.linear(resnet_out.to(torch.float32))
        return out
    
class ResnetDepthEncoder(nn.Module):
    def __init__(self, input_height, input_width, device="cuda", train_resnet=True, **kwargs):
        super().__init__()
        self.device = device

        self.train_resnet = train_resnet

        device = "cuda:0"
        self.resnet18 = torchvision.models.resnet18(
            weights=torchvision.models.ResNet18_Weights.IMAGENET1K_V1
        ).to(torch.bfloat16)
        # remove last 2 layers of resnet18
        self.resnet18.fc = nn.Identity()
        self.resnet18.avgpool = nn.Identity()

        if train_resnet:
            self.resnet18.train().to(device)
        else:
            self.resnet18.eval().to(device)

        self.transform = get_standard_transform(self.device)

        # Linear layers
        self.linear = nn.Sequential(
            nn.Linear(40960, 16384)
        )

    def forward(self, x, train_encoder=True):
        x = x.to(torch.bfloat16)

        # Stack to 3 channels
        if x.shape[1] == 1:
            x = x.repeat(1, 3, 1, 1)

        if train_encoder:
            x = self.transform(x)
            resnet_out = self.resnet18(x)
        else:
            with torch.no_grad():
                x = self.transform(x)
                resnet_out = self.resnet18(x)
        out = self.linear(resnet_out.to(torch.float32))
        return out


class Resnet_GFMEncoder(nn.Module):
    def __init__(self,  input_height, input_width, device="cuda", ckpt_path=None, backbone="resnet18",  out_channels=128, freeze_backbone=True):
        super().__init__()

        self.device = device
        self.backbone = backbone
        self.out_channels = out_channels
        self.freeze_backbone = freeze_backbone

        self.model = ResNetBiFPN(backbone_name=backbone, out_channels=out_channels)

        # Resize input image to the closest multiple of 32
        self.input_height = input_height
        self.input_width = input_width
        self.img_height = (input_height + 31) // 32 * 32
        self.img_width = (input_width + 31) // 32 * 32

        self.resize_transform = transforms.Resize((self.img_height, self.img_width))

        print(f"Resizing input from ({input_height}, {input_width}) to ({self.img_height}, {self.img_width})")

        self.load_pretrained_weights(ckpt_path)

        # freeze backbone
        if freeze_backbone:
            self.model.eval()
            for param in self.model.backbone.parameters():
                param.requires_grad = False
            print("Backbone frozen")


        self.normalize_depth = transforms.Normalize(mean=[0.248880, 0.495620, 0.492858],
                                                    std=[0.139357, 0.271314, 0.297177])
        
        embed_dim = self.model.embed_dim # This is the global pooled CLS token

        # Compute the output dim
        self.output_dim = (self.img_height // 16) * (self.img_width // 16) * out_channels + embed_dim

        print(f"Resnet_GFMEncoder output dim: {self.output_dim}")

        # Linear layers
        self.linear = nn.Sequential(
            nn.Linear(self.output_dim, 16384)
        )


    def load_pretrained_weights(self, ckpt_path=None):
        if ckpt_path is not None:
            print(f"Loading ResNetBiFPN weights from {ckpt_path}")
            state_dict = torch.load(ckpt_path, map_location="cpu")

            # Handle Lightning checkpoints
            if "state_dict" in state_dict:
                state_dict = state_dict["state_dict"]

            # Keep only student.backbone.* keys
            backbone_state = {}
            for k, v in state_dict["teacher"].items():
                if k.startswith("backbone."):
                    new_k = k.replace("backbone.", "", 1)  # strip prefix
                    backbone_state[new_k] = v
            
            missing, unexpected = self.model.load_state_dict(backbone_state, strict=True)

            if missing:
                print(f"⚠️ Missing keys when loading: {missing[:10]}... (total {len(missing)})")
            elif unexpected:
                print(f"⚠️ Unexpected keys when loading: {unexpected[:10]}... (total {len(unexpected)})")
            else:
                print("Weights loaded successfully.")
        else:
            print("No checkpoint path provided, using random weights.")

    @property
    def is_blind(self):
        return False

    def convert_metric_to_three_channel_depth(self, metric_depth: torch.Tensor, max_depth_ch0: float = 100.0, max_depth_ch1: float = 10.0) -> torch.Tensor:
        if metric_depth.dim() == 4:
            metric_depth = metric_depth.squeeze(1)  # (B, H, W)
        elif metric_depth.dim() != 3:
            raise ValueError("metric_depth must have shape (B, 1, H, W) or (B, H, W)")

        metric_depth = torch.clamp(metric_depth, min=1e-6)

        log_depth = torch.log1p(metric_depth)

        # TorchScript-safe tensor creation
        max0 = torch.tensor(max_depth_ch0, dtype=metric_depth.dtype, device=metric_depth.device)
        max1 = torch.tensor(max_depth_ch1, dtype=metric_depth.dtype, device=metric_depth.device)

        channel_1 = log_depth / torch.log1p(max0)
        channel_2 = torch.clamp(log_depth / torch.log1p(max1), 0.0, 1.0)

        min_log_depth = torch.amin(log_depth, dim=(1, 2), keepdim=True)
        max_log_depth = torch.amax(log_depth, dim=(1, 2), keepdim=True)
        denom = max_log_depth - min_log_depth

        channel_3 = torch.where(
            denom > 0,
            (log_depth - min_log_depth) / denom,
            torch.zeros_like(log_depth)
        )

        three_channel_depth = torch.stack([channel_1, channel_2, channel_3], dim=1)
        three_channel_depth = self.normalize_depth(three_channel_depth)

        return three_channel_depth

    def forward(self, x, train_encoder=True):
        depth = x  # [B, H, W, C]

        # Here the depth needs to be metric Depth and not scaled between 0-1

        # Need to convert to 3 channel
        depth = depth.permute(0, 3, 1, 2)  # B x C x H x W
        depth = self.resize_transform(depth)
        depth = self.convert_metric_to_three_channel_depth(depth)

        out = self.model(depth)
        spatial_tokens = out["dense_bifpn"]["P4"]  # [B, out_channels, H//16, W//16]
        cls_token = out["global_backbone"]

        spatial_tokens = torch.flatten(spatial_tokens, start_dim=1)
        output = torch.cat([spatial_tokens, cls_token], dim=1)

        output = self.linear(output)

        return output


class ConvNextEncoder(nn.Module):
    def __init__(self, input_height, input_width, device="cuda", train_resnet=True, **kwargs):
        super().__init__()
        self.device = device

        self.train_resnet = train_resnet

        device = "cuda:0"
        self.convnext = torchvision.models.convnext_tiny(
            weights=torchvision.models.ConvNeXt_Tiny_Weights.DEFAULT
        ).to(torch.bfloat16)
        # remove last 2 layers of convnext
        self.convnext.avgpool = nn.Identity()
        self.convnext.classifier = nn.Identity()

        if train_resnet:
            self.convnext.train().to(device)
        else:
            self.convnext.eval().to(device)

        self.transform = get_standard_transform(self.device)

        self.reduce_channels = nn.Sequential(
            nn.Conv2d(in_channels=768, out_channels=128, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, 
                      stride=1, padding=0),
            nn.ReLU(inplace=True)
        )

        # Linear layers


    def forward(self, x, train_encoder=True):
        x = x.to(torch.bfloat16)

        if train_encoder:
            x = self.transform(x)
            convnext_out = self.convnext(x)
        else:
            with torch.no_grad():
                x = self.transform(x)
                convnext_out = self.convnext(x)
        out = self.reduce_channels(convnext_out.to(torch.float32))
        return out.reshape(x.shape[0], 128, -1)


MODEL_SETTINGS = {
    "scratch": {
        "n_embd": 128,
        "num_tokens": 234,
        "model": CustomCNN,
    },
    "resnet": {
        "n_embd": 128,
        "num_tokens": 128,
        "model": ResnetEncoder,
    },
    "convnext": {
        "n_embd": 40,
        "num_tokens": 128,
        "model": ConvNextEncoder,
    },
    "resnet_gfm": {
        "n_embd": 128,
        "num_tokens": 128,
        "model": Resnet_GFMEncoder,
    },
    "resnet_depth": {
        "n_embd": 128,
        "num_tokens": 128,
        "model": ResnetDepthEncoder,
    },
}

class CrossOnlyAttention(nn.Module):
    def __init__(
        self,
        n_embd,               # embedding dimension
        n_head,               # number of attention heads
        attn_pdrop=0.1,       # dropout rate for attention
        resid_pdrop=0.1,      # dropout rate for feed-forward/ residual
        T1=234,               # number of tokens from image 1
        T2=234                # number of tokens from image 2
    ):
        super().__init__()

        self.n_embd = n_embd
        self.n_head = n_head
        self.T1 = T1
        self.T2 = T2

        # key, query, value projections
        self.c_attn = nn.Linear(n_embd, 3 * n_embd)
        self.c_proj = nn.Linear(n_embd, n_embd)

        # dropouts
        self.attn_dropout = nn.Dropout(attn_pdrop)
        self.resid_dropout = nn.Dropout(resid_pdrop)

        # Precompute the cross-only mask if T1, T2 are fixed
        mask_2d = self.create_cross_attention_mask(T1, T2)  # shape (T, T)
        # shape => (1, 1, T, T) so it broadcasts across (B, n_head, T, T).
        self.register_buffer("cross_mask", mask_2d.view(1, 1, *mask_2d.shape))

    def create_cross_attention_mask(self, T1, T2):
        """
        Returns a tensor of shape (T, T) with 1s for cross-token positions,
        and 0s for same-image positions.
        """
        T = T1 + T2
        img_mask = torch.zeros(T, T)  # shape [T, T]

        # Image 1 attends to Image 2
        img_mask[0:T1, T1:T] = 1
        # Image 2 attends to Image 1
        img_mask[T1:T, 0:T1] = 1

        mask = torch.ones(T+1, T+1)
        mask[1:T+1, 1:T+1] = img_mask


        return mask

    def forward(self, x):
        """
        x shape: (B, T, n_embd), where T = T1 + T2
        """
        B, T, C = x.size()
        head_size = C // self.n_head

        # Project to Q, K, V
        q, k, v = self.c_attn(x).split(self.n_embd, dim=2)

        # Reshape => (B, n_head, T, head_size)
        q = q.view(B, T, self.n_head, head_size).transpose(1, 2)
        k = k.view(B, T, self.n_head, head_size).transpose(1, 2)
        v = v.view(B, T, self.n_head, head_size).transpose(1, 2)

        # (B, n_head, T, head_size) x (B, n_head, head_size, T) => (B, n_head, T, T)
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(head_size))

        # Cross-only mask => disallow attending to same-image tokens
        # att = att.masked_fill(self.cross_mask == 0, float('-inf'))

        # Softmax and dropout
        att = F.softmax(att, dim=-1)
        att = self.attn_dropout(att)

        # Weighted sum over values => (B, n_head, T, head_size)
        y = att @ v

        # Reassemble => (B, T, C)
        y = y.transpose(1, 2).contiguous().view(B, T, C)

        # Output projection and dropout
        y = self.resid_dropout(self.c_proj(y))
        return y


class SquaredReLU(nn.Module):
    def forward(self, x):
        # ReLU(x) squared
        return F.relu(x).pow(2)


class KeypointModule(nn.Module):
    def __init__(self, device):
        super().__init__()
        self.device = device
        self.fc1 = nn.Linear(16384, 128)
        self.fc2 = nn.Linear(128, 2)

    def forward(self, x):
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        x = F.sigmoid(x)
        return x


class Block(nn.Module):
    def __init__(self, n_embd, n_head, n_tokens, attn_pdrop=0.1, resid_pdrop=0.1):
        super().__init__()
        self.ln1 = nn.LayerNorm(n_embd)
        self.attn = CrossOnlyAttention(
            n_embd, n_head, attn_pdrop,
            resid_pdrop, T1=n_tokens, T2=n_tokens
        )
        self.ln2 = nn.LayerNorm(n_embd)
        self.mlp = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd, bias=False),
            nn.GELU(),
            nn.Linear(4 * n_embd, n_embd, bias=False),
            nn.Dropout(resid_pdrop),
        )
    
    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x


class Transformer(nn.Module):
    def __init__(
        self, in_dim, out_dim, ctx_len, n_embd, n_head, num_layer, attn_pdrop=0.1, resid_pdrop=0.1
    ):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.ctx_len = ctx_len
        self.n_embd = n_embd
        self.n_head = n_head
        self.attn_pdrop = attn_pdrop
        self.resid_pdrop = resid_pdrop
        self.num_layer = num_layer

        self.input_layer = nn.Sequential(
            nn.Linear(in_dim, n_embd),
            nn.Dropout(resid_pdrop),
        )
        self.weight_pos_embed = nn.Embedding(ctx_len, n_embd)
        self.blocks = nn.Sequential(
            *[
                Block(
                    n_embd, n_head, ctx_len,
                    attn_pdrop, resid_pdrop
                )
                for _ in range(num_layer)
            ],
        )
        self.output_layer = nn.Sequential(
            nn.LayerNorm(n_embd),
            nn.Linear(n_embd, out_dim),
        )
        self.embd_token = nn.Parameter(torch.randn(1, 1, n_embd))

    def forward(self, x):
        # (B, T, in_dim)
        x = self.input_layer(x)
        # pose embeds for left and right images
        pos_embeds = self.weight_pos_embed(
            torch.arange(self.ctx_len, device=x.device)
        ).unsqueeze(0)
        x = x + pos_embeds
        x = torch.cat([self.embd_token.repeat(x.shape[0], 1, 1), x], dim=1)
        x = self.blocks(x)
        x = self.output_layer(x)
        return x


class MonoEncoder(nn.Module):
    def __init__(
        self, backbone, img_height, img_width, n_embd, n_head, attn_pdrop=0.1, resid_pdrop=0.1
    ):
        super().__init__()
        self.backbone = backbone
        self.cnn = MODEL_SETTINGS[backbone]["model"](img_height, img_width, "cuda", ckpt_path=CKPT_PATH)
        self.num_tokens = MODEL_SETTINGS[backbone]["num_tokens"]
        if n_embd is None:
            n_embd = MODEL_SETTINGS[backbone]["n_embd"]
        self.out_embd = n_embd # 8
        self.transformer = Transformer(n_embd, self.out_embd, self.num_tokens, n_embd, n_head, 2)
        self.n_embd = n_embd
        # self.keypoint_head = KeypointModule("cuda")

        self.out_layer = nn.Sequential(
            nn.Linear(self.out_embd, 128),
            nn.GELU(),
            nn.Linear(128, 32),
        ) # 288, 128, 13, 18

    def forward(self, x, finetune_backbone=True):
        batch_size = x.shape[0]
        x = self.cnn(x, train_encoder=finetune_backbone)
        if self.backbone == "convnext":
            x = x.reshape(1, batch_size, -1, self.n_embd)
            x = x.permute(1, 0, 2, 3)
            x = x.reshape(batch_size, -1, self.n_embd)
        else:
            x = x.view(1, batch_size, self.n_embd, -1)
            x = x.permute(1, 0, 2, 3) # B, 2, 128, -1
            x = x.reshape(batch_size, -1, self.n_embd)
        x = self.transformer(x)
        # x = self.out_layer(x.view(batch_size, -1))
        x = self.out_layer(x[:, 0, :])
        # return x, kpt_left, kpt_right
        return x


def main():
    # im_left = Image.open("left_img.png")
    # batch_size = 144

    # img_tensor_left = torch.tensor(np.array(im_left)).permute(2, 0, 1).unsqueeze(0) / 255.
    # imgs = img_tensor_left.repeat(batch_size, 1, 1, 1).to("cuda")
    # backbone = "convnext"
    # mono_encoder = MonoEncoder(
    #     backbone=backbone,
    #     img_height=240, img_width=320,
    #     n_embd=MODEL_SETTINGS[backbone]["n_embd"], n_head=4
    # ).to("cuda")
    # out = mono_encoder(imgs)
    # breakpoint()

    batch_size = 512
    

    device = torch.device("cuda:0")


    # imgs = torch.rand(batch_size, 3, 240, 320).to("cuda") * 255.0
    # mono_encoder = MonoEncoder(
    #     backbone="resnet",
    #     img_height=240,
    #     img_width=320,
    #     n_embd=None, n_head=4
    # ).to(device)

    imgs = torch.rand(batch_size, 240, 320, 1).to("cuda") * 10.0
    mono_encoder = MonoEncoder(
        backbone="resnet_gfm",
        img_height=240,
        img_width=320,
        n_embd=None, n_head=4
    ).to(device)

    # warmup runs (not timed) to stabilize kernels / caches
    for _ in range(10):
        _ = mono_encoder(imgs)
    torch.cuda.synchronize()

    n_iters = 100

    # timed runs
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(n_iters):
        _ = mono_encoder(imgs)
    torch.cuda.synchronize()
    t1 = time.perf_counter()

    total = t1 - t0
    avg_ms = total / n_iters * 1000.0
    print(f"time taken for {n_iters} forward passes: {total:.4f}s, avg {avg_ms:.2f} ms", flush=True)

if __name__ == "__main__":
    main()
