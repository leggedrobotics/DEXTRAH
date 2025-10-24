import torch
import torch.nn as nn
from .resnet_bifpn import ResNetBiFPN
from torchvision import transforms


CKPT_PATH = "/media/sharedaccess/Manthan_RSL_SSD/clariden_models/vitl14_data_all_distill_multi_resnet18_34_50_625k_webd_bs_2048/resnet18_ema_checkpoint.pth"

class Resnet_GFMEncoder(nn.Module):
    def __init__(self, ckpt_path=None, backbone="resnet18", out_channels=128, freeze_backbone=True, out_feat_dim=128):
        super().__init__()

        self.backbone = backbone
        self.out_channels = out_channels
        self.freeze_backbone = freeze_backbone

        self.model = ResNetBiFPN(backbone_name=backbone, out_channels=out_channels)

        self.load_pretrained_weights(ckpt_path)

        # freeze backbone
        if freeze_backbone:
            self.model.eval()
            for param in self.model.backbone.parameters():
                param.requires_grad = False
            print("Backbone frozen")

        # Adjust this Convolutions according to the output feature map size you want
        # The out["dense_bifpn"]["P4"] feature map is at 1/16 resolution of input
        # So for input 224x224, it will be 14x14
        # The below compression will reduce it to 7x7 and then 4x4
        # These layers need to be trained !!

        # self.compression = nn.Sequential(
        #     nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=2, padding=1),  # 16 -> 8
        #     nn.ReLU(inplace=True),
        #     nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=2, padding=1),  # 8 -> 4
        #     nn.ReLU(inplace=True),
        # )

        self.normalize_depth = transforms.Normalize(mean=[0.248880, 0.495620, 0.492858],
                                                    std=[0.139357, 0.271314, 0.297177])
        
        embed_dim = self.model.embed_dim
        self.fc = nn.Linear(embed_dim, out_feat_dim)

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

    def forward(self, observations):
        depth = observations["depth"]  # [B, H, W, C]

        # Here the depth needs to be metric Depth and not scaled between 0-1

        # Need to convert to 3 channel
        depth = depth.permute(0, 3, 1, 2)  # B x C x H x W
        depth = self.convert_metric_to_three_channel_depth(depth)

        out = self.model(depth)
        # spatial_tokens = out["dense_bifpn"]["P4"]  # [B, out_channels, H//16, W//16]
        # output = self.compression(spatial_tokens)
        output = out["global_backbone"]

        output = self.fc(output)

        return output

if __name__ == "__main__":

    # Dummy forward pass 
    resnet_encoder = Resnet_GFMEncoder(ckpt_path=CKPT_PATH, backbone="resnet18", out_channels=128, freeze_backbone=True)

    observations = {}
    observations["depth"] =  torch.rand(8, 240, 320, 1) * 10.0  # Random depth between 0 and 10 meters
    # Move to GPU if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    resnet_encoder.to(device)
    observations["depth"] = observations["depth"].to(device)

    output = resnet_encoder(observations)
    print("Output shape:", output.shape)
