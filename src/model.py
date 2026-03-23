import torch
import torch.nn as nn
import timm


class BirdModel(nn.Module):
    def __init__(self, num_classes=234, model_name="efficientnet_b0", pretrained=True):
        super().__init__()

        self.backbone = timm.create_model(
            model_name,
            pretrained=pretrained,
            in_chans=1,       # grayscale spectrogram input
            num_classes=0,    # remove original classification head
        )

        # Get the output feature size of the backbone
        backbone_out = self.backbone.num_features

        # Custom head
        self.head = nn.Sequential(
            nn.Linear(backbone_out, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        features = self.backbone(x)
        output = self.head(features)
        return output
        # NOTE: No sigmoid here — we use BCEWithLogitsLoss during training
        # At inference time: probs = torch.sigmoid(output)


if __name__ == "__main__":
    # Quick test — run with: python src/model.py
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    model = BirdModel(num_classes=234, model_name="efficientnet_b0")
    model = model.to(device)

    # Simulate a batch of 4 spectrograms
    dummy_input = torch.randn(4, 1, 128, 500).to(device)
    output = model(dummy_input)

    print(f"Input shape:  {dummy_input.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Expected:     torch.Size([4, 234])")

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total params:     {total_params:,}")
    print(f"Trainable params: {trainable_params:,}")
    print("model.py OK!")