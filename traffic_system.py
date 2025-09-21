# traffic_system.py
from typing import Dict, Any, Optional
import os
import torch
import torch.nn as nn
import torch.nn.functional as F


# 1) Multi-class vehicle classifier (car, truck, bus, motorcycle, ambulance, fire_truck)
class TrafficCNN(nn.Module):
    def __init__(self, num_classes: int = 6):
        super(TrafficCNN, self).__init__()

        self.conv_layers = nn.Sequential(
            # Block 1
            nn.Conv2d(3, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),

            # Block 2
            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),

            # Block 3
            nn.Conv2d(128, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
        )

        # Use adaptive pooling to be size-agnostic
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d((7, 7)),
            nn.Flatten(),
            nn.Linear(256 * 7 * 7, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes)  # logits
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feats = self.conv_layers(x)
        logits = self.classifier(feats)
        return logits


# 2) Multi-task CNN (vehicle presence map, emergency vs normal, density map)
class TrafficManagementCNN(nn.Module):
    def __init__(self):
        super(TrafficManagementCNN, self).__init__()

        self.backbone = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            nn.Conv2d(64, 128, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
        )

        # Heads
        self.vehicle_detector = nn.Conv2d(128, 1, kernel_size=1)  # presence (logits)
        self.emergency_classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(128, 2)  # emergency vs normal (logits)
        )
        self.density_estimator = nn.Conv2d(128, 1, kernel_size=1)  # density (logits or raw)

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        features = self.backbone(x)
        vehicle_map_logits = self.vehicle_detector(features)
        emergency_logits = self.emergency_classifier(features)
        density_map = self.density_estimator(features)
        return {
            "vehicles": torch.sigmoid(vehicle_map_logits),  # [B,1,H',W'] in [0,1]
            "emergency": emergency_logits,                  # [B,2] (logits)
            "density": F.relu(density_map)                  # non-negative density
        }


# 3) Emergency Vehicle Detection CNN (ambulance, fire_truck, police)
class EmergencyVehicleCNN(nn.Module):
    def __init__(self, num_classes: int = 3):
        super(EmergencyVehicleCNN, self).__init__()

        self.feature_extractor = nn.Sequential(
            # Larger receptive field for light patterns/text
            nn.Conv2d(3, 16, 7, padding=3),
            nn.ReLU(inplace=True),

            nn.Conv2d(16, 32, 5, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            nn.Conv2d(64, 128, 3, padding=1),
            nn.ReLU(inplace=True),
        )

        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d((4, 4)),
            nn.Flatten(),                 # -> 128*4*4 = 2048
            nn.Linear(128 * 4 * 4, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes)   # logits: [ambulance, fire_truck, police]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feats = self.feature_extractor(x)
        logits = self.classifier(feats)
        return logits


# 4) Lane-wise Vehicle Counting via attention-weighted density
class LaneCountingCNN(nn.Module):
    def __init__(self, num_lanes: int = 4):
        super(LaneCountingCNN, self).__init__()
        self.num_lanes = num_lanes

        # Soft lane attention masks
        self.lane_attention = nn.Sequential(
            nn.Conv2d(3, 64, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, num_lanes, 1),  # one channel per lane
        )

        # Density predictor
        self.counter = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 1, 1),
            nn.ReLU(inplace=True),  # ensure non-negative
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # lane masks normalized across lanes
        lane_logits = self.lane_attention(x)           # [B,L,H,W]
        lane_masks = F.softmax(lane_logits, dim=1)     # sum over L == 1 per pixel

        density_map = self.counter(x)                  # [B,1,H,W]

        # Per-lane counts by integrating density weighted by that lane's mask
        per_lane_counts = []
        for i in range(lane_masks.size(1)):
            lane_density = density_map * lane_masks[:, i:i+1]  # [B,1,H,W]
            count = torch.sum(lane_density, dim=(2, 3))        # [B,1] -> integrate
            per_lane_counts.append(count)

        # [B, L, 1] -> [B, L]
        counts = torch.cat(per_lane_counts, dim=1)
        return counts


# 5) Processing Pipeline
class CNNTrafficSystem:
    def __init__(self,
                 num_lanes: int = 4,
                 emergency_threshold: float = 0.8,
                 max_green_seconds: int = 75,
                 device: Optional[str] = None,
                 # optional weight paths
                 vehicle_ckpt: str = "traffic_cnn.pth",
                 emergency_ckpt: str = "emergency_cnn.pth",
                 lane_ckpt: str = "lane_counting_cnn.pth"):
        self.device = torch.device(device) if device else torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.vehicle_detector = TrafficCNN().to(self.device)
        self.emergency_detector = EmergencyVehicleCNN().to(self.device)
        self.lane_counter = LaneCountingCNN(num_lanes=num_lanes).to(self.device)

        # Simple per-lane timer (mocked here; in practice, drive it from your controller clock)
        self.lane_timers = torch.zeros(num_lanes, dtype=torch.int32, device=self.device)

        self.emergency_threshold = emergency_threshold
        self.max_green_seconds = max_green_seconds

        self.load_models(vehicle_ckpt, emergency_ckpt, lane_ckpt)

    def _try_load(self, model: nn.Module, path: str):
        if path and os.path.isfile(path):
            state = torch.load(path, map_location=self.device)
            model.load_state_dict(state)
            model.eval()

    def load_models(self, vehicle_ckpt: str, emergency_ckpt: str, lane_ckpt: str):
        """
        Loads pretrained weights if the files exist (no error if they don't).
        """
        self._try_load(self.vehicle_detector, vehicle_ckpt)
        self._try_load(self.emergency_detector, emergency_ckpt)
        self._try_load(self.lane_counter, lane_ckpt)

    @torch.no_grad()
    def process_intersection(self, frame: torch.Tensor) -> Dict[str, Any]:
        """
        frame: [B,3,H,W] float tensor (0..1 recommended)
        Returns dict with raw outputs and a control decision string.
        """
        frame = frame.to(self.device)

        # 1) Detect all vehicles (multi-class logits per image)
        vehicle_logits = self.vehicle_detector(frame)          # [B,6]
        vehicle_probs = F.softmax(vehicle_logits, dim=1)       # [B,6]

        # 2) Check for emergency vehicles (3-way logits)
        emergency_logits = self.emergency_detector(frame)      # [B,3]
        emergency_probs = F.softmax(emergency_logits, dim=1)   # [B,3]
        # Max emergency confidence across classes
        max_emerg_conf, emerg_pred = torch.max(emergency_probs, dim=1)  # [B]

        # 3) Count vehicles per lane
        lane_counts = self.lane_counter(frame)                 # [B,L]

        # 4) Decide traffic control per sample (assume batch size 1 here)
        decision = self.traffic_controller(max_emerg_conf, lane_counts[0])

        return {
            "vehicle_probs": vehicle_probs,               # [B,6]
            "emergency_probs": emergency_probs,           # [B,3]
            "lane_counts": lane_counts,                   # [B,L]
            "decision": decision
        }

    def traffic_controller(self, max_emergency_conf: torch.Tensor, counts_1b: torch.Tensor) -> str:
        """
        max_emergency_conf: scalar tensor (confidence for the most likely emergency class) for a single frame
        counts_1b: [L] per-lane counts for a single frame
        """
        # Emergency override
        if max_emergency_conf.item() > self.emergency_threshold:
            return "EMERGENCY_OVERRIDE"

        # Normal logic: pick the lane with the largest queue and check its timer
        lane_idx = int(torch.argmax(counts_1b).item())

        # Increment that lane's timer; decay others (toy logic)
        self.lane_timers[lane_idx] += 1
        for i in range(self.lane_timers.numel()):
            if i != lane_idx and self.lane_timers[i] > 0:
                self.lane_timers[i] -= 1

        if int(self.lane_timers[lane_idx].item()) >= self.max_green_seconds:
            self.lane_timers[lane_idx] = 0
            return f"SWITCH_TO_LANE_{lane_idx}"

        return "CONTINUE_CURRENT"


# --------- Smoke test (run this file directly) ----------
if __name__ == "__main__":
    torch.manual_seed(0)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    system = CNNTrafficSystem(num_lanes=4, device=device)

    # Dummy input: batch size 1, RGB, 224x224 (any size works due to adaptive pooling)
    dummy_frame = torch.rand(1, 3, 224, 224)

    out = system.process_intersection(dummy_frame)
    print("Vehicle probs shape:", out["vehicle_probs"].shape)    # [1,6]
    print("Emergency probs shape:", out["emergency_probs"].shape)  # [1,3]
    print("Lane counts shape:", out["lane_counts"].shape)        # [1,4]
    print("Decision:", out["decision"])