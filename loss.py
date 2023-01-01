import torch
import torch.nn as nn
from utils import intersection_over_union

class YoloLoss(nn.Module):
    
    def __init__(self, S=7, B=2, C=20):
        """_summary_

        Args:
            S (int, optional): number of grid_cell. Defaults to 7.
            B (int, optional): number of boxes. Defaults to 2.
            C (int, optional): number of classes . Defaults to 20.
        """
        super(YoloLoss, self).__init__()
        self.mse = nn.MSELoss(reduction="sum")
        
        self.S = S
        self.B = B
        self.C = C
        
        self.lambda_noobj = 0.5
        self.lambda_coord = 5
        
    def forward(self, predictions, target):
        
        # target
        # class_probability, objectness, bbox_coord

        batch_size = target.size(0)
        
        # predictions are shaped (BATCH_SIZE, S*S(C+B*5) when inputted
        predictions = predictions.reshape(-1, self.S, self.S, self.C + self.B * 5)
        
        iou_b1 = intersection_over_union(predictions[..., 21:25], target[..., 21:25])
        iou_b2 = intersection_over_union(predictions[..., 26:30], target[..., 21:25])
        ious = torch.cat([iou_b1.unsqueeze(0), iou_b2.unsqueeze(0)], dim=0)
        
        # Take the box with highest IoU out of the two prediction
        iou_maxes, bestbox = torch.max(ious, dim=0)
        # this is Iobj_i
        exists_box = target[..., 20].unsqueeze(3)
        
        # 1. box coordinate loss
        box_predictions = exists_box * (
            (1 - bestbox) * predictions[..., 21:25] 
            + bestbox * predictions[..., 26:30] 
        )
        
        box_targets = exists_box * target[..., 21:25]
        
        ## Take sqrt of width, height
        #  When a negative number enters sqrt, nan, and therefore applies after abs. And add 1e-6 so that it does not fall into negative
        box_predictions[..., 2:4] = torch.sign(box_predictions[..., 2:4]) * torch.sqrt(torch.abs(box_predictions[..., 2:4] + 1e-6))
        
        box_targets[..., 2:4] = torch.sqrt(box_targets[..., 2:4])
        
        # (N, S, S, 4) -> (N*S*S, 4)
        box_loss = self.mse(torch.flatten(box_predictions, end_dim=-2),
                            torch.flatten(box_targets, end_dim=-2))
        
        # 2. obj loss
        pred_box = exists_box * (
            (1 - bestbox) * predictions[..., 20:21] 
            + bestbox * predictions[..., 25:26] 
        )
        
        # It's the last one, so I'll skip the end_dim
        # object_loss = self.mse(
        #     torch.flatten(pred_box),
        #     torch.flatten(exists_box),
        # )
        
        object_loss = self.mse(
            torch.flatten(exists_box * pred_box),
            torch.flatten(exists_box * target[..., 20:21] * iou_maxes),
        )
        
        no_object_loss = self.mse(
            torch.flatten((1-exists_box) * predictions[..., 20:21], start_dim=1),
            torch.flatten((1-exists_box) * target[..., 20:21], start_dim=1),
        )
        
        no_object_loss += self.mse(
            torch.flatten((1-exists_box) * predictions[..., 25:26], start_dim=1),
            torch.flatten((1-exists_box) * target[..., 20:21], start_dim=1),
        )
        
        # 3. cls loss
        cls_loss = self.mse(
            torch.flatten(exists_box * predictions[..., :20], end_dim=-2),
            torch.flatten(exists_box * target[..., :20], end_dim=-2),
        )
        
        loss = (
            self.lambda_coord * box_loss
            + object_loss
            + self.lambda_noobj * no_object_loss
            + cls_loss
        )
        
        return loss / batch_size