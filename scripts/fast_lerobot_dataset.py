"""Cache small numeric action columns; retain LeRobot image decoding and boundaries."""
import numpy as np
import torch
from lerobot.datasets.lerobot_dataset import LeRobotDataset

class CachedActionDataset(LeRobotDataset):
    include_task_time = False
    action_prior = None

    def __getitem__(self, index):
        sample = super().__getitem__(index)
        if self.action_prior is not None:
            frame = round(float(sample['timestamp'])*50)
            action = sample['action']
            indices = torch.arange(frame, frame + (len(action) if action.ndim == 2 else 1)).clamp(max=len(self.action_prior)-1)
            prior = self.action_prior[indices]
            sample['action'] = action - (prior if action.ndim == 2 else prior[0])
        if self.include_task_time:
            # Observable episode clock, never an expert stage or object pose.
            clock = torch.as_tensor(sample['timestamp']).float().reshape(1) / 60.
            sample['observation.state'] = torch.cat((sample['observation.state'], clock))
        return sample

    def _query_hf_dataset(self, query_indices):
        if not hasattr(self,'_action_cache'):
            column = self.hf_dataset.select_columns(['action']).with_format('numpy')[:]['action']
            self._action_cache = torch.from_numpy(np.asarray(column,dtype=np.float32).copy())
        result = {}
        remaining = {}
        for key, indices in query_indices.items():
            if key != 'action':
                remaining[key] = indices
                continue
            if self._absolute_to_relative_idx is not None:
                indices = [self._absolute_to_relative_idx[i] for i in indices]
            result[key] = self._action_cache[indices]
        if remaining: result.update(super()._query_hf_dataset(remaining))
        return result
