"""Neural network models and policies."""

import torch
import torch.nn as nn
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
try:
    import gymnasium as gym
    from gymnasium import spaces
except ImportError:
    import gym
    from gym import spaces


def create_custom_cnn_policy():
    """Create a custom CNN policy for visual observations."""
    class CustomCNN(BaseFeaturesExtractor):
        def __init__(self, observation_space: spaces.Box, features_dim: int = 256):
            super().__init__(observation_space, features_dim)
            
            # Assuming 160x144 Game Boy screen, grayscale
            n_input_channels = observation_space.shape[0]
            
            self.cnn = nn.Sequential(
                nn.Conv2d(n_input_channels, 32, kernel_size=8, stride=4, padding=0),
                nn.ReLU(),
                nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=0),
                nn.ReLU(),
                nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=0),
                nn.ReLU(),
                nn.Flatten(),
            )
            
            # Compute shape by doing one forward pass
            with torch.no_grad():
                n_flatten = self.cnn(
                    torch.as_tensor(observation_space.sample()[None]).float()
                ).shape[1]
            
            self.linear = nn.Sequential(
                nn.Linear(n_flatten, features_dim),
                nn.ReLU()
            )
        
        def forward(self, observations: torch.Tensor) -> torch.Tensor:
            return self.linear(self.cnn(observations))
    
    return CustomCNN
