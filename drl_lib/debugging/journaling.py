import torch.nn as nn
import os 
import datetime
class Journal:
     # ANSI color codes
    COLORS = {
        'RED': '\033[91m',
        'YELLOW': '\033[93m',
        'GREEN': '\033[92m',
        'BLUE': '\033[94m',
        'RESET': '\033[0m',
        'BOLD': '\033[1m',
    }
    def __init__(self, directory: str, action_bounds: tuple, experiment_name: str):
        self.entries = []
        self.directory = directory
        file_dir = os.path.join(os.getcwd(), directory)
        
        # Create directory if it doesn't exist
        if not os.path.exists(file_dir):
            os.makedirs(file_dir, exist_ok=True)
            print(f"{self.COLORS['GREEN']}Directory created successfully{self.COLORS['RESET']}")
            
        self.file_dir = file_dir

        current_time = datetime.datetime.now()
        formatted_time = current_time.strftime('%Y-%m-%d_%H-%M-%S')
        self.save_path = os.path.join(file_dir, experiment_name + ".txt")
        self.box_width = 70
        self.separator = "=" * self.box_width

        self.action_bound_lower = action_bounds[0]
        self.action_bound_upper = action_bounds[1]
        
        # Gradient monitoring thresholds
        self.gradient_threshold_explode = 1e6  # Threshold for exploding gradients
        self.gradient_threshold_vanish = 1e-6  # Threshold for vanishing gradients

    def format_line(self, content: str) -> str:
        return f"|{content.ljust(self.box_width - 4)}|"

    
    def _actor_debug(self, sample_action):

        bound_check = (sample_action > self.action_bound_lower) or (sample_action < self.action_bound_upper) 

        if bound_check:
            self.entries.append(f"Action out of bounds: {sample_action}")

    def log(self, message: str, level: str = "INFO"):
        if level == "WARNING":
            print(f"{self.COLORS['YELLOW']}{self.format_line(message)}{self.COLORS['RESET']}")
        elif level == "ERROR":
            print(f"{self.COLORS['RED']}{self.format_line(message)}{self.COLORS['RESET']}")
        else:
            print(f"{self.format_line(message)}")

    def _check_gradient(self, model: nn.Module):

        total_norm = 0
        param_count = 0

        # Iterate through all named parameters (weights & biases) in the neural network model
        # name: str - parameter name (e.g. "layer1.weight")
        # param: Parameter - the actual parameter tensor
        for name, param in model.named_parameters():
            if param.grad is not None:
                param_norm = param.grad.data.norm(2).item()
                total_norm += param_norm ** 2
                param_count += 1

        total_norm = total_norm ** (1/2)

        # Exploding gradient check
        if total_norm > self.gradient_threshold_explode:
            self.log(f"Gradient norm too high ({total_norm:.4e}). Potential exploding gradients.", level="WARNING")
        
        # Vanishing gradient check
        if total_norm < self.gradient_threshold_vanish:
            self.log(f"Gradient norm very low ({total_norm:.4e}). Potential vanishing gradients.", level="WARNING")


        

        



