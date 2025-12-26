
import torch
import torch.nn as nn
from torchvision import models

# 1. SimpleCNN Model
class SimpleCNN(nn.Module):
    def __init__(self, num_classes):
        super(SimpleCNN, self).__init__()

        self.fc_dropout = nn.Dropout(0.5) # Drop 50% of neurons randomly
        # Convolutional Block 1
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, padding=1)
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # Convolutional Block 2
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, padding=1)
        self.cnn_dropout = nn.Dropout(0.25)
        
        # Convolutional Block 3
        self.conv3 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1)
        
        # Fully Connected Layers
        # Calculation: Image is 128x128. 
        # After 3 MaxPools (dividing by 2 three times), size is 128 / 2 / 2 / 2 = 16.
        # So feature map is 16x16 with 64 channels.
        self.fc1 = nn.Linear(64 * 16 * 16, 512)
        self.fc2 = nn.Linear(512, num_classes) # Output layer

    def forward(self, x):
        # Pass through Conv blocks
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        x = self.cnn_dropout(x)
        
        x = self.pool(self.relu(self.conv3(x)))
        
        # Flatten for Dense layers
        x = x.view(-1, 64 * 16 * 16) 
        
        # Pass through Dense layers
        x = self.relu(self.fc1(x))
        x = self.fc_dropout(x)
        x = self.fc2(x)
        return x

# 2. CNNLSTM Model
class CNNLSTM(nn.Module):
    def __init__(self, num_classes):
        super(CNNLSTM, self).__init__()
        
        # --- 1. CNN LAYERS (Feature Extractor) ---
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool2d(2, 2)
        self.cnn_dropout = nn.Dropout(0.25)
        
        
        # --- 2. LSTM LAYER ---
        # Calculation for LSTM Input:
        # Image input: 128x128
        # After 3 pools (div by 8): Feature map is 16x16
        # Channels: 64
        # We treat 'Height' (16) as the Sequence Length (Time steps)
        # We treat 'Width * Channels' (16 * 64 = 1024) as the Input Features per step
        
        self.lstm_input_size = 16 * 64 # 1024
        self.lstm_hidden_size = 256
        
        # batch_first=True means input format is (Batch, Seq, Feature)
        self.lstm = nn.LSTM(input_size=self.lstm_input_size, 
                            hidden_size=self.lstm_hidden_size,
                            # bidirectional=True,
                            num_layers=2,
                            dropout=0.4,
                            batch_first=True)
        
        # --- 3. CLASSIFIER ---
        self.fc = nn.Linear(self.lstm_hidden_size, num_classes)

    def forward(self, x):
        # 1. CNN Forward Pass
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        x = self.cnn_dropout(x)
        x = self.pool(self.relu(self.conv3(x)))
        
        # Current Shape: (Batch, 64, 16, 16) [C, H, W]
        
        # 2. Reshape for LSTM
        # We want (Batch, Sequence, Features) -> (Batch, 16, 1024)
        
        # Permute to put Height (Sequence) first: (Batch, Height, Width, Channels)
        x = x.permute(0, 2, 3, 1) 
        
        # Merge Width and Channels into one dimension
        # (Batch, 16, 16, 64) -> (Batch, 16, 1024)
        x = x.reshape(x.size(0), 16, -1)
        
        # 3. LSTM Forward Pass
        # out shape: (Batch, Seq_Len, Hidden_Size)
        # _ (hidden states): not needed here
        out, _ = self.lstm(x)
        
        # 4. Take the last time step
        # We only care about the LSTM's final conclusion after seeing all rows
        out = out[:, -1, :] 
        
        # 5. Classifier
        out = self.fc(out)
        return out

# 3. ViT Helper
def get_vit_model(num_classes):
    # Load architecture (weights not needed as we load our own state dict)
    # But usually it's safer to load default weights to initialize structure correctly if state dict doesn't cover everything
    # However, since we load a full state_dict, we can start with default or empty.
    # To be safe and match notebook:
    model = models.vit_b_16(weights=None) 
    
    # Replace head
    input_features = model.heads.head.in_features
    model.heads.head = nn.Linear(input_features, num_classes)
    
    return model
