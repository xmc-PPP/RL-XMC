"""训练脚本"""
import sys
import warnings
warnings.filterwarnings('ignore')

sys.path.insert(0, '.')
from main import train

print("Starting training...")
train(max_episodes=50)
print("Training completed!")
