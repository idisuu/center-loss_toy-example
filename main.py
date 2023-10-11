import torch.optim as optim

from model.lenets import LeNetsPlus
from utils.datasets import load_mnist_datasets
from utils.trainer import Trainer

import matplotlib.pyplot as plt

# Load model
model = LeNetsPlus()

# Load datasets
train_dataset, test_dataset = load_mnist_datasets()

# Hyper parameter setting
epochs = 3
batch_size = 256
lr = 0.001
optimizer = optim.SGD
save_path = None # './checkpoint/test.pth'
center_loss_config = {'alpha': 1, 'lambda': 1, 'warmup_steps': 0}
use_softmax_initially =  0
manually_update_center=False

trainer = Trainer(
    epochs = epochs,
    batch_size=batch_size,
    lr= lr,
    model = model,
    train_dataset=train_dataset,
    test_dataset=test_dataset,
    optimizer=optimizer,
    save_path=save_path,
    center_loss_config=center_loss_config,    
    use_softmax_initially=use_softmax_initially,
    manually_update_center=manually_update_center,
)

# train
result = trainer.train()


softmax_loss_list = []
center_loss_list = []
total_loss_list = []

train_acc_list = []
test_acc_list = []

for item in result:
    if item.get('softmax_loss'):
        softmax_loss_list.append(item['softmax_loss'])
    if item.get('center_loss'):
        center_loss_list.append(item['center_loss'])
    if item.get('total_loss'):
        total_loss_list.append(item['total_loss'])

    train_acc_list.append(item['train_acc'])
    test_acc_list.append(item['test_acc'])    

fig, axes = plt.subplots(1, 5, figsize=(15, 5))

# Softmax Loss
axes[0].plot(softmax_loss_list)
axes[0].set_title('Softmax Loss')
axes[0].grid(True)

# Center Loss
axes[1].plot(center_loss_list)
axes[1].set_title('Center Loss')
axes[1].grid(True)

# Total Loss
axes[2].plot(total_loss_list)
axes[2].set_title('Total Loss')
axes[2].grid(True)

# Train Acc
axes[3].plot(train_acc_list)
axes[3].set_title('Train Acc')
axes[3].grid(True)

# Total Loss
axes[4].plot(test_acc_list)
axes[4].set_title('Test Acc')
axes[4].grid(True)
plt.show()

# Visualize distribution
trainer.validate(mode='train', visualize=True)
trainer.validate(mode='test', visualize=True)
