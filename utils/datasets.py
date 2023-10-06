from torchvision import datasets, transforms

def load_mnist_datasets(dataset_path="./data"):
    # 데이터 전처리 설정
    train_transform = transforms.Compose([transforms.RandomHorizontalFlip(),
                                          transforms.ToTensor(),
                                          transforms.Normalize((0.5,), (0.5,))
                                         ])
    
    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    
    # MNIST 데이터셋 로드
    train_dataset = datasets.MNIST('./data', download=True, train=True, transform=train_transform)
    test_dataset = datasets.MNIST('./data', download=True, train=False, transform=test_transform)
    return train_dataset, test_dataset
