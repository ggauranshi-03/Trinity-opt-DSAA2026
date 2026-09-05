import torch
import torchvision
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import numpy as np

class LongTailedCIFAR10(datasets.CIFAR10):
    def __init__(self, imb_factor=0.01, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.imb_factor = imb_factor
        if self.train:
            self.img_num_list = self._get_img_num_per_cls(self.targets, imb_factor)
            self._gen_imbalanced_data(self.img_num_list)
        else:
            self.img_num_list = None

    def _get_img_num_per_cls(self, targets, imb_factor):
        num_cls = len(set(targets))
        img_max = len(targets) / num_cls
        img_num_per_cls = []
        for cls_idx in range(num_cls):
            num = img_max * (imb_factor**(cls_idx / (num_cls - 1.0)))
            img_num_per_cls.append(int(num))
        return img_num_per_cls

    def _gen_imbalanced_data(self, img_num_per_cls):
        new_data, new_targets = [], []
        targets_np = np.array(self.targets, dtype=np.int64)
        classes = np.unique(targets_np)
        
        for the_class, the_img_num in zip(classes, img_num_per_cls):
            idx = np.where(targets_np == the_class)[0]
            np.random.shuffle(idx)
            selec_idx = idx[:the_img_num]
            new_data.append(self.data[selec_idx, ...])
            new_targets.extend([the_class, ] * the_img_num)
            
        self.data = np.vstack(new_data)
        self.targets = new_targets
        print(f"[*] Generated Long-Tailed CIFAR-10 (Imbalance: {self.imb_factor}). Total train samples: {len(self.targets)}")

class LongTailedCIFAR100(datasets.CIFAR100):
    def __init__(self, imb_factor=0.01, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.imb_factor = imb_factor
        if self.train:
            self.img_num_list = self._get_img_num_per_cls(self.targets, imb_factor)
            self._gen_imbalanced_data(self.img_num_list)
        else:
            self.img_num_list = None

    def _get_img_num_per_cls(self, targets, imb_factor):
        num_cls = len(set(targets))
        img_max = len(targets) / num_cls
        img_num_per_cls = []
        for cls_idx in range(num_cls):
            num = img_max * (imb_factor**(cls_idx / (num_cls - 1.0)))
            img_num_per_cls.append(int(num))
        return img_num_per_cls

    def _gen_imbalanced_data(self, img_num_per_cls):
        new_data, new_targets = [], []
        targets_np = np.array(self.targets, dtype=np.int64)
        classes = np.unique(targets_np)
        
        for the_class, the_img_num in zip(classes, img_num_per_cls):
            idx = np.where(targets_np == the_class)[0]
            np.random.shuffle(idx)
            selec_idx = idx[:the_img_num]
            new_data.append(self.data[selec_idx, ...])
            new_targets.extend([the_class, ] * the_img_num)
            
        self.data = np.vstack(new_data)
        self.targets = new_targets
        print(f"[*] Generated Long-Tailed CIFAR-100 (Imbalance: {self.imb_factor}). Total train samples: {len(self.targets)}")

class PerClassEvaluator:
    def __init__(self, num_classes, img_num_per_cls=None):
        self.num_classes = num_classes
        self.img_num_per_cls = img_num_per_cls # Number of training samples per class
        self.reset()
        
    def reset(self):
        self.correct_per_class = np.zeros(self.num_classes)
        self.total_per_class = np.zeros(self.num_classes)
        
    def update(self, preds, targets):
        """
        preds: (B,) tensor of predicted class indices
        targets: (B,) tensor of true class indices
        """
        preds = preds.cpu().numpy()
        targets = targets.cpu().numpy()
        for p, t in zip(preds, targets):
            if p == t:
                self.correct_per_class[t] += 1
            self.total_per_class[t] += 1
            
    def compute(self):
        # Prevent division by zero
        acc_per_class = np.divide(self.correct_per_class, self.total_per_class, 
                                  out=np.zeros_like(self.correct_per_class), where=self.total_per_class!=0)
        
        results = {
            'per_class_acc': acc_per_class.tolist(),
            'overall_acc': 100.0 * np.sum(self.correct_per_class) / max(1, np.sum(self.total_per_class))
        }
        
        if self.img_num_per_cls is not None:
            # Calculate Head, Medium, Tail accuracies based on training sample frequencies
            many_shot_classes = []
            medium_shot_classes = []
            few_shot_classes = []
            
            for i, num in enumerate(self.img_num_per_cls):
                if num > 100:
                    many_shot_classes.append(i)
                elif 20 <= num <= 100:
                    medium_shot_classes.append(i)
                else:
                    few_shot_classes.append(i)
            
            def get_group_acc(classes):
                if len(classes) == 0:
                    return 0.0
                total = np.sum(self.total_per_class[classes])
                correct = np.sum(self.correct_per_class[classes])
                return 100.0 * correct / max(1, total)

            results['many_shot_acc'] = get_group_acc(many_shot_classes)
            results['medium_shot_acc'] = get_group_acc(medium_shot_classes)
            results['few_shot_acc'] = get_group_acc(few_shot_classes)
            
        return results

def get_dataloaders(dataset_name='cifar10', imb_factor=0.01, batch_size=128, num_workers=2):
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    
    if dataset_name.lower() == 'cifar10':
        trainset = LongTailedCIFAR10(imb_factor=imb_factor, root='./data', train=True, download=True, transform=transform_train)
        testset = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)
        num_classes = 10
    elif dataset_name.lower() == 'cifar100':
        trainset = LongTailedCIFAR100(imb_factor=imb_factor, root='./data', train=True, download=True, transform=transform_train)
        testset = datasets.CIFAR100(root='./data', train=False, download=True, transform=transform_test)
        num_classes = 100
    else:
        raise ValueError("Unsupported dataset")
        
    trainloader = DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)
    testloader = DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
    
    return trainloader, testloader, trainset.img_num_list, num_classes
