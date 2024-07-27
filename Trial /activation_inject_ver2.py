import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
import torch.utils.data
import random
from collections import namedtuple
import time
import numpy as np

random.seed(time.time())
QTensor = namedtuple("QTensor", ["tensor", "scale", "zero_point"])

class LeNet5(nn.Module):
    def __init__(self, num_classes=10):
        super(LeNet5, self).__init__()
        self.conv1 = nn.Conv2d(1, 6, kernel_size=5, stride=1, padding=0)
        self.relu1 = nn.ReLU()
        self.mp1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(6, 16, kernel_size=5, stride=1, padding=0)
        self.relu2 = nn.ReLU()
        self.mp2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.fc = nn.Linear(400, 120)
        self.relu3 = nn.ReLU()
        self.fc1 = nn.Linear(120, 84)
        self.relu4 = nn.ReLU()
        self.fc2 = nn.Linear(84, num_classes)
        self.activations = {}  # Khởi tạo dictionary để lưu activations
        
    def quantize_and_inject_faults(self, x, num_faults):
        scale, zero_point = self.compute_scale_zero_point(x.min(), x.max(), num_bits=8)
        q_x = zero_point + x / scale
        q_x.clamp_(0, 255).round_()
        q_x = q_x.byte()
        
        # Tiêm lỗi
        #print("Before injecting faults:")
        #print(q_x.flatten()) 
        # Tiêm lỗi
        faulted_q_x = self.inject_tensor_faults(q_x, num_faults)
        #print("After injecting faults:")
        #print(faulted_q_x.flatten())  
        deq_x = scale * (faulted_q_x.float() - zero_point)
        
        return deq_x

    def compute_scale_zero_point(self, min_val, max_val, num_bits=8):
        qmin, qmax = 0, 2**num_bits - 1
        scale = (max_val - min_val) / (qmax - qmin)
        zero_point = qmin - min_val / scale
        zero_point = max(qmin, min(qmax, zero_point))
        return scale, zero_point

    def inject_tensor_faults_test(self, tensor, num_faults):
        num_tensor = tensor.numel()
        tensor_flat = tensor.flatten().tolist()
        indices = list(range(num_tensor))
        selected_indices = []  # Lưu các chỉ số được chọn để tiêm lỗi
        
        while num_faults > 0 and indices:
            i = random.choice(indices)
            num_faults_in_tensor = random.randint(1, min(8, num_faults))
            tensor_bits = tensor_flat[i]
            bit_positions = random.sample(range(8), num_faults_in_tensor)
            
            # Lưu chỉ số và giá trị trước khi tiêm lỗi
            selected_indices.append((i, tensor_flat[i]))

            for pos in bit_positions:
                tensor_bits ^= (1 << pos)
            tensor_flat[i] = tensor_bits
            num_faults -= num_faults_in_tensor
            indices.remove(i)

        tensor = torch.tensor(tensor_flat, dtype=torch.uint8).view(tensor.shape)

        # In các giá trị được chọn trước và sau khi tiêm lỗi
        print("Selected indices and values before injecting faults:")
        for idx, val in selected_indices:
            print(f"Index {idx}: Before {val} - After {tensor.flatten()[idx]}")

        return tensor

    def inject_tensor_faults(self, tensor, num_faults):
        num_tensor = tensor.numel()
        tensor_flat = tensor.flatten().tolist()
        indices = list(range(num_tensor))
        while num_faults > 0 and indices:
            if not indices:
                break  
            i = random.choice(indices)
            num_faults_in_tensor = random.randint(1, min(8, num_faults))
            tensor_bits = tensor_flat[i]
            bit_positions = random.sample(range(8), num_faults_in_tensor)
            for pos in bit_positions:
                #tensor_bits ^= (1 << pos)
                tensor_bits = 0
            tensor_flat[i] = tensor_bits
            num_faults -= num_faults_in_tensor
            indices.remove(i)
        tensor = torch.tensor(tensor_flat, dtype=torch.uint8).view(tensor.shape)
        return tensor

    def calculate_total_faults(self, total_activation, fault_rate):
        total_bits = total_activation * 8  # Mỗi activation là 8 bits
        total_faults = int(total_bits * fault_rate)
        return total_faults        

    def distribute_faults_randomly(self, total_faults, num_parts=4):
        faults = np.random.multinomial(total_faults, np.ones(num_parts) / num_parts)
        return faults.tolist()    



    def forward(self, x, total_activation_count):
        total_faults = self.calculate_total_faults(total_activation_count, fault_rate=0.1)
        fault_per_layer = self.distribute_faults_randomly(total_faults, num_parts=4)

        x = self.conv1(x)
        x = self.quantize_and_inject_faults(x, 0)
        x = self.relu1(x)
        #print('Tensor after relu1:')
        #print(x.detach().clone())
        self.activations['relu1'] = x = self.quantize_and_inject_faults(x, fault_per_layer[0])
        #print('Tensor after injecting faults in relu1:')
        #print(x.detach().clone()) 
        x = self.mp1(x)
        x = self.quantize_and_inject_faults(x, 0)
        
        x = self.conv2(x)
        x = self.quantize_and_inject_faults(x, 0)
        x = self.relu2(x)
        self.activations['relu2'] = x = self.quantize_and_inject_faults(x, fault_per_layer[1])
        x = self.mp2(x)
        x = self.quantize_and_inject_faults(x, 0)

        x = x.view(x.size(0), -1)
        x = self.fc(x)
        x = self.quantize_and_inject_faults(x, 0)
        x = self.relu3(x)
        self.activations['relu3'] = x = self.quantize_and_inject_faults(x, fault_per_layer[2])

        x = self.fc1(x)
        x = self.quantize_and_inject_faults(x, 0)
        x = self.relu4(x)
        self.activations['relu4'] = x = self.quantize_and_inject_faults(x, fault_per_layer[3])

        x = self.fc2(x)
        x = self.quantize_and_inject_faults(x, 0)
        output = F.log_softmax(x, dim=1)
        return output


    

# Load the dataset and models
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
kwargs = {"num_workers": 1, "pin_memory": True}
test_loader = torch.utils.data.DataLoader(
    datasets.MNIST(
        "./data",
        train=False,
        transform=transforms.Compose([
            transforms.Resize((32, 32)),
            transforms.ToTensor(),
            transforms.Normalize((0.1325,), (0.3105,))
        ]),
    ),
    batch_size=1,
    shuffle=True,
    **kwargs
)

model_3 = LeNet5().to(device)
model_3.load_state_dict(torch.load('./output/lenet5_weights.pth', map_location=device))

# Lấy một mẫu từ test_loader
#data, target = next(iter(test_loader))
#data, target = data.to(device), target.to(device)

# Chạy mẫu qua mô hình
#output = model_3(data, total_activation_count=6508)

# Tính toán dự đoán
#pred = output.argmax(dim=1, keepdim=True)

# Kiểm tra dự đoán có đúng hay không
#is_correct = pred.eq(target.view_as(pred)).item()

# In kết quả
#print(f'The prediction is {"correct" if is_correct else "incorrect"}.')

# Evaluate the model
correct = 0
total = 0
for data, target in test_loader:
#for data, target in test_loader: 
    data, target = data.to(device), target.to(device)
    output = model_3(data, total_activation_count=6508)
    pred = output.argmax(dim=1, keepdim=True)
    correct += pred.eq(target.view_as(pred)).sum().item()
    total += target.size(0)

#print(f'correct: {correct}')
accuracy = correct / total
print(f'Accuracy: {accuracy * 100:.2f}%')
