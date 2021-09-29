import os
import torch
from torch import nn, optim
import torch.nn.functional as F
from torchvision import datasets, transforms
from torchvision.transforms.transforms import ToTensor
from torchsummary import summary
from tqdm import tqdm

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CLASSES = ('cats', 'dogs')
ROOT_DIR = os.path.abspath(".")
MODEL_PATH = os.path.join(ROOT_DIR, "torch_model")
BATCH_SIZE = 32
EPOCHS = 1


class PetClassifier(nn.Module):

    def __init__(self, params={}):
                
        super(PetClassifier, self).__init__()

        self.cnn_layers = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=5, stride=3, padding=2),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(64, 16, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2)
        )

        self.dense_layers = nn.Sequential(
            nn.Linear(16 * 2 * 2, 32),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )
    def forward(self, x):
        

        x = self.cnn_layers(x)
        x = x.view(x.size(0), -1)
        x = self.dense_layers(x)

        return x



def get_model():

    classifier = PetClassifier()

    optimizer = optim.Adam(classifier.parameters(), lr=2e-3)
    # loss = nn.NLLLoss()
    # loss = nn.CrossEntropyLoss()
    loss = nn.BCEWithLogitsLoss()

    return (classifier, optimizer, loss)



def data_augmenter():

    train_trans = transforms.Compose([transforms.Resize((64, 64)),
                                    transforms.RandomRotation(30),
                                    transforms.RandomResizedCrop(64),
                                    transforms.RandomHorizontalFlip(),
                                    transforms.ToTensor(),
                                    transforms.Normalize([0.5, 0.5, 0.5], 
                                                        [0.5, 0.5, 0.5])])

    test_trans = transforms.Compose([transforms.Resize(64),
                                    transforms.ToTensor(),
                                    transforms.Normalize([0.5, 0.5, 0.5],
                                    [0.5, 0.5, 0.5])])

    training_set = datasets.ImageFolder(root="dataset/training_set", transform=train_trans)
    test_set = datasets.ImageFolder(root="dataset/test_set", transform=test_trans)

    print("Classes loaded...")
    print(training_set.classes) #show class names sorted alphabetically
    print(training_set.class_to_idx) #show dict(class_name, class_index)
    
    return (training_set, test_set)


def train(training_set, test_set, epochs=1, save=True):


    classifier, optimizer, loss = get_model()
    n_train = len(training_set)
    n_test = len(test_set)

    for e in range(epochs):
        correct = 0
        classifier.train()
        print(f"\nRunning Epoch {e}:")    
        for x, y in tqdm(training_set):

            optimizer.zero_grad()

            y_hat = classifier(x.unsqueeze(0))
            y = (torch.tensor([y], dtype=torch.float32)).unsqueeze(0)
            l = loss(y_hat, y)
            l.backward()
            
            optimizer.step()
            _, prediction = torch.max(y_hat, 1)
            correct += (prediction == y).sum().item()

        train_accuracy = correct / n_train
        print(f"Training Accuracy: {train_accuracy}")


        correct = 0
        classifier.eval()

        for x_test, y_test in test_set:

            y_hat = classifier(x.unsqueeze(0))
            _,prediction = torch.max(y_hat, 1)
            correct += (prediction == y_test).sum().item()

        val_accuracy = correct / n_test
        print(f"Validation Accuracy: {val_accuracy}")

        if e % 5:
            torch.save(classifier.state_dict(), MODEL_PATH)

    torch.save(classifier.state_dict(), MODEL_PATH)

def load_model(PATH="models/"):

    classifier, optimizer, loss = get_model()
    classifier.load_state_dict(torch.load(PATH))
    classifier.eval()

    return (classifier, optimizer, loss)

if __name__ == "__main__":


    
    training_set, test_set = data_augmenter()
    cl = PetClassifier()
    summary(cl, (3, 64, 64))
    print(training_set[0][0].shape)
    train(training_set, test_set, 2, save=True)