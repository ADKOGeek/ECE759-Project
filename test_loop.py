# device = CPU/GPU
import torch
def test(model, device, loss_func, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    total_samples = 0
    with torch.no_grad():
        for data in test_loader:
            # data, target = data.to(device), target.to(device)
            
            inputs = data['data']
            labels = data['label']
            
            output = model(inputs)
            test_loss += loss_func(output, labels).item()

            predicted = output.argmax(1, dim = 1, keepdim=False)
            correct += (predicted == labels).sum().item()
            total_samples += len(labels)
            total_samples += labels.size(0)

            # test_loss += F.nll_loss(output, target, reduction='sum').item()  # sum up batch loss
            # pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            # correct += pred.eq(target.view_as(pred)).sum().item()
    
    test_loss = test_loss / len(dataloader)
    accuracy = correct / total_samples * 100
    print(f"Test Loss: {test_loss}, Accuracy: {accuracy}")
    # print("Test_loop")