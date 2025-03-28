
optimiser_func = torch.optim.Adam(params)
loss_func = torch.nn.L1Loss()

def train(num_epochs, params):
    
    for epoch in range(num_epochs):
        epoch_loss = train_one_epoch(epoch, tb_writer)
        # test_loss = test_one_epoch(epoch, tb_writer)
        # print('Epoch {} test loss: {}'.format(epoch, test_loss))
        print(f"Epoch {epoch+1} completed with Loss: {epoch_loss}")

    print("Training complete")


def train_one_epoch(epoch_index, tb_writer):
    running_loss = 0
    last_loss = 0 

    for i, data in enumerate(dataloader):
        sample_data = data['data']
        #inputs = data['inputs'].to(device)
        #labels = sample_data['labels'].to(device)

        optimiser_func.zero_grad()

        outputs = model(inputs)

        loss = loss_func(outputs, labels)

        loss.backward()
        optimiser_func.step()

        running_loss += loss.item()

        epoch_loss = running_loss / len(dataloader)
        # print(f"Epoch {epoch_index+1} completed with Loss: {epoch_loss}")
        return epoch_loss

        # if i % 1000 == 999:
        #     last_loss = running_loss / 1000 # loss per batch
        #     print('  batch {} loss: {}'.format(i + 1, last_loss))
        #     tb_x = epoch_index * len(data_loader) + i + 1
        #     tb_writer.add_scalar('Loss/train', last_loss, tb_x)
        #     running_loss = 0.

    # return last_loss

