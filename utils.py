from lib import *
from config import *

def train_model(net, dataloader_dict, critetion, optimizer, num_epochs):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("device: ", device)
    
    history = {
        "epochs": [],
        "phase": [],
        "loss": [],
        "accuracy": []
    }

    for epoch in range(num_epochs):
        print("Epoch {}/{}".format(epoch, num_epochs))

        net.to(device)
        torch.backends.cudnn.benchmark = True

        for phase in ['train', 'val']:
            if phase == 'train':
                net.train()
            else:
                net.eval()

            epoch_loss = 0.0
            epoch_corrents = 0

            #Skip first epoch trainning if derised
            if (epoch == 0) and (phase == "traim"):
                continue

            for inputs, labels in tqdm(dataloader_dict[phase]):
                inputs = inputs.to(device)
                labels = labels.to(device)

                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == "train"):
                    outputs = net(inputs)

                    # No using torch.tensor(outputs)
                    outputs = outputs.float()
                    labels = labels.long()

                    loss = critetion(outputs, labels)
                    _, preds = torch.max(outputs, 1)

                    if phase == "train":
                        loss.backward()
                        optimizer.step()

                    epoch_loss += loss.item() * inputs.size(0)
                    epoch_corrents += torch.sum(preds == labels.data)
                
                epoch_loss = epoch_loss / len(dataloader_dict[phase].dataset)
                epoch_accuracy = epoch_corrents.double() / len(dataloader_dict[phase].dataset)

                print("{} loss: {:.4f} Acc: {:.4f}".format(phase, epoch_loss, epoch_accuracy))

                history["epochs"].append(epoch)
                history["phase"].append(phase)
                history["loss"].append(epoch_loss)
                history["accuracy"].append(epoch_accuracy.item())

    torch.save(net.state_dict(), save_path)

    df = pd.DataFrame(history)
    df.to_csv(csv_path, index = False)
    print(f"Trainning history save to {csv_path}")
                
