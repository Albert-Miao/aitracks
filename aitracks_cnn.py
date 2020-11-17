from baseline_cnn import *


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)


load_model = True
model_path = 'AI_Tracks_v0.0.1'

# accuracies_path = 'accuracies_11_14_19_baseline/'
# if not os.path.isdir(accuracies_path):
#     os.mkdir(accuracies_path)
#     for i in range(0, 200):
#         with open(accuracies_path + str(i) + ".csv", 'a+') as f:
#             f.write("Accuracy, Precision, Recall, BCR\n")
#
#     for i in range(0, 200):
#         with open(accuracies_path + "train_" + str(i) + ".csv", 'a+') as f:
#             f.write("Accuracy, Precision, Recall, BCR\n")
#
#     with open(accuracies_path + "avg.csv", 'a+') as f:
#         f.write("Accuracy, Precision, Recall, BCR\n")
#
#     with open(accuracies_path + "train_avg.csv", 'a+') as f:
#         f.write("Accuracy, Precision, Recall, BCR\n")
#
#     with open(accuracies_path + "train_loss.csv", 'a+') as f:
#         f.write("Average Loss per Epoch\n")
#
#     with open(accuracies_path + "val_loss.csv", 'a+') as f:
#         f.write("Average Loss per Epoch\n")

# Check if your system supports CUDA
use_cuda = torch.cuda.is_available()

# Setup GPU optimization if CUDA is supported
if use_cuda:
    computing_device = torch.device("cuda")
    extras = {"num_workers": 1, "pin_memory": True}
    print("CUDA is supported")
else:  # Otherwise, train on the CPU
    computing_device = torch.device("cpu")
    extras = False
    print("CUDA NOT supported")

net = Nnet().to(computing_device)

if load_model:
    net.load_state_dict(torch.load(model_path))
else:
    net.apply(weights_init)

# Print the model
print(net)

# loss criteria are defined in the torch.nn package
criterion = nn.MSELoss()

# Instantiate the gradient descent optimizer - use Adam optimizer with default parameters
optimizer = optim.Adam(net.parameters(), lr=0.001)

transform = transforms.Compose([transforms.ToTensor()])
dataset = loader('0_0_1_train.csv', '', transform=transform)
batch_size = 64
validation_split = .2
shuffle_dataset = False
random_seed = 42

# Creating data indices for training and validation splits:
dataset_size = len(dataset)
indices = list(range(dataset_size))
split = int(np.floor(validation_split * dataset_size))
if shuffle_dataset:
    np.random.seed(random_seed)
    np.random.shuffle(indices)
train_indices, val_indices = indices[split:], indices[:split]

# Creating PT data samplers and loaders:
train_sampler = SubsetRandomSampler(train_indices)
valid_sampler = SubsetRandomSampler(val_indices)

train_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size,
                                           sampler=train_sampler)
validation_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size,
                                                sampler=valid_sampler)

# Track the loss across training
total_loss = []
N = 50

total_val_loss = []
val_N = 20

# train_class_acc = np.empty((100, 200))
# train_class_prec = np.empty((100, 200))
# train_class_rec = np.empty((100, 200))
# train_class_bcr = np.empty((100, 200))
#
# train_avg_acc = np.empty(100)
# train_avg_prec = np.empty(100)
# train_avg_rec = np.empty(100)
# train_avg_bcr = np.empty(100)
#
# val_class_acc = np.empty((100, 200))
# val_class_prec = np.empty((100, 200))
# val_class_rec = np.empty((100, 200))
# val_class_bcr = np.empty((100, 200))
#
# val_avg_acc = np.empty(100)
# val_avg_prec = np.empty(100)
# val_avg_rec = np.empty(100)
# val_avg_bcr = np.empty(100)

def train_network():
    for epoch in range(100):
        N_loss = 0.0

        eps = np.finfo(float).eps

        # Get the next minibatch of images, labels for training
        for minibatch_count, ((images, prev_coords), labels) in enumerate(train_loader, 0):
            # Zero out the stored gradient (buffer) from the previous iteration
            optimizer.zero_grad()
            # Put the minibatch data in CUDA Tensors and run on the GPU if supported
            images, prev_coords, labels = images.to(computing_device), \
                                          prev_coords.to(computing_device), \
                                          labels.to(computing_device)
            # Perform the forward pass through the network and compute the loss
            outputs = net(images, prev_coords)

            loss = criterion(outputs, labels)
            # Automagically compute the gradients and backpropagate the loss through the network
            loss.backward()

            # Update the weights
            optimizer.step()
            # Add this iteration's loss to the total_loss
            N_loss += loss.item()

            print("mini_batch", minibatch_count, loss)

            if minibatch_count % N == N - 1:
                # Print the loss averaged over the last N mini-batches
                print('Epoch %d, average minibatch %d loss: %.3f' % (epoch + 1, minibatch_count + 1,
                                                                     N_loss / (minibatch_count * batch_size)))

        total_loss.append(N_loss)
        N_loss = 0.0
        print("Finished", epoch + 1, "epochs of training")

        with torch.no_grad():
            for minibatch_count, ((images, prev_coords), labels) in enumerate(validation_loader, 0):
                images, prev_coords, labels = images.to(computing_device), \
                                              prev_coords.to(computing_device), \
                                              labels.to(computing_device)
                outputs = net(images, prev_coords)

                loss = criterion(outputs, labels)


                N_loss += loss.item()

                print("val mini_batch", minibatch_count, loss)

            total_val_loss.append(N_loss)

        if len(total_val_loss) <= 1 or N_loss <= total_val_loss[len(total_val_loss) - 1]:
            torch.save(net.state_dict(), model_path)

        # train_class_acc[epoch] = (train_true_pos + train_true_neg) / (train_true_pos + train_true_neg + train_false_pos
        #                                                               + train_false_neg)
        # train_class_prec[epoch] = train_true_pos / (train_true_pos + train_false_pos)
        # train_class_rec[epoch] = train_true_pos / (train_true_pos + false_neg)
        # train_class_bcr[epoch] = np.mean((train_class_prec[epoch], train_class_rec[epoch]), axis=0)
        #
        # val_class_acc[epoch] = (true_pos + true_neg) / (true_pos + true_neg + false_pos + false_neg)
        # val_class_prec[epoch] = true_pos / (true_pos + false_pos)
        # val_class_rec[epoch] = true_pos / (true_pos + false_neg)
        # val_class_bcr[epoch] = np.mean((val_class_prec[epoch], val_class_rec[epoch]), axis=0)
        #
        # for i in range(0, 200):
        #     acc_string = ''
        #     acc_string += str(val_class_acc[epoch][i]) + ', '
        #     acc_string += str(val_class_prec[epoch][i]) + ', '
        #     acc_string += str(val_class_rec[epoch][i]) + ', '
        #     acc_string += str(val_class_bcr[epoch][i]) + '\n'
        #     with open(accuracies_path + str(i) + ".csv", 'a+') as f:
        #         f.write(acc_string)
        #
        # for i in range(0, 200):
        #     acc_string = ''
        #     acc_string += str(train_class_acc[epoch][i]) + ', '
        #     acc_string += str(train_class_prec[epoch][i]) + ', '
        #     acc_string += str(train_class_rec[epoch][i]) + ', '
        #     acc_string += str(train_class_bcr[epoch][i]) + '\n'
        #     with open(accuracies_path + 'train_' + str(i) + ".csv", 'a+') as f:
        #         f.write(acc_string)
        #
        # val_avg_acc[epoch] = np.mean(val_class_acc[epoch])
        # val_avg_prec[epoch] = np.mean(val_class_prec[epoch])
        # val_avg_rec[epoch] = np.mean(val_class_rec[epoch])
        # val_avg_bcr[epoch] = np.mean(val_class_bcr[epoch])
        #
        # acc_string = ''
        # acc_string += str(val_avg_acc[epoch]) + ', '
        # acc_string += str(val_avg_prec[epoch]) + ', '
        # acc_string += str(val_avg_rec[epoch]) + ', '
        # acc_string += str(val_avg_bcr[epoch]) + '\n'
        # with open(accuracies_path + "avg.csv", 'a+') as f:
        #     f.write(acc_string)
        #
        # train_avg_acc[epoch] = np.mean(train_class_acc[epoch])
        # train_avg_prec[epoch] = np.mean(train_class_prec[epoch])
        # train_avg_rec[epoch] = np.mean(train_class_rec[epoch])
        # train_avg_bcr[epoch] = np.mean(train_class_bcr[epoch])
        #
        # acc_string = ''
        # acc_string += str(train_avg_acc[epoch]) + ', '
        # acc_string += str(train_avg_prec[epoch]) + ', '
        # acc_string += str(train_avg_rec[epoch]) + ', '
        # acc_string += str(train_avg_bcr[epoch]) + '\n'
        # with open(accuracies_path + "train_avg.csv", 'a+') as f:
        #     f.write(acc_string)
        #
        # with open(accuracies_path + "train_loss.csv", 'a+') as f:
        #     f.write(str(total_loss[epoch] / dataset_size) + '\n')
        #
        # with open(accuracies_path + "val_loss.csv", 'a+') as f:
        #     f.write(str(total_val_loss[epoch] / dataset_size) + '\n')
        #
        # print('Epoch %d, average class accuracy, %.3f' % (epoch + 1, val_avg_acc[epoch]))

if __name__ == '__main__':
    train_network()