from baseline_cnn_0_2_0 import *


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)


load_model = True
model_path = 'AI_Tracks_v0.2.0'
validation_split = .2
random_seed = 42

img_means = (0.5455, 0.5684, 0.5804)
img_stds = (0.1738, 0.1632, 0.1821)

out_means = (32.702289134844705, -117.2294848913561)
out_stds = (0.007713758442119858, 0.017891982624611145)

data_to_load = [
    'data/NN_data_skeletons/6_train.csv',
    'data/NN_data_skeletons/7_train.csv'
]

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

transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(img_means, img_stds)])

const_reset_inds = [0]
df = pd.DataFrame()
for file in data_to_load:
    df = pd.concat([df, pd.read_csv(file, names=['imgpath', 'xtl', 'ytl', 'xbr', 'ybr', 'lat', 'lon'])])
    const_reset_inds.append(len(df))

df = df.reset_index()
const_reset_inds = np.array(const_reset_inds)
num_frames = len(df)

split_ind = int(num_frames * (1 - validation_split))
train_df = df.iloc[:split_ind]
val_df = df.iloc[split_ind:]

# Track the loss across training
total_loss = []
N = 50

total_val_loss = []
val_N = 20


def train_network():

    num_resets = 3600

    for epoch in range(100):
        N_loss = 0.0

        eps = np.finfo(float).eps

        num_resets -= epoch

        additional_inds = np.random.choice(np.arange(1, num_frames), num_resets, replace=False)
        reset_inds = np.concatenate((const_reset_inds, additional_inds))
        reset_inds.sort()

        # Get the next minibatch of images, labels for training
        for minibatch_count, row in train_df.iterrows():

            # Zero out the stored gradient (buffer) from the previous iteration
            optimizer.zero_grad()

            image = transform(Image.open(row['imgpath']))[None, :, :, :]
            target = torch.FloatTensor(((row[['lat', 'lon']] - out_means) / out_stds).to_list())[None, :]
            bb_coords = torch.FloatTensor(row[['xtl', 'ytl', 'xbr', 'ybr']].to_list())[None, :]
            image, bb_coords, target = image.to(computing_device), \
                                       bb_coords.to(computing_device), \
                                       target.to(computing_device)

            # Put the minibatch data in CUDA Tensors and run on the GPU if supported
            if minibatch_count in reset_inds:
                output = net(image, bb_coords=bb_coords, prev_out=None)

            else:
                output = Variable(output.data)
                output = output.to(computing_device)
                output = net(image, bb_coords=None, prev_out=output)

            loss = criterion(output, target)
            # Automagically compute the gradients and backpropagate the loss through the network

            if minibatch_count + 1 in reset_inds:
                loss.backward()
            else:
                loss.backward(retain_graph=True)

            # Update the weights
            optimizer.step()
            # Add this iteration's loss to the total_loss
            N_loss = N_loss + loss.item()

            #print("mini_batch", minibatch_count, loss)

            if minibatch_count % N == N - 1:
                # Print the loss averaged over the last N mini-batches
                print('Epoch %d, average minibatch %d loss: %.9f' % (epoch + 1, minibatch_count + 1,
                                                                     N_loss / (minibatch_count)))

        total_loss.append(N_loss)
        N_loss = 0.0
        print("Finished", epoch + 1, "epochs of training")

        with torch.no_grad():
            for minibatch_count, row in val_df.iterrows():

                image = transform(Image.open(row['imgpath']))[None, :, :, :]
                target = torch.FloatTensor(((row[['lat', 'lon']] - out_means) / out_stds).to_list())[None, :]
                bb_coords = torch.FloatTensor(row[['xtl', 'ytl', 'xbr', 'ybr']].to_list())[None, :]
                image, bb_coords, target = image.to(computing_device), \
                                           bb_coords.to(computing_device), \
                                           target.to(computing_device)

                if minibatch_count in reset_inds:
                    output = net(image, bb_coords=bb_coords, prev_out=None)

                else:
                    output = Variable(output.data)
                    output = output.to(computing_device)
                    output = net(image, bb_coords=None, prev_out=output)

                loss = criterion(output, target)

                N_loss += loss.item()

                print("val mini_batch", minibatch_count, loss)

            total_val_loss.append(N_loss)

        if len(total_val_loss) <= 1 or N_loss <= total_val_loss[len(total_val_loss) - 1]:
            torch.save(net.state_dict(), model_path)


if __name__ == '__main__':
    train_network()
