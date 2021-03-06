from baseline_cnn_0_3_0 import *
from torchvision import models
from aitracks_cnn_0_3_0 import const_reset_inds


import matplotlib.pyplot as plt

model_path = 'AI_Tracks_v0.3.0'
random_seed = 42

img_means = (0.5455, 0.5684, 0.5804)
img_stds = (0.1738, 0.1632, 0.1821)

out_means = (32.702289134844705, -117.2294848913561)
out_stds = (0.007713758442119858, 0.017891982624611145)

gen_fp = [
    'data/generated_coords/6.csv'
]

data_to_load = [
    'data/NN_data_skeletons/6_train.csv'
]

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
net.load_state_dict(torch.load(model_path))
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(img_means, img_stds)])

for fp in gen_fp:
    baseline_coords = pd.read_csv(fp)
    min_lat = min(baseline_coords['fake_lat'])
    max_lat = max(baseline_coords['fake_lat'])
    min_lon = min(baseline_coords['fake_lon'])
    max_lon = max(baseline_coords['fake_lon'])

    plt.scatter(baseline_coords['fake_lat'], baseline_coords['fake_lon'], zorder=1, alpha=0.2, c='b', s=10)

# plt.xlim(min_lat, max_lat)
# plt.ylim(min_lon, max_lon)

df = pd.DataFrame()
for file in data_to_load:
    df = pd.concat([df, pd.read_csv(file, names=['imgpath', 'xtl', 'ytl', 'xbr', 'ybr', 'lat', 'lon'])])

df = df.reset_index()

with torch.no_grad():
    eps = np.finfo(float).eps

    trans = transforms.ToPILImage()
    x = []
    y = []
    for minibatch_count, row in df.iterrows():

        images = []
        targets = []
        bb_coords = []

        for i in range(len(df)):
            row = df.iloc[i]
            images.append(transform(Image.open(row['imgpath'])))
            targets.append(torch.FloatTensor(((row[['lat', 'lon']] - out_means) / out_stds).to_list()))
            bb_coords.append(torch.FloatTensor(row[['xtl', 'ytl', 'xbr', 'ybr']].to_list()))

        images = torch.stack(images)
        targets = torch.stack(targets)
        bb_coords = torch.stack(bb_coords)

        h = Variable(torch.zeros(2, 1, 200))
        h = h.to(computing_device)

        images, bb_coords, targets = images.to(computing_device), \
                                     bb_coords.to(computing_device), \
                                     targets.to(computing_device)

        # Put the minibatch data in CUDA Tensors and run on the GPU if supported
        output, h = net(images, bb_coords=bb_coords, prev_out=h)

        x.append(output[0][0].item() * out_stds[0] + out_means[0])
        y.append(output[0][1].item() * out_stds[1] + out_means[1])
        print(minibatch_count)

    plt.plot(x, y,  color='r')

plt.savefig('0_3_0_output/11_15_20')
plt.clf()
