# Environment
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from sklearn import preprocessing
import torch.utils.data as data_utils

from utils import *
from usad import *

device = get_default_device()

print(f"Using device {device}")

# Downsample
def downsample(input_arr: np.ndarray, m: int):
    '''
    Given an (n,d) numpy array, down-sample the array by calculating the median
    of every m data points

    Tested
    '''    
    # 0: calculate d
    d = input_arr.shape[1]
    # 1: cut off end
    new_len = input_arr.shape[0] - (input_arr.shape[0] % m)
    A = input_arr[0:new_len,:]
    # 2: reshape
    B = A.T
    C = B.reshape(d, -1, m)
    # 3: median
    D = np.median(C ,2)
    # 4: Reshape again
    E = D.T
    return E

#Read data
normal = pd.read_csv("input/SWaT_Dataset_Normal_v1.csv")#, nrows=1000)
normal = normal.drop(["Timestamp" , "Normal/Attack" ] , axis = 1)

# Transform all columns into float64
for i in list(normal): 
    normal[i]=normal[i].apply(lambda x: str(x).replace("," , "."))
normal = normal.astype(float)

# Normalization
from sklearn import preprocessing
min_max_scaler = preprocessing.MinMaxScaler()

x = normal.values
x_scaled = min_max_scaler.fit_transform(x)
normal = pd.DataFrame(x_scaled)

# Attack
# Read data
attack = pd.read_csv("input/SWaT_Dataset_Attack_v0.csv",sep=";")#, nrows=1000)
labels = [ float(label!= 'Normal' ) for label  in attack["Normal/Attack"].values]
attack = attack.drop(["Timestamp" , "Normal/Attack" ] , axis = 1)

print("Attack shape:")
print(attack.shape)

# Transform all columns into float64
for i in list(attack):
    attack[i]=attack[i].apply(lambda x: str(x).replace("," , "."))
attack = attack.astype(float)

# Normalization
x = attack.values 
x_scaled = min_max_scaler.transform(x)
attack = pd.DataFrame(x_scaled)

# Windows
window_size=12

print()
normal_vals = downsample(normal.values, 5)
print("Train shape:")
print(normal.values.shape)
print(normal_vals.shape)
print("Training data sample:")
print(normal.values[0:10,0])
print("Training data sum:")
print(list(normal.values.sum(0)))
print()

attack_vals = downsample(attack.values, 5)
print("Test shape:")
print(attack.values.shape)
print(attack_vals.shape)
print("Testing data sample:")
print(attack.values[0:10,0])
print("Testing data sum:")
print(list(attack.values.sum(0)))
print()

windows_normal=normal_vals[np.arange(window_size)[None, :] + np.arange(normal_vals.shape[0]-window_size)[:, None]]
windows_attack=attack_vals[np.arange(window_size)[None, :] + np.arange(attack_vals.shape[0]-window_size)[:, None]]

# Training
BATCH_SIZE =  4096
N_EPOCHS = 100
hidden_size = 100

w_size=windows_normal.shape[1]*windows_normal.shape[2]
#z_size=windows_normal.shape[1]*hidden_size
z_size=hidden_size

windows_normal_train = windows_normal[:int(np.floor(.8 *  windows_normal.shape[0]))]
windows_normal_val = windows_normal[int(np.floor(.8 *  windows_normal.shape[0])):int(np.floor(windows_normal.shape[0]))]

train_loader = torch.utils.data.DataLoader(data_utils.TensorDataset(
    torch.from_numpy(windows_normal_train).float().view(([windows_normal_train.shape[0],w_size]))
) , batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

val_loader = torch.utils.data.DataLoader(data_utils.TensorDataset(
    torch.from_numpy(windows_normal_val).float().view(([windows_normal_val.shape[0],w_size]))
) , batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

test_loader = torch.utils.data.DataLoader(data_utils.TensorDataset(
    torch.from_numpy(windows_attack).float().view(([windows_attack.shape[0],w_size]))
) , batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

if 0:
    # Skip training
    model = UsadModel(w_size, z_size)
    model = to_device(model,device)

    history = training(N_EPOCHS,model,train_loader,val_loader)

    #plot_history(history)

    torch.save({
                'encoder': model.encoder.state_dict(),
                'decoder1': model.decoder1.state_dict(),
                'decoder2': model.decoder2.state_dict()
                }, "model.pth")


# Testing
if 0:
    # Load from saved model trained here
    checkpoint = torch.load("model.pth")

    model.encoder.load_state_dict(checkpoint['encoder'])
    model.decoder1.load_state_dict(checkpoint['decoder1'])
    model.decoder2.load_state_dict(checkpoint['decoder2'])
else:
    # Load from saved model trained from mypkg
    checkpoint = torch.load('/home/alexey/School/Research/Checkpoints/USAD_SWaT_model.ckpt')
    w_size = checkpoint['hp_dict']['w_size']
    z_size = checkpoint['hp_dict']['z_size']
    print("w_size", w_size)
    print("z_size", z_size)
    model = UsadModel(w_size, z_size)
    model = to_device(model,device)
    model.load_state_dict(checkpoint['model_states'])

results=testing(model,test_loader)

if 0:
    print(len(results))
    for i in range(len(results)):
        plt.plot(results[i].to('cpu'))
        plt.savefig(f'./results_{i}')
        plt.close()

windows_labels=[]
labels_vals = downsample(np.asarray(labels).reshape(-1,1), 5)
for i in range(len(labels_vals)-window_size):
    windows_labels.append(list(np.int_(labels_vals[i:i+window_size])))


y_test = [1.0 if (np.sum(window) > 0) else 0 for window in windows_labels ]

y_pred=np.concatenate([torch.stack(results[:-1]).flatten().detach().cpu().numpy(),
                              results[-1].flatten().detach().cpu().numpy()])

print(y_pred.shape)

threshold=ROC(y_test,y_pred)

with open(f"/home/alexey/School/Research/submodules/usad/output/mypkg_test_loss", 'rb') as f:
    mypkg_y_pred = pickle.load(f)
plt.plot(mypkg_y_pred, label='mypkg')
plt.plot(y_pred, label='usad')
plt.legend()
plt.savefig('figs/y_pred.pdf')
plt.close()

with open(f"/home/alexey/School/Research/submodules/usad/output/mypkg_test_labels", 'rb') as f:
    mypkg_y_test = pickle.load(f)
plt.plot(mypkg_y_test, label='mypkg')
plt.plot(y_test, label='usad')
plt.legend()
plt.savefig('figs/y_test.pdf')
plt.close()

plt.plot(mypkg_y_test[0:1000], label='mypkg')
plt.plot(y_test[0:1000], label='usad')
plt.legend()
plt.savefig('figs/y_test_zoom.pdf')
plt.close()
