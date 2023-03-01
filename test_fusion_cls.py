import torch
from torch.nn.modules.linear import Linear
from model import GRU, TempCNN, Linear_Net, CoordCNNTSM_Model, GRU_Mod, DecoderCNN

import numpy as np
import matplotlib.animation as animation
import matplotlib.image as mpimg
import matplotlib.pyplot as plt

from torch.utils.data import DataLoader
from play_datasets import NBA_Play_Video_Dataset, NBA_Real_Dataset

import utils

if __name__ == "__main__":
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print("Using device:", device)

    # Recovery Net Definition
    recover_net_cnn = CoordCNNTSM_Model().to(device)
    recover_net_temp_cnn = TempCNN().to(device)
    recover_net_gru = GRU_Mod().to(device)
    recover_net_deconv = DecoderCNN().to(device)

    # Tactic Net Definition
    net_cnn = CoordCNNTSM_Model().to(device)
    net_temp_cnn = TempCNN().to(device)
    net_gru = GRU().to(device)
    net_linear = Linear_Net().to(device)

    checkpoint = torch.load('weights/fusion_supervised/recover_net.pth.tar', map_location=device)
    recover_net_cnn.load_state_dict(checkpoint['cnn_state_dict'], strict=False)
    recover_net_temp_cnn.load_state_dict(checkpoint['temp_cnn_state_dict'], strict=False)
    recover_net_gru.load_state_dict(checkpoint['gru_state_dict'], strict=False)
    recover_net_deconv.load_state_dict(checkpoint['deconv_state_dict'], strict=False)


    checkpoint = torch.load('weights/fusion_supervised/tactic_net.pth.tar', map_location=device)
    net_cnn.load_state_dict(checkpoint['cnn_state_dict'], strict=False)
    net_temp_cnn.load_state_dict(checkpoint['temp_cnn_state_dict'], strict=False)
    net_gru.load_state_dict(checkpoint['gru_state_dict'], strict=False)
    net_linear.load_state_dict(checkpoint['linear_state_dict'], strict=False)

    train_dataset = NBA_Play_Video_Dataset(mode="train")

    train_loader = DataLoader(train_dataset, batch_size=1,
                            num_workers=0, drop_last=False, shuffle=True)

    test_dataset = NBA_Play_Video_Dataset(mode="test")

    test_loader = DataLoader(test_dataset, batch_size=3,
                            num_workers=0, drop_last=False, shuffle=False)


    recover_net_cnn.eval()
    recover_net_temp_cnn.eval()
    recover_net_gru.eval()
    recover_net_deconv.eval()

    net_cnn.eval()
    net_temp_cnn.eval()
    net_gru.eval()
    net_linear.eval()

    num_test = 10

    total_top1 = 0
    total_top5 = 0

    with torch.no_grad():
        for test in range(num_test):
            top1_accuracy = 0
            top5_accuracy = 0

            for iter, (heatmap, category, play_keypoints, heatmap_noise) in enumerate(test_loader):
                heatmap_noise = heatmap_noise.to(device)
                category = category.to(device)
                play_keypoints = play_keypoints.to(device)
                # heatmap = heatmap.to(device)

                # --- RecoveryNet forward pass ---
                h = recover_net_cnn(heatmap_noise)
                temp_features = recover_net_temp_cnn(h)
                x, hidden_state = recover_net_gru(temp_features)
                rec_heatmap = recover_net_deconv(x)

                # --- Tactics Model forward pass ---
                h = net_cnn(rec_heatmap)
                temp_features = net_temp_cnn(h)
                out, hidden_state = net_gru(temp_features)  
                pred = torch.nn.Softmax(dim=1)(net_linear(hidden_state))
                
                top1, top5 = utils.accuracy(pred, category, topk=(1,5))

                top1_accuracy += top1[0]
                top5_accuracy += top5[0]

            top1_accuracy /= (iter + 1)
            top5_accuracy /= (iter + 1)

            print(f"Test {test}\tTop1 Test acc: {top1_accuracy.item()}\tTop5 test acc: {top5_accuracy.item()}")

            total_top1 += top1_accuracy.item()
            total_top5 += top5_accuracy.item()

    print("Average Top1 Acc: ", total_top1/num_test)
    print("Average Top5 Acc: ", total_top5/num_test)
    