import _init_paths

import argparse
from tqdm import tqdm

import torch
import torch.nn as nn

from datasets.dataset import PointDataset
from lib.mvtn import mvtn

parser = argparse.ArgumentParser()
parser.add_argument('--dataset_dir', type=str, default='./datasets', help='modelnet40')
parser.add_argument('--batch_size', type=int, default=2, help='batch size')
parser.add_argument('--lr', default=0.05, help='learning rate')
parser.add_argument('--lr_rate', default=0.3, help='learning rate decay rate')
parser.add_argument('--nepoch', type=int, default=10, help='max number of epochs to train')
parser.add_argument('--start_epoch', type=int, default=1, help='which epoch to start')
opt = parser.parse_args()

if __name__ == '__main__':

	device = torch.device("cuda:0")

	criterion = nn.CrossEntropyLoss()
	model = mvtn(device=device)
	model = nn.DataParallel(model, device_ids=[0]).cuda()
	optimizer = torch.optim.Adam(model.parameters(), lr=opt.lr)

	train_set = PointDataset(dataset_dir=opt.dataset_dir, partition='train')
	train_loader = torch.utils.data.DataLoader(train_set, batch_size=opt.batch_size, shuffle=True)
	batch_size = train_loader.batch_size

	for epoch in range(opt.start_epoch, opt.nepoch + 1):
	    model.train()
	    total_loss, total_num, train_bar = 0.0, 0, tqdm(train_loader)
	    
	    for points, label in train_bar:
	        points = points.cuda()
	        label = label.cuda()

	        optimizer.zero_grad()

	        pre = model(points)

	        loss = criterion(pre, label.flatten())
	        loss.backward()
	        optimizer.step()

	        total_num += batch_size
	        total_loss += loss.item()
	        train_bar.set_description('Train Epoch: [{}/{}] Loss: {:.4f}'.format(epoch, opt.nepoch, total_loss / total_num))

