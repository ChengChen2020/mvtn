import _init_paths

import argparse
import pandas as pd
from tqdm import tqdm

import torch
import torch.nn as nn

from datasets.dataset import PointDataset
from lib.mvtn import mvtn

parser = argparse.ArgumentParser()
parser.add_argument('--dataset_dir', type=str, default='./datasets', help='modelnet40')
parser.add_argument('--batch_size', type=int, default=16, help='batch size')
parser.add_argument('--lr', default=0.05, help='learning rate')
parser.add_argument('--lr_rate', default=0.3, help='learning rate decay rate')
parser.add_argument('--wd_rate', default=0.01, help='weight dacay')
parser.add_argument('--nepoch', type=int, default=10, help='max number of epochs to train')
parser.add_argument('--start_epoch', type=int, default=1, help='which epoch to start')
parser.add_argument('--log_dir', type=str, default='./experiments/logs')
parser.add_argument('--resume', type=str, default='')
opt = parser.parse_args()

if __name__ == '__main__':

	device = torch.device("cuda:0")

	criterion = nn.CrossEntropyLoss()
	model = mvtn(device=device)
	model = nn.DataParallel(model, device_ids=[0]).cuda()
	optimizer = torch.optim.Adam(model.parameters(), lr=opt.lr)

	epoch_start = opt.start_epoch

	if opt.resume != '':
		checkpoint = torch.load('{0}/{1}'.format(opt.log_dir, opt.resume))
		epoch_start = checkpoint['epoch'] + 1
		model.load_state_dict(checkpoint['state_dict'])
		print('Loaded from: {}'.format(opt.resume))

	train_set = PointDataset(dataset_dir=opt.dataset_dir, partition='train')
	train_loader = torch.utils.data.DataLoader(train_set, batch_size=opt.batch_size, shuffle=True)

	test_set = PointDataset(dataset_dir=opt.dataset_dir, partition='test')
	test_loader = torch.utils.data.DataLoader(test_set, batch_size=opt.batch_size, shuffle=False)

	best_acc = 0.0
	results = {'train_loss': [], 'test_acc': []}

	for epoch in range(epoch_start, opt.nepoch + 1):
		model.train()
		total_loss, total_num, train_bar = 0.0, 0, tqdm(train_loader)

		for points, label in train_bar:
			points, label = points.cuda(), label.cuda()

			optimizer.zero_grad()

			out = model(points)

			loss = criterion(out, label.flatten())
			loss.backward()
			optimizer.step()

			total_loss += loss.item()
			total_num += points.size(0)
			train_bar.set_description('Train Epoch: [{}/{}] Loss: {:.4f}'.format(epoch, opt.nepoch, total_loss / total_num))

		results['train_loss'].append(total_loss / total_num)

		model.eval()

		total_top1, total_num, test_bar = 0.0, 0, tqdm(test_loader)
		with torch.no_grad():

			for points, label in test_bar:
				points, label = points.cuda(), label.cuda()

				out = model(points)

				_, pred = out.topk(1, 1, True)
				correct = pred.T.eq(label.view(1, -1))

				total_top1 += correct.float().sum().item()
				total_num += points.size(0)
				test_bar.set_description('Test Epoch: [{}/{}] Acc:{:.2f}%'.format(epoch, opt.nepoch, total_top1 / total_num * 100))

		test_acc_1 = total_top1 / total_num * 100
		results['test_acc'].append(test_acc_1)

		data_frame = pd.DataFrame(data=results, index=range(epoch_start, epoch + 1))
		data_frame.to_csv(opt.log_dir + '/log.csv', index_label='epoch')
		
		if test_acc_1 > best_acc:
			best_acc = test_acc_1
			torch.save({'epoch': epoch, 'state_dict': model.state_dict()}, '{0}/model_{1}_{2}.pth'.format(opt.log_dir, epoch, test_acc_1))


