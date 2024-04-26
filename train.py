import argparse

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.metrics import classification_report
from torch_geometric.datasets import Twitch
from torch_geometric.nn import GCNConv


class GCN(torch.nn.Module):
	def __init__(self, num_node_features, num_classes, hidden=256):
		super().__init__()
		self.conv1 = GCNConv(num_node_features, hidden)
		self.conv2 = GCNConv(hidden, hidden)
		self.conv3 = GCNConv(hidden, num_classes)


	def forward(self, data):
		x, edge_index = data.x, data.edge_index

		x = self.conv1(x, edge_index)
		x = F.relu(x)
		x = F.dropout(x, training=self.training)
		x = self.conv2(x, edge_index)
		x = F.relu(x)
		x = F.dropout(x, training=self.training)
		x = self.conv3(x, edge_index)

		return F.softmax(x, dim=1)


if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('--option', type=str, default='basic', help='Options: basic, transfer')
	args = parser.parse_args()

	datasets = [
		Twitch(root='data/Twitch', name='DE'),
		Twitch(root='data/Twitch', name='EN'),
		Twitch(root='data/Twitch', name='ES'),
		Twitch(root='data/Twitch', name='FR'),
		Twitch(root='data/Twitch', name='PT'),
		Twitch(root='data/Twitch', name='RU'),
	]
	train_split = 0.8
	val_split = 0.1
	device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

	accuracies, precisions, recalls, f1_scores = [], [], [], []

	for dataset_1_idx, dataset_1 in enumerate(datasets):
		print(f'Training on {dataset_1.name}...')
		model = GCN(dataset_1.num_node_features, dataset_1.num_classes, hidden=512).to(device)
		data_1 = dataset_1[0].to(device)

		optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=0.001)

		# 80/10/10 train/val/test split masks for the dataset
		train_mask = torch.zeros(data_1.num_nodes, dtype=torch.bool, device=device)
		train_mask[:int(train_split * data_1.num_nodes)] = 1
		val_mask = torch.zeros(data_1.num_nodes, dtype=torch.bool, device=device)
		val_mask[int(train_split * data_1.num_nodes):int((train_split + val_split) * data_1.num_nodes)] = 1

		# training loop over one language dataset
		global_step = 0
		for i, epoch in enumerate(range(200)):
			model.train()
			optimizer.zero_grad()
			out = model(data_1)
			loss = F.nll_loss(out[train_mask], data_1.y[train_mask])
			loss.backward()
			optimizer.step()

			model.eval()
			out = model(data_1)
			val_loss = F.nll_loss(out[val_mask], data_1.y[val_mask])
			pred = out.argmax(dim=1)

			correct = (pred[val_mask] == data_1.y[val_mask]).sum()
			acc = int(correct) / int(val_mask.sum())
			global_step += 1

		print(f'Finished training, saving model...')
		torch.save(model, f'models/model_{dataset_1.name}.pt')
		# model = torch.load(f'models/model_{dataset_1.name}.pt')

		model.eval()
		print('Testing...')
		if args.option == 'basic':
			test_mask = torch.zeros(data_1.num_nodes, dtype=torch.bool, device=device)
			test_mask[int((train_split + val_split) * data_1.num_nodes):] = 1

			pred = model(data_1).argmax(dim=1)
			with open(f'outputs/{dataset_1.name}_{dataset_1.name}.txt', 'w') as f:
				for p in pred:
					f.write(f'{p}\n')
			report = classification_report(data_1.y[test_mask].cpu().numpy(), pred[test_mask].cpu().numpy(), target_names=['0', '1'], output_dict=True, zero_division=np.nan)
			accuracies.append(report['accuracy'])
			precisions.append(report['weighted avg']['precision'])
			recalls.append(report['weighted avg']['recall'])
			f1_scores.append(report['weighted avg']['f1-score'])

			correct = (pred[test_mask] == data_1.y[test_mask]).sum()
			acc = int(correct) / int(test_mask.sum())
		elif args.option == 'transfer':
			for dataset_2_idx, dataset_2 in enumerate(datasets):
				if dataset_1_idx == dataset_2_idx:
					accuracies.append('-')
					precisions.append('-')
					recalls.append('-')
					f1_scores.append('-')
					continue
				data_2 = dataset_2[0].to(device)
				test_mask = torch.ones(data_2.num_nodes, dtype=torch.bool, device=device)
				test_mask[int((train_split + val_split) * data_2.num_nodes):] = 1

				pred = model(data_2).argmax(dim=1)
				with open(f'outputs/{dataset_1.name}_{dataset_2.name}.txt', 'w') as f:
					for p in pred:
						f.write(f'{p}\n')
				report = classification_report(data_2.y[test_mask].cpu().numpy(), pred[test_mask].cpu().numpy(), target_names=['0', '1'], output_dict=True, zero_division=np.nan)
				accuracies.append(report['accuracy'])
				precisions.append(report['weighted avg']['precision'])
				recalls.append(report['weighted avg']['recall'])
				f1_scores.append(report['weighted avg']['f1-score'])

				correct = (pred[test_mask] == data_2.y[test_mask]).sum()
				acc = int(correct) / int(test_mask.sum())

	print('Finished testing, saving report...')
	labels = [f'{d1.name}-{d2.name}' for d1 in datasets for d2 in datasets] \
		if args.option == 'transfer' \
		else [f'{d1.name}-{d1.name}' for d1 in datasets]
	mega_report = pd.DataFrame({
		'Trained-Tested': labels,
		'Accuracy': accuracies,
		'Precision': precisions,
		'Recall': recalls,
		'F1-Score': f1_scores
	})
	mega_report.to_csv(f'{args.option}_report.csv')
	print('Done.')