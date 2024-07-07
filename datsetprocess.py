from torch.utils.data import Dataset, DataLoader
from utils import *

class MyDataset(Dataset):
	def __init__(self, scene_directory, transforms=None, is_training=True, noise=None):
		list = os.listdir(scene_directory)
		self.image_list = []
		self.num = 0
		self.transforms = transforms
		self.noise = noise
		self.is_training = is_training
		for scene in range(len(list)):
			expo_path = os.path.join(scene_directory, list[scene], 'exposure.txt')
			file_path = list_all_files_sorted(os.path.join(scene_directory, list[scene]), '.tif')
			label_path = os.path.join(scene_directory, list[scene])
			self.image_list += [[expo_path, file_path, label_path]]
			self.num = self.num + 1

	def __getitem__(self, idx):
		expoTimes = ReadExpoTimes(self.image_list[idx][0])
		imgs = ReadImages(self.image_list[idx][1])
		label = ReadLabel(self.image_list[idx][2])
		if self.is_training:
			image = np.concatenate([imgs[0], imgs[1], imgs[2], label], axis=2)
			image = self.transforms(image=image)['image']
			imgs[0] = image[:, :, 0:3]
			imgs[1] = image[:, :, 3:6]
			imgs[2] = image[:, :, 6:9]
			label = image[:, :, 9:12]
		pre_img0 = LDR_to_HDR(imgs[0], expoTimes[0], 2.2)
		pre_img1 = LDR_to_HDR(imgs[1], expoTimes[1], 2.2)
		pre_img2 = LDR_to_HDR(imgs[2], expoTimes[2], 2.2)
		output0 = np.concatenate((imgs[0], pre_img0), 2)
		output1 = np.concatenate((imgs[1], pre_img1), 2)
		output2 = np.concatenate((imgs[2], pre_img2), 2)
		crop_size = 256
		H, W, _ = imgs[0].shape
		x = np.random.randint(0, H - crop_size - 1)
		y = np.random.randint(0, W - crop_size - 1)

		im1 = output0[x:x + crop_size, y:y + crop_size, :].astype(np.float32).transpose(2, 0, 1)
		im2 = output1[x:x + crop_size, y:y + crop_size, :].astype(np.float32).transpose(2, 0, 1)
		im3 = output2[x:x + crop_size, y:y + crop_size, :].astype(np.float32).transpose(2, 0, 1)
		im4 = label[x:x + crop_size, y:y + crop_size, :].astype(np.float32).transpose(2, 0, 1)

		im1 = torch.from_numpy(im1)
		im2 = torch.from_numpy(im2)
		im3 = torch.from_numpy(im3)
		im4 = torch.from_numpy(im4)

		sample = {'input1': im1, 'input2': im2, 'input3': im3, 'label': im4}

		return sample

	def __len__(self):
		return self.num




