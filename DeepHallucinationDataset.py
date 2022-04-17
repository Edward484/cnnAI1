import torchvision.transforms as torch_transmorms
from PIL import Image as Im
from os import walk
from torch.utils.data import Dataset, DataLoader
import imageio as im
import random


class DeepHallucinationDataset(Dataset):
    def __init__(self, data_dir, image_txt, image_dir, role="Train", transform=None, enhancements=None):
        #directory where all of the files are
        self.data_dir = data_dir

        #directory of the images that are to be read
        self.image_dir = image_dir
        self.images = 0
        if role != "Test":
            self.images, self.labels = read_images(image_txt, image_dir)
        if role == "Test":
            self.images, self.labels = read_test(image_txt, image_dir)
        self.role = role

        #basic transformations made for all of the data
        self.transform = transform

        #eventual transformations made to train data to diversify it
        self.enhancements = enhancements

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        #getter for image with it's own label from dataset
        current_image = self.images[index]
        current_image_label = self.labels[index]

        #apply transformations
        current_image_transformed = self.transform(current_image)
        current_image_enhanced = current_image_transformed
        if self.role == 'Train':
            current_image_enhanced = self.enhancements(current_image_transformed)

        #returned transformed images
        return current_image_enhanced, current_image_label



def create_dataloader(data_dir, image_txt, image_dir, role, shuffle, batch_size):
    # todo hyperparameter here
    transmorms = torch_transmorms.Compose([
        torch_transmorms.ToTensor(),
        torch_transmorms.Normalize(
            mean=(0.5, 0.5, 0.5),
            std=(0.2, 0.1, 0.3)
        )
    ])

    p1 = random.randint(3, 7) * 0.1
    p2 = random.randint(3, 7) * 0.1
    p3 = random.randint(3, 7) * 0.1

    enhancements = torch_transmorms.Compose([
        torch_transmorms.RandomHorizontalFlip(p=p1),
        # torch_transmorms.RandomVerticalFlip(p=p2),
    ])

    #create dataset and dataloader
    dataset = DeepHallucinationDataset(data_dir, image_txt, image_dir, role, transmorms, enhancements)
    dataloader = DataLoader(dataset, batch_size, shuffle)

    return dataloader


def read_images(file_txt, image_dir):
    # reads filenames and labels
    file = open(file_txt)
    file_all_lines = file.readlines()
    file_all_lines.sort()

    # keeps the filenames and labels in the txt
    filenames_and_labels = []
    for line in file_all_lines[1:]:
        filenames_and_labels.append((line.split(',')[0], line.split(',')[1].rstrip('\n')))

    # makes a dictionary to images, labels and filenames sorted sorted
    images_dict = {}
    image_filenames = next(walk(image_dir), (None, None, []))[2]
    for filename in image_filenames:
        filename_label = [item for item in filenames_and_labels if item[0] == filename]

        # enters if only if the current filename is in filenames_and_labels
        if filename_label:
            img = im.imread(f'{image_dir}/{filename}')
            im_pil = Im.fromarray(img)
            images_dict[filename] = (tuple(filename_label)[0][0], tuple(filename_label)[0][1], im_pil)

    images = []
    labels = []
    for pair in list(images_dict.values()):
        images.append(pair[2])
        labels.append(int(pair[1]))

    return images, labels

def read_test(file_txt, image_dir,):
    #reads filenames
    file = open(file_txt)
    file_all_lines = file.readlines()
    file_all_lines.sort()
    test_filenames = []
    for line in file_all_lines[1:]:
        test_filenames.append(line.rstrip('\n'))

        #makes a directory to keep images and filenames organized
    test_images_dict = {}
    test_filenames = next(walk(image_dir), (None, None, []))[2]
    for filename in test_filenames:
        #read with imageio
        img = im.imread(f'{image_dir}/{filename}')
        #convert to PIL
        im_pil = Im.fromarray(img)
        test_images_dict[filename] = (0, im_pil)
    test_images = []
    test_names = []
    for pair in list(test_images_dict.items()):
        test_images.append(pair[1][1])
        test_names.append(pair[0])
    #test_names will be read in labels later on
    return test_images, test_names

if __name__ == '__main__':
    create_dataloader('../data','../data/test.txt', '../data/test',role="Test",shuffle=False, batch_size=16)
