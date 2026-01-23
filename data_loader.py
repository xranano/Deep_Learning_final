import os
import torch
import random
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset
from PIL import Image
import torchvision.transforms as transforms

class Vocabulary:
    def __init__(self, freq_threshold):
        self.itos = {0: "<PAD>", 1: "<SOS>", 2: "<EOS>", 3: "<UNK>"}
        self.stoi = {"<PAD>": 0, "<SOS>": 1, "<EOS>": 2, "<UNK>": 3}
        self.freq_threshold = freq_threshold

    def __len__(self):
        return len(self.itos)

    @staticmethod
    def tokenizer_eng(text):
        return [tok.lower() for tok in text.replace(".", " .").replace(",", " ,").split()]

    def build_vocabulary(self, sentence_list):
        frequencies = {}
        idx = 4

        for sentence in sentence_list:
            for word in self.tokenizer_eng(sentence):
                if word not in frequencies:
                    frequencies[word] = 1
                else:
                    frequencies[word] += 1

                if frequencies[word] == self.freq_threshold:
                    self.stoi[word] = idx
                    self.itos[idx] = word
                    idx += 1

    def numericalize(self, text):
        tokenized_text = self.tokenizer_eng(text)

        return [
            self.stoi[token] if token in self.stoi else self.stoi["<UNK>"]
            for token in tokenized_text
        ]


class FlickrDataset(Dataset):
    def __init__(self, root_dir, imgs, captions, vocab, transform=None):
        self.root_dir = root_dir
        self.imgs = imgs
        self.captions = captions
        self.vocab = vocab
        self.transform = transform

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, index):
        caption = self.captions[index]
        img_id = self.imgs[index]
        img_path = os.path.join(self.root_dir, img_id)
        
        image = Image.open(img_path).convert("RGB")

        if self.transform is not None:
            image = self.transform(image)

        numericalized_caption = [self.vocab.stoi["<SOS>"]]
        numericalized_caption += self.vocab.numericalize(caption)
        numericalized_caption.append(self.vocab.stoi["<EOS>"])

        return image, torch.tensor(numericalized_caption)


class MyCollate:
    def __init__(self, pad_idx):
        self.pad_idx = pad_idx

    def __call__(self, batch):
        imgs = [item[0].unsqueeze(0) for item in batch]
        imgs = torch.cat(imgs, dim=0)
        targets = [item[1] for item in batch]
        targets = pad_sequence(targets, batch_first=False, padding_value=self.pad_idx)

        return imgs, targets


def get_transforms(train=True):
    if train:
        return transforms.Compose([
            transforms.Resize(256),
            transforms.RandomCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ])
    else:
        return transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ])

def get_loaders(
    root_folder,
    annotation_file,
    transform=None,
    train_transform=None,
    val_transform=None,
    batch_size=32,
    num_workers=2,
    shuffle=True,
    pin_memory=True,
    test_size=0.1,
    val_size=0.1,
    freq_threshold=5
):
    # Handle transform arguments
    if train_transform is None:
        train_transform = transform
        
    if val_transform is None:
        val_transform = transform
        
    # If still None, we could default to get_transforms() but let's assume caller provides
    
    all_imgs = []
    all_captions = []
    
    with open(annotation_file, 'r', encoding='utf-8') as f:
        lines = f.readlines()
        if lines and "image,caption" in lines[0]:
            lines = lines[1:]
        
        for line in lines:
            parts = line.strip().split(',', 1) 
            if len(parts) == 2:
                all_imgs.append(parts[0])
                all_captions.append(parts[1])

    unique_imgs = list(set(all_imgs))
    random.seed(42)
    random.shuffle(unique_imgs)
    
    total_imgs = len(unique_imgs)
    v_count = int(total_imgs * val_size)
    t_count = int(total_imgs * test_size)
    train_count = total_imgs - v_count - t_count
    
    train_img_ids = set(unique_imgs[:train_count])
    val_img_ids = set(unique_imgs[train_count:train_count+v_count])
    test_img_ids = set(unique_imgs[train_count+v_count:])
    
    train_imgs, train_caps = [], []
    val_imgs, val_caps = [], []
    test_imgs, test_caps = [], []
    
    for img, cap in zip(all_imgs, all_captions):
        if img in train_img_ids:
            train_imgs.append(img)
            train_caps.append(cap)
        elif img in val_img_ids:
            val_imgs.append(img)
            val_caps.append(cap)
        elif img in test_img_ids:
            test_imgs.append(img)
            test_caps.append(cap)
            
    vocab = Vocabulary(freq_threshold)
    vocab.build_vocabulary(train_caps)
    
    # Use specific transforms
    train_dataset = FlickrDataset(root_folder, train_imgs, train_caps, vocab, transform=train_transform)
    val_dataset = FlickrDataset(root_folder, val_imgs, val_caps, vocab, transform=val_transform)
    test_dataset = FlickrDataset(root_folder, test_imgs, test_caps, vocab, transform=val_transform)
    
    pad_idx = vocab.stoi["<PAD>"]

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, num_workers=num_workers, 
        shuffle=shuffle, pin_memory=pin_memory, collate_fn=MyCollate(pad_idx),
        persistent_workers=True 
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, num_workers=num_workers, 
        shuffle=False, pin_memory=pin_memory, collate_fn=MyCollate(pad_idx),
        persistent_workers=True
    )
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, num_workers=num_workers, 
        shuffle=False, pin_memory=pin_memory, collate_fn=MyCollate(pad_idx),
        persistent_workers=True
    )

    return train_loader, val_loader, test_loader, vocab


if __name__ == "__main__":
    pass
