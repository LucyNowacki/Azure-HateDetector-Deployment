# Define ModelSaverReader class
import os
import torch

class ModelSaverReader:
    def __init__(self, save_directory):
        self.save_directory = save_directory
        if not os.path.exists(save_directory):
            os.makedirs(save_directory)
            os.chmod(save_directory, 0o777)  # Set directory permissions to be accessible

    def save_model(self, model, model_name):
        save_path = os.path.join(self.save_directory, f"{model_name}.pth")
        torch.save(model.state_dict(), save_path)
        print(f"Model saved to {save_path}")

    def save_checkpoint(self, model, optimizer, scheduler, scaler, step, epoch, context_length, model_name):
        save_path = os.path.join(self.save_directory, f"{model_name}_checkpoint.pth")
        checkpoint = {
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scaler_state_dict': scaler.state_dict(),
            'step': step,
            'epoch': epoch,
            'context_length': context_length
        }
        if scheduler is not None:
            checkpoint['scheduler_state_dict'] = scheduler.state_dict()
        torch.save(checkpoint, save_path)
        print(f"Checkpoint saved to {save_path}")

    def load_checkpoint(self, model, optimizer, scaler, model_name, scheduler=None):
        load_path = os.path.join(self.save_directory, f"{model_name}_checkpoint.pth")
        if not os.path.exists(load_path):
            raise FileNotFoundError(f"No checkpoint found at {load_path}")
        checkpoint = torch.load(load_path)
        model.load_state_dict(checkpoint['model_state_dict'], strict=False)  # Use strict=False to allow partial loading

        # Ensure optimizer parameter groups match before loading the state
        saved_optimizer_state = checkpoint['optimizer_state_dict']
        if len(optimizer.param_groups) != len(saved_optimizer_state['param_groups']):
            optimizer.param_groups = saved_optimizer_state['param_groups']
        optimizer.load_state_dict(saved_optimizer_state)

        scaler.load_state_dict(checkpoint['scaler_state_dict'])
        if 'scheduler_state_dict' in checkpoint and scheduler is not None:
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        step = checkpoint['step']
        epoch = checkpoint['epoch']
        context_length = checkpoint['context_length']
        return model, optimizer, scaler, step, epoch, context_length


    def get_model_name(self):
        model_name = input("Enter the model name: ")
        return model_name

    def save_current_model_name(self, model_name):
        with open("current_model_name.txt", "w") as file:
            file.write(model_name)

    def load_current_model_name(self):
        if os.path.exists("current_model_name.txt"):
            with open("current_model_name.txt", "r") as file:
                return file.read().strip()
        return None

    def load_model(self, model_class, model_name, config, map_location=None):
        load_path = os.path.join(self.save_directory, f"{model_name}.pth")
        if not os.path.exists(load_path):
            raise FileNotFoundError(f"No model found at {load_path}")
        model = model_class(config)
        model.load_state_dict(torch.load(load_path, map_location=map_location))
        print(f"Model loaded from {load_path}")
        return model

from torch.utils.data import Dataset, DataLoader, random_split

class TweetDatasetToken(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length):
        self.texts = [str(text) for text in texts if text is not None]  # Ensure all texts are strings and non-null
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.data = self._tokenize_texts()

    def _tokenize_texts(self):
        # Tokenize the texts based on the current max_length
        tokenized_texts = []
        for text in self.texts:
            inputs = self.tokenizer.encode_plus(
                text,
                add_special_tokens=True,
                max_length=self.max_length,
                padding='max_length',
                truncation=True,
                return_tensors='pt'
            )
            input_ids = inputs['input_ids'].squeeze()
            if input_ids.size(0) == 0:  # Check if input_ids are empty
                input_ids = torch.zeros(self.max_length, dtype=torch.long)
            tokenized_texts.append(input_ids)
        return tokenized_texts

    def set_max_length(self, max_length):
        self.max_length = max_length
        self.data = self._tokenize_texts()  # Re-tokenize texts with the new max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        input_ids = self.data[idx]
        # Shift the input_ids to create labels (next token prediction)
        labels = torch.cat((input_ids[1:], torch.tensor([self.tokenizer.pad_token_id], dtype=torch.long)))
        return input_ids, labels
    

class TweetDatasetBinary(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length):
        self.texts = [str(text) for text in texts if text is not None]  # Ensure all texts are strings and non-null
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.data = self._tokenize_texts()

    def _tokenize_texts(self):
        # Tokenize the texts based on the current max_length
        tokenized_texts = []
        for text in self.texts:
            inputs = self.tokenizer.encode_plus(
                text,
                add_special_tokens=True,
                max_length=self.max_length,
                padding='max_length',
                truncation=True,
                return_tensors='pt'
            )
            input_ids = inputs['input_ids'].squeeze()
            if input_ids.size(0) == 0:  # Check if input_ids are empty
                input_ids = torch.zeros(self.max_length, dtype=torch.long)
            tokenized_texts.append(input_ids)
        return tokenized_texts

    def set_max_length(self, max_length):
        self.max_length = max_length
        self.data = self._tokenize_texts()  # Re-tokenize texts with the new max_length
        print(f"Dataset context length updated to {self.max_length}")

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        input_ids = self.data[idx]
        label = torch.tensor(self.labels[idx], dtype=torch.float).unsqueeze(-1)  # Ensure label is of float type and reshaped
        # Debugging print statements
        #print(f"Index: {idx}, Input IDs shape: {input_ids.shape}, Label shape: {label.shape}")
        return input_ids, label
    

class TweetDatasetBianry(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length):
        self.texts = [str(text) for text in texts if text is not None]  # Ensure all texts are strings and non-null
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.data = self._tokenize_texts()

    def _tokenize_texts(self):
        # Tokenize the texts based on the current max_length
        tokenized_texts = []
        for text in self.texts:
            inputs = self.tokenizer.encode_plus(
                text,
                add_special_tokens=True,
                max_length=self.max_length,
                padding='max_length',
                truncation=True,
                return_tensors='pt'
            )
            input_ids = inputs['input_ids'].squeeze()
            if input_ids.size(0) == 0:  # Check if input_ids are empty
                input_ids = torch.zeros(self.max_length, dtype=torch.long)
            tokenized_texts.append(input_ids)
        return tokenized_texts

    def set_max_length(self, max_length):
        self.max_length = max_length
        self.data = self._tokenize_texts()  # Re-tokenize texts with the new max_length
        print(f"Dataset context length updated to {self.max_length}")

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        input_ids = self.data[idx]
        label = torch.tensor(self.labels[idx], dtype=torch.float).unsqueeze(-1)  # Ensure label is of float type and reshaped
        # Debugging print statements
        #print(f"Index: {idx}, Input IDs shape: {input_ids.shape}, Label shape: {label.shape}")
        return input_ids, label
    

def signal_handler(sig, frame):
    global interrupted
    interrupted = True