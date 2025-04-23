import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import os

# ----------------------------
# Model Definition
# ----------------------------
class BLSTM_NER(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, linear_dim, num_tags, dropout=0.33):
        """
        Args:
            vocab_size: Vocabulary size.
            embedding_dim: Embedding layer dimension (100).
            hidden_dim: LSTM hidden layer dimension (256).
            linear_dim: Linear layer output dimension (128).
            num_tags: Number of NER tags.
            dropout: Dropout rate for LSTM layer.
        """
        super(BLSTM_NER, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, num_layers=1,
                            bidirectional=True, dropout=dropout, batch_first=True)
        self.linear = nn.Linear(hidden_dim * 2, linear_dim)
        self.elu = nn.ELU()
        self.classifier = nn.Linear(linear_dim, num_tags)

    def forward(self, x):
        embeds = self.embedding(x)
        lstm_out, _ = self.lstm(embeds)
        linear_out = self.linear(lstm_out)
        elu_out = self.elu(linear_out)
        logits = self.classifier(elu_out)
        return logits

# ----------------------------
# Data Loading
# ----------------------------
class NERDataset(Dataset):
    def __init__(self, file_path, word_to_idx=None, tag_to_idx=None, build_vocab=False):
        """
        Args:
            file_path: Path to dataset (sentences separated by blank lines).
            word_to_idx, tag_to_idx: Pre-built mappings if available, else use build_vocab.
            build_vocab: Flag to build vocabulary and tag mappings from data.
        """
        self.sentences = []
        self.tags = []
        with open(file_path, 'r', encoding='utf-8') as f:
            sentence_words = []
            sentence_tags = []
            for line in f:
                line = line.strip()
                if line == "":
                    if sentence_words:
                        self.sentences.append(sentence_words)
                        self.tags.append(sentence_tags)
                        sentence_words = []
                        sentence_tags = []
                else:
                    parts = line.split()
                    if len(parts) >= 3:
                        sentence_words.append(parts[1])
                        sentence_tags.append(parts[2])
            if sentence_words:
                self.sentences.append(sentence_words)
                self.tags.append(sentence_tags)

        if build_vocab:
            self.build_vocab()
        else:
            self.word_to_idx = word_to_idx
            self.tag_to_idx = tag_to_idx

    def build_vocab(self):
        words = {word for sent in self.sentences for word in sent}
        tags = {tag for tag_seq in self.tags for tag in tag_seq}
        self.word_to_idx = {word: i + 2 for i, word in enumerate(sorted(words))}
        self.word_to_idx["<PAD>"] = 0
        self.word_to_idx["<UNK>"] = 1
        self.tag_to_idx = {tag: i for i, tag in enumerate(sorted(tags))}

    def __len__(self):
        return len(self.sentences)

    def __getitem__(self, idx):
        word_indices = [self.word_to_idx.get(w, self.word_to_idx["<UNK>"]) for w in self.sentences[idx]]
        tag_indices = [self.tag_to_idx[t] for t in self.tags[idx]]
        return torch.tensor(word_indices, dtype=torch.long), torch.tensor(tag_indices, dtype=torch.long)

def pad_collate(batch):
    words, tags = zip(*batch)
    max_len = max(len(seq) for seq in words)
    padded_words = [torch.cat([w, torch.zeros(max_len - len(w), dtype=torch.long)]) for w in words]
    padded_tags = [torch.cat([t, torch.full((max_len - len(t),), -100, dtype=torch.long)]) for t in tags]
    return torch.stack(padded_words), torch.stack(padded_tags), [len(seq) for seq in words]

# ----------------------------
# Training Function
# ----------------------------
def train_model(model, train_loader, optimizer, scheduler, criterion, device, num_epochs=10):
    model.train()
    for epoch in range(num_epochs):
        epoch_loss = 0.0
        for words, tags, lengths in train_loader:
            words, tags = words.to(device), tags.to(device)
            optimizer.zero_grad()
            outputs = model(words).view(-1, model.classifier.out_features)
            loss = criterion(outputs, tags.view(-1))
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        scheduler.step(epoch_loss / len(train_loader))
        print(f"Epoch {epoch+1} Loss: {epoch_loss / len(train_loader):.4f}")

# ----------------------------
# Generate Test Predictions
# ----------------------------
class NERTestDataset(Dataset):
    def __init__(self, file_path, word_to_idx):
        self.word_to_idx = word_to_idx
        self.sentences = []
        self.raw_lines = []
        with open(file_path, 'r', encoding='utf-8') as f:
            sentence_words, sentence_lines = [], []
            for line in f:
                if line.strip() == "":
                    if sentence_words:
                        self.sentences.append(sentence_words)
                        self.raw_lines.append(sentence_lines)
                        sentence_words, sentence_lines = [], []
                else:
                    parts = line.strip().split()
                    if len(parts) >= 2:
                        sentence_words.append(parts[1])
                        sentence_lines.append((parts[0], parts[1]))
            if sentence_words:
                self.sentences.append(sentence_words)
                self.raw_lines.append(sentence_lines)
    
    def __len__(self):
        return len(self.sentences)
    
    def __getitem__(self, idx):
        return torch.tensor([self.word_to_idx.get(w, self.word_to_idx["<UNK>"]) for w in self.sentences[idx]], dtype=torch.long), self.raw_lines[idx]

def pad_collate_test(batch):
    words, raw_lines = zip(*batch)
    max_len = max(len(seq) for seq in words)
    padded_words = [torch.cat([w, torch.zeros(max_len - len(w), dtype=torch.long)]) for w in words]
    return torch.stack(padded_words), [len(seq) for seq in words], raw_lines

def generate_test_predictions(model, test_dataset, device, output_file):
    model.eval()
    idx_to_tag = {0: "O", 1: "B-PER", 2: "I-PER", 3: "B-LOC", 4: "I-LOC", 5: "B-ORG", 6: "I-ORG"}
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, collate_fn=pad_collate_test)
    with open(output_file, "w", encoding="utf-8") as outf:
        with torch.no_grad():
            for words, lengths, raw_lines in test_loader:
                predictions = torch.argmax(model(words.to(device)), dim=-1).squeeze(0)[:lengths[0]].cpu().tolist()
                for ((idx, word), pred_idx) in zip(raw_lines[0], predictions):
                    outf.write(f"{idx} {word} {idx_to_tag.get(pred_idx, 'O')}\n")
                outf.write("\n")
    print(f"Predictions saved to {output_file}")

# ----------------------------
# Main function
# ----------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load training dataset and build vocabulary/tag mapping
train_dataset = NERDataset("data/train", build_vocab=True)
# Load development dataset using the same vocabulary/tag mapping as training data
dev_dataset = NERDataset("data/dev", word_to_idx=train_dataset.word_to_idx, tag_to_idx=train_dataset.tag_to_idx)

train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True, collate_fn=pad_collate)

vocab_size = len(train_dataset.word_to_idx)
num_tags = len(train_dataset.tag_to_idx)

model = BLSTM_NER(vocab_size=vocab_size,
                  embedding_dim=100,
                  hidden_dim=256,
                  linear_dim=128,
                  num_tags=num_tags,
                  dropout=0.33)
model.to(device)

# Use CrossEntropyLoss and ignore padded labels (-100)
criterion = nn.CrossEntropyLoss(ignore_index=-100)

# Use SGD with momentum optimization
optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)

# Use ReduceLROnPlateau scheduler for adaptive learning rate
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3, verbose=True)

num_epochs = 10
train_model(model, train_loader, optimizer, scheduler, criterion, device, num_epochs)
torch.save(model.state_dict(), "blstm1.pt")

# Generate prediction file for evaluation
dev_pred_file = "dev1.out"
generate_prediction_file(model, dev_dataset, device, dev_pred_file)

# Call the provided eval.py script for evaluation (gold file is data/dev)
cmd = f"python eval.py -p {dev_pred_file} -g data/dev"
print("Calling evaluation script...")
os.system(cmd)

test_dataset = NERTestDataset("data/test", train_dataset.word_to_idx)
generate_test_predictions(model, test_dataset, device, output_file="test1.out")


model.load_state_dict(torch.load("blstm1.pt"))
additional_epochs = 10

print(f"Starting additional {additional_epochs} epochs training...")
train_model(model, train_loader, optimizer, scheduler, criterion, device, num_epochs=additional_epochs)
torch.save(model.state_dict(), "blstm1.pt")

# Generate prediction file for evaluation
generate_prediction_file(model, dev_dataset, device, dev_pred_file)

# Call the provided eval.py script for evaluation (gold file is data/dev)
cmd = f"python eval.py -p {dev_pred_file} -g data/dev"
print("Calling evaluation script...")
os.system(cmd)

generate_test_predictions(model, test_dataset, device, output_file="test1.out")

# ----------------------------
# Function to load GloVe embeddings
# ----------------------------
def load_glove_embeddings(glove_path, word_to_idx, embedding_dim=100):
    """
    Loads GloVe embeddings and creates an embedding matrix.
    For each word in the vocabulary, its lower-case version is used to look up in GloVe.
    Words not found in GloVe are randomly initialized.
    """
    embeddings = np.random.randn(len(word_to_idx), embedding_dim).astype(np.float32)
    glove_dict = {}
    with gzip.open(glove_path, 'rt', encoding='utf-8') as f:
        for line in f:
            tokens = line.strip().split()
            if len(tokens) == embedding_dim + 1:
                word = tokens[0]
                vector = np.array(tokens[1:], dtype=np.float32)
                glove_dict[word] = vector
    # Initialize embeddings: use lower-case lookup for each word in our vocabulary.
    for word, idx in word_to_idx.items():
        glove_vector = glove_dict.get(word.lower())
        if glove_vector is not None:
            embeddings[idx] = glove_vector
    return torch.tensor(embeddings)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load training dataset and build vocabulary/tag mapping.
train_dataset = NERDataset("data/train", build_vocab=True)
# Load development dataset using the same vocabulary/tag mapping as training data.
dev_dataset = NERDataset("data/dev", word_to_idx=train_dataset.word_to_idx, tag_to_idx=train_dataset.tag_to_idx)

train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True, collate_fn=pad_collate)

vocab_size = len(train_dataset.word_to_idx)
num_tags = len(train_dataset.tag_to_idx)

model = BLSTM_NER(vocab_size=vocab_size,
                  embedding_dim=100,
                  hidden_dim=256,
                  linear_dim=128,
                  num_tags=num_tags,
                  dropout=0.33)
model.to(device)

# Load GloVe embeddings and initialize the embedding layer.
glove_path = "glove.6B.100d.gz"
glove_weights = load_glove_embeddings(glove_path, train_dataset.word_to_idx, embedding_dim=100)
model.embedding.weight.data.copy_(glove_weights)

# Use CrossEntropyLoss and ignore padded labels (-100).
criterion = nn.CrossEntropyLoss(ignore_index=-100)

# Use SGD with momentum.
optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)

# Use ReduceLROnPlateau scheduler for adaptive learning rate.
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3, verbose=True)

num_epochs = 10
train_model(model, train_loader, optimizer, scheduler, criterion, device, num_epochs)
torch.save(model.state_dict(), "blstm2.pt")

# Generate prediction file for evaluation
dev_pred_file = "dev2.out"
generate_prediction_file(model, dev_dataset, device, dev_pred_file)

# Call the provided eval.py script for evaluation (gold file is data/dev)
cmd = f"python eval.py -p {dev_pred_file} -g data/dev"
print("Calling evaluation script...")
os.system(cmd)

test_dataset = NERTestDataset("data/test", train_dataset.word_to_idx)
generate_test_predictions(model, test_dataset, device, output_file="test2.out")


# optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
model.load_state_dict(torch.load("blstm2.pt"))
additional_epochs = 5

print(f"Starting additional {additional_epochs} epochs training...")
train_model(model, train_loader, optimizer, scheduler, criterion, device, num_epochs=additional_epochs)
torch.save(model.state_dict(), "blstm2.pt")

# Generate prediction file for evaluation
generate_prediction_file(model, dev_dataset, device, dev_pred_file)

# Call the provided eval.py script for evaluation (gold file is data/dev)
cmd = f"python eval.py -p {dev_pred_file} -g data/dev"
print("Calling evaluation script...")
os.system(cmd)

generate_test_predictions(model, test_dataset, device, output_file="test2.out")


# ----------------------------
# Dataset with Character-level Information
# ----------------------------
class NERDatasetWithChar(Dataset):
    def __init__(self, file_path, word_to_idx=None, tag_to_idx=None, char_to_idx=None, build_vocab=False):
        """
        Args:
            file_path: Path to the data file.
                      For training/dev: each non-blank line has "index word tag".
                      For test: each non-blank line has "index word".
                      Sentences are separated by blank lines.
            word_to_idx: Pre-built word vocabulary mapping (if available).
            tag_to_idx: Pre-built tag mapping (if available; not used for test).
            char_to_idx: Pre-built character vocabulary mapping (if available).
            build_vocab: Whether to build vocabularies from this file (usually on training data).
        """
        self.sentences = []  # list of list of words
        self.tags = []       # list of list of tags; for test files, this will be empty lists
        self.indices = []    # list of list of (index, word) tuples (to preserve original order)
        self.has_tags = True

        with open(file_path, 'r', encoding='utf-8') as f:
            sent_words = []
            sent_tags = []
            sent_idx_word = []
            for line in f:
                line = line.strip()
                if line == "":
                    if sent_words:
                        self.sentences.append(sent_words)
                        self.tags.append(sent_tags)
                        self.indices.append(sent_idx_word)
                        sent_words = []
                        sent_tags = []
                        sent_idx_word = []
                    continue
                tokens = line.split()
                # Determine file type by number of tokens
                if len(tokens) == 3:
                    # training/dev file: index word tag
                    idx, word, tag = tokens
                    sent_words.append(word)
                    sent_tags.append(tag)
                    sent_idx_word.append((idx, word))
                elif len(tokens) == 2:
                    # test file: index word (no tag)
                    idx, word = tokens
                    sent_words.append(word)
                    sent_tags.append(None)
                    sent_idx_word.append((idx, word))
                    self.has_tags = False
                else:
                    continue
            if sent_words:
                self.sentences.append(sent_words)
                self.tags.append(sent_tags)
                self.indices.append(sent_idx_word)

        # If building vocabularies, do it now.
        if build_vocab:
            self.build_vocab()
        else:
            self.word_to_idx = word_to_idx
            self.tag_to_idx = tag_to_idx
            self.char_to_idx = char_to_idx

    def build_vocab(self):
        # Build word vocabulary (case-sensitive)
        words = set()
        tags = set()
        chars = set()
        for sent, tag_seq in zip(self.sentences, self.tags):
            for word in sent:
                words.add(word)
                for ch in word:
                    chars.add(ch)
            # Only add tags if available (for training/dev)
            if self.has_tags:
                for tag in tag_seq:
                    tags.add(tag)
        # Reserve indices: 0 for <PAD>, 1 for <UNK>
        self.word_to_idx = {word: i+2 for i, word in enumerate(sorted(words))}
        self.word_to_idx["<PAD>"] = 0
        self.word_to_idx["<UNK>"] = 1
        if self.has_tags:
            self.tag_to_idx = {tag: i for i, tag in enumerate(sorted(tags))}
        else:
            self.tag_to_idx = None
        # Build character vocabulary similarly (0 for <PAD>, 1 for <UNK>)
        self.char_to_idx = {ch: i+2 for i, ch in enumerate(sorted(chars))}
        self.char_to_idx["<PAD>"] = 0
        self.char_to_idx["<UNK>"] = 1

    def __len__(self):
        return len(self.sentences)

    def __getitem__(self, idx):
        # Convert words to indices and also produce character indices for each word.
        words = self.sentences[idx]
        word_indices = [self.word_to_idx.get(w, self.word_to_idx["<UNK>"]) for w in words]
        # For each word, convert characters to indices
        char_indices = []
        for word in words:
            ch_idx = [self.char_to_idx.get(ch, self.char_to_idx["<UNK>"]) for ch in word]
            char_indices.append(ch_idx)
        if self.has_tags:
            tag_seq = self.tags[idx]
            tag_indices = [self.tag_to_idx[t] for t in tag_seq]
        else:
            tag_indices = None
        return torch.tensor(word_indices, dtype=torch.long), \
               (torch.tensor(tag_indices, dtype=torch.long) if tag_indices is not None else None), \
               char_indices

# ----------------------------
# Collate function for batching with character sequences
# ----------------------------
def pad_collate_with_char(batch):
    """
    Pads word sequences, tag sequences, and character sequences.
    Returns:
      padded_words: LongTensor of shape (batch_size, max_seq_len)
      padded_tags: LongTensor of shape (batch_size, max_seq_len) or None if tags not available
      lengths: list of original sentence lengths
      padded_chars: LongTensor of shape (batch_size, max_seq_len, max_word_len)
    """
    batch_size = len(batch)
    word_seqs = [item[0] for item in batch]  # each is a tensor of word indices
    tag_seqs = [item[1] for item in batch]     # each is a tensor of tag indices or None
    char_seqs = [item[2] for item in batch]      # list of list of lists

    lengths = [len(seq) for seq in word_seqs]
    max_seq_len = max(lengths)
    
    # Pad word sequences
    padded_words = []
    for seq in word_seqs:
        pad_size = max_seq_len - seq.size(0)
        if pad_size > 0:
            padded_seq = torch.cat([seq, torch.zeros(pad_size, dtype=torch.long)])
        else:
            padded_seq = seq
        padded_words.append(padded_seq)
    padded_words = torch.stack(padded_words)
    
    # Pad tag sequences if available
    if tag_seqs[0] is not None:
        padded_tags = []
        for seq in tag_seqs:
            pad_size = max_seq_len - seq.size(0)
            if pad_size > 0:
                padded_seq = torch.cat([seq, torch.full((pad_size,), -100, dtype=torch.long)])
            else:
                padded_seq = seq
            padded_tags.append(padded_seq)
        padded_tags = torch.stack(padded_tags)
    else:
        padded_tags = None

    # For character sequences: first determine maximum word length in the batch
    max_word_len = 0
    for sent in char_seqs:
        for word in sent:
            if len(word) > max_word_len:
                max_word_len = len(word)
    # Pad each word's character list, and also pad sentences with fewer words.
    padded_chars = []
    for sent in char_seqs:
        # Pad each word in the sentence
        padded_sent = []
        for word in sent:
            pad_len = max_word_len - len(word)
            padded_word = word + [0]*pad_len
            padded_sent.append(padded_word)
        # If sentence length < max_seq_len, add padding for missing words.
        for _ in range(max_seq_len - len(sent)):
            padded_sent.append([0]*max_word_len)
        padded_chars.append(padded_sent)
    padded_chars = torch.tensor(padded_chars, dtype=torch.long)  # shape: (batch_size, max_seq_len, max_word_len)
    
    return padded_words, padded_tags, lengths, padded_chars

# ----------------------------
# LSTM-CNN NER Model with Character-level CNN
# ----------------------------
class LSTM_CNN_NER(nn.Module):
    def __init__(self, vocab_size, char_vocab_size, embedding_dim, char_embedding_dim,
                 char_cnn_out_dim, hidden_dim, linear_dim, num_tags, dropout=0.33):
        """
        Args:
            vocab_size: Size of the word vocabulary.
            char_vocab_size: Size of the character vocabulary.
            embedding_dim: Dimension of the word embeddings (e.g., 100).
            char_embedding_dim: Dimension of the character embeddings (set to 30).
            char_cnn_out_dim: Output dimension of the character-level CNN (e.g., 50).
            hidden_dim: Hidden dimension of the BLSTM (256).
            linear_dim: Dimension of the intermediate Linear layer (128).
            num_tags: Number of NER tags.
            dropout: Dropout rate for the BLSTM.
        """
        super(LSTM_CNN_NER, self).__init__()
        # Word embedding layer (will be initialized with GloVe later)
        self.word_embedding = nn.Embedding(vocab_size, embedding_dim)
        # Character embedding layer (randomly initialized, padding index=0)
        self.char_embedding = nn.Embedding(char_vocab_size, char_embedding_dim, padding_idx=0)
        # CNN module for characters: use 1D convolution with kernel size 3 and padding=1
        self.char_cnn = nn.Conv1d(in_channels=char_embedding_dim, out_channels=char_cnn_out_dim, kernel_size=3, padding=1)
        # BLSTM layer: input dimension is word_embedding + char_cnn output (100+char_cnn_out_dim)
        self.lstm = nn.LSTM(embedding_dim + char_cnn_out_dim, hidden_dim, num_layers=1,
                            bidirectional=True, dropout=dropout, batch_first=True)
        self.linear = nn.Linear(hidden_dim * 2, linear_dim)
        self.elu = nn.ELU()
        self.classifier = nn.Linear(linear_dim, num_tags)

    def forward(self, word_inputs, char_inputs):
        """
        Args:
            word_inputs: LongTensor of shape (batch_size, seq_len) containing word indices.
            char_inputs: LongTensor of shape (batch_size, seq_len, max_word_len) containing character indices.
        Returns:
            logits: Tensor of shape (batch_size, seq_len, num_tags)
        """
        batch_size, seq_len = word_inputs.size()
        # Word embeddings: (batch_size, seq_len, embedding_dim)
        word_embeds = self.word_embedding(word_inputs)
        
        # Process character-level inputs
        # char_inputs shape: (batch_size, seq_len, max_word_len)
        # Reshape to (batch_size*seq_len, max_word_len)
        char_inputs = char_inputs.view(-1, char_inputs.size(2))
        # Get char embeddings: (batch_size*seq_len, max_word_len, char_embedding_dim)
        char_embeds = self.char_embedding(char_inputs)
        # Permute to (batch_size*seq_len, char_embedding_dim, max_word_len) for CNN
        char_embeds = char_embeds.permute(0, 2, 1)
        # Apply CNN: output shape -> (batch_size*seq_len, char_cnn_out_dim, max_word_len)
        char_cnn_out = self.char_cnn(char_embeds)
        # Apply ReLU
        char_cnn_out = torch.relu(char_cnn_out)
        # Apply max pooling over time dimension (kernel = entire sequence length)
        char_rep, _ = torch.max(char_cnn_out, dim=2)  # shape: (batch_size*seq_len, char_cnn_out_dim)
        # Reshape back to (batch_size, seq_len, char_cnn_out_dim)
        char_rep = char_rep.view(batch_size, seq_len, -1)
        
        # Concatenate word embeddings and character-level representations: (batch_size, seq_len, embedding_dim + char_cnn_out_dim)
        combined = torch.cat([word_embeds, char_rep], dim=2)
        
        # BLSTM layer
        lstm_out, _ = self.lstm(combined)  # shape: (batch_size, seq_len, hidden_dim*2)
        # Linear, ELU, and classifier layers
        linear_out = self.linear(lstm_out)  # shape: (batch_size, seq_len, linear_dim)
        elu_out = self.elu(linear_out)
        logits = self.classifier(elu_out)   # shape: (batch_size, seq_len, num_tags)
        return logits

# ----------------------------
# Training function
# ----------------------------
def train_model(model, train_loader, optimizer, scheduler, criterion, device, num_epochs=10):
    model.train()
    for epoch in range(num_epochs):
        epoch_loss = 0.0
        for word_inputs, tag_inputs, lengths, char_inputs in train_loader:
            word_inputs = word_inputs.to(device)
            char_inputs = char_inputs.to(device)
            tag_inputs = tag_inputs.to(device)
            
            optimizer.zero_grad()
            outputs = model(word_inputs, char_inputs)  # (batch_size, seq_len, num_tags)
            outputs = outputs.view(-1, outputs.size(-1))
            tag_inputs = tag_inputs.view(-1)
            loss = criterion(outputs, tag_inputs)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        
        avg_loss = epoch_loss / len(train_loader)
        # Update scheduler based on the average epoch loss
        scheduler.step(avg_loss)
        print(f"Epoch {epoch+1} Loss: {avg_loss:.4f}")

# ----------------------------
# Generate prediction file (for dev/test) using eval.py
# ----------------------------
def generate_prediction_file(model, dataset, device, pred_file_path):
    """
    Generates a prediction file.
    The output format per line is: index word predicted_tag
    Blank lines separate sentences.
    """
    model.eval()
    # Build mapping from index to tag
    idx_to_tag = {v: k for k, v in dataset.tag_to_idx.items()} if dataset.has_tags else None
    # To store predicted tag sequences for each sentence
    predicted_sentences = []
    
    # Use DataLoader with batch_size=1 to preserve order
    loader = DataLoader(dataset, batch_size=1, shuffle=False, collate_fn=pad_collate_with_char)
    with torch.no_grad():
        for word_inputs, tag_inputs, lengths, char_inputs in loader:
            word_inputs = word_inputs.to(device)
            char_inputs = char_inputs.to(device)
            outputs = model(word_inputs, char_inputs)  # (1, seq_len, num_tags)
            predictions = torch.argmax(outputs, dim=-1).squeeze(0)  # (seq_len,)
            seq_len = lengths[0]
            pred_seq = predictions[:seq_len].cpu().tolist()
            if idx_to_tag is not None:
                pred_tags = [idx_to_tag[idx] for idx in pred_seq]
            else:
                # For test data, if tag mapping not available, output a dummy tag (e.g., "O")
                pred_tags = ["O"] * seq_len
            predicted_sentences.append(pred_tags)
    
    # Write predictions in the original file format (using dataset.indices for index and word)
    with open(pred_file_path, "w", encoding="utf-8") as f:
        for sent_pred, sent_raw in zip(predicted_sentences, dataset.indices):
            for (idx, word), tag in zip(sent_raw, sent_pred):
                f.write(f"{idx} {word} {tag}\n")
            f.write("\n")

# ----------------------------
# Main function
# ----------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ----------------------------
# Load training data and build vocabularies
# ----------------------------
train_file = "data/train"
dev_file = "data/dev"
test_file = "data/test"

train_dataset = NERDatasetWithChar(train_file, build_vocab=True)
# Use the same vocabularies for dev and test
dev_dataset = NERDatasetWithChar(dev_file, word_to_idx=train_dataset.word_to_idx,
                                  tag_to_idx=train_dataset.tag_to_idx,
                                  char_to_idx=train_dataset.char_to_idx, build_vocab=False)
test_dataset = NERDatasetWithChar(test_file, word_to_idx=train_dataset.word_to_idx,
                                   tag_to_idx=train_dataset.tag_to_idx,  # not used in test
                                   char_to_idx=train_dataset.char_to_idx, build_vocab=False)

train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True, collate_fn=pad_collate_with_char)

vocab_size = len(train_dataset.word_to_idx)
char_vocab_size = len(train_dataset.char_to_idx)
num_tags = len(train_dataset.tag_to_idx)
    
# ----------------------------
# Build the LSTM-CNN NER Model
# ----------------------------
model = LSTM_CNN_NER(vocab_size=vocab_size,
                     char_vocab_size=char_vocab_size,
                     embedding_dim=100,
                     char_embedding_dim=30,
                     char_cnn_out_dim=50,
                     hidden_dim=256,
                     linear_dim=128,
                     num_tags=num_tags,
                     dropout=0.33)
model.to(device)

# Initialize word embeddings with GloVe
glove_path = "glove.6B.100d.gz"
glove_weights = load_glove_embeddings(glove_path, train_dataset.word_to_idx, embedding_dim=100)
model.word_embedding.weight.data.copy_(glove_weights)

# Loss, optimizer (SGD with momentum) and scheduler (ReduceLROnPlateau)
criterion = nn.CrossEntropyLoss(ignore_index=-100)
optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3, verbose=True)

num_epochs = 10
train_model(model, train_loader, optimizer, scheduler, criterion, device, num_epochs)

# ----------------------------
# Generate predictions on dev data and evaluate using provided eval.py
# ----------------------------
dev_pred_file = "predictions_bonus.txt"
generate_prediction_file(model, dev_dataset, device, dev_pred_file)
print("Calling evaluation script on dev data...")
os.system(f"python eval.py -p {dev_pred_file} -g {dev_file}")

# ----------------------------
# Generate predictions on test data and save to file "pred"
# ----------------------------
test_pred_file = "pred"
generate_prediction_file(model, test_dataset, device, test_pred_file)
print("Test predictions saved to file 'pred'.")


additional_epochs = 5
print(f"Starting additional {additional_epochs} epochs training...")
train_model(model, train_loader, optimizer, scheduler, criterion, device, additional_epochs)

generate_prediction_file(model, dev_dataset, device, dev_pred_file)
print("Calling evaluation script on dev data...")
os.system(f"python eval.py -p {dev_pred_file} -g {dev_file}")

test_pred_file = "pred"
generate_prediction_file(model, test_dataset, device, test_pred_file)
print("Test predictions saved to file 'pred'.")
