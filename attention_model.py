import torch
import torch.nn as nn
import torchvision.models as models
from torchvision.models import ResNet101_Weights

class EncoderCNN(nn.Module):
    def __init__(self, embed_size, train_CNN=False):
        super(EncoderCNN, self).__init__()
        self.train_CNN = train_CNN
        
        # Load pre-trained ResNet101
        resnet = models.resnet101(weights=ResNet101_Weights.DEFAULT)
        
        # We need the features from the last conv layer, not the final FC layer.
        # ResNet101 structure: (conv1 -> bn1 -> relu -> maxpool -> layer1 -> layer2 -> layer3 -> layer4 -> avgpool -> fc)
        # We want everything up to layer4.
        modules = list(resnet.children())[:-2] # Remove avgpool and fc
        self.resnet = nn.Sequential(*modules)
        
        # Adaptive pooling to ensure fixed output size if image size varies (though we use 224x224 usually)
        # Output will be (batch, 2048, 14, 14) for 224x224
        self.adaptive_pool = nn.AdaptiveAvgPool2d((14, 14))
        
        self.fine_tune(train_CNN)

    def forward(self, images):
        features = self.resnet(images) # (batch, 2048, 7, 7) or similar depending on input
        features = self.adaptive_pool(features) # (batch, 2048, 14, 14)
        features = features.permute(0, 2, 3, 1) # (batch, 14, 14, 2048)
        features = features.view(features.size(0), -1, features.size(3)) # (batch, 196, 2048)
        return features

    def fine_tune(self, fine_tune=True):
        """
        Allow or prevent the computation of gradients for convolutional blocks 2 through 4 of the encoder.
        :param fine_tune: Allow?
        """
        for p in self.resnet.parameters():
            p.requires_grad = False
            
        # If fine-tuning, only fine-tune convolutional blocks 2 through 4
        if fine_tune:
            for c in list(self.resnet.children())[5:]:
                for p in c.parameters():
                    p.requires_grad = fine_tune

class Attention(nn.Module):
    """
    Attention Network.
    """
    def __init__(self, encoder_dim, decoder_dim, attention_dim):
        super(Attention, self).__init__()
        self.encoder_att = nn.Linear(encoder_dim, attention_dim)  # linear layer to transform encoded image
        self.decoder_att = nn.Linear(decoder_dim, attention_dim)  # linear layer to transform decoder's output
        self.full_att = nn.Linear(attention_dim, 1)  # linear layer to calculate values to be softmax-ed
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=1)  # softmax layer to calculate weights

    def forward(self, encoder_out, decoder_hidden):
        """
        Forward propagation.
        :param encoder_out: encoded images, a tensor of dimension (batch_size, num_pixels, encoder_dim)
        :param decoder_hidden: previous decoder output, a tensor of dimension (batch_size, decoder_dim)
        :return: attention weighted encoding, weights
        """
        att1 = self.encoder_att(encoder_out)  # (batch_size, num_pixels, attention_dim)
        att2 = self.decoder_att(decoder_hidden)  # (batch_size, attention_dim)
        att = self.full_att(self.relu(att1 + att2.unsqueeze(1)))  # (batch_size, num_pixels, 1)
        alpha = self.softmax(att)  # (batch_size, num_pixels, 1)
        attention_weighted_encoding = (encoder_out * alpha).sum(dim=1)  # (batch_size, encoder_dim)

        return attention_weighted_encoding, alpha

class DecoderRNNWithAttention(nn.Module):
    """
    Decoder.
    """
    def __init__(self, attention_dim, embed_dim, decoder_dim, vocab_size, encoder_dim=2048, dropout=0.5):
        super(DecoderRNNWithAttention, self).__init__()
        self.encoder_dim = encoder_dim
        self.attention_dim = attention_dim
        self.embed_dim = embed_dim
        self.decoder_dim = decoder_dim
        self.vocab_size = vocab_size
        self.dropout = dropout

        self.attention = Attention(encoder_dim, decoder_dim, attention_dim)  # attention network

        self.embedding = nn.Embedding(vocab_size, embed_dim)  # embedding layer
        self.dropout = nn.Dropout(p=self.dropout)
        
        # LSTMCell instead of LSTM because we need to manually step through time and apply attention
        self.decode_step = nn.LSTMCell(embed_dim + encoder_dim, decoder_dim, bias=True)  # decoding LSTMCell
        self.init_h = nn.Linear(encoder_dim, decoder_dim)  # linear layer to find initial hidden state of LSTMCell
        self.init_c = nn.Linear(encoder_dim, decoder_dim)  # linear layer to find initial cell state of LSTMCell
        self.f_beta = nn.Linear(decoder_dim, encoder_dim)  # linear layer to create a sigmoid-activated gate
        self.sigmoid = nn.Sigmoid()
        self.fc = nn.Linear(decoder_dim, vocab_size)  # linear layer to find scores over vocabulary
        self.init_weights()  # initialize some layers with the uniform distribution

    def init_weights(self):
        """
        Initialize some parameters with values from the uniform distribution, for easier convergence.
        """
        self.embedding.weight.data.uniform_(-0.1, 0.1)
        self.fc.bias.data.fill_(0)
        self.fc.weight.data.uniform_(-0.1, 0.1)

    def load_pretrained_embeddings(self, embeddings):
        """
        Loads embedding layer with pre-trained embeddings.
        :param embeddings: pre-trained embeddings
        """
        self.embedding.weight = nn.Parameter(embeddings)

    def fine_tune_embeddings(self, fine_tune=True):
        """
        Allow fine-tuning of embedding layer? (Only makes sense with pre-trained embeddings)
        :param fine_tune: Allow?
        """
        for p in self.embedding.parameters():
            p.requires_grad = fine_tune

    def init_hidden_state(self, encoder_out):
        """
        Creates the initial hidden and cell states for the decoder's LSTM based on the encoded images.
        :param encoder_out: encoded images, a tensor of dimension (batch_size, num_pixels, encoder_dim)
        :return: hidden state, cell state
        """
        mean_encoder_out = encoder_out.mean(dim=1)
        h = self.init_h(mean_encoder_out)  # (batch_size, decoder_dim)
        c = self.init_c(mean_encoder_out)
        return h, c

    def forward(self, encoder_out, encoded_captions, caption_lengths):
        """
        Forward propagation.
        :param encoder_out: encoded images, a tensor of dimension (batch_size, num_pixels, encoder_dim)
        :param encoded_captions: encoded captions, a tensor of dimension (batch_size, max_caption_length)
        :param caption_lengths: caption lengths, a tensor of dimension (batch_size, 1)
        :return: scores for vocabulary, sorted encoded captions, decode lengths, weights, sort indices
        """
        batch_size = encoder_out.size(0)
        encoder_dim = encoder_out.size(-1)
        vocab_size = self.vocab_size

        # Flatten image
        encoder_out = encoder_out.view(batch_size, -1, encoder_dim)  # (batch_size, num_pixels, encoder_dim)
        num_pixels = encoder_out.size(1)

        # Sort input data by decreasing lengths; why? apparent below
        caption_lengths, sort_ind = caption_lengths.sort(dim=0, descending=True)
        encoder_out = encoder_out[sort_ind]
        encoded_captions = encoded_captions[sort_ind]

        # Embedding
        embeddings = self.embedding(encoded_captions)  # (batch_size, max_caption_length, embed_dim)

        # Initialize LSTM state
        h, c = self.init_hidden_state(encoder_out)  # (batch_size, decoder_dim)

        # We won't decode at the <end> position, since we've finished generating as soon as we generate <end>
        # So, decoding lengths are actual lengths - 1
        decode_lengths = (caption_lengths - 1).tolist()

        # Create tensors to hold word prediction scores and alphas
        predictions = torch.zeros(batch_size, max(decode_lengths), vocab_size).to(encoder_out.device)
        alphas = torch.zeros(batch_size, max(decode_lengths), num_pixels).to(encoder_out.device)

        # At each time-step, decode by
        # attention-weighing the encoder's output based on the decoder's previous hidden state output
        # then generate a new word in the decoder with the previous word and the attention weighted encoding
        for t in range(max(decode_lengths)):
            batch_size_t = sum([l > t for l in decode_lengths])
            attention_weighted_encoding, alpha = self.attention(encoder_out[:batch_size_t], h[:batch_size_t])
            gate = self.sigmoid(self.f_beta(h[:batch_size_t]))  # gating scalar, (batch_size_t, encoder_dim)
            attention_weighted_encoding = gate * attention_weighted_encoding
            h, c = self.decode_step(
                torch.cat([embeddings[:batch_size_t, t, :], attention_weighted_encoding], dim=1),
                (h[:batch_size_t], c[:batch_size_t])
            )  # (batch_size_t, decoder_dim)
            preds = self.fc(self.dropout(h))  # (batch_size_t, vocab_size)
            predictions[:batch_size_t, t, :] = preds
            alphas[:batch_size_t, t, :] = alpha.squeeze(2)

        return predictions, encoded_captions, decode_lengths, alphas, sort_ind
    
    def caption_image(self, encoder_out, vocabulary, max_length=50, device="cuda"):
        """
        Generates caption for a given image features.
        """
        batch_size = encoder_out.size(0)
        encoder_dim = encoder_out.size(-1)
        
        # Flatten image
        encoder_out = encoder_out.view(batch_size, -1, encoder_dim) 
        
        h, c = self.init_hidden_state(encoder_out)
        
        # Start with <SOS>
        start_token = vocabulary.stoi["<SOS>"]
        
        # Current input word
        word = torch.tensor([start_token]).to(device) # (1)
        
        captions = []
        alphas = []
        
        for i in range(max_length):
            embeddings = self.embedding(word) # (1, embed_dim)
            
            attention_weighted_encoding, alpha = self.attention(encoder_out, h)
            gate = self.sigmoid(self.f_beta(h))
            attention_weighted_encoding = gate * attention_weighted_encoding
            
            h, c = self.decode_step(
                torch.cat([embeddings, attention_weighted_encoding], dim=1),
                (h, c)
            )
            
            preds = self.fc(self.dropout(h)) # (1, vocab_size)
            predicted_word_idx = preds.argmax(dim=1)
            
            captions.append(vocabulary.itos[predicted_word_idx.item()])
            alphas.append(alpha)
            
            if vocabulary.itos[predicted_word_idx.item()] == "<EOS>":
                break
            
            # Next input is current prediction
            word = predicted_word_idx
            
        return [c for c in captions if c != "<SOS>"]

class AttentionModel(nn.Module):
    def __init__(self, attention_dim, embed_dim, decoder_dim, vocab_size, encoder_dim=2048, dropout=0.5, train_CNN=False):
        super(AttentionModel, self).__init__()
        self.encoder = EncoderCNN(embed_size=embed_dim, train_CNN=train_CNN) # embed_size arg is dummy for compatibility if reused
        self.decoder = DecoderRNNWithAttention(attention_dim, embed_dim, decoder_dim, vocab_size, encoder_dim, dropout)

    def forward(self, images, captions, caption_lengths):
        encoder_out = self.encoder(images)
        return self.decoder(encoder_out, captions, caption_lengths)

    def caption_image(self, image, vocabulary, max_length=50, device="cuda"):
        # image expected 4D tensor (1, 3, 224, 224)
        self.eval()
        with torch.no_grad():
            encoder_out = self.encoder(image)
            result = self.decoder.caption_image(encoder_out, vocabulary, max_length, device)
        self.train()
        return result
