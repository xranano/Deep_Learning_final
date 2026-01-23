import torch
import torch.nn as nn
import torchvision.models as models
from torchvision.models import ResNet101_Weights


class EncoderCNN(nn.Module):
    def __init__(self, embed_size, train_CNN=False):
        super(EncoderCNN, self).__init__()
        self.train_CNN = train_CNN
        
       
        resnet = models.resnet101(weights=ResNet101_Weights.DEFAULT)
        
        modules = list(resnet.children())[:-1]
        self.resnet = nn.Sequential(*modules)
        
        self.embed = nn.Linear(resnet.fc.in_features, embed_size)
        
        if not train_CNN:
            for param in self.resnet.parameters():
                param.requires_grad = False
        else:
            for param in self.resnet.parameters():
                param.requires_grad = False

    def forward(self, images):
        features = self.resnet(images)
        features = features.view(features.size(0), -1)
        features = self.embed(features)
        return features
    
    def unfreeze_encoder(self):
        
        for param in self.resnet[7].parameters():
            param.requires_grad = True
            
        for param in self.embed.parameters():
            param.requires_grad = True


class DecoderRNN(nn.Module):
    def __init__(self, embed_size, hidden_size, vocab_size, num_layers, rnn_type="LSTM"):
        super(DecoderRNN, self).__init__()
        self.rnn_type = rnn_type
        self.embed = nn.Embedding(vocab_size, embed_size)
        
        if rnn_type == "LSTM":
            self.rnn = nn.LSTM(embed_size, hidden_size, num_layers, batch_first=False)
        elif rnn_type == "GRU":
            self.rnn = nn.GRU(embed_size, hidden_size, num_layers, batch_first=False)
        else:
            raise ValueError(f"Unsupported RNN type: {rnn_type}")
            
        self.linear = nn.Linear(hidden_size, vocab_size)
        self.dropout = nn.Dropout(0.5)

    def forward(self, features, captions):
        features = features.unsqueeze(0)
        embeddings = self.embed(captions)
        embeddings = torch.cat((features, embeddings), dim=0)
        
        hiddens, _ = self.rnn(embeddings)
        outputs = self.linear(hiddens)
        return outputs


class CNNtoRNN(nn.Module):
    def __init__(self, embed_size, hidden_size, vocab_size, num_layers, rnn_type="LSTM"):
        super(CNNtoRNN, self).__init__()
        self.encoderCNN = EncoderCNN(embed_size)
        self.decoderRNN = DecoderRNN(embed_size, hidden_size, vocab_size, num_layers, rnn_type)

    def forward(self, images, captions):
        features = self.encoderCNN(images)
        outputs = self.decoderRNN(features, captions)
        return outputs

    def caption_image(self, image, vocabulary, max_length=50, device="cuda"):
        result_caption = []

        with torch.no_grad():
            x = self.encoderCNN(image).unsqueeze(0)
            states = None

            for _ in range(max_length):
                hiddens, states = self.decoderRNN.rnn(x, states)
                output = self.decoderRNN.linear(hiddens.squeeze(0))
                predicted = output.argmax(1)
                
                if vocabulary.itos[predicted.item()] == "<EOS>":
                    break
                
                # Ignore <SOS> if generated
                if vocabulary.itos[predicted.item()] != "<SOS>":
                    result_caption.append(vocabulary.itos[predicted.item()])
                    
                x = self.decoderRNN.embed(predicted).unsqueeze(0)

        return result_caption

    def beam_search_caption(self, image, vocabulary, max_length=50, beam_width=3, device="cuda"):
        """
        Generates a caption using Beam Search.
        """
        with torch.no_grad():
            # Initial encoding
            features = self.encoderCNN(image).unsqueeze(0)  # (1, 1, embed_size)
            
            # (score, current_input, states, sequence)
            # score: cumulative log probability
            # current_input: last word index (tensor)
            # states: RNN hidden states
            # sequence: list of word indices generated so far
            
            # Initialize with <SOS> (if your model uses it implicitly via features in the first step)
            # actually our model uses features as the first input to RNN.
            # So first step is manually done.
            
            hiddens, states = self.decoderRNN.rnn(features, None)
            # hiddens: (1, 1, hidden_size)
            output = self.decoderRNN.linear(hiddens.squeeze(0)) # (1, vocab_size)
            log_probs = torch.nn.functional.log_softmax(output, dim=1)
            
            # Get top k (beam_width) starting words
            top_k_probs, top_k_indices = log_probs.topk(beam_width, 1)
            
            beams = []
            for i in range(beam_width):
                word_idx = top_k_indices[0][i]
                score = top_k_probs[0][i].item()
                beams.append((score, word_idx.unsqueeze(0), states, [word_idx.item()]))
            
            for _ in range(max_length - 1):
                candidates = []
                
                for score, input_word, old_states, seq in beams:
                    if vocabulary.itos[seq[-1]] == "<EOS>":
                        candidates.append((score, input_word, old_states, seq))
                        continue
                        
                    # Embed the last word
                    x = self.decoderRNN.embed(input_word.unsqueeze(0)).unsqueeze(0) # (1, 1, embed_size)
                    
                    hiddens, new_states = self.decoderRNN.rnn(x, old_states)
                    output = self.decoderRNN.linear(hiddens.squeeze(0))
                    log_probs = torch.nn.functional.log_softmax(output, dim=1)
                    
                    top_k_probs, top_k_indices = log_probs.topk(beam_width, 1)
                    
                    for i in range(beam_width):
                        new_word_idx = top_k_indices[0][i]
                        new_score = score + top_k_probs[0][i].item()
                        new_seq = seq + [new_word_idx.item()]
                        candidates.append((new_score, new_word_idx, new_states, new_seq))
                
                # Sort candidates by score (highest first) and keep top k
                ordered = sorted(candidates, key=lambda x: x[0], reverse=True)
                beams = ordered[:beam_width]
                
                # Check if all top beams have ended
                all_ended = True
                for _, _, _, seq in beams:
                    if vocabulary.itos[seq[-1]] != "<EOS>":
                        all_ended = False
                        break
                if all_ended:
                    break
            
            # Choose the best beam
            best_score, _, _, best_seq = beams[0]
            
            # Convert indices to words
            result_caption = [vocabulary.itos[idx] for idx in best_seq if vocabulary.itos[idx] not in ["<EOS>", "<SOS>"]]
            return result_caption
