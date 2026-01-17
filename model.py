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
    def __init__(self, embed_size, hidden_size, vocab_size, num_layers):
        super(DecoderRNN, self).__init__()
        self.embed = nn.Embedding(vocab_size, embed_size)
        self.lstm = nn.LSTM(embed_size, hidden_size, num_layers, batch_first=False) 
        
        self.linear = nn.Linear(hidden_size, vocab_size)
        self.dropout = nn.Dropout(0.5)

    def forward(self, features, captions):
        
        features = features.unsqueeze(0)
        
        embeddings = self.embed(captions) 
        
        embeddings = torch.cat((features, embeddings), dim=0)
        
        hiddens, _ = self.lstm(embeddings)
        
        outputs = self.linear(hiddens)
        return outputs


class CNNtoRNN(nn.Module):
    def __init__(self, embed_size, hidden_size, vocab_size, num_layers):
        super(CNNtoRNN, self).__init__()
        self.encoderCNN = EncoderCNN(embed_size)
        self.decoderRNN = DecoderRNN(embed_size, hidden_size, vocab_size, num_layers)

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
                hiddens, states = self.decoderRNN.lstm(x, states)
                output = self.decoderRNN.linear(hiddens.squeeze(0))
                predicted = output.argmax(1)
                
                if vocabulary.itos[predicted.item()] == "<EOS>":
                    break
                
                result_caption.append(vocabulary.itos[predicted.item()])
                x = self.decoderRNN.embed(predicted).unsqueeze(0)

        return result_caption
