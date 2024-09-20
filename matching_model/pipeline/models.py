from transformers import BertForSequenceClassification
from sklearn.metrics import top_k_accuracy_score
import torchvision.models as models
from torch.nn import BatchNorm1d
import torch.nn.functional as F
from torch.nn import Parameter
import torch.nn as nn
import numpy as np
import torch
import math




class ArcMarginProduct(nn.Module):
    """
    Класс модели аркфейс для сближения похожих товаров и отдаления различных 
    Подробнее: 
    https://habr.com/ru/companies/ntechlab/articles/531842/
    """
    
    def __init__(self, in_features, out_features, s=30.0, m=0.50, easy_margin=False):
        super(ArcMarginProduct, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.s = s
        self.m = m
        self.weight = Parameter(torch.FloatTensor(out_features, in_features))
        nn.init.xavier_uniform_(self.weight)

        self.easy_margin = easy_margin
        self.cos_m = math.cos(m)
        self.sin_m = math.sin(m)
        self.th = math.cos(math.pi - m)
        self.mm = math.sin(math.pi - m) * m

    def forward(self, input, label):
        # --------------------------- cos(theta) & phi(theta) ---------------------------
        cosine = F.linear(F.normalize(input), F.normalize(self.weight))
        sine = torch.sqrt((1.0 - torch.pow(cosine, 2)).clamp(0, 1))
        phi = cosine * self.cos_m - sine * self.sin_m
        if self.easy_margin:
            phi = torch.where(cosine > 0, phi, cosine)
        else:
            phi = torch.where(cosine > self.th, phi, cosine - self.mm)
        # --------------------------- convert label to one-hot ---------------------------
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        one_hot = torch.zeros(cosine.size(), requires_grad=True, device=device)
        # -------------torch.where(out_i = {x_i if condition_i else y_i) -------------
        output = (one_hot * phi) + ((1.0 - one_hot) * cosine)  # you can use torch.where if your torch.__version__ is 0.4
        output *= self.s
        
        return output




class NameBert(nn.Module):
    """Класс берта, на входе принимает название товара, возвращает эмбеддинг """
    def __init__(self, count_cat):
        super().__init__()
        self.m = BertForSequenceClassification.from_pretrained('cointegrated/rubert-tiny2')
        
        for param in self.m.parameters():
            param.requires_grad = False
            
        self.m.classifier = torch.nn.Linear(self.m.classifier.in_features, count_cat)

        for param in self.m.classifier.parameters():
            param.requires_grad = True

    def forward(self, text):
        # на вход подаётся уже токенизированный текст

        with torch.no_grad():
            model_output = self.m(**{k: v.to(self.m.device) for k, v in text.items()}, output_hidden_states=True)

        """
        embeddings содержит 4 ембеддинга из разных слоёв. Стоит попробовать использовать каждый по отдельности, либо же MaxPooling, AveragePooling, GlobalPooling. нужно тестить , пока оставлю последний слой

        каждый такой эмбеддинг содержит ещё 240 слоёв, тоже можно попробовать пулинг
        """
        embeddings = model_output.hidden_states[-1][:, 0, :]
        embeddings = torch.nn.functional.normalize(embeddings)
        return embeddings.detach()




class AttributeBert(nn.Module):
    """Класс берта, на входе принимает атрибуты товара, возвращает эмбеддинг"""
    def __init__(self, count_cat):
        super().__init__()
        self.m = BertForSequenceClassification.from_pretrained('cointegrated/rubert-tiny2')
        
        for param in self.m.parameters():
            param.requires_grad = False

        self.m.classifier = torch.nn.Linear(self.m.classifier.in_features, count_cat)
        for param in self.m.classifier.parameters():
            param.requires_grad = True

    def forward(self, text):
        # на вход подаётся уже токенизированный текст

        with torch.no_grad():
            model_output = self.m(**{k: v.to(self.m.device) for k, v in text.items()}, output_hidden_states=True)

        """
        embeddings содержит 4 ембеддинга из разных слоёв. Стоит попробовать использовать каждый по отдельности, либо же MaxPooling, AveragePooling, GlobalPooling. нужно тестить , пока оставлю последний слой

        каждый такой эмбеддинг содержит ещё 240 слоёв, тоже можно попробовать пулинг
        """
        embeddings = model_output.hidden_states[-1][:, 0, :]
        embeddings = torch.nn.functional.normalize(embeddings)
        return embeddings.detach()





class CNN(nn.Module):
    """ResNet43 для получения эмбеддингов изображений товаров"""
    def __init__(self, count_cat, device, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.resnet = models.resnet34(pretrained=True)

        for param in self.resnet.parameters():
            param.requires_grad = False

        self.resnet.fc = torch.nn.Linear(self.resnet.fc.in_features, count_cat)
        for param in self.resnet.fc.parameters():
            param.requires_grad = True
        self.device = device

    def forward(self, img):
        embedding = self.get_embedding(img)

        # возвращаем ембеддинг
        return embedding


    def get_embedding(self, img_tensor):
        img_tensor = img_tensor.to(self.device)
        layer = self.resnet._modules.get('avgpool')
        outputs = []

        def copy_embeddings(m, i, o):
            """Copy embeddings from the penultimate layer.
            """
            o = o[:, :, 0, 0].cpu().detach().numpy().tolist()
            outputs.append(o)
            hook.remove()

        # attach hook to the penulimate layer
        hook = layer.register_forward_hook(copy_embeddings)


        self.resnet.eval()

        self.resnet(img_tensor)
        list_embeddings = [item for sublist in outputs for item in sublist]
        self.embedding = np.array(list_embeddings)
        return self.embedding


    
class Embedding(nn.Module):
    """Класс для склеивания и нормализации эмбеддингов. 
    Думается, что склейка эмбеддингов - довольно топорно, для улучшения можно попробовать берты для тексто-визуальной информации"""
    def __init__(self):
        super().__init__()


    def forward(self, name_bert, cnn, name, img, num_features, device):
        name = name_bert(name)
        img = torch.from_numpy(cnn(img)).to(device)

        self.BN = BatchNorm1d(num_features)
        embedding_raw = torch.cat((name, img), dim=1)
        self.BN.to(device, dtype=float)
        embedding = self.BN(embedding_raw.to(device, dtype=float))
        return embedding



class CustomLoss(nn.Module):
    """Класс лосса по формуле из статьи: https://habr.com/ru/companies/ozontech/articles/648231/"""
    def __init__(self):
        super(CustomLoss, self).__init__()
        self.alpha = lambda x: 0.5 / np.exp(x/np.exp(1))  # [0, 0.5]?? or [0, 1]


    def forward(self, cnn_loss, names_bert_loss, arc_loss, epoch):
        main_loss = self.alpha(epoch)*(cnn_loss + names_bert_loss) + (1 - self.alpha(epoch)) * arc_loss
        return main_loss



class Losses:
    """Класс всех лоссов для записи их в файл во время обучения (см. Pipeline.save_everything)"""
    def __init__(self) -> None:
        self.losses = [[], [], [], [], []]
        self.losses_val = [[], [], [], [], []]
        self.Loss_history = [[], [], [], [], []]
        self.Loss_history_val = [[], [], [], [], []]

        self.acc = [[], [], [], [], []]
        self.acc5 = [[], [], [], [], []]
        self.acc_val = [[], [], [], [], []]
        self.Accuracy_history = [[], [], [], [], []]
        self.Accuracy_history_val = [[], [], [], [], []]




    def append(self, cnn_loss, names_bert_loss, attr_bert_loss, arc_loss,  loss, cnn_outputs, names_bert_outputs, attr_bert_outputs, output, labels_cat1, cat1_list, labels_cat2, cat2_list):
        self.losses[0].append(cnn_loss.item())
        self.losses[1].append(names_bert_loss.item())
        self.losses[2].append(attr_bert_loss.item())

        cnn_accuracy1 = top_k_accuracy_score(labels_cat1.cpu().detach().numpy(), cnn_outputs.cpu().detach().numpy(), k = 1, labels=cat1_list)
        names_bert_accuracy1 = top_k_accuracy_score(labels_cat1.cpu().detach().numpy(), names_bert_outputs.logits.cpu().detach().numpy(), k = 1, labels=cat1_list)
        attr_bert_accuracy1 = top_k_accuracy_score(labels_cat1.cpu().detach().numpy(), attr_bert_outputs.logits.cpu().detach().numpy(), k = 1, labels=cat1_list)

        # считаем acc@5 на кат1
        cnn_accuracy5 = top_k_accuracy_score(labels_cat1.cpu().detach().numpy(), cnn_outputs.cpu().detach().numpy(), k = 5, labels=cat1_list)
        names_bert_accuracy5 = top_k_accuracy_score(labels_cat1.cpu().detach().numpy(), names_bert_outputs.logits.cpu().detach().numpy(), k = 5, labels=cat1_list)
        attr_bert_accuracy5 = top_k_accuracy_score(labels_cat1.cpu().detach().numpy(), attr_bert_outputs.logits.cpu().detach().numpy(), k = 5, labels=cat1_list)


        self.acc[0].append(cnn_accuracy1.item())
        self.acc[1].append(names_bert_accuracy1.item())
        self.acc[2].append(attr_bert_accuracy1.item())

        self.acc5[0].append(cnn_accuracy5.item())
        self.acc5[1].append(names_bert_accuracy5.item())
        self.acc5[2].append(attr_bert_accuracy5.item())


        arc_accuracy1 = top_k_accuracy_score(labels_cat2.cpu().detach().numpy(), output.cpu().detach().numpy(), k = 1, labels=cat2_list)
        arc_accuracy5 = top_k_accuracy_score(labels_cat2.cpu().detach().numpy(), output.cpu().detach().numpy(), k = 5, labels=cat2_list)


        self.losses[3].append(arc_loss.item())
        self.losses[4].append(loss.item())
        self.acc[3].append(arc_accuracy1.item())
        self.acc5[3].append(arc_accuracy5.item())