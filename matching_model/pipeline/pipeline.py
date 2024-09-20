from __future__ import print_function
from __future__ import division

from sklearn.metrics.pairwise import cosine_similarity
from transformers import AutoTokenizer
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import plotly.express as px
import torch.optim as optim
from statistics import mean
from tqdm import tqdm
from umap import UMAP
import pandas as pd
import numpy as np
import zipfile
import dataset
import shutil
import pickle
import models
import torch
import ast
import os






class Pipeline():
    """
    Класс пайплана модели поиска идентичных товаров

    хранилище файлов:
    https://drive.google.com/drive/folders/1ujWJjaJufBJBbzX2NRtfEIUaUupfzeBO?usp=sharing

    Методы:
    train - метод обучения пайплайна. по итогу обученные модели сохраняются в директории (см. метод self._save_everything)
    
    predict - метод предсказания идентичных товаров на обученных моделях. возвращает индексы одинаковых товаров.  

    _prepare_data - внутренний метод предобработки датасета. по идеи должен получать сырые датасеты после парсинга 
    и обрабатывать для использовании в обучении и предсказаниях.

    extract_zip - метод для распаковки архивов в текущую директорию ./
    
    umap_vector_visualization - визуализация результата обучения с помощью понижения размерности ембеддингов алгоритмом UMAP

    attribute_identical_search - метод поиска идентичных товаров по эмбеддингам атрибутов

    look_for_identical_products_main - метод поиска идентичных товаров по основным эмбеддингам

    _look_for_identical_products_main_optimized_experimental - попытка ускорить look_for_identical_products_main. работает некорректно

    _save_everything - метод для сохранения результатов обучения. нужен из-за того, что обучается модель на google.colab

    _show_imgs - метод для вывода пары изображений на одном графике (для юпитера)

    clear_images - метод для очистки датасета от товаров, для которых нет фотографий

    compare_names - метод для поиска идентичных товаров по совпадающим именам (бейзлайн)

    """


    DATA = '/content/drive/MyDrive/sarawan_data/data/prepared_data/'
    MODELS = '/content/drive/MyDrive/sarawan_data/models/'
    ZIP_PHOTO_DIR = '/content/drive/MyDrive/sarawan_data/data/photo/auchan_perekrestok.zip'

    def __init__(self, data, img_dir):
        """
        При инициализации должны быть либо распакованы фотографии в img_dir
        либо подключен колаб с папкой sarawan_data в корне

        data - датафрейм с фотографиями. Обязательные поля см. в dataset.py CustomMatchingDataset

        img_dir - папка с фотографиями. Если фотографии не распакованы, то будут распкованы из переменной self.ZIP_PHOTO_DIR
                  чтобы увидеть, куда распаковываются фотографии см. self.extract_zip 
        """

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.data = data
        self.img_dir = img_dir

        try: 
            os.listdir(img_dir)
        except:
            self.extract_zip(self.ZIP_PHOTO_DIR)
        
        self.data = self.clear_images()

    def train(self,
            num_epoch=60,
            momentum=0.9,
            batch_size=10,
            learning_rate=1e-3, s=30, m=0.5):

        custom_dataset = dataset.CustomMatchingDataset(data=self.data, img_dir=self.img_dir)
        data_loader = torch.utils.data.DataLoader(custom_dataset, batch_size=batch_size, shuffle=False)
        
        cnn = models.CNN(len(custom_dataset.le_cat1.classes_), self.device)
        names_bert = models.NameBert(len(custom_dataset.le_cat1.classes_),)
        attr_bert = models.AttributeBert(len(custom_dataset.le_cat1.classes_),)
        emb = models.Embedding()
        tokenizer = AutoTokenizer.from_pretrained("cointegrated/rubert-tiny2")
        main_loss = models.CustomLoss()
        loss_monitor = models.Losses()
        

        final_embedding_lenght = cnn.resnet.fc.in_features + names_bert.m.classifier.in_features
        metric_fc = models.ArcMarginProduct(final_embedding_lenght, len(custom_dataset.le_cat2.classes_), s=s, m=m) # cat1_num_classes or cat2_num_classes
                    

        cnn_criterion = torch.nn.CrossEntropyLoss()
        names_bert_criterion = torch.nn.CrossEntropyLoss()
        attr_bert_criterion = torch.nn.CrossEntropyLoss()
        arcface_criterion = torch.nn.CrossEntropyLoss()

        # оптимизаторы
        optimizer1 = optim.SGD(cnn.resnet.fc.parameters(), lr=learning_rate)
        optimizer2 = optim.Adam(names_bert.m.classifier.parameters(), lr=learning_rate)
        optimizer3 = optim.Adam(attr_bert.m.classifier.parameters(), lr=learning_rate)
        optimizer = optim.SGD(metric_fc.parameters(), lr=learning_rate)

        cnn.resnet = cnn.resnet.to(self.device)
        names_bert.m = names_bert.m.to(self.device)
        attr_bert.m = attr_bert.m.to(self.device)
        emb.to(self.device)
        metric_fc.to(self.device)

        n_batches = round(len(self.data) / batch_size)

        cat1_list = list(custom_dataset.data['cat1_enc'].unique())
        cat1_list.sort()
        
        cat2_list = list(custom_dataset.data['cat2_enc'].unique())
        cat2_list.sort()

        for epoch in range(num_epoch):
            cnn.resnet.train()
            names_bert.m.train()
            attr_bert.m.train()

            embbb = None
            cat1 = None
            cat2 = None
            names_store = None
            data_with_embeddings = None
            
        
            for names, attributes, images, index, labels_cat1, labels_cat2 in tqdm(data_loader):
                
                names = tokenizer(names, padding=True, truncation=True, return_tensors='pt')
                attributes = tokenizer(attributes, padding=True, truncation=True, return_tensors='pt')

                names = names.to(self.device)
                attributes = attributes.to(self.device)
                images = images.to(self.device)
                index = index.to(self.device)
                labels_cat1 = labels_cat1.to(self.device)
                labels_cat2 = labels_cat2.to(self.device)

                # обнуляем градиенты
                optimizer1.zero_grad()
                optimizer2.zero_grad()
                optimizer3.zero_grad()
                optimizer.zero_grad()

                # получаем предсказания
                cnn_outputs = cnn.resnet(images.to(self.device))
                names_bert_outputs = names_bert.m(**{k: v.to(self.device) for k, v in names.to(self.device).items()})
                attr_bert_outputs = attr_bert.m(**{k: v.to(self.device) for k, v in attributes.to(self.device).items()})


                # считаем лоссы на кат1
                cnn_loss = cnn_criterion(cnn_outputs.to(self.device), labels_cat1.to(self.device))
                names_bert_loss = names_bert_criterion(names_bert_outputs.logits.to(self.device), labels_cat1.to(self.device))
                attr_bert_loss = attr_bert_criterion(attr_bert_outputs.logits.to(self.device), labels_cat1.to(self.device))


                cnn_loss.backward(retain_graph=True)
                names_bert_loss.backward(retain_graph=True)
                attr_bert_loss.backward(retain_graph=True)

                # получаем ембеддинг от трёх моделей
                embedding = emb(names_bert, cnn, names.to(self.device), images.to(self.device), final_embedding_lenght, self.device)

                # получаем предсказание кат2 от АркФейса
                output = metric_fc(embedding.float(), labels_cat2) 

                # считаем лосс по кат2
                arc_loss = arcface_criterion(output, labels_cat2.to(self.device)) 
                # считаем финальный лосс
                loss = main_loss(cnn_loss, names_bert_loss, arc_loss, epoch)

                arc_loss.backward(retain_graph=True)
                loss.backward(retain_graph=True)

                loss_monitor.append(cnn_loss, names_bert_loss, attr_bert_loss, arc_loss,  loss, 
                                    cnn_outputs, names_bert_outputs, attr_bert_outputs, output, 
                                    labels_cat1, cat1_list,
                                     labels_cat2, cat2_list)


                optimizer1.step()
                optimizer2.step()
                optimizer3.step()
                optimizer.step()

                data = pd.DataFrame(self.data.loc[index.cpu().detach().numpy(), :])
                # записываем
                if (epoch+1) % 5 == 0 or epoch == 0:
                    if data_with_embeddings is None:
                        embedding = embedding.cpu().detach().numpy()
                        emb_attr = attr_bert_outputs.logits.cpu().detach().numpy()
                        data['emb_main'] = [embedding[i] for i in range(len(embedding))]
                        data['emb_attr'] = [emb_attr[i] for i in range(len(emb_attr))]
                        data_with_embeddings = data
                    else:
                        embedding = embedding.cpu().detach().numpy()
                        emb_attr = attr_bert_outputs.logits.cpu().detach().numpy()
                        data['emb_main'] = [embedding[i] for i in range(len(embedding))]
                        data['emb_attr'] = [emb_attr[i] for i in range(len(emb_attr))]
                        data_with_embeddings = pd.concat((data_with_embeddings, data), axis=0)
                        

            with open('loss.txt', 'a') as f:
                f.write(f"Epoch [{epoch}], img_loss: {mean(loss_monitor.losses[0][-n_batches:])}, attr_loss: {mean(loss_monitor.losses[2][-n_batches:])}, names_bert_loss: {mean(loss_monitor.losses[1][-n_batches:])}, arc_loss: {mean(loss_monitor.losses[3][-n_batches:])}, main_loss: {mean(loss_monitor.losses[4][-n_batches:])}\n")
                f.write(f"Epoch [{epoch}], img_accuracy1: {mean(loss_monitor.acc[0][-n_batches:])}, attr_accuracy1: {mean(loss_monitor.acc[2][-n_batches:])}, names_bert_accuracy1: {mean(loss_monitor.acc[1][-n_batches:])}, arc_accuracy1: {mean(loss_monitor.acc[3][-n_batches:])}\n")
                f.write(f"Epoch [{epoch}], img_accuracy5: {mean(loss_monitor.acc5[0][-n_batches:])}, attr_accuracy5: {mean(loss_monitor.acc5[2][-n_batches:])}, names_bert_accuracy5: {mean(loss_monitor.acc5[1][-n_batches:])}, arc_accuracy5: {mean(loss_monitor.acc5[3][-n_batches:])}\n\n")

            if (epoch+1) % 5 == 0 or epoch == 0:
                self._save_everything(epoch, data_with_embeddings)
                torch.save(cnn.resnet.state_dict(), '/content/drive/MyDrive/sarawan_data/models/test/resnet34.pth')
                torch.save(names_bert.m.state_dict(), '/content/drive/MyDrive/sarawan_data/models/test/names_bert.pth')
                torch.save(attr_bert.m.state_dict(), '/content/drive/MyDrive/sarawan_data/models/test/attr_bert.pth')

            print(f"Epoch [{epoch}], img_loss: {mean(loss_monitor.losses[0][-n_batches:])}, attr_loss: {mean(loss_monitor.losses[2][-n_batches:])}, names_bert_loss: {mean(loss_monitor.losses[1][-n_batches:])}, arc_loss: {mean(loss_monitor.losses[3][-n_batches:])}, main_loss: {mean(loss_monitor.losses[4][-n_batches:])}\n")
            print(f"Epoch [{epoch}], img_accuracy1: {mean(loss_monitor.acc[0][-n_batches:])}, attr_accuracy1: {mean(loss_monitor.acc[2][-n_batches:])}, names_bert_accuracy1: {mean(loss_monitor.acc[1][-n_batches:])}, arc_accuracy1: {mean(loss_monitor.acc[3][-n_batches:])}\n")
            print(f"Epoch [{epoch}], img_accuracy5: {mean(loss_monitor.acc5[0][-n_batches:])}, attr_accuracy5: {mean(loss_monitor.acc5[2][-n_batches:])}, names_bert_accuracy5: {mean(loss_monitor.acc5[1][-n_batches:])}, arc_accuracy5: {mean(loss_monitor.acc5[3][-n_batches:])}\n\n")
            




    def predict(self, data=None, batch_size=10, first_threshold=0.7, last_threshold=0.999):
        if data is None:
            data = self.data


        custom_dataset = dataset.CustomMatchingDataset(data=data, img_dir=self.img_dir)
        data_loader = torch.utils.data.DataLoader(custom_dataset, batch_size=batch_size, shuffle=False)

        cnn = models.CNN(len(custom_dataset.le_cat1.classes_), self.device)
        name_bert = models.NameBert(len(custom_dataset.le_cat1.classes_),)
        attr_bert = models.AttributeBert(len(custom_dataset.le_cat1.classes_),)
        emb = models.Embedding()
        tokenizer = AutoTokenizer.from_pretrained("cointegrated/rubert-tiny2")
        final_embedding_lenght = cnn.resnet.fc.in_features + name_bert.m.classifier.in_features

        name_bert.to(self.device)
        attr_bert.to(self.device)
        cnn.to(self.device)
        emb.to(self.device)

        try:
            cnn.resnet.load_state_dict(torch.load(self.MODELS + "test/resnet34.pth"))
            name_bert.m.load_state_dict(torch.load(self.MODELS + "test/names_bert.pth"))
            attr_bert.m.load_state_dict(torch.load(self.MODELS + "test/attr_bert.pth"))

        
        except Exception as e:
            print(e, f'\n\n Сначала обучите или поместите модели в {self.MODELS}')
            return    


        self.data_with_embeddings = None
        for name, attribute, image, index, _, _ in tqdm(data_loader):
            """создание ембеддингов"""
            torch.cuda.empty_cache()
            name = tokenizer(name, padding=True, truncation=True, return_tensors='pt')
            attribute = tokenizer(attribute, padding=True, truncation=True, return_tensors='pt')
            emb_main = emb(name_bert, cnn, name.to(self.device), image.to(self.device), final_embedding_lenght, self.device)
            attr_bert_outputs = attr_bert.m(**{k: v.to(self.device) for k, v in attribute.to(self.device).items()})
            emb_main = emb_main.cpu().detach().numpy()
            emb_attr = attr_bert_outputs.logits.cpu().detach().numpy()

            data = pd.DataFrame(self.data.loc[index.cpu().detach().numpy(), :])

            if self.data_with_embeddings is None:
                data['emb_main'] = [emb_main[i] for i in range(len(emb_main))]
                data['emb_attr'] = [emb_attr[i] for i in range(len(emb_attr))]
                self.data_with_embeddings = data
            else:
                data['emb_main'] = [emb_main[i] for i in range(len(emb_main))]
                data['emb_attr'] = [emb_attr[i] for i in range(len(emb_attr))]
                self.data_with_embeddings = pd.concat((self.data_with_embeddings, data), axis=0)

        self.data_with_embeddings.to_json(self.DATA + 'data_with_embeddings.json')

        similar_products, count_in_cats = self.look_for_identical_products_main(self.data_with_embeddings, first_threshold)
        identical_products = self.attribute_identical_search(similar_products, last_threshold)

      
        print('\n\nНайденные идентичные продукты на 1-ом уровне с threshold = ' + str(first_threshold) + ': ' + str(count_in_cats))
        print('Найденные идентичные продукты на 2-ом уровне с threshold = ' + str(last_threshold) + ': ' + str(len(identical_products)))

        return identical_products

    def _prepare_data(self, data):
        # Подготовка данных в init
        tranformer = dataset.DataPreparation(data)
        prepared = tranformer.prepare()
        pass



    def extract_zip(self, zip_path):
        "метод для извлечения фотографий из архива"
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall('./')




    def umap_vector_visualization(self, dataframe:pd.DataFrame, show=None, color=None, dot_size=3, title=None, text=None, write_to_html_path=None):

        reducer = UMAP(n_components=3)
        vector = reducer.fit_transform(dataframe) # понижаем размерность
        fig = px.scatter_3d(x=vector[:, 0],
                            y=vector[:, 1],
                            z=vector[:, 2],
                            color=color,
                            title=title,
                            text=text
                            )
        fig.update_traces(marker=dict(size=dot_size))  # задаём размер точек

        if write_to_html_path:
            fig.write_html(write_to_html_path)   # если передан путь к html, файл будет записан в него

        if show:
            fig.show()  # отображение в  jupiter notebook


    def attribute_identical_search(self, identical_products_first, threshold_2):
       
        identical_products_second = []
        for sim_main, i, j in identical_products_first:
            similarity = cosine_similarity(np.array(self.data_with_embeddings['emb_attr'][i]).reshape(1, -1), np.array(self.data_with_embeddings['emb_attr'][j]).reshape(1, -1))
            if similarity > threshold_2:
                identical_products_second.append([sim_main, similarity, i, j])
        return identical_products_second


    def look_for_identical_products_main(self, data, threshold):
        """поиск идентичных товаров по основным ембеддингам"""
        count_in_cats = []
        similar_products = []
        cats = list(set(data['cat1']))
        
        for cat in tqdm(cats):
            count = 0
            indexes = data[data['cat1'] == cat].index
            emb = pd.DataFrame(np.vstack(data.loc[indexes, 'emb_main'].values))
            emb.set_index(indexes, inplace=True)
            similarity = pd.DataFrame(cosine_similarity(emb))
            similarity.set_index(indexes, inplace=True)
            similarity.columns = similarity.index
            
            for row in indexes:
                for col in indexes:
                    if row != col and [similarity[row][col], col, row] not in similar_products and similarity[row][col] >= threshold:
                        similar_products.append([similarity[row][col], row, col])
                        count += 1
            count_in_cats.append([cat, count])
        return similar_products, count_in_cats



    def _look_for_identical_products_main_optimized_experimental(self, data, threshold):
        count_in_cats = []
        similar_products = []
        cats = data['cat1'].unique()
        
        for cat in cats:
            d = data[data['cat1'] == cat]
            emb = np.vstack(d['emb_main'].values)
            similarity = cosine_similarity(emb)
            np.fill_diagonal(similarity, 0)  # диагональные элементы в 0
            
            similar_indices = np.argwhere(similarity >= threshold)
            similar_indices = [(row, col) for row, col in similar_indices if row < col]  
            similar_products.extend([(row, col) for row, col in similar_indices])
            count_in_cats.append([cat, len(similar_indices)])
            
        return similar_products, count_in_cats




    def _save_everything(self, epoch, data_with_embeddings):
        shutil.copy2('./loss.txt', f'/content/drive/MyDrive/sarawan_data/objs/loss.txt')
        data_with_embeddings.to_json(f'/content/drive/MyDrive/sarawan_data/objs/data_with_embeddings_{epoch}.json')

        self.umap_vector_visualization(np.vstack(data_with_embeddings['emb_main'].values), color=data_with_embeddings['cat2'], write_to_html_path=f'/content/drive/MyDrive/sarawan_data/objs/cat1_{epoch}.html')



    def _show_imgs(self, data, index1, index2):
        """вывод изображений"""
        plt.figure(figsize=(20, 10))

        photo1 = data.loc[index1, 'image']
        photo2 = data.loc[index2, 'image']

        # Путь к файлам с изображениями
        path_to_image1 = f'{self.img_dir}{photo1}'
        path_to_image2 = f'{self.img_dir}{photo2}'

        # Загрузка изображений
        image1 = mpimg.imread(path_to_image1)
        image2 = mpimg.imread(path_to_image2)

        # Создание нового графика
        plt.figure()

        # Первое изображение
        plt.subplot(1, 2, 1)  
        plt.imshow(image1)
        plt.title(data.loc[index1, 'name'], fontsize=8)
        plt.axis('off')  # Отключение осей

        # Второе изображение
        plt.subplot(1, 2, 2)  
        plt.imshow(image2)
        plt.title(data.loc[index2, 'name'], fontsize=8)
        plt.axis('off')
        plt.show('image.png')



    def clear_images(self):
        """убирает из датафрейма товары, если их фоток нет в папке self.img_dir"""
        image_folder = os.listdir(self.img_dir)
        bad_df = pd.DataFrame(columns = self.data.columns)
        print('Исходный датасет: ', self.data.shape)

        for i in range(len(self.data)):
            try:
                if not self.data.loc[i, 'image'] in image_folder:
                    bad_df.loc[i, :] = self.data.loc[i, :]
            except:
                print(self.data.loc[i, 'image'])
                break

        self.data.drop(index=bad_df.index, inplace=True)
        self.data.reset_index(drop=True, inplace=True)
        print('Очиенный:', self.data.shape)
        return self.data
    

    def compare_names(self, data=None):
        "алгоритм поиска идентичных товаров по именам"
        if data is None:
            data = self.data

        similarity = []
        cats = list(set(data['cat1']))
        
        for cat in tqdm(cats):
            d = data[data['cat1'] == cat]
            indexes = d.index
            
            for idx, i in enumerate(indexes[:-1]):
                for j in indexes[idx + 1:]:
                    if d['name'][i].lower() == d['name'][j].lower():
                        similarity.append([i, j])
        return similarity
