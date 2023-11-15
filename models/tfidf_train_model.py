import os
# from dotenv import load_dotenv
import nlpaug.augmenter.word as naw
import sklearn
import re
import pandas as pd
from data_sampling import data_sampling

sampling = data_sampling.DataSampling()

MODEL_DIR = './models'

class TrainTFIDF():
    @classmethod
    def setUpClass(cls):
        # passar por todas as variaveis de ambiente configurando elas e depos instanciar a classe e treinar pelo metodo _train_tfidf
        env_config_path = os.path.abspath(os.path.join(
            os.path.dirname(__file__), '..', '..', '..', '.env'))
        print("# env_config_path:", env_config_path)
        # load_dotenv(env_config_path)

        cls.word2vec_model_path = os.path.join(MODEL_DIR, 'tfidf')
        print("########", cls.word2vec_model_path)
        cls.word2vec_model = naw.WordEmbsAug(model_type='word2vec', model_path=cls.word2vec_model_path)
        cls.context_word_embs_model = naw.ContextualWordEmbsAug()

        cls.tfidf_model_path = os.path.join(MODEL_DIR, 'word', 'tfidf')

        cls.train_tfidf(cls)

    @classmethod
    def tearDownClass(self):
        os.remove(os.path.join(self.tfidf_model_path, 'tfidfaug_w2idf.txt'))
        os.remove(os.path.join(self.tfidf_model_path, 'tfidfaug_w2tfidf.txt'))

    def train_tfidf(self):
        import sklearn.datasets
        import re
        import nlpaug.model.word_stats as nmw

        tfidf_model_path = os.path.join(MODEL_DIR, 'tfidf')
        print("####tfidf_model_path:", tfidf_model_path)
        def _tokenizer(text, token_pattern=r"(?u)\b\w\w+\b"):
            token_pattern = re.compile(token_pattern)
            return token_pattern.findall(text)

        # Load sample data
        train_x, _ = sampling.get_dataset_sample('./Tweets_dataset.csv')
        # dataset =  pd.read_csv('./Tweets_dataset.csv', encoding ='utf-8')

        # Tokenize input
        train_x_tokens = [_tokenizer(x) for x in train_x]

        # Train TF-IDF model
        if not os.path.exists(tfidf_model_path):
            os.makedirs(tfidf_model_path)

        tfidf_model = nmw.TfIdf()
        tfidf_model.train(train_x_tokens)
        tfidf_model.save(tfidf_model_path)

# def _tokenizer(text, token_pattern=r"(?u)\b\w\w+\b"):
#     token_pattern = re.compile(token_pattern)
#     return token_pattern.findall(text)

# destiny_path=os.path.join('.', 'models')

# # Load sample data
# train_data = sklearn.datasets.fetch_20newsgroups(subset='train', remove=('headers', 'footers', 'quotes'))
# train_x = train_data.data

# # Tokenize input
# train_x_tokens = [_tokenizer(x) for x in train_x]

# # Train TF-IDF model
# tfidf_model = nmw.TfIdf()
# tfidf_model.train(train_x_tokens)
# tfidf_model.save(destiny_path)

# Load TF-IDF augmenter
# aug = naw.TfIdfAug(model_path=destiny_path, tokenizer=_tokenizer)

# texts = [
#     'The quick brown fox jumps over the lazy dog',
#     'asdasd test apple dog asd asd'
# ]

# for text in texts:
#     augmented_text = aug.augment(text)
    
#     print('-'*20)
#     print('Original Input:{}'.format(text))
#     print('Agumented Output:{}'.format(augmented_text))