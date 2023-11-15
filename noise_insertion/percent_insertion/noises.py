
import nlpaug.augmenter.char as nac
import nlpaug.augmenter.word as naw
import nlpaug.augmenter.sentence as nas
import nlpaug.flow as nafc
import string
import random
from random import randrange
from nlpaug.util import Action
import os

def tokenizer(text):
    return [text]

def reverse_tokenizer(token_list):
    return ''.join(token_list).strip()

def no_noise(text_lists, aug_level=0):
    return text_lists

# character noises

def Keyboard(text_lists,aug_level=0.3):
    augmented_texts = []

    aug = nac.KeyboardAug(aug_char_p=aug_level,
                      aug_char_max=None,
                      tokenizer = tokenizer,
                      reverse_tokenizer=reverse_tokenizer)

    for text in text_lists:
        size = len(text)

        augmented_text = aug.augment(text, n=1)
        augmented_texts.append(augmented_text)
    
    return augmented_texts


def OCR(text_lists, aug_level=0.3):
    aug = nac.OcrAug(aug_char_p=aug_level,
                    aug_char_max=None,
                    aug_word_max=None,
                    tokenizer = tokenizer,
                    reverse_tokenizer=reverse_tokenizer)

    augmented_texts = []

    for text in text_lists:
        augmented_text = aug.augment(text, n=1)
        augmented_texts.append(augmented_text)

    return augmented_texts

def RandomCharReplace(text_lists,aug_level=0.3):
    aug = nac.RandomCharAug(aug_char_p=aug_level,
                    action=Action.SUBSTITUTE,
                    aug_char_max=None,
                    tokenizer = tokenizer,
                    reverse_tokenizer=reverse_tokenizer,
                    spec_char='!@#$%^&*()_+.'
                    )

    augmented_texts = []

    for text in text_lists:
        augmented_text = aug.augment(text, n=1)
        augmented_texts.append(augmented_text)

    return augmented_texts

def CharSwap(text_lists,aug_level=0.3):
    """
    Noise insertion by swapping characters.

    Notes:
        - noise level grows up in even number increments because in each swap two characters are changed
        - it seems that in some situations it repeats the swap in the same characters, making them go back to what they were before and decreasing the noise level 
    :param text_lists: list of texts to generate noise in
    :aug_level: noise level
    """
    aug = nac.RandomCharAug(action=Action.SWAP,
                    swap_mode='adjacent', # adjacent, middle or random
                    aug_char_p=aug_level/2, # needs to be divided because nlpaug interpretation of augmentation unit
                    aug_char_max=None,
                    tokenizer = tokenizer,
                    reverse_tokenizer= reverse_tokenizer)

    augmented_texts = []

    for text in text_lists:
        augmented_text = aug.augment(text, n=1)
        augmented_texts.append(augmented_text)

    return augmented_texts

# word augmenters

def Antonym(text_lists, aug_level=0.3):
    aug = naw.AntonymAug(name='Antonym_Aug', aug_min=0, aug_max=None,
                aug_p=aug_level, lang='eng', stopwords=None,
                stopwords_regex=None, verbose=1)

    augmented_texts = aug.augment(text_lists, n=1, num_thread=1)

    return augmented_texts

def Synonym(text_lists, aug_level=0.3):
    aug = naw.SynonymAug(aug_p=aug_level,aug_min=0,aug_max=None)

    augmented_texts = aug.augment(text_lists, n=1, num_thread=1)

    return augmented_texts

def WordEmbeddings(text_lists, aug_level=0.3):
    aug = naw.WordEmbsAug(model_type='glove',
                          model_path='models/glove.twitter.27B.100d.txt',
                          aug_p=aug_level,aug_min=0,aug_max=None)

    augmented_texts = aug.augment(text_lists, n=1, num_thread=1)

    return augmented_texts

def ContextualWordEmbs(text_lists, aug_level=0.3):
    aug = naw.ContextualWordEmbsAug(aug_p=aug_level,aug_min=0,aug_max=None, verbose=True,device="cpu")

    augmented_texts = aug.augment(text_lists, n=1, num_thread=1)

    return augmented_texts

def WordSwap(text_lists, aug_level=0.3):
    aug = naw.RandomWordAug(action=Action.SWAP, aug_p=aug_level,aug_min=0,aug_max=None,verbose=True)

    augmented_texts = aug.augment(text_lists, n=1, num_thread=1)

    return augmented_texts

def Spelling(text_lists, aug_level=0.3):
    aug = naw.SpellingAug(dict_path='./models/en.natural.txt', aug_p=aug_level,aug_min=0,aug_max=None)

    augmented_texts = aug.augment(text_lists, n=1, num_thread=1)

    return augmented_texts

def WordSplit(text_lists, aug_level=0.3):
    aug = naw.SplitAug(aug_p=aug_level, aug_min=1, aug_max=1000)

    augmented_texts = aug.augment(text_lists, n=1, num_thread=1)

    return augmented_texts

## loads a pre-trained model
def TfIdfWord(text_lists, aug_level=0.3):
    aug = naw.TfIdfAug(model_path='./models/tfidf', aug_p=aug_level,aug_min=0,aug_max=None)

    augmented_texts = aug.augment(text_lists, n=1, num_thread=1)

    return augmented_texts

## just do word replacement, not used
def ReservedAug(text_lists, aug_level=0.3):
    aug = naw.ReservedAug(reserved_tokens=[],aug_min=1, aug_max=None, aug_p=aug_level )

    augmented_texts = aug.augment(text_lists, n=1, num_thread=1)

    return augmented_texts

## most dataset instances is only a sentence length, so is not used
def SentenceShuffle(text_lists, aug_level=0.3):
    aug = nas.RandomSentAug(aug_p=aug_level, tokenizer = None, mode="left")

    augmented_texts = aug.augment(text_lists, n=1, num_thread=1)

    return augmented_texts

def OBackTranslation(text_lists, aug_level=0.3):
    aug = naw.BackTranslationAug(device='cpu', max_length=200, batch_size=1, force_reload=True)

    augmented_texts = aug.augment(text_lists, n=1, num_thread=1)

    return augmented_texts

# OBS: Other's NLPAug where not used because of the lack of controll of noise level