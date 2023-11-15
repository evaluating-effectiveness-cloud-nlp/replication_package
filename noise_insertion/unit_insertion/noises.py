from typing import List
import nlpaug.augmenter.char as nac
import nlpaug.augmenter.sentence as nas
import nlpaug.augmenter.word as naw
from noise_insertion.utils import return_sentence_similarity

def test_noise(noise_func, units_to_alter, text = "The white fox jumps over the blue wall."):
    print("before: ", text)

    result = noise_func(text_lists=[text], aug_level=units_to_alter)

    print("after:  ", result[0])
    print(return_sentence_similarity(text, result[0]))

def tokenizer(text):
    return [text]

def reverse_tokenizer(token_list):
    return ''.join(token_list).strip()

def OCR(text_lists, aug_level=4):
    if(int(aug_level)==0): return text_lists

    aug = nac.OcrAug(
                    aug_char_min=int(aug_level),
                    aug_char_max=int(aug_level),
                    aug_word_max=None,
                    min_char=0,
                    tokenizer=tokenizer,
                    reverse_tokenizer=reverse_tokenizer
        )

    augmented_texts = []

    for text in text_lists:
        augmented_text = aug.augment(text, n=1)
        augmented_texts.append(augmented_text)

    return augmented_texts

def Keyboard(text_lists, aug_level=4) -> List[str]:
    if(int(aug_level)==0): return text_lists

    aug = nac.KeyboardAug(
                    aug_char_min=int(aug_level),
                    aug_char_max=int(aug_level),
                    aug_word_max=None,
                    min_char=0,
                    tokenizer=tokenizer,
                    reverse_tokenizer=reverse_tokenizer
        )

    augmented_texts = []

    for text in text_lists:
        augmented_text = aug.augment(text, n=1)
        augmented_texts.append(augmented_text)

    return augmented_texts

def WordSwap(text_lists, aug_level=4) -> List[str]:
    if(int(aug_level)==0): return text_lists

    aug = naw.RandomWordAug(action='swap',
                            aug_min=int(aug_level), \
                            aug_max=int(aug_level))

    augmented_texts = []

    for text in text_lists:
        augmented_text = aug.augment(text, n=1)
        augmented_texts.append(augmented_text)

    return augmented_texts

def RandomCharReplace(text_lists, aug_level=4):
    if(int(aug_level)==0): return text_lists

    aug = nac.RandomCharAug(
                    action='substitute',
                    spec_char='!@#$%^&*()_+.',
                    aug_char_min=int(aug_level),
                    aug_char_max=int(aug_level),
                    aug_word_max=None,
                    min_char=0,
                    tokenizer=tokenizer,
                    reverse_tokenizer=reverse_tokenizer
        )

    augmented_texts = []

    for text in text_lists:
        augmented_text = aug.augment(text, n=1)
        augmented_texts.append(augmented_text)

    return augmented_texts

def CharSwap(text_lists, aug_level=4):
    if(int(aug_level)==0): return text_lists

    aug = nac.RandomCharAug(action='swap',
                            swap_mode='adjacent',
                            spec_char='!@#$%^&*()_+.',
                            aug_char_min=int(aug_level),
                            aug_char_max=int(aug_level),
                            aug_word_max=None,
                            min_char=0,
                            tokenizer=tokenizer,
                            reverse_tokenizer=reverse_tokenizer
        )

    augmented_texts = []

    for text in text_lists:
        augmented_text = aug.augment(text, n=1)
        augmented_texts.append(augmented_text)

    return augmented_texts

def Antonym(text_lists, aug_level=4):
    if(int(aug_level)==0): return text_lists

    aug = naw.AntonymAug(name='Antonym_Aug', 
                         aug_min=int(aug_level), 
                         aug_max=int(aug_level),
                         lang='eng',
                         stopwords=None,
                         stopwords_regex=None,
                         verbose=1)

    augmented_texts = aug.augment(text_lists, n=1, num_thread=1)

    return augmented_texts

def WordEmbeddings(text_lists, aug_level=4):
    if(int(aug_level)==0): return text_lists

    aug = naw.WordEmbsAug(model_type='glove',
                          model_path='models/glove.twitter.27B.100d.txt',
                          aug_min=int(aug_level), 
                          aug_max=int(aug_level))

    augmented_texts = aug.augment(text_lists, n=1, num_thread=1)

    return augmented_texts

def ContextualWordEmbs(text_lists, aug_level=0.3):
    aug = naw.ContextualWordEmbsAug(aug_min=int(aug_level),
                                    aug_max=int(aug_level),
                                    verbose=True,
                                    device="cpu")

    augmented_texts = aug.augment(text_lists, n=1, num_thread=1)

    return augmented_texts

def TfIdfWord(text_lists, aug_level=0.3):
    if(int(aug_level)==0): return text_lists

    aug = naw.TfIdfAug(model_path='./models/tfidf',
                       aug_min=int(aug_level), 
                       aug_max=int(aug_level))

    augmented_texts = aug.augment(text_lists, n=1, num_thread=1)

    return augmented_texts

def Spelling(text_lists, aug_level=4):
    if(int(aug_level)==0): return text_lists

    aug = naw.SpellingAug(dict_path='./models/en.natural.txt',
                          aug_min=int(aug_level), 
                          aug_max=int(aug_level)
                          )

    augmented_texts = aug.augment(text_lists, n=1, num_thread=1)

    return augmented_texts

def WordSplit(text_lists, aug_level=4):
    if(int(aug_level)==0): return text_lists

    aug = naw.SplitAug(aug_min=int(aug_level), 
                       aug_max=int(aug_level),
                       min_char=3, # best value is 2, we're using 2 because of an nlpaug's bug
                    #    tokenizer=tokenizer,
                    #    reverse_tokenizer=reverse_tokenizer
                       )

    augmented_texts = aug.augment(text_lists, n=1, num_thread=1)

    return augmented_texts


def Synonym(text_lists, aug_level=4):
    if(int(aug_level)==0): return text_lists

    aug = naw.SynonymAug(aug_min=int(aug_level), 
                         aug_max=int(aug_level))

    augmented_texts = aug.augment(text_lists, n=1, num_thread=1)

    return augmented_texts

def SentenceShuffle(text_lists, aug_level=4):
    if(int(aug_level)==0): return text_lists

    aug = nas.RandomSentAug(aug_min=0, # necessary to work with text with small number of sentences
                            aug_max=int(aug_level),
                            mode="random")

    augmented_texts = aug.augment(text_lists, n=1, num_thread=1)

    return augmented_texts