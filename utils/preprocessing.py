import re
from bs4 import BeautifulSoup


# Funções Auxiliares:
# Removendo as tags htmls:
def strip_html(text):
    soup = BeautifulSoup(text, "html.parser")
    return soup.get_text()

# Removendo alguns caracteres especiais como colchetes
def remove_between_square_brackets(text):
    return re.sub('\[[^]]*\]', '', text)

# Função que remove os caracteres especiais:
def remove_special_characters(text, remove_digits=True):
    pattern=r'[^a-zA-z0-9\s]'
    text=re.sub(pattern,'',text)
    return text

# cleans text
def denoise_text(text):
    text = strip_html(text)
    if(len(text)> 5120):
        text = text[:5120]
    text = remove_between_square_brackets(text)
    # text = remove_special_characters(text)
    return text