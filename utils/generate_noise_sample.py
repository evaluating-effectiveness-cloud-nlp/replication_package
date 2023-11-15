from os import sep
from noise_insertion import noises

text = "The few scenes that actually attempt a depiction of revolutionary. This struggle resemble a hirsute Boy Scout troop meandering tentatively between swimming holes."
noise_level = 0.8

noise_algorithms=[
    # noises.aug.AntonymAug,
    # noises.aug.RandomSentAug,
    # noises.aug.RandomWordAug,
    # noises.aug.ReservedAug, # removido
    # noises.aug.SpellingAug,
    # noises.aug.SplitAug,
    # noises.aug.SynonymAug,
    # noises.aug.TfldfAug,
    noises.aug.WordEmbsAug,
    # noise_insertion.aug.BackTranslation, # error
    # noise_insertion.aug.LambadaAug # error
]

noise_results = []

print(text)
for noise in noise_algorithms:
    print(noise.__name__,'...',end='')
    noised_text = noise([text], noise_level)
    print("\r",noise.__name__, ";", noised_text[0])
    noise_results.append({'noise': noise.__name__, 'result': noised_text})

print(text)
print(noise_results)