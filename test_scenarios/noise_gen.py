from noise_insertion.unit_insertion import noises

# noises.test_noise(noises.WordEmbeddings, 7)
print(1)
noises.test_noise(noises.ContextualWordEmbs, 1)

print(2)
noises.test_noise(noises.ContextualWordEmbs, 2)

print(4)
noises.test_noise(noises.ContextualWordEmbs, 3)

print(6)
noises.test_noise(noises.ContextualWordEmbs, 4)

print(8)
noises.test_noise(noises.ContextualWordEmbs, 5)

print(10)
noises.test_noise(noises.ContextualWordEmbs, 6)

print(11)
noises.test_noise(noises.ContextualWordEmbs, 6)

exit(0)