from gensim.models import KeyedVectors

model = KeyedVectors.load("data/text8-word2vec.bin")
word_vectors = model.wv

words = word_vectors.key_to_index.keys()
print([x for i, x in enumerate(words) if i < 10])
assert "king" in words


def print_similar_words(word, topn=10):
    print("Similar words to %s:" % word)
    for similar_word, similarity in word_vectors.most_similar(word, topn=topn):
        print("%s (%.4f)" % (similar_word, similarity))


print_similar_words("king")

print(word_vectors.doesnt_match(["hindus", "parsis", "singapore", "christians"]))

for word in ["woman", "dog", "whale", "tree"]:
    print(
        "similarity({:s}, {:s}) = {:.3f}".format(
            "man", word, word_vectors.similarity("man", word)
        )
    )

print(
    "distance(singapore, malaysia) = {:.3f}".format(
        word_vectors.distance("singapore", "malaysia")
    )
)
