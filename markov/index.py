from markov_chain import MarkovChain

chain = MarkovChain("Simple test MarkovChain instance")
chain.init(["One fish two fish red fish blue fish."])
print(chain)
# chain.create()
# print(chain.generate())