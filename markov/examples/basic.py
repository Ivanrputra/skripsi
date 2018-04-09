#!C:\Users\Pandaiman\AppData\Local\Programs\Python\Python35\python.exe

import sys

# relative import.
sys.path.insert(0, "..")
from chain._main import MarkovChain

from common import functions


def basic_usage(text):
    """Basic usage example."""
    chain = MarkovChain("Simple test MarkovChain instance")
    chain.init(text)
    chain.create()
    print(chain.generate())


def instance_arguments(text):
    """Instance arguments example."""
    chain = MarkovChain("Hard test MarkovChain instance", window=2)
    chain.init(text)
    chain.create()
    print(chain.generate())


def generation_arguments(text):
    """Generation arguments example."""
    chain = MarkovChain("Insane test MarkovChain instance")
    chain.init(text)
    chain.create()
    return chain.generate(start="You", max_words=10,max_length=20)


if __name__ == '__main__':
    print("Content-Type: text/html")
    print()
    text = functions.read_json("./common/text.json")

    # basic_usage(text["text"]["simple"])
    # instance_arguments(text["text"]["hard"])
    word = generation_arguments(text["text"]["insane"])

    print('\n'+str(word))
else:
    print('ERROR :: Enter either train or classify after file name......!!')
