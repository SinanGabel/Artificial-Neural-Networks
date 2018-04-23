# summul.ai
Public AI (neural network) experiments

## name inspiration, history
"Summa and its diminutive summula was a generic category of text popularized in the thirteenth century Europe. In its most simplest sense, they might be considered texts that 'sum up' knowledge in a field, such as the compendiums of theology, philosophy and canon law.". Source: wikipedia.

# About
The experiments will be inspired by the teachings of Andrew Ng and his sources. I will do my best to provide all relevant sources.

# Code
I will try to work in Javascript at first, although I know that Andrew's deep learning courses and many others work with Python (perhaps other languages). Perhaps later I will switch to Python too in this repository.

Currently there is a 2 layered neural network fully inspired by: `http://cs231n.github.io/neural-networks-case-study/`

# Run

Open test.html in a standard web browser, open the developer console and type, for example: 

`neuralnetwork2L(numeric.transpose(X2), Y2, 50, 10000, 0.05, 0.01)`

corresponding to: neuralnetwork2L(X, Y, hidden_units, iterations, epsilon, lambda)

It is not currently converging, perhaps you can help find out why.
