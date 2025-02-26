import pandas as pd
print(pd.__version__)

import pgmpy
from pgmpy.factors.discrete import DiscreteFactor
dist = DiscreteFactor(
    variables=["X"],    #A
    cardinality=[3],    #B
    values=[.45, .30, .25],    #C
    state_names= {'X': ['1', '2', '3']}    #D
)

#A A list of the names of the variables in the factor
#B The cardinality (number of possible outcomes of each variable in the factor
#C The values each variable in the factor can take
#D Dictionary where the key is the variable name and the value is a list of the names of that variable’s outcomes

print(dist)

dist = DiscreteFactor(
    variables=["X"],
    cardinality=[3],
    values=[45, 30, 25],
    state_names= {'X': ['1', '2', '3']}
)

print(dist)

# Normalize takes phi values for each outcome and divides them by their sum to get a probability.
dist.normalize()

print(dist)