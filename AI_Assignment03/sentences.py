# ---- Examples Formula ----#

from logic import *

# Sentence 01: "If it's rains, it's wet".
def rainWet():
    Rain = Atom('Rain') # whether it's raining
    Wet = Atom('Wet') # whether it's wet
    return Implies(Rain, Wet)

# Sentence 02: "There is a light that shines."
def lightShines():
    def Light(x): return Atom('Light', x)    # whether x is lit
    def Shines(x): return Atom('Shines', x)  # whether x is shining
    return Exists('$x', And(Light('$x'), Shines('$x')))

# Defining Parent in terms of Child.
def parentChild():
    def Parent(x, y): return Atom('Parent', x, y)  # whether x has a parent y
    def Child(x, y): return Atom('Child', x, y)    # whether x has a child y
    return Forall('$x', Forall('$y', Equiv(Parent('$x', '$y'), Child('$y', '$x'))))
