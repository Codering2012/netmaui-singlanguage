import inspect
from rtmlib import Hand

print("Hand __init__ signature:")
print(inspect.signature(Hand.__init__))
print("Hand docstring:")
print(Hand.__doc__)
