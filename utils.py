import numpy

CLASS = ["apple", "banana", "orange", "strawberry"]

def getNoise(batchSize:int, nNoise:int):
  return numpy.random.normal(0, 1, (batchSize, nNoise))
