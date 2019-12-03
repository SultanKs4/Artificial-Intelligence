import distance as dis
import numpy as np


def main():
    b1 = np.array([50000, 4, 1], dtype=np.float64)
    b2 = np.array([15000, 1, 5], dtype=np.float64)
    b3 = np.array([60000, 3, 2], dtype=np.float64)
    b4 = np.array([25000, 5, 3], dtype=np.float64)
    b5 = np.array([78000, 4, 5], dtype=np.float64)
    # Cosine
    print("Cosine\nCosine B5, B4 : " + str(dis.cosine(b5, b4)))
    print("Cosine B5, B3 : " + str(dis.cosine(b5, b3)))
    print("Cosine B5, B2 : " + str(dis.cosine(b5, b2)))
    print("Cosine B5, B1 : " + str(dis.cosine(b5, b1)))
    #  Jaccard
    print("Jaccard\nJaccard B5, B4 : " + str(dis.jaccard(b5, b4)))
    print("Jaccard B5, B3 : " + str(dis.jaccard(b5, b3)))
    print("Jaccard B5, B2 : " + str(dis.jaccard(b5, b2)))
    print("Jaccard B5, B1 : " + str(dis.jaccard(b5, b1)))
    # Dice
    print("Dice\nDice B5, B4 : " + str(dis.dice(b5, b4)))
    print("Dice B5, B3 : " + str(dis.dice(b5, b3)))
    print("Dice B5, B2 : " + str(dis.dice(b5, b2)))
    print("Dice B5, B1 : " + str(dis.dice(b5, b1)))
    # Euclidean
    print("Euclidean\nEuclidean B5, B4 : " + str(dis.euclidean(b5, b4)))
    print("Euclidean B5, B3 : " + str(dis.euclidean(b5, b3)))
    print("Euclidean B5, B2 : " + str(dis.euclidean(b5, b2)))
    print("Euclidean B5, B1 : " + str(dis.euclidean(b5, b1)))
    # Manhattan
    print("Manhattan\nManhattan B5, B4 : " + str(dis.manhattan(b5, b4)))
    print("Manhattan B5, B3 : " + str(dis.manhattan(b5, b3)))
    print("Manhattan B5, B2 : " + str(dis.manhattan(b5, b2)))
    print("Manhattan B5, B1 : " + str(dis.manhattan(b5, b1)))


if __name__ == "__main__":
    main()
