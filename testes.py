import numpy as np
import random
from IA import RedeNeural


def convertResult(result):
    aux = []
    for i in range(len(result[0],)):
        aux.append(round(result[0][i] * 10 + i + 1))

    return aux

def convertTarget(arr):
    aux = []
    for i in range(len(arr)):
        aux.append((arr[i] - i+1) / 10)
    
    return aux

nn = RedeNeural(15, 15, 10, 15)

db = [[2, 3, 5, 6, 9, 10, 11, 13, 14, 16, 18, 20, 23, 24, 25],
[1, 4, 5, 6, 7, 9, 11, 12, 13, 15, 16, 19, 20, 23, 24],
[1, 4, 6, 7, 8, 9, 10, 11, 12, 14, 16, 17, 20, 23, 24],
[1, 2, 4, 5, 8, 10, 12, 13, 16, 17, 18, 19, 23, 24, 25],
[1, 2, 4, 8, 9, 11, 12, 13, 15, 16, 19, 20, 23, 24, 25],
[1, 2, 4, 5, 6, 7, 10, 12, 15, 16, 17, 19, 21, 23, 25],
[1, 4, 7, 8, 10, 12, 14, 15, 16, 18, 19, 21, 22, 23, 25],
[1, 5, 6, 8, 9, 10, 13, 15, 16, 17, 18, 19, 20, 22, 25],
[3, 4, 5, 9, 10, 11, 13, 15, 16, 17, 19, 20, 21, 24, 25],
[2, 3, 4, 5, 6, 8, 9, 10, 11, 12, 14, 19, 20, 23, 24],
[2, 6, 7, 8, 9, 10, 11, 12, 16, 19, 20, 22, 23, 24, 25],
[1, 2, 4, 5, 7, 8, 9, 10, 11, 12, 14, 16, 17, 24, 25],
[3, 5, 6, 7, 8, 9, 10, 11, 13, 14, 15, 16, 17, 19, 23],
[1, 2, 5, 6, 7, 9, 13, 14, 15, 18, 19, 20, 21, 23, 25],
[1, 2, 4, 6, 8, 10, 12, 15, 16, 18, 19, 21, 23, 24, 25],
[2, 5, 6, 7, 8, 10, 12, 13, 15, 17, 19, 21, 23, 24, 25],
[1, 2, 3, 5, 6, 7, 9, 13, 14, 16, 17, 18, 19, 20, 21],
[2, 6, 7, 8, 10, 11, 14, 15, 17, 18, 19, 20, 22, 23, 24],
[2, 5, 6, 7, 8, 10, 11, 13, 14, 15, 16, 17, 20, 23, 24],
[3, 4, 6, 7, 8, 9, 10, 14, 16, 17, 18, 19, 20, 23, 24],
[1, 2, 4, 5, 8, 11, 14, 16, 18, 19, 20, 22, 23, 24, 25],
[1, 2, 3, 4, 5, 6, 7, 9, 10, 12, 13, 14, 15, 22, 25],
[1, 3, 4, 5, 6, 8, 10, 11, 12, 14, 16, 17, 18, 19, 20],
[1, 2, 3, 5, 7, 10, 11, 14, 17, 19, 20, 21, 23, 24, 25],
[1, 2, 3, 4, 5, 6, 7, 9, 13, 14, 16, 20, 22, 23, 24],
[5, 7, 8, 9, 10, 11, 13, 14, 16, 17, 19, 20, 21, 22, 23],
[3, 6, 8, 10, 11, 12, 13, 14, 15, 18, 20, 21, 22, 24, 25],
[1, 3, 7, 9, 10, 11, 12, 13, 14, 16, 17, 18, 19, 20, 21],
[1, 4, 5, 6, 8, 9, 13, 14, 16, 17, 19, 20, 21, 22, 24],
[1, 2, 3, 4, 6, 7, 8, 11, 14, 17, 19, 20, 21, 22, 23],
[1, 2, 3, 4, 9, 13, 14, 15, 17, 19, 20, 21, 22, 24, 25],
[1, 2, 4, 6, 7, 9, 10, 11, 14, 15, 16, 17, 20, 22, 23],
[1, 2, 5, 7, 8, 10, 11, 12, 14, 16, 19, 20, 21, 23, 24],
[1, 2, 4, 7, 8, 9, 10, 11, 15, 16, 18, 19, 20, 21, 23],
[1, 4, 5, 6, 11, 12, 13, 14, 16, 17, 19, 21, 22, 23, 25],
[1, 4, 5, 7, 8, 10, 11, 14, 17, 19, 20, 21, 22, 23, 24],
[1, 3, 4, 5, 8, 9, 10, 11, 13, 15, 20, 21, 22, 23, 24],
[1, 2, 3, 6, 7, 8, 9, 10, 13, 14, 15, 16, 22, 24, 25],
[2, 7, 8, 9, 11, 13, 14, 15, 17, 18, 19, 21, 22, 23, 24],
[1, 2, 5, 7, 8, 10, 12, 13, 14, 16, 17, 20, 21, 22, 24],
[2, 3, 4, 9, 12, 13, 15, 16, 17, 18, 19, 20, 21, 23, 25],
[1, 3, 4, 5, 6, 8, 11, 12, 14, 15, 17, 21, 22, 24, 25],
[1, 3, 6, 7, 8, 9, 10, 14, 17, 18, 19, 20, 22, 23, 24],
[3, 4, 5, 6, 10, 11, 12, 13, 14, 18, 19, 21, 23, 24, 25],
[1, 2, 3, 5, 7, 9, 14, 16, 17, 18, 19, 21, 23, 24, 25],
[1, 2, 4, 5, 6, 8, 10, 11, 14, 18, 19, 21, 23, 24, 25],
[1, 2, 3, 4, 5, 6, 8, 10, 11, 13, 15, 18, 20, 22, 25],
[2, 5, 6, 7, 11, 13, 15, 16, 17, 18, 19, 20, 21, 22, 23],
[2, 4, 5, 6, 8, 11, 13, 15, 16, 19, 20, 21, 22, 23, 24],
[1, 2, 3, 6, 7, 9, 10, 11, 12, 13, 19, 20, 21, 23, 25],
[1, 3, 5, 6, 7, 8, 11, 13, 14, 16, 17, 20, 21, 22, 23],
[1, 2, 4, 8, 9, 11, 12, 13, 15, 16, 21, 22, 23, 24, 25],
[1, 2, 3, 6, 7, 9, 11, 12, 14, 17, 18, 19, 20, 23, 24],
[2, 4, 5, 6, 7, 8, 9, 12, 14, 16, 18, 20, 21, 22, 24],
[2, 3, 4, 5, 6, 7, 8, 9, 12, 13, 15, 16, 18, 19, 25],
[1, 2, 5, 9, 12, 13, 14, 15, 16, 17, 19, 20, 22, 23, 24],
[1, 2, 3, 4, 6, 7, 8, 11, 12, 17, 19, 20, 21, 22, 25],
[3, 4, 5, 6, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 22],
[1, 3, 5, 6, 7, 8, 10, 11, 13, 14, 15, 19, 20, 23, 25],
[1, 2, 3, 4, 5, 8, 11, 16, 17, 18, 19, 22, 23, 24, 25],
[1, 4, 5, 9, 11, 12, 13, 14, 15, 16, 19, 20, 22, 23, 25],
[1, 3, 7, 8, 9, 11, 13, 14, 15, 16, 17, 18, 20, 24, 25],
[4, 8, 9, 10, 11, 12, 13, 16, 17, 19, 20, 21, 22, 24, 25],
[1, 2, 3, 6, 7, 8, 9, 10, 13, 14, 17, 18, 20, 23, 25],
[1, 2, 4, 7, 10, 15, 16, 17, 18, 19, 20, 21, 22, 23, 25],
[1, 2, 4, 8, 10, 11, 14, 15, 16, 17, 20, 21, 22, 23, 24],
[5, 7, 8, 9, 10, 11, 12, 13, 15, 16, 20, 21, 22, 23, 24],
[1, 2, 3, 5, 6, 7, 8, 9, 10, 12, 16, 17, 22, 23, 25],
[1, 4, 7, 8, 9, 10, 11, 15, 17, 18, 19, 20, 21, 23, 24],
[1, 2, 3, 5, 6, 9, 10, 12, 14, 15, 16, 19, 20, 24, 25],
[1, 3, 5, 6, 8, 9, 11, 13, 14, 15, 16, 18, 21, 22, 23],
[1, 2, 5, 7, 8, 11, 12, 13, 15, 16, 17, 18, 23, 24, 25],
[1, 2, 5, 7, 10, 11, 13, 14, 15, 16, 17, 18, 19, 21, 25],
[1, 2, 3, 4, 5, 6, 8, 11, 13, 15, 16, 18, 22, 23, 25],
[1, 3, 4, 5, 6, 7, 8, 10, 11, 13, 15, 16, 17, 20, 25],
[1, 3, 5, 8, 9, 10, 12, 13, 15, 16, 17, 19, 21, 23, 25],
[2, 3, 4, 6, 8, 9, 11, 14, 17, 18, 20, 21, 22, 24, 25],
[1, 2, 3, 5, 6, 7, 8, 12, 13, 15, 16, 17, 18, 20, 23],
[1, 2, 3, 5, 9, 12, 14, 15, 17, 18, 20, 22, 23, 24, 25],
[1, 2, 3, 4, 5, 7, 9, 10, 14, 15, 17, 18, 20, 22, 25],
[1, 2, 3, 4, 5, 6, 7, 10, 11, 13, 15, 16, 17, 19, 21],
[1, 4, 5, 6, 8, 9, 12, 13, 15, 17, 18, 19, 20, 22, 23],
[1, 2, 3, 5, 6, 8, 10, 11, 12, 13, 14, 20, 23, 24, 25],
[1, 2, 4, 6, 8, 12, 15, 16, 17, 19, 20, 22, 23, 24, 25],
[1, 2, 5, 7, 8, 9, 11, 13, 14, 15, 16, 18, 21, 22, 23],
[2, 4, 5, 6, 8, 9, 12, 13, 14, 17, 18, 19, 23, 24, 25],
[1, 2, 3, 5, 8, 9, 10, 11, 13, 15, 18, 20, 22, 24, 25],
[1, 2, 4, 5, 8, 9, 10, 11, 12, 14, 17, 18, 19, 24, 25],
[1, 6, 11, 12, 13, 15, 16, 17, 19, 20, 21, 22, 23, 24, 25],
[1, 2, 3, 8, 10, 11, 12, 15, 17, 20, 21, 22, 23, 24, 25],
[1, 4, 9, 10, 11, 12, 13, 14, 15, 16, 17, 20, 22, 23, 25],
[1, 2, 4, 6, 8, 12, 13, 14, 15, 17, 18, 19, 21, 22, 24],
[2, 4, 5, 6, 7, 9, 10, 13, 17, 18, 19, 20, 21, 23, 24],
[2, 4, 8, 9, 11, 12, 13, 15, 16, 17, 18, 21, 22, 23, 24],
[1, 2, 3, 4, 5, 10, 11, 12, 14, 15, 17, 18, 19, 20, 22],
[3, 4, 6, 9, 10, 12, 14, 16, 17, 18, 19, 21, 22, 23, 24],
[3, 4, 6, 10, 12, 13, 14, 15, 17, 19, 20, 22, 23, 24, 25],
[1, 2, 5, 6, 8, 10, 11, 15, 17, 18, 20, 22, 23, 24, 25],
[1, 2, 3, 4, 7, 10, 11, 13, 14, 16, 17, 20, 21, 24, 25],
[1, 3, 4, 8, 9, 10, 11, 12, 13, 14, 16, 17, 22, 23, 25]]

# print(random.randint(0, 2))



# print(target)

# for i in range(10000):
#     print(i)
#     index = random.randint(0, 48)
#     # print(index)
#     # print(db[index])
#     input = convertTarget(db[index])
#     target = convertTarget(db[index+1])
#     nn.train([input], [target])

result = nn.predict([db[25]]).tolist()

formatedResult = convertResult(result)

print('result: ',result)
print('formated result: ', formatedResult)
# with open('config.txt', 'r+') as configFile:
#     lines = configFile.read().split('#\n')
#     bias = lines[0],.split('_')
#     weigths = lines[1],.split('_')
    # bias = [],

    # for i in range(len(lines)):
    #     bias.append(eval(lines[i],.replace('\n', '')))

# input = np.array(np.transpose([[10, 20],],))
# weigths_ih = eval(weigths[0],)
    
# print(type(weigths_ih))
# print(type(input))

# print(weigths_ih)

# print('weigths_ih:', weigths_ih)
# print('input:', input)
# print(np.dot(weigths_ih, input))

# print(round(0.6 * 10 + 2))
# print((8 - 2) / 10)

# with open('Resultados.txt', 'r') as db:
#     results = db.readlines()

# print(np.array(eval(results[0],)))

