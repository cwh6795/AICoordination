import numpy as np

men_top = np.loadtxt('men_top.csv', delimiter=',', dtype=object)
men_bot = np.loadtxt('men_bot.csv', delimiter=',', dtype=object)
women_top = np.loadtxt('women_top.csv', delimiter=',', dtype=object)
women_bot = np.loadtxt('women_bot.csv', delimiter=',', dtype=object)

men_top = men_top[:, 1:]
men_bot = men_bot[:, 1:]
women_top = women_top[:, 1:]
women_bot = women_bot[:, 1:]


men_type_matrix = np.zeros((4, 8))
men_texture_matrix = np.zeros((19, 19))
men_style_matrix = np.zeros((7, 7))
women_type_matrix = np.zeros((6, 8))
women_texture_matrix = np.zeros((19, 19))
women_style_matrix = np.zeros((7, 7))


men_type_row = {'chinos':0, 'jean':1, 'shorts':2, 'sweatpants':3}
men_type_column = {'blouse':0, 'button-down':1, 'henley':2, 'hoodie':3, 'sweater':4, 'tank':5, 'tee':6, 'turtleneck':7}

women_type_row = {'chinos':0, 'culottes':1, 'jean':2, 'shorts':3, 'skirt':4, 'sweatpants':5}
women_type_column = {'blouse':0, 'button-down':1, 'henley':2, 'hoodie':3, 'sweater':4, 'tank':5, 'tee':6, 'turtleneck':7}

texture_row = {'abstract':0, 'animal':1, 'bird':2, 'butterfly':3, 'camo':4, 'camouflage':5, 'colorblock':6, 'daisy':7, 'destroyed':8, 'diamond':9, 'dot':10, 'grid':11, 
                'lace':12, 'marled':13, 'palm':14, 'ringer':15, 'stripe':16, 'tonal':17, 'zigzag':18}
texture_column = {'abstract':0, 'animal':1, 'bird':2, 'butterfly':3, 'camo':4, 'camouflage':5, 'colorblock':6, 'daisy':7, 'destroyed':8, 'diamond':9, 'dot':10, 'grid':11, 
                'lace':12, 'marled':13, 'palm':14, 'ringer':15, 'stripe':16, 'tonal':17, 'zigzag':18}

style_row = {'athletic':0, 'elegant':1, 'life':2, 'retro':3, 'sweet':4, 'trench':5, 'youth':6}
style_column = {'athletic':0, 'elegant':1, 'life':2, 'retro':3, 'sweet':4, 'trench':5, 'youth':6}


for top, bot in zip(men_top, men_bot):
    men_type_matrix[men_type_row[bot[0]], men_type_column[top[0]]] += 1
    men_texture_matrix[texture_row[bot[1]], texture_column[top[1]]] += 1
    men_style_matrix[style_row[bot[2]], style_column[top[2]]] += 1

for top, bot in zip(women_top, women_bot):
    women_type_matrix[women_type_row[bot[0]], women_type_column[top[0]]] += 1
    women_texture_matrix[texture_row[bot[1]], texture_column[top[1]]] += 1
    women_style_matrix[style_row[bot[2]], style_column[top[2]]] += 1

for string in men_type_matrix:
    print(string)

for string in men_texture_matrix:
    print(string)

for string in men_style_matrix:
    print(string)

for string in women_type_matrix:
    print(string)

for string in women_texture_matrix:
    print(string)

for string in women_style_matrix:
    print(string)

def file_writer(filename, matrix):
    with open(filename, 'w') as filetxt:
        for row in matrix:
            for i, dot in enumerate(row):
                if i != len(row) - 1:
                    filetxt.write(str(dot) + ',')
                else:
                    filetxt.write(str(dot))
            filetxt.write('\n')

file_writer('data/men_type.csv', men_type_matrix)
file_writer('data/men_texture.csv', men_texture_matrix)
file_writer('data/men_style.csv', men_style_matrix)
file_writer('data/women_type.csv', women_type_matrix)
file_writer('data/women_texture.csv', women_texture_matrix)
file_writer('data/women_style.csv', women_style_matrix)

# with open('test.csv', 'w') as filetxt:
#     for row in men_type_matrix:
#         for i, dot in enumerate(row):
#             if i != len(row) - 1:
#                 filetxt.write(str(dot) + ',')
#             else:
#                 filetxt.write(str(dot))
#         filetxt.write('\n')
