import numpy as np
import time
from RKO_v2 import RKO

def ler_instance(name):
    path = 'C:\\Users\\felip\\OneDrive\\Documentos\\GitHub\\RKO\\C++\\Instances\\KP\\'
    
    items = []
    
    line_atual = 0
    with open(path + name, 'r') as file:

        for line in file:
            linha = line.split()
            if line_atual == 0:
                items_numero = linha[0]
                capacidade = int(linha[1])
            else:                
                items.append((int(linha[0]), int(linha[1])))

            line_atual += 1
    return [capacidade, items]

# print(ler_instance('kp10.txt'))

class KnapsackProblem:
    def __init__(self, instance):
        self.instance_name = f'{instance}'

        self.capacidade, self.items = ler_instance(instance)

        self.tam_solution = len(self.items)

        self.LS_type = 'Best'

        self.max_time = 50

        self.dict_best = None

    def decoder(self, keys):

        solution = np.argsort(keys)
        # print(solution)
        return solution
    
    def cost(self, solution):

        capacidade_usada = 0
        custo_total = 0

        for item in solution:
            if capacidade_usada + self.items[item][1] <= self.capacidade:
                capacidade_usada += self.items[item][1]
                custo_total += self.items[item][0]
            else:
                break

        return -1 * custo_total

class KnapsackProblem_2:
    def __init__(self, instance):
        self.instance_name = f'{instance}'

        self.capacidade, self.items = ler_instance(instance)

        self.tam_solution = len(self.items)

        self.LS_type = 'Best'

        self.max_time = 50

        self.dict_best = None
    def decoder(self, keys):
        return keys

    def cost(self, keys):
        selected_items_indices = []
        for i, key in enumerate(keys):
            if key > 0.5:
                selected_items_indices.append(i)

        temp_value = 0
        temp_weight = 0
        for idx in selected_items_indices:
            temp_value += self.items[idx][0]
            temp_weight += self.items[idx][1]

        if temp_weight > self.capacidade:
            penalty = (temp_weight - self.capacidade) * 1000000
            current_value = temp_value - penalty
        else:
            current_value = temp_value

        return -1 * current_value

if __name__ == '__main__':
    tempo = 50
    env = KnapsackProblem('kp50.txt')
    solver = RKO(env)

    time_start = time.time()
    k = 0
    while time.time() - time_start < 60:
        k+=1
        print(k)
        env.cost(env.decoder(solver.random_keys())) 
    cost,sol, temp = solver.solve(50,0.3,0.5,tempo,5,1,1,1,1,1)
        
