import os
import numpy as np
import time
import random
import copy
import math
import datetime
import bisect
from multiprocessing import Manager, Process, cpu_count

class SolutionPool():
    def __init__(self, size):
        self.size = size
        self.pool = []
        self.best = None
        self.best_cost = float('inf')
        
    def insert(self, entry):
        if entry[0] < self.best_cost:
            self.best_cost = entry[0]
            self.best = entry[1]
            
        bisect.insort(self.pool, entry)
        if len(self.pool) > self.size:
            self.pool.pop()

class RKO():
    def __init__(self, env):
        self.env = env
        self.__MAX_KEYS = self.env.tam_solution
        self.LS_type = self.env.LS_type
        self.start_time = time.time()
        self.max_time = self.env.max_time
        
        
    
    def random_keys(self):
        return np.random.random(self.__MAX_KEYS)        
    
    def shaking(self, keys, beta_min, beta_max):
        
        beta = random.uniform(beta_min, beta_max)
        new_keys = copy.deepcopy(keys)
        
        numero_pertubacoes = max(1, int(self.__MAX_KEYS * beta))
        for _ in range(numero_pertubacoes):
            
            tipo = random.choice(['Swap', 'SwapN', 'Invert', 'Random'])
            
            
            if tipo == 'Swap':
                idx1, idx2 = random.sample(range(self.__MAX_KEYS), 2)
                new_keys[idx1], new_keys[idx2] = new_keys[idx2], new_keys[idx1]
                
            
            elif tipo == 'SwapN':
                idx = random.randint(0, self.__MAX_KEYS - 1)
                
                if idx == 0:
                    new_keys[idx], new_keys[idx + 1] = new_keys[idx + 1], new_keys[idx]                   
                elif idx == self.__MAX_KEYS - 1:
                    new_keys[idx], new_keys[idx - 1] = new_keys[idx - 1], new_keys[idx]                    
                else:
                    idx2 = random.choice([idx - 1, idx + 1])
                    new_keys[idx], new_keys[idx2] = new_keys[idx2], new_keys[idx]
                               
            elif tipo == 'Invert':
                idx = random.randint(0, self.__MAX_KEYS - 1)
                
                key = new_keys[idx]
                new_keys[idx] = 1 - key  
                            
            elif tipo == 'Random':                
                idx = random.randint(0, self.__MAX_KEYS - 1)
                new_keys[idx] = random.random()
                
            # print(f"Perturbação: {tipo} - Chave: {len(new_keys) == self.__MAX_KEYS} - Valor: {self.env.cost(self.env.decoder(new_keys))}")
        
        return new_keys
        
    def SwapLS(self, keys):
        
        if self.LS_type == 'Best':
            
            swap_order = [i for i in range(self.__MAX_KEYS)]
            random.shuffle(swap_order)
            
            best_keys = copy.deepcopy(keys)
            best_cost = self.env.cost(self.env.decoder(best_keys))
            
            
            k = 0
            for idx1 in swap_order:
                for idx2 in reversed(swap_order):
                    
                    if self.env.dict_best is not None and best_cost == self.env.dict_best[self.env.instance_name]:
                        return best_keys
                    if time.time() - self.start_time > self.max_time:
                        return best_keys

                    k+=1
                    
                    new_keys = copy.deepcopy(best_keys)
                    new_keys[idx1], new_keys[idx2] = new_keys[idx2], new_keys[idx1]
                    new_cost = self.env.cost(self.env.decoder(new_keys))
                    
                    if new_cost < best_cost:
                        best_keys = new_keys
                        best_cost = new_cost    

            return best_keys
        elif self.LS_type == 'First':
            
            swap_order = [i for i in range(self.__MAX_KEYS)]
            random.shuffle(swap_order)
            
            best_keys = copy.deepcopy(keys)
            best_cost = self.env.cost(self.env.decoder(best_keys))
            
            for idx1 in swap_order:
                for idx2 in reversed(swap_order):
                    
                    if self.env.dict_best is not None and best_cost == self.env.dict_best[self.env.instance_name]:
                        return best_keys
                    if time.time() - self.start_time > self.max_time:
                        return best_keys
                        
                    new_keys = copy.deepcopy(best_keys)
                    new_keys[idx1], new_keys[idx2] = new_keys[idx2], new_keys[idx1]
                    new_cost = self.env.cost(self.env.decoder(new_keys))
                    
                    if new_cost < best_cost:
                        best_keys = new_keys
                        best_cost = new_cost
                        
                        return best_keys
                    
            return best_keys
            
    def FareyLS(self, keys):
        
        Farey_Squence = [0.00, 0.142857, 0.166667, 0.20, 0.25, 0.285714, 0.333333, 0.40, 0.428571, 0.50, 
                                 0.571429, 0.60, 0.666667, 0.714286, 0.75, 0.80, 0.833333, 0.857143, 1.0]
        if self.LS_type == 'Best':
            
            swap_order = [i for i in range(self.__MAX_KEYS)]
            random.shuffle(swap_order)
            
            best_keys = copy.deepcopy(keys)
            best_cost = self.env.cost(self.env.decoder(best_keys))
            
            
            k = 0
            for idx in swap_order:
                for i in range(len(Farey_Squence) - 1):
                    
                    if self.env.dict_best is not None and best_cost == self.env.dict_best[self.env.instance_name]:
                        return best_keys
                    if time.time() - self.start_time > self.max_time:
                        return best_keys

                    k+=1
                    
                    new_keys = copy.deepcopy(best_keys)
                    new_keys[idx] = random.uniform(Farey_Squence[i], Farey_Squence[i+1])
                    new_cost = self.env.cost(self.env.decoder(new_keys))
                    
                    if new_cost < best_cost:
                        best_keys = new_keys
                        best_cost = new_cost    
 
            return best_keys
        elif self.LS_type == 'First':
            
            swap_order = [i for i in range(self.__MAX_KEYS)]
            random.shuffle(swap_order)
            
            best_keys = copy.deepcopy(keys)
            best_cost = self.env.cost(self.env.decoder(best_keys))
            
            for idx in swap_order:
                for i in range(len(Farey_Squence) - 1):
                    
                    if self.env.dict_best is not None and best_cost == self.env.dict_best[self.env.instance_name]:
                        return best_keys
                    if time.time() - self.start_time > self.max_time:
                        return best_keys
                        
                    new_keys = copy.deepcopy(best_keys)
                    new_keys[idx] = random.uniform(Farey_Squence[i], Farey_Squence[i+1])
                    new_cost = self.env.cost(self.env.decoder(new_keys))
                    
                    if new_cost < best_cost:
                        best_keys = new_keys
                        best_cost = new_cost    
                        return best_keys
                        
            return best_keys
    
    def InvertLS(self, keys):
        if self.LS_type == 'Best':
            
            swap_order = [i for i in range(self.__MAX_KEYS)]
            random.shuffle(swap_order)
            
            best_keys = copy.deepcopy(keys)
            best_cost = self.env.cost(self.env.decoder(best_keys))
            
            
            k = 0
            for idx in swap_order:
                
                if self.env.dict_best is not None and best_cost == self.env.dict_best[self.env.instance_name]:
                    return best_keys
                if time.time() - self.start_time > self.max_time:
                    return best_keys

                k+=1
                new_keys = copy.deepcopy(best_keys)
                new_keys[idx] = 1 - new_keys[idx]
                new_cost = self.env.cost(self.env.decoder(new_keys))
                
                if new_cost < best_cost:
                    best_keys = new_keys
                    best_cost = new_cost    

            return best_keys    
        elif self.LS_type == 'First':
            
            swap_order = [i for i in range(self.__MAX_KEYS)]
            random.shuffle(swap_order)
            
            best_keys = copy.deepcopy(keys)
            best_cost = self.env.cost(self.env.decoder(best_keys))
            
            for idx in swap_order:
            
                if self.env.dict_best is not None and best_cost == self.env.dict_best[self.env.instance_name]:
                    return best_keys
                if time.time() - self.start_time > self.max_time:
                    return best_keys
                        
                new_keys = copy.deepcopy(best_keys)
                new_keys[idx] = 1 - new_keys[idx]
                new_cost = self.env.cost(self.env.decoder(new_keys))
                    
                if new_cost < best_cost:
                    best_keys = new_keys
                    best_cost = new_cost
                    
                    return best_keys
                
            return best_keys
    # def Blending(self, keys1, keys2, factor):
  
    #     new_keys = np.zeros(self.__MAX_KEYS)
        
    #     for i in range(self.__MAX_KEYS):
            
    #         if random.random() < 0.02: 
    #             new_keys[i] = random.random()
                
    #         else:               
    #             if random.random() < 0.5:
    #                 new_keys[i] = keys1[i]
    #             else:
    #                 if factor == -1:
    #                     new_keys[i] = 1 - keys2[i]
    #                 else:
    #                     new_keys[i] = keys2[i] 
        
    #     return new_keys
    
    
    # def NelderMeadSearch(self, keys, pool):
    #     keys1 = copy.deepcopy(keys)
    #     keys_S = random.sample(list(pool), 1)[0][1]
    #     keys_H = random.sample(list(pool), 1)[0][1]
    #     while keys_S == keys_H: 
    #         keys_S = random.sample(list(pool), 1)[0][1]
    #         keys_H = random.sample(list(pool), 1)[0][1]
            
        
    #     fit1 = self.env.cost(self.env.decoder(keys1))
    #     fit_S = self.env.cost(self.env.decoder(keys_S))
    #     fit_H = self.env.cost(self.env.decoder(keys_H))
        
    #     if fit1 < fit_S and fit1 < fit_H:
    #         x1 = keys1
    #         if fit_S < fit_H:
    #             x2 = keys_S
    #             x3 = keys_H
    #         else:
    #             x2 = keys_H
    #             x3 = keys_S
            
    #     elif fit_S < fit1 and fit_S < fit_H:
    #         x1 = keys_S
    #         if fit1 < fit_H:
    #             x2 = keys1
    #             x3 = keys_H
    #         else:
    #             x2 = keys_H
    #             x3 = keys1
    #     else:
    #         x1 = keys_H
    #         if fit1 < fit_S:
    #             x2 = keys1
    #             x3 = keys_S
    #         else:
    #             x2 = keys_S
    #             x3 = keys1
                
    #     xBest = x1
    #     CostBest = fit1
        
    #     x0 = self.Blending(x1, x2, 1)
    #     fit0 = self.env.cost(self.env.decoder(x0))
    #     if fit0 < CostBest:
    #         xBest = x0
    #         CostBest = fit0
    #         melhorou = 1
            
    #     iter = 1
        
    #     max_iter = math.exp(-2) * self.__MAX_KEYS
    #     while iter < max_iter:
    #         pass
        
    def Blending(self, keys1, keys2, factor):
        new_keys = np.zeros(self.__MAX_KEYS)
        
        for i in range(self.__MAX_KEYS):
            # Mutação com 2% de probabilidade
            if random.random() < 0.02: 
                new_keys[i] = random.random()
            else:               
                if random.random() < 0.5:
                    new_keys[i] = keys1[i]
                else:
                    if factor == -1:
                        # Aplicar clamp como no C++: std::clamp(1.0 - s2.rk[j], 0.0, 0.9999999)
                        new_keys[i] = max(0.0, min(1.0 - keys2[i], 0.9999999))
                    else:
                        new_keys[i] = keys2[i] 
        
        return new_keys
    
    def NelderMeadSearch(self, keys, pool = None):
        improved = 0
        improvedX1 = 0
        keys_origem = copy.deepcopy(keys)
        

        
        x1 = copy.deepcopy(keys)
        if pool is None:
            x2 = self.random_keys()
            x3 = self.random_keys()
        else:
            
            x2 = random.sample(list(pool), 1)[0][1]  
            x3 = random.sample(list(pool), 1)[0][1]
            while x2 == x3:
                x2 = random.sample(list(pool), 1)[0][1]
                x3 = random.sample(list(pool), 1)[0][1]
        
        # Calcular fitness
        fit1 = self.env.cost(self.env.decoder(x1))
        fit2 = self.env.cost(self.env.decoder(x2))
        fit3 = self.env.cost(self.env.decoder(x3))
        
        # Ordenar pontos: x1 (melhor) <= x2 <= x3 (pior)
        if fit1 > fit2:
            x1, x2 = x2, x1
            fit1, fit2 = fit2, fit1
            
        if fit1 > fit3:
            x1, x3 = x3, x1
            fit1, fit3 = fit3, fit1
            
        if fit2 > fit3:
            x2, x3 = x3, x2
            fit2, fit3 = fit3, fit2
        
        xBest = copy.deepcopy(x1)
        fitBest = fit1
        
        # Calcular centroide do simplex
        x0 = self.Blending(x1, x2, 1)
        fit0 = self.env.cost(self.env.decoder(x0))
        if fit0 < fitBest:
            xBest = copy.deepcopy(x0)
            fitBest = fit0
            improved = 1
            
        iter_count = 1
        eval_count = 0
        
        # Critério de parada igual ao C++
        max_iter = int(self.__MAX_KEYS * math.exp(-2))
        
        while iter_count <= max_iter:
            # print(f"FIT1: {fit1} - FIT2: {fit2} - FIT3: {fit3}")
            shrink = 0
            
            # Ponto de reflexão (r)
            x_r = self.Blending(x0, x3, -1)
            fit_r = self.env.cost(self.env.decoder(x_r))
            if fit_r < fitBest:
                xBest = copy.deepcopy(x_r)
                fitBest = fit_r
                improved = 1
                improvedX1 = 1
            eval_count += 1
            
            # x_r é melhor que x1 (melhor ponto)
            if fit_r < fit1:
                # Ponto de expansão (e)
                x_e = self.Blending(x_r, x0, -1)
                fit_e = self.env.cost(self.env.decoder(x_e))
                if fit_e < fitBest:
                    xBest = copy.deepcopy(x_e)
                    fitBest = fit_e
                    improved = 1
                    improvedX1 = 1
                eval_count += 1
                
                if fit_e < fit_r:
                    # Expandir
                    x3 = copy.deepcopy(x_e)
                    fit3 = fit_e
                else:
                    # Refletir
                    x3 = copy.deepcopy(x_r)
                    fit3 = fit_r
                    
            # x_r NÃO é melhor que x1
            else:
                # x_r é melhor que x2 (segundo melhor)
                if fit_r < fit2:
                    # Refletir
                    x3 = copy.deepcopy(x_r)
                    fit3 = fit_r
                else:
                    # x_r é melhor que x3 (pior)
                    if fit_r < fit3:
                        # Ponto de contração (c)
                        x_c = self.Blending(x_r, x0, 1)
                        fit_c = self.env.cost(self.env.decoder(x_c))
                        if fit_c < fitBest:
                            xBest = copy.deepcopy(x_c)
                            fitBest = fit_c
                            improved = 1
                            improvedX1 = 1
                        eval_count += 1
                        
                        if fit_c < fit_r:
                            # Contrair para fora
                            x3 = copy.deepcopy(x_c)
                            fit3 = fit_c
                        else:
                            # Encolher
                            shrink = 1
                    else:
                        # Ponto de contração (c)
                        x_c = self.Blending(x0, x3, 1)
                        fit_c = self.env.cost(self.env.decoder(x_c))
                        if fit_c < fitBest:
                            xBest = copy.deepcopy(x_c)
                            fitBest = fit_c
                            improved = 1
                            improvedX1 = 1
                        eval_count += 1
                        
                        if fit_c < fit3:
                            # Contrair para dentro
                            x3 = copy.deepcopy(x_c)
                            fit3 = fit_c
                        else:
                            # Encolher
                            shrink = 1
            
            # Operação de encolhimento
            if shrink:
                x2 = self.Blending(x1, x2, 1)
                fit2 = self.env.cost(self.env.decoder(x2))
                if fit2 < fitBest:
                    xBest = copy.deepcopy(x2)
                    fitBest = fit2
                    improved = 1
                    improvedX1 = 1
                eval_count += 1
                
                x3 = self.Blending(x1, x3, 1)
                fit3 = self.env.cost(self.env.decoder(x3))
                if fit3 < fitBest:
                    xBest = copy.deepcopy(x3)
                    fitBest = fit3
                    improved = 1
                    improvedX1 = 1
                eval_count += 1
            
            # Reordenar pontos
            if fit1 > fit2:
                x1, x2 = x2, x1
                fit1, fit2 = fit2, fit1
                
            if fit1 > fit3:
                x1, x3 = x3, x1
                fit1, fit3 = fit3, fit1
                
            if fit2 > fit3:
                x2, x3 = x3, x2
                fit2, fit3 = fit3, fit2
            
            # Calcular novo centroide
            x0 = self.Blending(x1, x2, 1)
            fit0 = self.env.cost(self.env.decoder(x0))
            if fit0 < fitBest:
                xBest = copy.deepcopy(x0)
                fitBest = fit0
                improved = 1
                improvedX1 = 1
            
            # Controle de iterações
            if improved == 1:
                improved = 0
                iter_count = 0
            else:
                iter_count += 1
            
            # Verificar condição de parada (equivalente ao stop_execution.load())
            # if self.should_stop():
            #     return keys_origem if improvedX1 == 0 else xBest
        
        # Retornar melhor solução encontrada
        if improvedX1 == 1:
            return xBest
        else:
            return keys_origem
            
        
        
 
    def RVND(self, keys):
        
        best_keys = copy.deepcopy(keys)
        best_cost = self.env.cost(self.env.decoder(best_keys))
        
        
        
        neighborhoods = ['SwapLS', 'NelderMeadSearch','FareyLS', 'InvertLS']
        not_used_nb = ['SwapLS', 'NelderMeadSearch','FareyLS', 'InvertLS']
        
        while not_used_nb:
            
            current_neighborhood = random.choice(not_used_nb)
            
            
            if current_neighborhood == 'SwapLS':
                new_keys = self.SwapLS(best_keys)
            elif current_neighborhood == 'NelderMeadSearch':               
                new_keys = self.NelderMeadSearch(best_keys)             
            elif current_neighborhood == 'FareyLS':
                new_keys = self.FareyLS(best_keys)
            elif current_neighborhood == 'InvertLS':
                new_keys = self.InvertLS(best_keys)
                
            new_cost = self.env.cost(self.env.decoder(new_keys))
            # print(f"Neighborhood: {current_neighborhood} - Custo: {new_cost}")
            
            if new_cost < best_cost:
                best_keys = new_keys
                best_cost = new_cost
                if self.env.dict_best is not None and best_cost == self.env.dict_best[self.env.instance_name]:
                    return best_keys
                not_used_nb = copy.deepcopy(neighborhoods)
                
            else:
                not_used_nb.remove(current_neighborhood)
            
        
        return best_keys
            
            

    
    def inserir_em_elite(self, elite, fitness_elite, key, fitness, tam_elite, modo='maiores'):
        if modo == 'maiores':
            # Lista em ordem decrescente (maior no começo, menor no fim)
            comparar = lambda novo, atual: novo > atual
        elif modo == 'menores':
            # Lista em ordem crescente (menor no começo, maior no fim)
            comparar = lambda novo, atual: novo < atual
        else:
            raise ValueError("modo deve ser 'maiores' ou 'menores'")

        i = 0
        for value in fitness_elite:
            if comparar(fitness, value):
                break
            i += 1

        # Se ainda tem espaço, só insere na posição correta
        if len(elite) < tam_elite:
            elite.insert(i, key)
            fitness_elite.insert(i, fitness)
        else:
            # Se o novo fitness é melhor que o último (pior da elite), substitui
            if comparar(fitness, fitness_elite[-1]):
                elite.pop(-1)
                fitness_elite.pop(-1)
                elite.insert(i, key)
                fitness_elite.insert(i, fitness)

        if modo == 'maiores':
            elite.sort(key=lambda x: fitness_elite[elite.index(x)], reverse=True)
            fitness_elite.sort(reverse=True)
        else:
            elite.sort(key=lambda x: fitness_elite[elite.index(x)])
            fitness_elite.sort()
        return elite, fitness_elite


    def vizinhos(self, keys, min = 0.1, max = 0.25):
   
        new_keys = keys.copy()
        N = len(new_keys)

        # número de posições a perturbar
        min_p = int(min * N)
        max_p = int(max * N)
        n_perturb = random.randint(min_p, max_p)

        for _ in range(n_perturb):
            i = random.randrange(N)
            new_keys[i] = random.random()

        return new_keys
    def pertubacao(self, keys): #Realiza perturbações nas chaves, gerando vizinhos, soluções proximas/parecidas, para a solução atual.
        new_keys = copy.deepcopy(keys)
        prob = random.random()
        if prob < 0.5:
            alteracoes = math.ceil(self.__MAX_KEYS * 0.1) # altera 10% das chaves
            for _ in range(int(alteracoes)):
                idx1, idx2 = random.sample(range(self.__MAX_KEYS), 2)  # Escolhe dois índices distintos aleatórios
                new_keys[idx1], new_keys[idx2] = new_keys[idx2], new_keys[idx1]  # Realiza o swap


        elif prob < 0.7:
            alteracoes = math.ceil(self.__MAX_KEYS * 0.25) # altera 25% das chaves
            for _ in range(int(alteracoes)):
                idx1, idx2 = random.sample(range(self.__MAX_KEYS), 2)  # Escolhe dois índices distintos aleatórios
                new_keys[idx1], new_keys[idx2] = new_keys[idx2], new_keys[idx1]  # Realiza o swap


        elif  prob < 0.9:     
            for i, key in enumerate(new_keys):
                if random.random() > 0.5:
                    new_keys[i] = key + random.uniform(-0.5*key, 0.5*(1-key)) # aumenta ou diminui metade das chaves, mas com valores baixos
            
        else:
            
            for i, key in enumerate(new_keys):
                if random.random() > 0.7: # gera aleatouramente novos valores para 30% das chaves
                    new_keys[i] = random.random()
                    
        return new_keys
    
    def LocSearch(self, keys, x): # Busca local, recebe uma chave/solução e busca ao redor dela outras soluções, para ver se existe algum solução melhor
        iter = 0
        best_keys = keys
        solution = self.env.decoder(keys)
        best_cost = self.env.cost(solution)
        while iter < x:
            new_keys = self.vizinhos(best_keys)
            new_solution = self.env.decoder(new_keys)           
            new_cost = self.env.cost(new_solution)
            print(f"x {iter}, Custo: {new_cost}")
            
            if new_cost < best_cost: 
                best_keys = new_keys
                best_cost = new_cost
                iter = 0
            else:
                iter += 1
        
        return best_keys
    
    def MultiStart(self,max_iter,x, tempo, tag,  pool,lock,best): # Multi Start, gera várias soluções aleatórias e aplica a busca local em cada uma delas, retornando a melhor solução encontrada
        best_keys = None
        best_cost = float('inf')
        best_ini_cost = float('inf')
        start_time = time.time()
        iter = 0
        random_keys = self.random_keys()
        ini_solution = self.env.decoder(random_keys)
        ini_cost = self.env.cost(ini_solution)
        with lock:
            entry = (ini_cost, random_keys)
            
            bisect.insort(pool, entry)       
            if len(pool) > 10:
                pool.pop() 

        while time.time() - start_time < tempo:
            
            
            iter += 1
            k = 0
            while True:
                k+=1
                # print(pool)
                with lock:
                    if len(pool) > 0:
                        # print("POOL")
                        break
            random_keys = random.sample(list(pool), 1)[0][1]
            ini_solution = self.env.decoder(random_keys)
            ini_cost = self.env.cost(ini_solution)
            if ini_cost < best_ini_cost:
                best_ini_cost = ini_cost
            
            keys = self.LocSearch(random_keys,x)

            solution = self.env.decoder(keys)
            cost = self.env.cost(solution)
            with lock:
                entry = (cost, keys)
                
                bisect.insort(pool, entry)       
                if len(pool) > 20:
                    pool.pop()

            # with lock:
            #     if cost < best[0]:
            #         best[0] = cost
            #         best[1] = keys
            #         print(f"\n MS {tag} NOVO MELHOR: {cost} BEST:{self.env.dict_best[self.env.instance_name]} - GAP: {round((cost - self.env.dict_best[self.env.instance_name]) / self.env.dict_best[self.env.instance_name] * 100, 2)}%") 
            
            print(f"\rtempo = {round(time.time() - start_time,2)}", end="")
            
            if cost < best_cost:
                best_cost = cost
                best_keys = keys
                
                print(f"\n MS {tag} NOVO MELHOR: {cost} BEST:{self.env.dict_best[self.env.instance_name]} - GAP: {round((cost - self.env.dict_best[self.env.instance_name]) / self.env.dict_best[self.env.instance_name] * 100, 2)}%") 
                if best_cost == self.env.dict_best[self.env.instance_name]:
                        # print(f" \n{tag} MELHOR: {fitness} - BEST:{self.env.dict_best[self.env.instance_name]} - GAP: {round((fitness - self.env.dict_best[self.env.instance_name]) / self.env.dict_best[self.env.instance_name] * 100, 2)}% -  Tempo: {round(time.time() - start_time,2)}s")

                        
                        solution = self.env.decoder(best_keys)
                        cost = self.env.cost(solution, True)  
                        
                            
                            
                        return self.env.bins_usados, best_keys, best_cost
        solution = self.env.decoder(best_keys)
        cost = self.env.cost(solution, True)
        # print(f"Melhor Custo: {best_cost}, Melhor Custo Inicial: {best_ini_cost}, tempo = {round(time.time() - start_time,2)}")  

        return self.env.bins_usados,best_keys, best_cost
        

    def SimulatedAnnealing(self, SAmax=50, Temperatura=10000, alpha=0.99, tempo_max=200, beta_min=0.05, beta_max=0.25, tag = 0):
        metaheuristic_name = "SA"

        keys = self.random_keys()
        keys = self.RVND(keys)

        best_keys = keys

        solution = self.env.decoder(keys)
        cost = self.env.cost(solution)
        best_cost = cost
        
        elapsed_time = time.time() - self.start_time # Tempo inicial para o primeiro print
        print(f"\n{metaheuristic_name} NOVO MELHOR: {best_cost:.2f} - Tempo: {elapsed_time:.2f}s")
        if self.env.dict_best is not None:
            if best_cost == self.env.dict_best[self.env.instance_name]:
                elapsed_time = time.time() - self.start_time
                optimal_cost = self.env.dict_best[self.env.instance_name]
                if optimal_cost != 0:
                    gap = ((best_cost - optimal_cost) / optimal_cost) * 100
                    print(f"\n{metaheuristic_name} {tag} ÓTIMO ENCONTRADO: {best_cost:.2f} (BEST: {optimal_cost:.2f} - GAP: {gap:.2f}%) - Tempo: {elapsed_time:.2f}s")
                else:
                    print(f"\n{metaheuristic_name} {tag} ÓTIMO ENCONTRADO: {best_cost:.2f} (BEST: {optimal_cost:.2f}) - Tempo: {elapsed_time:.2f}s")
                print(f"{metaheuristic_name} {tag} ENCERRADO. Melhor Custo Final: {best_cost:.2f} - Tempo Total: {elapsed_time:.2f}s")
                return [], best_keys, best_cost
        start_time = time.time()
        T = Temperatura
        iter_at_temp = 0

        while time.time() - start_time < tempo_max:
            while iter_at_temp < SAmax:
                # print(f"\rTempo: {round(time.time() - start_time, 2):.1f}s - Temperatura {T:.1f} - \n", end="")
                iter_at_temp += 1

                new_keys = self.shaking(keys, beta_min, beta_max)
                new_solution = self.env.decoder(new_keys)
                new_cost = self.env.cost(new_solution)
                
                delta = new_cost - cost
                
                if new_cost < best_cost:
                    best_keys = new_keys
                    best_cost = new_cost
                    elapsed_time = time.time() - start_time
                    print(f"\n{metaheuristic_name} NOVO MELHOR: {best_cost:.2f} - Tempo: {elapsed_time:.2f}s")
                    if self.env.dict_best is not None:
                        if best_cost == self.env.dict_best[self.env.instance_name]:
                            elapsed_time = time.time() - start_time
                            optimal_cost = self.env.dict_best[self.env.instance_name]
                            if optimal_cost != 0:
                                gap = ((best_cost - optimal_cost) / optimal_cost) * 100
                                print(f"\n{metaheuristic_name} {tag} ÓTIMO ENCONTRADO: {best_cost:.2f} (BEST: {optimal_cost:.2f} - GAP: {gap:.2f}%) - Tempo: {elapsed_time:.2f}s")
                            else:
                                print(f"\n{metaheuristic_name} {tag} ÓTIMO ENCONTRADO: {best_cost:.2f} (BEST: {optimal_cost:.2f}) - Tempo: {elapsed_time:.2f}s")
                            print(f"{metaheuristic_name} {tag} ENCERRADO. Melhor Custo Final: {best_cost:.2f} - Tempo Total: {elapsed_time:.2f}s")
                            return [], best_keys, best_cost

                if delta <= 0:
                    keys = new_keys
                    cost = new_cost
                else:
                    if random.random() < math.exp(-delta / T):
                        keys = new_keys
                        cost = new_cost

            iter_at_temp = 0
            T = T * alpha
            keys = self.RVND(keys)

        final_cost_solution = self.env.decoder(best_keys)
        final_cost_value = self.env.cost(final_cost_solution, True)
        
        elapsed_time = time.time() - start_time
        print(f"{metaheuristic_name} ENCERRADO. Melhor Custo Final: {final_cost_value:.2f} - Tempo Total: {elapsed_time:.2f}s")

        return [], best_keys, final_cost_value


    def VNS(self, limit_time, x, tag, pool, lock, best, beta_min=0.05, k_max=6):
        metaheuristic_name = "VNS"

        idx_k = 0
        start_time = time.time()
        bests_S = []

        keys = self.random_keys()
        keys = self.RVND(keys)
        best_keys = keys
        best_cost = self.env.cost(self.env.decoder(keys))
        bests_S.append(keys)

        elapsed_time_init = time.time() - start_time # Tempo inicial para o primeiro print do VNS
        print(f"{metaheuristic_name} {tag} NOVO MELHOR: {best_cost:.2f} - Tempo: {elapsed_time_init:.2f}s")
        if self.env.dict_best is not None:
            if best_cost == self.env.dict_best[self.env.instance_name]:
                elapsed_time = time.time() - start_time
                optimal_cost = self.env.dict_best[self.env.instance_name]
                if optimal_cost != 0:
                    gap = ((best_cost - optimal_cost) / optimal_cost) * 100
                    print(f"\n{metaheuristic_name} {tag} ÓTIMO ENCONTRADO: {best_cost:.2f} (BEST: {optimal_cost:.2f} - GAP: {gap:.2f}%) - Tempo: {elapsed_time:.2f}s")
                else:
                    print(f"\n{metaheuristic_name} {tag} ÓTIMO ENCONTRADO: {best_cost:.2f} (BEST: {optimal_cost:.2f}) - Tempo: {elapsed_time:.2f}s")
                print(f"{metaheuristic_name} {tag} ENCERRADO. Melhor Custo Final: {best_cost:.2f} - Tempo Total: {elapsed_time:.2f}s")
                return [], best_keys, best_cost

        while time.time() - start_time < limit_time:
            # print(f"\rTempo decorrido: {round(time.time() - start_time, 2):.1f}s - Vizinhança k={idx_k + 1}/{k_max} ", end="")

            if idx_k >= k_max:
                idx_k = 0

            s1 = self.shaking(best_keys, idx_k * beta_min, (idx_k + 1) * beta_min)
            s2 = self.RVND(s1)
            sol2 = self.env.decoder(s2)
            cost = self.env.cost(sol2)

            if cost <= best_cost:
                best_cost = cost
                best_keys = s2
                bests_S.append(s2)

                with lock:
                    entry = (best_cost, list(best_keys))
                    bisect.insort(pool, entry)

                    if len(pool) > 20:
                        pool.pop()

                    if best_cost < best[0]:
                        best[0] = best_cost
                        best[1] = list(best_keys)
                        elapsed_time = time.time() - start_time
                        print(f"{metaheuristic_name} {tag} NOVO MELHOR: {best_cost:.2f} - Tempo: {elapsed_time:.2f}s")

                if self.env.dict_best is not None:
                    if best_cost == self.env.dict_best[self.env.instance_name]:
                        elapsed_time = time.time() - start_time
                        optimal_cost = self.env.dict_best[self.env.instance_name]
                        if optimal_cost != 0:
                            gap = ((best_cost - optimal_cost) / optimal_cost) * 100
                            print(f"\n{metaheuristic_name} {tag} ÓTIMO ENCONTRADO: {best_cost:.2f} (BEST: {optimal_cost:.2f} - GAP: {gap:.2f}%) - Tempo: {elapsed_time:.2f}s")
                        else:
                            print(f"\n{metaheuristic_name} {tag} ÓTIMO ENCONTRADO: {best_cost:.2f} (BEST: {optimal_cost:.2f}) - Tempo: {elapsed_time:.2f}s")
                        print(f"{metaheuristic_name} {tag} ENCERRADO. Melhor Custo Final: {best_cost:.2f} - Tempo Total: {elapsed_time:.2f}s")
                        return [], best_keys, best_cost
            else:
                idx_k += 1

        final_cost_solution = self.env.decoder(best_keys)
        final_cost_value = self.env.cost(final_cost_solution)
        
        elapsed_time = time.time() - start_time
        print(f"{metaheuristic_name} {tag} ENCERRADO. Melhor Custo Final: {final_cost_value:.2f} - Tempo Total: {elapsed_time:.2f}s")

        return [], best_keys, final_cost_value


    def BRKGA(self, pop_size, elite_pop, chance_elite, limit_time, tag, pool, lock, best):
        metaheuristic_name = "BRKGA"

        generation = 0
        tam_elite = int(pop_size * elite_pop)
        half_time_restart_done = False

        population = [self.random_keys() for _ in range(pop_size)]
        best_keys_overall = None
        best_fitness_overall = float('inf')

        start_time = time.time()

        while time.time() - start_time < limit_time:
            if not half_time_restart_done and (time.time() - start_time) > (limit_time / 2):
                population = [self.random_keys() for _ in range(pop_size)]
                half_time_restart_done = True

            generation += 1
            
            evaluated_population = []
            for key in population:
                sol = self.env.decoder(key)
                fitness = self.env.cost(sol)
                evaluated_population.append((key, sol, fitness))

                if fitness < best_fitness_overall:
                    best_fitness_overall = fitness
                    best_keys_overall = key
                    
                    with lock:
                        entry = (best_fitness_overall, list(best_keys_overall))
                        bisect.insort(pool, entry)
                        if len(pool) > 20:
                            pool.pop()
                        if best_fitness_overall < best[0]:
                            best[0] = best_fitness_overall
                            best[1] = list(best_keys_overall)
                            elapsed_time = time.time() - start_time
                            print(f"{metaheuristic_name} {tag} NOVO MELHOR: {best_fitness_overall:.2f} - Tempo: {elapsed_time:.2f}s")

                if self.env.dict_best is not None:
                    if fitness == self.env.dict_best[self.env.instance_name]:
                        elapsed_time = time.time() - start_time
                        optimal_cost = self.env.dict_best[self.env.instance_name]
                        if optimal_cost != 0:
                            gap = ((fitness - optimal_cost) / optimal_cost) * 100
                            print(f"\n{metaheuristic_name} {tag} ÓTIMO ENCONTRADO: {fitness:.2f} (BEST: {optimal_cost:.2f} - GAP: {gap:.2f}%) - Tempo: {elapsed_time:.2f}s")
                        else:
                            print(f"\n{metaheuristic_name} {tag} ÓTIMO ENCONTRADO: {fitness:.2f} (BEST: {optimal_cost:.2f}) - Tempo: {elapsed_time:.2f}s")
                        print(f"{metaheuristic_name} {tag} ENCERRADO. Melhor Custo Final: {fitness:.2f} - Tempo Total: {elapsed_time:.2f}s")
                        return [], best_keys_overall, fitness
            
            evaluated_population.sort(key=lambda x: x[2])
            
            elite_keys = [item[0] for item in evaluated_population[:tam_elite]]

            new_population = [elite_keys[0]]

            while len(new_population) < pop_size:
                parent1_source = random.random()
                if parent1_source < 0.5 and len(pool) > 0:
                    parent1 = random.sample(list(pool), 1)[0][1]
                else:
                    parent1 = random.sample(population, 1)[0]

                parent2_source = random.random()
                if parent2_source < 0.5 and len(elite_keys) > 0:
                    parent2 = random.sample(elite_keys, 1)[0]
                else:
                    parent2 = random.sample(population, 1)[0]

                child = np.zeros(self.__MAX_KEYS)
                for i in range(len(child)):
                    if random.random() < chance_elite:
                        child[i] = parent2[i]
                    else:
                        child[i] = parent1[i]
                
                for idx in range(len(child)):
                    if random.random() < 0.05:
                        child[idx] = random.random()

                new_population.append(child)
            
            population = new_population[:pop_size]

            # print(f"\r{metaheuristic_name} {tag} Geração {generation}: Melhor fitness = {best_fitness_overall:.2f} - Tempo: {round(time.time() - start_time, 2)}s", end="")

        final_cost_solution = self.env.decoder(best_keys_overall)
        final_cost_value = self.env.cost(final_cost_solution)

        elapsed_time = time.time() - start_time
        print(f"{metaheuristic_name} {tag} ENCERRADO. Melhor Custo Final: {final_cost_value:.2f} - Tempo Total: {elapsed_time:.2f}s")

        return [], best_keys_overall, best_fitness_overall

    def GRASP(self,max_iter,x, tempo = None): # GRASP, gera várias soluções semi gulosas e aplica a busca local em cada uma delas, retornando a melhor solução encontrada
        best_keys = None
        best_cost = float('inf')
        best_ini_cost = float('inf')
        start_time = time.time()
        iter = 0
        while iter < max_iter or time.time() - start_time < tempo:
            iter += 1
            random_keys = self.env.greedy_solution(10)
            ini_solution = self.env.decoder(random_keys)
            ini_cost = self.env.cost(ini_solution)
            if ini_cost < best_ini_cost:
                best_ini_cost = ini_cost
            
            keys = self.LocSearch(random_keys,x)

            solution = self.env.decoder(keys)
            cost = self.env.cost(solution)
            
            print(f"Iteração {iter}, Custo Inicial: {ini_cost}, Custo Final: {cost}, tempo = {time.time() - start_time}")
            
            if cost < best_cost:
                best_cost = cost
                best_keys = keys
        
        solution = self.env.decoder(best_keys)
        cost = self.env.cost(solution, True)
        print(f"Melhor Custo: {best_cost}, Melhor Custo Inicial: {best_ini_cost}, tempo = {time.time() - start_time}")  

        return best_keys, best_cost
    def ILS(self, limit_time, x, tag, pool, lock, best, beta_min=0.05, beta_max=0.25):
        metaheuristic_name = "ILS"

        start_time = time.time()
        bests_S = []

        keys = self.random_keys()
        keys = self.RVND(keys)
        best_keys = keys
        best_cost = self.env.cost(self.env.decoder(keys))
        bests_S.append(keys)
        
        elapsed_time_init = time.time() - start_time
        print(f"{metaheuristic_name} {tag} NOVO MELHOR: {best_cost:.2f} - Tempo: {elapsed_time_init:.2f}s")
        if self.env.dict_best is not None:
            if best_cost == self.env.dict_best[self.env.instance_name]:
                elapsed_time = time.time() - start_time
                optimal_cost = self.env.dict_best[self.env.instance_name]
                if optimal_cost != 0:
                    gap = ((best_cost - optimal_cost) / optimal_cost) * 100
                    print(f"\n{metaheuristic_name} {tag} ÓTIMO ENCONTRADO: {best_cost:.2f} (BEST: {optimal_cost:.2f} - GAP: {gap:.2f}%) - Tempo: {elapsed_time:.2f}s")
                else:
                    print(f"\n{metaheuristic_name} {tag} ÓTIMO ENCONTRADO: {best_cost:.2f} (BEST: {optimal_cost:.2f}) - Tempo: {elapsed_time:.2f}s")
                print(f"{metaheuristic_name} {tag} ENCERRADO. Melhor Custo Final: {best_cost:.2f} - Tempo Total: {elapsed_time:.2f}s")
                return [], best_keys, best_cost

        while time.time() - start_time < limit_time:
            # print(f"\rTempo decorrido: {round(time.time() - start_time, 2):.1f}s ", end="")

            s1 = self.shaking(best_keys, beta_min, beta_max)
            s2 = self.RVND(s1)
            
            sol2 = self.env.decoder(s2)
            cost = self.env.cost(sol2)

            if cost <= best_cost:
                best_cost = cost
                best_keys = s2
                
                bests_S.append(s2)
                with lock:
                    entry = (best_cost, list(best_keys))
                    bisect.insort(pool, entry)
                    
                    if len(pool) > 20:
                        pool.pop()
                    if best_cost < best[0]:
                        best[0] = best_cost
                        best[1] = list(best_keys)
                        elapsed_time = time.time() - start_time
                        print(f"ILS {tag} NOVO MELHOR: {best_cost:.2f} - Tempo: {elapsed_time:.2f}s")

                if self.env.dict_best is not None:
                    if best_cost == self.env.dict_best[self.env.instance_name]:
                        elapsed_time = time.time() - start_time
                        optimal_cost = self.env.dict_best[self.env.instance_name]
                        if optimal_cost != 0:
                            gap = ((best_cost - optimal_cost) / optimal_cost) * 100
                            print(f"\n{metaheuristic_name} {tag} ÓTIMO ENCONTRADO: {best_cost:.2f} (BEST: {optimal_cost:.2f} - GAP: {gap:.2f}%) - Tempo: {elapsed_time:.2f}s")
                        else:
                            print(f"\n{metaheuristic_name} {tag} ÓTIMO ENCONTRADO: {best_cost:.2f} (BEST: {optimal_cost:.2f}) - Tempo: {elapsed_time:.2f}s")
                        print(f"{metaheuristic_name} {tag} ENCERRADO. Melhor Custo Final: {best_cost:.2f} - Tempo Total: {elapsed_time:.2f}s")
                        return [], best_keys, best_cost

        final_cost_solution = self.env.decoder(best_keys)
        final_cost_value = self.env.cost(final_cost_solution)
        
        elapsed_time = time.time() - start_time
        print(f"{metaheuristic_name} {tag} ENCERRADO. Melhor Custo Final: {final_cost_value:.2f} - Tempo Total: {elapsed_time:.2f}s")
        return [], best_keys, final_cost_value

    def solve(self, pop_size, elite_pop, chance_elite, limit_time, n_workers=None,brkga=1, ms=1, sa=1, vns=1, ils=1):
        """Roda múltiplas instâncias de BRKGA em paralelo e compartilha apenas best_solution."""
        if n_workers is None:
            n_workers = cpu_count()
        self.max_time = limit_time


        manager = Manager()
        shared = manager.Namespace()
        shared.best_keys = None
        shared.best_fitness = float('inf')
        shared.best_pair = manager.list([float('inf'), None])
        shared.best_pool = manager.list()
        shared.best_time = None 
        lock = manager.Lock()
        processes = []
        tag = 0
        for _ in range(brkga):
            p = Process(
                target=_brkga_worker,
                args=(self.env, pop_size, elite_pop, chance_elite, limit_time, shared, lock,tag)
            )
            tag += 1
            processes.append(p)
            p.start()
        for _ in range(ms):
            p = Process(
                target=_MS_worker,
                args=(self.env,10000,100,limit_time, shared, lock,tag)
            )
            tag += 1
            processes.append(p)
            p.start()
        for _ in range(sa):
            p = Process(
                target=_SA_worker,
                args=(self.env, pop_size, elite_pop, chance_elite, limit_time, shared, lock,tag)
            )
            tag += 1
            processes.append(p)
            p.start()
        for _ in range(vns):
            p = Process(
                target=_VNS_worker,
                args=(self.env, limit_time,pop_size, shared, lock,tag)
            )
            tag += 1
            processes.append(p)
            p.start()
        for _ in range(ils):
            p = Process(
                target=_ILS_worker,
                args=(self.env, limit_time,pop_size, shared, lock,tag)
            )
            tag += 1
            processes.append(p)
            p.start()

        for p in processes:
            p.join()


        solution = self.env.decoder(shared.best_keys)
        
        cost = self.env.cost(solution)
        return cost, solution, shared.best_time
        
def _brkga_worker(env, pop_size, elite_pop, chance_elite, limit_time, shared, lock,tag):
    runner = RKO(env)
    _, local_keys, local_best = runner.BRKGA(pop_size, elite_pop, chance_elite, limit_time,tag,shared.best_pool,lock,shared.best_pair)
    
    with lock:
        if local_best < shared.best_fitness:
            shared.best_fitness = local_best
            shared.best_keys = local_keys
            shared.best_time = time.time() - env.start_time

def _MS_worker(env, max_itr, x, limit_time, shared, lock,tag):
    runner = RKO(env)
    _, local_keys, local_best = runner.MultiStart( max_itr,x,limit_time,tag,shared.best_pool,lock,shared.best_pair)
    
    with lock:
        if local_best < shared.best_fitness:
            shared.best_fitness = local_best
            shared.best_keys = local_keys
            shared.best_time = time.time() - env.start_time
def _GRASP_worker(env, max_itr, x, limit_time, shared, lock,tag):
    runner = RKO(env)
    _, local_keys, local_best = runner.MultiStart( max_itr,x,limit_time,tag,shared.best_pool,lock,shared.best_pair)
    
    with lock:
        if local_best < shared.best_fitness:
            shared.best_fitness = local_best
            shared.best_keys = local_keys
            shared.best_time = time.time() - env.start_time
            
def _VNS_worker(env, limit_time, x, shared, lock,tag):
    runner = RKO(env)
    _, local_keys, local_best = runner.VNS(limit_time,x,tag,shared.best_pool,lock,shared.best_pair)
    
    with lock:
        if local_best < shared.best_fitness:
            shared.best_fitness = local_best
            shared.best_keys = local_keys
            shared.best_time = time.time() - env.start_time
def _ILS_worker(env, limit_time, x, shared, lock,tag):
    runner = RKO(env)
    _, local_keys, local_best = runner.ILS(limit_time,x,tag,shared.best_pool,lock,shared.best_pair)
    
    with lock:
        if local_best < shared.best_fitness:
            shared.best_fitness = local_best
            shared.best_keys = local_keys
            shared.best_time = time.time() - env.start_time

def _SA_worker(env, pop_size, elite_pop, chance_elite, limit_time, shared, lock,tag):
    runner = RKO(env)
    _, local_keys, local_best = runner.SimulatedAnnealing( )
    
    with lock:
        if local_best < shared.best_fitness:
            shared.best_fitness = local_best
            shared.best_keys = local_keys
            shared.best_time = time.time() - env.start_time
      
        

