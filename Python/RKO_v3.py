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
    def __init__(self, size, pool, best_pair, lock=None):
        self.size = size
        self.pool = pool
        self.best_pair = best_pair
        self.lock = lock
        self.start_time = time.time()    
        
    def insert(self, entry_tuple, metaheuristic_name, tag): 

        fitness = entry_tuple[0]
        keys = entry_tuple[1]
        # print(f"INSERINDO: {fitness} - {metaheuristic_name} - {tag} - {len(self.pool)}")
        print(f"\rtempo = {round(time.time() - self.start_time,2)} ", end="")
        with self.lock:
            # print(f"\n{metaheuristic_name} {tag}")  
            if fitness < self.best_pair[0]: 
                self.best_pair[0] = fitness          
                self.best_pair[1] = list(keys)         
                self.best_pair[2] = round(self.start_time - time.time(), 2)
                
                print(f"\n{metaheuristic_name} {tag} NOVO MELHOR: {fitness} - BEST: {self.best_pair[0]} - Tempo: {round(self.start_time - time.time(), 2)}s - {len(self.pool)}")    
                               
            bisect.insort(self.pool, entry_tuple) 
            if len(self.pool) > self.size:
                self.pool.pop()

class RKO():
    def __init__(self, env):
        self.env = env
        self.__MAX_KEYS = self.env.tam_solution
        self.LS_type = self.env.LS_type
        self.start_time = time.time()
        self.max_time = self.env.max_time
        self.rate = 0.1
        
        
    
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
            
            swap_order = [i for i in range(int(self.rate * self.__MAX_KEYS))]
            random.shuffle(swap_order)
            
            best_keys = copy.deepcopy(keys)
            best_cost = self.env.cost(self.env.decoder(best_keys))
            
            
            k = 0
            for idx1 in swap_order:
                for idx2 in reversed(swap_order):
                    
                    if self.stop_condition(best_cost, "SwapLS", -1):
                            return best_keys

                           

                    k+=1
                    
                    new_keys = copy.deepcopy(best_keys)
                    new_keys[idx1], new_keys[idx2] = new_keys[idx2], new_keys[idx1]
                    new_cost = self.env.cost(self.env.decoder(new_keys))
                    
                    if new_cost < best_cost:
                        best_keys = new_keys
                        best_cost = new_cost    
            print(k)
            return best_keys
      
        elif self.LS_type == 'First':
            
            swap_order = [i for i in range(int(self.rate * self.__MAX_KEYS))]
            random.shuffle(swap_order)
            
            best_keys = copy.deepcopy(keys)
            best_cost = self.env.cost(self.env.decoder(best_keys))
            
            for idx1 in swap_order:
                for idx2 in reversed(swap_order):
                    
                    if self.stop_condition(best_cost, "SwapLS", -1):
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
            
            swap_order = [i for i in range(int(self.rate * self.__MAX_KEYS))]
            random.shuffle(swap_order)
            
            best_keys = copy.deepcopy(keys)
            best_cost = self.env.cost(self.env.decoder(best_keys))
            
            
            k = 0
            for idx in swap_order:
                for i in range(len(Farey_Squence) - 1):
                    # print(k, len(Farey_Squence) * len(swap_order))
                    
                    if self.stop_condition(best_cost, "FareyLS", -1):
                        return best_keys

                    k+=1
                    
                    new_keys = copy.deepcopy(best_keys)
                    new_keys[idx] = random.uniform(Farey_Squence[i], Farey_Squence[i+1])
                    new_cost = self.env.cost(self.env.decoder(new_keys))
                    
                    if new_cost < best_cost:
                        best_keys = new_keys
                        best_cost = new_cost    
            # print(k)
            return best_keys
        elif self.LS_type == 'First':
            
            swap_order = [i for i in range(int(self.rate * self.__MAX_KEYS))]
            random.shuffle(swap_order)
            
            best_keys = copy.deepcopy(keys)
            best_cost = self.env.cost(self.env.decoder(best_keys))
            
            for idx in swap_order:
                for i in range(len(Farey_Squence) - 1):
                    
                    if self.stop_condition(best_cost, "FareyLS", -1):
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
            
            swap_order = [i for i in range(int( self.__MAX_KEYS))]
            random.shuffle(swap_order)
            blocks = []
            while swap_order:
                block = []
                for i in range(int(self.rate * self.__MAX_KEYS)):
                    if not swap_order:
                        break
                    block.append(swap_order[0])
                    swap_order.pop(0)
                    
                blocks.append(block)
                
            
            k = 0
            inverts = []
            
            for i in range(int(1/self.rate)):
                one_invert = []
                
            
            
            best_keys = copy.deepcopy(keys)
            best_cost = self.env.cost(self.env.decoder(best_keys))
            # print(f"Best Cost: {best_cost}")
            
            k = 0
            for block in blocks:
                # print(k, len(swap_order))
                
                if self.stop_condition(best_cost, "InvertLS", -1):
                    return best_keys

                k+=1
                new_keys = copy.deepcopy(best_keys)
                for idx in block:
                    new_keys[idx] = 1 - new_keys[idx]
                
                new_cost = self.env.cost(self.env.decoder(new_keys))
                
                if new_cost < best_cost:
                    best_keys = new_keys
                    best_cost = new_cost    
            # print(best_cost)
            # print(k)
            return best_keys    
        elif self.LS_type == 'First':
            
            swap_order = [i for i in range(int(self.rate * self.__MAX_KEYS))]
            random.shuffle(swap_order)
            
            best_keys = copy.deepcopy(keys)
            best_cost = self.env.cost(self.env.decoder(best_keys))
            
            for idx in swap_order:
            
                if self.stop_condition(best_cost, "InvertLS", -1):
                    return best_keys
                        
                new_keys = copy.deepcopy(best_keys)
                new_keys[idx] = 1 - new_keys[idx]
                new_cost = self.env.cost(self.env.decoder(new_keys))
                    
                if new_cost < best_cost:
                    best_keys = new_keys
                    best_cost = new_cost
                    
                    return best_keys
                
            return best_keys

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
            # print("POOL")
            x2 = random.sample(list(pool.pool), 1)[0][1]  
            x3 = random.sample(list(pool.pool), 1)[0][1]
            while x2 == x3:
                x2 = random.sample(list(pool.pool), 1)[0][1]
                x3 = random.sample(list(pool.pool), 1)[0][1]
        
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
        
        while iter_count <= (max_iter*self.rate):
            
            if self.stop_condition(fitBest, "NM", -1):
                return xBest
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
            
        
        
 
    def RVND(self, keys, pool=None):
        
        tag = random.randint(0, 100)
        
        best_keys = copy.deepcopy(keys)
        best_cost = self.env.cost(self.env.decoder(best_keys))
        
        
        
        neighborhoods = ['SwapLS', 'NelderMeadSearch','FareyLS', 'InvertLS']
        not_used_nb = ['SwapLS', 'NelderMeadSearch','FareyLS', 'InvertLS']
        
        while not_used_nb:
            
            current_neighborhood = random.choice(not_used_nb)
            
            # print(f"Inicio Neighborhood: {current_neighborhood} - Custo: {best_cost} - processo: {tag}")
            start = time.time()
            if current_neighborhood == 'SwapLS':
                new_keys = self.SwapLS(best_keys)
            elif current_neighborhood == 'NelderMeadSearch':               
                new_keys = self.NelderMeadSearch(best_keys, pool)             
            elif current_neighborhood == 'FareyLS':
                new_keys = self.FareyLS(best_keys)
            elif current_neighborhood == 'InvertLS':
                new_keys = self.InvertLS(best_keys)
                
            new_cost = self.env.cost(self.env.decoder(new_keys))
            # print(f"Fim Neighborhood: {current_neighborhood} - Custo: {new_cost} - processo: {tag} - tempo: {round(time.time() - start, 2)}s")
            
            if new_cost < best_cost:
                best_keys = new_keys
                best_cost = new_cost
                not_used_nb = copy.deepcopy(neighborhoods)
                pool.insert((best_cost, list(best_keys)), "RVND", -1)
                
            else:
                not_used_nb.remove(current_neighborhood)
            
            if self.stop_condition(best_cost, "RVND", -1):
                return best_keys
        
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
    
    def MultiStart(self, tag,  pool): # Multi Start, gera várias soluções aleatórias e aplica a busca local em cada uma delas, retornando a melhor solução encontrada
        metaheuristic_name = "MS"
        start_time = time.time()
        tempo_max = self.max_time
        keys = self.random_keys()
       

        best_keys = keys

        solution = self.env.decoder(keys)
        cost = self.env.cost(solution)
        best_cost = cost
        
      
        
        pool.insert((best_cost, list(best_keys)), metaheuristic_name, tag)
        if self.stop_condition(best_cost, metaheuristic_name, tag):
                return [], best_keys, best_cost
            
        

        while time.time() - start_time < tempo_max:
                k1 = random.sample(list(pool.pool), 1)[0][1]
                new_keys = self.shaking(k1, 0.1, 0.3)
                new_keys = self.RVND(pool = pool, keys = new_keys)
                
                new_solution = self.env.decoder(new_keys)
                new_cost = self.env.cost(new_solution)
                
                
                
                
                
                if new_cost < best_cost:
                    best_keys = new_keys
                    best_cost = new_cost
                    
                    
                    pool.insert((best_cost, list(best_keys)), metaheuristic_name, tag)
                
                if self.stop_condition(best_cost, metaheuristic_name, tag):
                        return [], best_keys, best_cost

               

            
            

        final_cost_solution = self.env.decoder(best_keys)
        final_cost_value = self.env.cost(final_cost_solution, True)
        

        return [], best_keys, final_cost_value
        

    def SimulatedAnnealing(self,QAgente, tag = 0, pool=None, ):
        
        metaheuristic_name = "SA"
        tempo_max = self.max_time
        keys = self.random_keys()
        keys = self.RVND(pool = pool, keys =keys)

        best_keys = keys

        solution = self.env.decoder(keys)
        cost = self.env.cost(solution)
        best_cost = cost
        
      
        
        pool.insert((best_cost, list(best_keys)), metaheuristic_name, tag)
        if self.stop_condition(best_cost, metaheuristic_name, tag):
                return [], best_keys, best_cost
        if True:
        # 1. Obter os PARÂMETROS INICIAIS do SA do agente Q-Learning
        # A primeira chamada a QTableAction() usa o estado inicial aleatório do agente.
            chosen_sa_parameters = QAgente.QTableAction()
            print(chosen_sa_parameters)
            SAmax = int(chosen_sa_parameters[0])
            alpha = chosen_sa_parameters[1]
            beta_min = chosen_sa_parameters[2]
            beta_max = chosen_sa_parameters[3]
            Temperatura = chosen_sa_parameters[4]    
        reward_from_previous_iteration = 0    
        start_time_mh_ql = time.time()
        current_runtime_SA = 0.0
        start_time = time.time()
        T = Temperatura
        

        k = 0
        while time.time() - start_time < tempo_max:
            k+=1
            best_cost_before_ql_phase = best_cost # Captura o melhor custo ANTES desta nova fase para o cálculo da recompensa
            # print(best_cost_before_ql_phase)
            if True:

                QAgente.UpdateQTable(reward_from_previous_iteration, current_runtime_SA)
                # print('entrou')
                chosen_sa_parameters = QAgente.QTableAction()
                # print('saiu')
                print(chosen_sa_parameters)
                

                SAmax = int(chosen_sa_parameters[0])
                alpha = chosen_sa_parameters[1]
                beta_min = chosen_sa_parameters[2]
                beta_max = chosen_sa_parameters[3]
                # Temperatura = chosen_sa_parameters[4]
            iter_at_temp = 0
            while iter_at_temp < SAmax:
                # print("entrou 2")
                
                iter_at_temp += 1

                # print('sa', k,iter_at_temp, T, best_cost)

                new_keys = self.shaking(keys, beta_min, beta_max)
                new_solution = self.env.decoder(new_keys)
                new_cost = self.env.cost(new_solution)
                
                delta = new_cost - cost
                
                if new_cost < best_cost:
                    best_keys = new_keys
                    best_cost = new_cost
                    elapsed_time = time.time() - start_time
                    
                    pool.insert((best_cost, list(best_keys)), metaheuristic_name, tag)
                if self.stop_condition(best_cost, metaheuristic_name, tag):
                        return [], best_keys, best_cost

                if delta <= 0:
                    keys = new_keys
                    cost = new_cost
                else:
                    if random.random() < math.exp(-delta / T):
                        keys = new_keys
                        cost = new_cost
            # print("saiu 2")
            iter_at_temp = 0
            T = T * alpha
            # print("entrou 3")
            # # keys = self.NelderMeadSearch(pool = pool, keys =keys)
            # print("saiu 3")
            new_solution = self.env.decoder(keys)
            new_cost = self.env.cost(new_solution)
            # print('SA',new_cost)
            
            delta = new_cost - cost
            
            if new_cost < best_cost:
                best_keys = keys
                best_cost = new_cost
                elapsed_time = time.time() - start_time
                
            pool.insert((best_cost, list(best_keys)), metaheuristic_name, tag)
            
            current_runtime_SA = time.time() - start_time 
            
            if best_cost < best_cost_before_ql_phase: # Se houve melhoria geral nesta fase QL
                reward_for_this_iteration = 1.0 
            else:
                # Recompensa relativa se não houve melhoria no QL-phase
                if best_cost_before_ql_phase != 0:
                    reward_for_this_iteration = (best_cost_before_ql_phase - best_cost) / abs(best_cost_before_ql_phase)
                else:
                    reward_for_this_iteration = 0.0 # Evita divisão por zero se o custo inicial for 0

            # Armazena a recompensa para a PRÓXIMA iteração (passada para UpdateQTable)
            reward_from_previous_iteration = reward_for_this_iteration
            

        final_cost_solution = self.env.decoder(best_keys)
        final_cost_value = self.env.cost(final_cost_solution, True)
        

        return [], best_keys, final_cost_value


    def VNS(self, limit_time, x, tag, pool, beta_min=0.05, k_max=6):
        metaheuristic_name = "VNS"

        idx_k = 0
        start_time = time.time()
        bests_S = []

        keys = self.random_keys()
        keys = self.RVND(pool = pool, keys =keys)
        best_keys = keys
        best_cost = self.env.cost(self.env.decoder(keys))
        bests_S.append(keys)

       
        
        pool.insert((best_cost, list(best_keys)), metaheuristic_name, tag)
        
        if self.stop_condition(best_cost, metaheuristic_name, tag):
                return [], best_keys, best_cost

        while time.time() - start_time < limit_time:
        

            if idx_k >= k_max:
                idx_k = 0

            s1 = self.shaking(best_keys, idx_k * beta_min, (idx_k + 1) * beta_min)
            s2 = self.RVND(pool = pool, keys =s1)
            sol2 = self.env.decoder(s2)
            cost = self.env.cost(sol2)

            if cost <= best_cost:
                best_cost = cost
                best_keys = s2
                bests_S.append(s2)

                pool.insert((best_cost, list(best_keys)), metaheuristic_name, tag)
            else:
                idx_k += 1
                
            
            if self.stop_condition(best_cost, metaheuristic_name, tag):
                return [], best_keys, best_cost

        final_cost_solution = self.env.decoder(best_keys)
        final_cost_value = self.env.cost(final_cost_solution)
        
        elapsed_time = time.time() - start_time
      

        return [], best_keys, final_cost_value


    def BRKGA(self, pop_size, elite_pop, chance_elite,  tag, pool):
        metaheuristic_name = "BRKGA"
        limit_time = self.max_time
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
                    
                    pool.insert((best_fitness_overall, list(best_keys_overall)), metaheuristic_name, tag)

                if self.stop_condition(best_fitness_overall, metaheuristic_name, tag):
                    return [], best_keys_overall, fitness
            
            evaluated_population.sort(key=lambda x: x[2])
            
            elite_keys = [item[0] for item in evaluated_population[:tam_elite]]

            new_population = [elite_keys[0]]

            while len(new_population) < pop_size:
                parent1_source = random.random()
                if parent1_source < 0.5 and len(pool.pool) > 0:
                    parent1 = random.sample(list(pool.pool), 1)[0][1]
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
            
    
            
            if cost < best_cost:
                best_cost = cost
                best_keys = keys
        
        solution = self.env.decoder(best_keys)
        cost = self.env.cost(solution, True)
       

        return best_keys, best_cost
    def ILS(self, limit_time, x, tag, pool, beta_min=0.1, beta_max=0.5):
        metaheuristic_name = "ILS"

        start_time = time.time()
        bests_S = []

        keys = self.random_keys()
        keys = self.RVND(pool = pool, keys =keys)
        best_keys = keys
        best_cost = self.env.cost(self.env.decoder(keys))
        bests_S.append(keys)
        
        
        pool.insert((best_cost, list(best_keys)), metaheuristic_name, tag)
      
        if self.stop_condition(best_cost, metaheuristic_name, tag):
                return [], best_keys, best_cost

        while time.time() - start_time < limit_time:
           

            s1 = self.shaking(best_keys, beta_min, beta_max)
            s2 = self.RVND(pool = pool, keys =s1)
            
            sol2 = self.env.decoder(s2)
            cost = self.env.cost(sol2)

            if cost <= best_cost:
                best_cost = cost
                best_keys = s2
                
                bests_S.append(s2)
                
                pool.insert((best_cost, list(best_keys)), metaheuristic_name, tag)

            if self.stop_condition(best_cost, metaheuristic_name, tag):
                    return [], best_keys, best_cost

        final_cost_solution = self.env.decoder(best_keys)
        final_cost_value = self.env.cost(final_cost_solution)
        
        elapsed_time = time.time() - start_time
        
        return [], best_keys, final_cost_value


    def stop_condition(self, best_cost, metaheuristic_name, tag):
        if time.time() - self.start_time > self.max_time:
            print(f"{metaheuristic_name} {tag}: ENCERRADO")
            
            return True
        if self.env.dict_best is not None:
            if best_cost == self.env.dict_best[self.env.instance_name]:
                print(f"Metaheurística {metaheuristic_name} com tag {tag} encontrou a melhor solução: {best_cost}")
                return True
            else:
                return False
        return False
        
        
    def solve(self, pop_size, elite_pop, chance_elite, time_total, n_workers=None,brkga=1, ms=1, sa=1, vns=1, ils=1, restart=1):
        
        limit_time = time_total * restart
        restarts = int(1/restart)
        # print(restarts)
        
        if n_workers is None:
            n_workers = cpu_count()
        self.max_time = limit_time
        
        print(self.max_time)


        manager = Manager()
        shared = manager.Namespace()
       
        
        shared.best_pair = manager.list([float('inf'), None, None])  # [cost, keys, time]
        shared.best_pool = manager.list()
        
        
        shared.pool = SolutionPool(20, shared.best_pool, shared.best_pair, lock=manager.Lock())
        for i in range(20):
            keys = self.random_keys()
            cost = self.env.cost(self.env.decoder(keys))
            shared.pool.insert((cost, list(keys)), 'pool', -1)
        lock = manager.Lock()
        processes = []
        tag = 0
        for k in range(restarts):
            self.start_time = time.time()
            shared.pool.pool = manager.list()
            for _ in range(brkga):
                p = Process(
                    target=_brkga_worker,
                    args=(self.env, pop_size, elite_pop, chance_elite, shared.pool,tag)
                )
                tag += 1
                processes.append(p)
                p.start()
            for _ in range(ms):
                p = Process(
                    target=_MS_worker,
                    args=(self.env,10000,100,shared.pool,tag)
                )
                tag += 1
                processes.append(p)
                p.start()
            for _ in range(sa):
                p = Process(
                    target=_SA_worker,
                    args=(self.env, pop_size, elite_pop, chance_elite, shared.pool,tag)
                )
                tag += 1
                processes.append(p)
                p.start()
            for _ in range(vns):
                p = Process(
                    target=_VNS_worker,
                    args=(self.env, limit_time,pop_size, shared.pool,tag)
                )
                tag += 1
                processes.append(p)
                p.start()
            for _ in range(ils):
                p = Process(
                    target=_ILS_worker,
                    args=(self.env, limit_time,pop_size, shared.pool,tag)
                )
                tag += 1
                processes.append(p)
                p.start()

            for p in processes:
                p.join()

        print(shared.pool.best_pair[0], shared.pool.best_pair[2])

        cost = shared.pool.best_pair[0]
        solution = shared.pool.best_pair[1]    
        time_elapsed = shared.pool.best_pair[2]
        return cost, solution, time_elapsed
        
def _brkga_worker(env, pop_size, elite_pop, chance_elite, pool,tag):
    runner = RKO(env)
    _, local_keys, local_best = runner.BRKGA(pop_size, elite_pop, chance_elite,tag, pool)
    


def _MS_worker(env, max_itr, x, pool,tag):
    runner = RKO(env)
    

    _, local_keys, local_best = runner.MultiStart(tag, pool)
    

def _GRASP_worker(env, max_itr, x, pool,tag):
    runner = RKO(env)
    _, local_keys, local_best = runner.MultiStart( max_itr,x,pool)
    

            
def _VNS_worker(env, limit_time, x, pool, tag):
    runner = RKO(env)
    _, local_keys, local_best = runner.VNS(limit_time,x,tag, pool)
    

def _ILS_worker(env, limit_time, x, pool, tag):
    runner = RKO(env)
    _, local_keys, local_best = runner.ILS(limit_time,x,tag, pool)
    


def _SA_worker(env, pop_size, elite_pop, chance_elite, pool,tag):
    runner = RKO(env)
    sa_params_options = env.SA_parameters_list 

    total_run_max_time_for_q_learning = env.max_time 


    QAgente_SA = QLearningAgent(
        mh_parameters_options=sa_params_options,
        total_run_max_time_for_q_learning=total_run_max_time_for_q_learning
    )
 
    _, local_keys, local_best = runner.SimulatedAnnealing(tag = tag, pool = pool, QAgente=QAgente_SA)
    

      
        


    



import random
import math
import itertools # Para gerar combinações de parâmetros (estados)

# --- Funções Auxiliares (Baseadas em Method.h do C++) ---
# Assegure-se de que estas funções estejam acessíveis (e.g., em um utilitário.py ou no mesmo arquivo)
def randomico(min_val, max_val):
    """Gera um número float aleatório entre min_val (inclusive) e max_val (exclusivo)."""
    return random.uniform(min_val, max_val)

def irandomico(min_val, max_val):
    """Gera um número inteiro aleatório entre min_val (inclusive) e max_val (inclusive)."""
    return random.randint(min_val, max_val)

# --- Classe QState (Representa um Estado na Q-Table) ---
class QState:
    """
    Representa um estado no espaço de estados do Q-Learning para uma meta-heurística.
    Equivalente à struct TState no C++.
    """
    def __init__(self, label, parameters_values):
        self.label = label                        # Identificador único do estado (índice)
        self.par = parameters_values              # Lista de valores de parâmetros (ex: [p, pe, pm, rhoe])
        self.Ai = []                              # Lista de índices de estados alcançáveis (ações)
        self.Qa = []                              # Lista de Q-values, Qa[j] é o Q-value para a ação Ai[j]
        
        # Inicialização para garantir que os atributos existam, serão preenchidos depois
        self.maxQ = 0.0
        self.maxA_idx = 0 
        
        # Atributos adicionais do TState do C++, se forem usados para algo além de debug
        self.ci = 0
        self.numN = 0
        self.sumQ = 0.0 # Será atualizado junto com Qa

    def add_action(self, next_state_label, initial_q_value):
        """Adiciona uma ação possível a partir deste estado."""
        self.Ai.append(next_state_label)
        self.Qa.append(initial_q_value)
        # Atualiza maxQ e maxA_idx se este novo Q-value for maior
        if initial_q_value > self.maxQ:
            self.maxQ = initial_q_value
            self.maxA_idx = len(self.Qa) - 1 # Índice da ação recém-adicionada

    def update_q_value_for_action(self, action_idx_in_Ai, reward, learning_rate, discount_factor, next_state_maxQ):
        """
        Atualiza o valor Q para uma ação específica e recalcula maxQ/maxA_idx do estado.
        action_idx_in_Ai: O índice da ação *dentro do vetor self.Ai* do estado atual.
        """
        current_q_s_a = self.Qa[action_idx_in_Ai]
        
        # Fórmula de atualização do Q-Learning (Equação de Bellman)
        updated_q_s_a = current_q_s_a + learning_rate * (reward + discount_factor * next_state_maxQ - current_q_s_a)
        
        self.Qa[action_idx_in_Ai] = updated_q_s_a
        
        # Re-calcular maxQ e maxA_idx após a atualização
        if self.Qa: # Garante que há ações para evitar erro em caso de lista vazia
            self.maxQ = max(self.Qa)
            self.maxA_idx = self.Qa.index(self.maxQ)
        else:
            # Isso não deveria acontecer se os estados forem criados corretamente com ações
            self.maxQ = 0.0
            self.maxA_idx = 0

# --- Função de Criação de Estados (Baseada em QLearning.h do C++) ---
def create_q_states_for_mh(parameters_options_for_mh):
    """
    Cria todos os estados possíveis para UMA meta-heurística e define suas ações.
    parameters_options_for_mh: Uma lista de listas, onde cada sub-lista contém os valores
                               possíveis para um parâmetro da MH.
                               Ex: [[p1_val1, p1_val2], [p2_val1, p2_val2, p2_val3]]
    Retorna uma lista de objetos QState.
    """
    states_list = []
    
    # 1. Gerar todas as combinações de parâmetros (estados)
    # itertools.product cria um iterador para o produto cartesiano
    all_combinations = list(itertools.product(*parameters_options_for_mh))
    
    # 2. Criar os objetos QState iniciais
    for i, combo in enumerate(all_combinations):
        states_list.append(QState(label=i, parameters_values=list(combo))) # Converter tupla para lista

    num_states = len(states_list)
    num_params_in_state = len(parameters_options_for_mh) # Número de parâmetros por estado

    # 3. Definir ações e inicializar Q-values para cada estado (baseado em distância de Hamming)
    for i in range(num_states):
        for j in range(num_states):
            # Calcular distância de Hamming
            distance = 0
            for k in range(num_params_in_state):
                if states_list[i].par[k] != states_list[j].par[k]:
                    distance += 1
            
            # Uma ação é viável se a distância de Hamming for <= 1
            if distance <= 1:
                # Adiciona a ação (que é o índice do próximo estado) e inicializa seu Q-value
                states_list[i].add_action(next_state_label=states_list[j].label,
                                          initial_q_value=randomico(0.01, 0.05)) # randomico(0.05,0.01) do C++

    return states_list

# --- Classe QLearningAgent (Seu 'MyQLearning' mais genérico) ---
class QLearningAgent:
    """
    Gerencia o processo de Q-Learning para UMA meta-heurística específica.
    Cada meta-heurística no RKO terá sua própria instância desta classe.
    """
    def __init__(self, mh_parameters_options, total_run_max_time_for_q_learning):
        """
        Inicializa o agente Q-Learning para uma meta-heurística.
        mh_parameters_options: Uma lista de listas de valores de parâmetros para esta MH.
                               Ex: [[p_opts], [pe_opts], [pm_opts], [rhoe_opts]]
        total_run_max_time_for_q_learning: O tempo total máximo para a run do RKO, usado para calcular Ti.
        """
        self.QTable_states = create_q_states_for_mh(mh_parameters_options)

        
        # O estado atual é inicializado aleatoriamente entre os estados disponíveis
        self.ActualState_idx = irandomico(0, len(self.QTable_states) - 1)
        
        # Variáveis para controle da atualização da Q-Table
        self.last_chosen_state_idx = self.ActualState_idx # Guarda o estado de onde a última ação partiu
        self.last_action_idx_in_Ai = -1 # Guarda o índice da ação no Ai do estado anterior

        # Parâmetros do Q-Learning que serão atualizados dinamicamente
        self.epsilon = 1.0          # Epsilon inicial (começa em epsilon_max)
        self.learning_rate = 1.0    # Taxa de aprendizado inicial
        self.discount_factor = 0.8  # Fator de desconto (fixo, como no C++)

        # Parâmetros de controle de epsilon decay e learning rate decay
        self.epsilon_max = 1.0
        self.epsilon_min = 0.1
        # Ti é 10% do tempo máximo de execução da run do RKO (como no C++)
        self.Ti = total_run_max_time_for_q_learning * 0.1 
        self.restart_epsilon_count = 1 # Contador de "épocas" para o restart de epsilon
        
        

    def _set_ql_parameters(self, current_runtime_of_mh):
        """
        Atualiza os parâmetros de aprendizado do Q-Learning (epsilon, learning_rate).
        Corresponde à função SetQLParameter do C++. É uma função interna.
        current_runtime_of_mh: Tempo decorrido desde o início da execução desta MH.
        """
        # Epsilon decay com warm restart (baseado em QLearning.h)
        if current_runtime_of_mh >= self.restart_epsilon_count * self.Ti:
            self.restart_epsilon_count += 1
            self.epsilon_max -= 0.1 # Reduz o valor máximo de epsilon gradualmente
            if self.epsilon_max < self.epsilon_min:
                self.epsilon_max = self.epsilon_min
            self.epsilon = self.epsilon_max # Reinicia epsilon para o novo epsilon_max
        else:
            # Decaimento em cosseno
            self.epsilon = self.epsilon_min + 0.5 * (self.epsilon_max - self.epsilon_min) * \
                           (1 + math.cos(((current_runtime_of_mh % self.Ti) / self.Ti) * math.pi))
        
        # Learning rate (alpha) decay (baseado em QLearning.h)
        self.learning_rate = 1 - (0.9 * current_runtime_of_mh / (self.Ti * 10)) # Aprox. MAXTIME total

    def UpdateQTable(self, reward_from_previous_action, current_runtime_of_mh):
        """
        Atualiza a Q-Table com a recompensa da ação ANTERIOR e o estado atual.
        Corresponde à lógica de atualização dentro das MHs no C++.
        
        reward_from_previous_action: A recompensa recebida pela ação tomada na iteração anterior.
        current_runtime_of_mh: O tempo atual da execução da meta-heurística.
        """
        # 1. Primeiro, atualiza os parâmetros do Q-Learning para a iteração ATUAL
        self._set_ql_parameters(current_runtime_of_mh)

        # 2. Em seguida, aplica a regra de atualização da Q-Table para a AÇÃO ANTERIOR
        # `self.last_chosen_state_idx` é o `st` (estado anterior de onde a ação partiu)
        # `self.last_action_idx_in_Ai` é o `at` (índice da ação no vetor Ai do estado anterior)
        # `self.ActualState_idx` é o `st_1` (o estado para o qual a ação levou na iteração anterior)

        if self.last_action_idx_in_Ai == -1:
            # Esta é a primeira iteração, não há ação anterior para atualizar.
       
            return 
        
        # Obter os objetos QState para o estado ANTERIOR e o estado ATUAL (que é o próximo estado)
        previous_q_state_object = self.QTable_states[self.last_chosen_state_idx]
        current_q_state_object = self.QTable_states[self.ActualState_idx] # Este é o st_1

        # Obter o valor Q máximo do próximo estado (st_1) para a fórmula de atualização
        max_q_next_state = current_q_state_object.maxQ

        # Chamar o método de atualização do QState para atualizar o Q-value específico
        previous_q_state_object.update_q_value_for_action(
            action_idx_in_Ai=self.last_action_idx_in_Ai,
            reward=reward_from_previous_action,
            learning_rate=self.learning_rate,
            discount_factor=self.discount_factor,
            next_state_maxQ=max_q_next_state
        )
        
        # self.ActualState_idx já foi atualizado para o 'next_state_label' na chamada anterior de QTableAction
        # então ele já aponta para o st_1.

    def QTableAction(self):
        """
        Escolhe uma ação (próximo estado de parâmetros) usando a política epsilon-greedy.
        Corresponde à função ChooseAction do C++.
        
        Retorna:
            - chosen_parameters_values: Uma lista dos valores de parâmetros do estado escolhido
                                        (que se torna o NOVO self.ActualState_idx).
        """
        # print("teste")
        current_q_state = self.QTable_states[self.ActualState_idx]
        # print(current_q_state)
        if not current_q_state.Ai:
            # Caso não haja ações possíveis, retornar os parâmetros do estado atual e não registrar ação
            print(f"Warning: State {self.ActualState_idx} has no available actions. Returning current parameters.")
            self.last_action_idx_in_Ai = -1 
            self.last_chosen_state_idx = self.ActualState_idx
            return current_q_state.par

        action_taken_idx_in_Ai = 0
        next_state_label = 0

        # Epsilon-greedy policy
        if random.uniform(0, 1) <= (1 - self.epsilon):
            # Explotação: escolher a ação com o maior valor Q (baseado em maxA_idx)
            action_taken_idx_in_Ai = current_q_state.maxA_idx
            next_state_label = current_q_state.Ai[action_taken_idx_in_Ai]
        else:
            # Exploração: escolher uma ação aleatória
            action_taken_idx_in_Ai = random.randint(0, len(current_q_state.Ai) - 1)
            next_state_label = current_q_state.Ai[action_taken_idx_in_Ai]
        
        # Salva o estado de onde a ação partiu e o índice da ação para a próxima atualização
        self.last_chosen_state_idx = self.ActualState_idx
        self.last_action_idx_in_Ai = action_taken_idx_in_Ai

        # ATUALIZA o estado atual do agente para o próximo estado escolhido
        self.ActualState_idx = next_state_label

        # Retorna os parâmetros do NOVO estado escolhido para a MH usar
        return self.QTable_states[self.ActualState_idx].par

