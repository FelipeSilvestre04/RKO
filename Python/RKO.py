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
        print(f"\rtempo = {round(time.time() - self.start_time,2)} ", end="")
        with self.lock:
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
        self.rate = 1
        
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
                
        return new_keys
    
    def SwapLS(self, keys):
        if self.LS_type == 'Best':
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
                        return new_keys
                    
            return best_keys
            
    def FareyLS(self, keys):
        Farey_Squence = [0.00, 0.142857, 0.166667, 0.20, 0.25, 0.285714, 0.333333, 0.40, 0.428571, 0.50, 
                         0.571429, 0.60, 0.666667, 0.714286, 0.75, 0.80, 0.833333, 0.857143, 1.0]
        
        if self.LS_type == 'Best':
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
                        return new_keys
                        
            return best_keys
    
    def InvertLS(self, keys):
        if self.LS_type == 'Best':
            swap_order = [i for i in range(int(self.__MAX_KEYS))]
            random.shuffle(swap_order)
            
            best_keys = copy.deepcopy(keys)
            best_cost = self.env.cost(self.env.decoder(best_keys))
            
            blocks = []
            while swap_order:
                block = swap_order[:int(self.rate * self.__MAX_KEYS)]
                swap_order = swap_order[int(self.rate * self.__MAX_KEYS):]
                blocks.append(block)

            for block in blocks:
                if self.stop_condition(best_cost, "InvertLS", -1):
                    return best_keys

                new_keys = copy.deepcopy(best_keys)
                for idx in block:
                    new_keys[idx] = 1 - new_keys[idx]
                
                new_cost = self.env.cost(self.env.decoder(new_keys))
                
                if new_cost < best_cost:
                    best_keys = new_keys
                    best_cost = new_cost    
            
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
                    return new_keys
                
            return best_keys
            
    def Blending(self, keys1, keys2, factor):
        new_keys = np.zeros(self.__MAX_KEYS)
        
        for i in range(self.__MAX_KEYS):
            if random.random() < 0.02: 
                new_keys[i] = random.random()
            else:               
                if random.random() < 0.5:
                    new_keys[i] = keys1[i]
                else:
                    if factor == -1:
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
            x2 = random.sample(list(pool.pool), 1)[0][1]  
            x3 = random.sample(list(pool.pool), 1)[0][1]
            while x2 == x3:
                x2 = random.sample(list(pool.pool), 1)[0][1]
        
        fit1 = self.env.cost(self.env.decoder(x1))
        fit2 = self.env.cost(self.env.decoder(x2))
        fit3 = self.env.cost(self.env.decoder(x3))
        
        if fit1 > fit2:
            x1, x2, fit1, fit2 = x2, x1, fit2, fit1
        if fit1 > fit3:
            x1, x3, fit1, fit3 = x3, x1, fit3, fit1
        if fit2 > fit3:
            x2, x3, fit2, fit3 = x3, x2, fit3, fit2
        
        xBest = copy.deepcopy(x1)
        fitBest = fit1
        
        x0 = self.Blending(x1, x2, 1)
        fit0 = self.env.cost(self.env.decoder(x0))
        if fit0 < fitBest:
            xBest, fitBest, improved = copy.deepcopy(x0), fit0, 1
            
        iter_count = 1
        
        max_iter = int(self.__MAX_KEYS * math.exp(-2))
        
        while iter_count <= (max_iter * self.rate):
            if self.stop_condition(fitBest, "NM", -1):
                return xBest
                
            shrink = 0
            
            x_r = self.Blending(x0, x3, -1)
            fit_r = self.env.cost(self.env.decoder(x_r))
            if fit_r < fitBest:
                xBest, fitBest, improved, improvedX1 = copy.deepcopy(x_r), fit_r, 1, 1

            if fit_r < fit1:
                x_e = self.Blending(x_r, x0, -1)
                fit_e = self.env.cost(self.env.decoder(x_e))
                if fit_e < fitBest:
                    xBest, fitBest, improved, improvedX1 = copy.deepcopy(x_e), fit_e, 1, 1
                
                if fit_e < fit_r:
                    x3, fit3 = copy.deepcopy(x_e), fit_e
                else:
                    x3, fit3 = copy.deepcopy(x_r), fit_r
            
            elif fit_r < fit2:
                x3, fit3 = copy.deepcopy(x_r), fit_r
            
            else:
                if fit_r < fit3:
                    x_c = self.Blending(x_r, x0, 1)
                    fit_c = self.env.cost(self.env.decoder(x_c))
                    if fit_c < fitBest:
                        xBest, fitBest, improved, improvedX1 = copy.deepcopy(x_c), fit_c, 1, 1
                    
                    if fit_c < fit_r:
                        x3, fit3 = copy.deepcopy(x_c), fit_c
                    else:
                        shrink = 1
                else:
                    x_c = self.Blending(x0, x3, 1)
                    fit_c = self.env.cost(self.env.decoder(x_c))
                    if fit_c < fitBest:
                        xBest, fitBest, improved, improvedX1 = copy.deepcopy(x_c), fit_c, 1, 1
                        
                    if fit_c < fit3:
                        x3, fit3 = copy.deepcopy(x_c), fit_c
                    else:
                        shrink = 1
            
            if shrink:
                x2 = self.Blending(x1, x2, 1)
                fit2 = self.env.cost(self.env.decoder(x2))
                if fit2 < fitBest:
                    xBest, fitBest, improved, improvedX1 = copy.deepcopy(x2), fit2, 1, 1

                x3 = self.Blending(x1, x3, 1)
                fit3 = self.env.cost(self.env.decoder(x3))
                if fit3 < fitBest:
                    xBest, fitBest, improved, improvedX1 = copy.deepcopy(x3), fit3, 1, 1
            
            if fit1 > fit2:
                x1, x2, fit1, fit2 = x2, x1, fit2, fit1
            if fit1 > fit3:
                x1, x3, fit1, fit3 = x3, x1, fit3, fit1
            if fit2 > fit3:
                x2, x3, fit2, fit3 = x3, x2, fit3, fit2
            
            x0 = self.Blending(x1, x2, 1)
            fit0 = self.env.cost(self.env.decoder(x0))
            if fit0 < fitBest:
                xBest, fitBest, improved, improvedX1 = copy.deepcopy(x0), fit0, 1, 1
            
            if improved == 1:
                improved = 0
                iter_count = 0
            else:
                iter_count += 1
        
        if improvedX1 == 1:
            return xBest
        else:
            return keys_origem       
        
    def RVND(self, keys, pool=None):
        best_keys = copy.deepcopy(keys)
        best_cost = self.env.cost(self.env.decoder(best_keys))

        neighborhoods = ['SwapLS', 'NelderMeadSearch', 'FareyLS', 'InvertLS']
        not_used_nb = copy.deepcopy(neighborhoods)
        
        while not_used_nb:
            current_neighborhood = random.choice(not_used_nb)
            
            if current_neighborhood == 'SwapLS':
                new_keys = self.SwapLS(best_keys)
            elif current_neighborhood == 'NelderMeadSearch':               
                new_keys = self.NelderMeadSearch(best_keys, pool)           
            elif current_neighborhood == 'FareyLS':
                new_keys = self.FareyLS(best_keys)
            elif current_neighborhood == 'InvertLS':
                new_keys = self.InvertLS(best_keys)
                
            new_cost = self.env.cost(self.env.decoder(new_keys))
            
            if new_cost < best_cost:
                best_keys = new_keys
                best_cost = new_cost
                
                not_used_nb = copy.deepcopy(neighborhoods)
                
                if pool is not None:
                    pool.insert((best_cost, list(best_keys)), "RVND", -1)
            else:
                not_used_nb.remove(current_neighborhood)
            
            if self.stop_condition(best_cost, "RVND", -1):
                return best_keys
        
        return best_keys
    
    def MultiStart(self, tag, pool):
        metaheuristic_name = "MS"
        start_time = time.time()
        tempo_max = self.max_time
        
        keys = self.random_keys()
        best_keys = keys
        solution = self.env.decoder(keys)
        best_cost = self.env.cost(solution)
        
        pool.insert((best_cost, list(best_keys)), metaheuristic_name, tag)
        if self.stop_condition(best_cost, metaheuristic_name, tag):
            return [], best_keys, best_cost
            
        while time.time() - start_time < tempo_max:
                k1 = random.sample(list(pool.pool), 1)[0][1]
                
                new_keys = self.shaking(k1, 0.1, 0.3)
                
                new_keys = self.RVND(pool=pool, keys=new_keys)
                
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
    
    def SimulatedAnnealing(self, SAmax=10, Temperatura=1000, alpha=0.95,  beta_min=0.05, beta_max=0.25, tag = 0, pool=None):
        metaheuristic_name = "SA"
        tempo_max = self.max_time

        keys = self.random_keys()
        keys = self.RVND(pool=pool, keys=keys)
        best_keys = keys

        solution = self.env.decoder(keys)
        cost = self.env.cost(solution)
        best_cost = cost
        
        pool.insert((best_cost, list(best_keys)), metaheuristic_name, tag)
        if self.stop_condition(best_cost, metaheuristic_name, tag):
                return [], best_keys, best_cost
            
        start_time = time.time()
        T = Temperatura

        while time.time() - start_time < tempo_max:
            iter_at_temp = 0
            while iter_at_temp < SAmax:
                iter_at_temp += 1

                new_keys = self.shaking(keys, beta_min, beta_max)
                new_solution = self.env.decoder(new_keys)
                new_cost = self.env.cost(new_solution)
                
                delta = new_cost - cost
                
                if new_cost < best_cost:
                    best_keys = new_keys
                    best_cost = new_cost
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

            T = T * alpha
            
            keys = self.NelderMeadSearch(pool=pool, keys=keys)
            new_cost = self.env.cost(self.env.decoder(keys))
            
            if new_cost < best_cost:
                best_keys = keys
                best_cost = new_cost
                pool.insert((best_cost, list(best_keys)), metaheuristic_name, tag)

        final_cost_solution = self.env.decoder(best_keys)
        final_cost_value = self.env.cost(final_cost_solution, True)
        
        return [], best_keys, final_cost_value

    def VNS(self, limit_time, x, tag, pool, beta_min=0.05, k_max=6):
        metaheuristic_name = "VNS"

        idx_k = 0
        start_time = time.time()
        
        keys = self.random_keys()
        keys = self.RVND(pool=pool, keys=keys)
        best_keys = keys
        best_cost = self.env.cost(self.env.decoder(keys))
        
        pool.insert((best_cost, list(best_keys)), metaheuristic_name, tag)
        if self.stop_condition(best_cost, metaheuristic_name, tag):
            return [], best_keys, best_cost

        while time.time() - start_time < limit_time:
            if idx_k >= k_max:
                idx_k = 0

            s1 = self.shaking(best_keys, idx_k * beta_min, (idx_k + 1) * beta_min)
            
            s2 = self.RVND(pool=pool, keys=s1)
            
            sol2 = self.env.decoder(s2)
            cost = self.env.cost(sol2)

            if cost <= best_cost:
                best_cost = cost
                best_keys = s2
                pool.insert((best_cost, list(best_keys)), metaheuristic_name, tag)
                idx_k = 0
            else:
                idx_k += 1
            
            if self.stop_condition(best_cost, metaheuristic_name, tag):
                return [], best_keys, best_cost

        final_cost_solution = self.env.decoder(best_keys)
        final_cost_value = self.env.cost(final_cost_solution)
        
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
                if random.random() < 0.5 and len(pool.pool) > 0:
                    parent1 = random.sample(list(pool.pool), 1)[0][1]
                else:
                    parent1 = random.sample(population, 1)[0]

                if random.random() < 0.5 and len(elite_keys) > 0:
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

        final_cost_solution = self.env.decoder(best_keys_overall)
        final_cost_value = self.env.cost(final_cost_solution)
        
        return [], best_keys_overall, best_fitness_overall

    def ILS(self, limit_time, x, tag, pool, beta_min=0.1, beta_max=0.5):
        metaheuristic_name = "ILS"
        start_time = time.time()
        
        keys = self.random_keys()
        keys = self.RVND(pool=pool, keys=keys)
        best_keys = keys
        best_cost = self.env.cost(self.env.decoder(keys))
        
        pool.insert((best_cost, list(best_keys)), metaheuristic_name, tag)
        if self.stop_condition(best_cost, metaheuristic_name, tag):
            return [], best_keys, best_cost

        while time.time() - start_time < limit_time:
            s1 = self.shaking(best_keys, beta_min, beta_max)
            
            s2 = self.RVND(pool=pool, keys=s1)
            
            sol2 = self.env.decoder(s2)
            cost = self.env.cost(sol2)

            if cost <= best_cost:
                best_cost = cost
                best_keys = s2
                pool.insert((best_cost, list(best_keys)), metaheuristic_name, tag)

            if self.stop_condition(best_cost, metaheuristic_name, tag):
                    return [], best_keys, best_cost

        final_cost_solution = self.env.decoder(best_keys)
        final_cost_value = self.env.cost(final_cost_solution)
        
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
        
        if n_workers is None:
            n_workers = cpu_count()
        self.max_time = limit_time
        
        manager = Manager()
        shared = manager.Namespace()
        
        shared.best_pair = manager.list([float('inf'), None, None])
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
    _, local_keys, local_best = runner.SimulatedAnnealing(tag = tag, pool = pool)