import torch
import time

def knapsack_decoder(random_keys, values, weights, capacity):
    """
    Decodificador para o problema da mochila usando PyTorch.
    
    Args:
        random_keys (torch.Tensor): Tensor de chaves aleatórias (random-keys) no intervalo [0, 1).
        values (torch.Tensor): Tensor com o valor (prize) de cada item.
        weights (torch.Tensor): Tensor com o peso (weight) de cada item.
        capacity (float): Capacidade máxima da mochila.
        
    Returns:
        torch.Tensor: O valor da função objetivo (custo) da solução.
    """
    # 1. Converte as chaves aleatórias em uma solução binária
    #    (1 se a chave for > 0.5, 0 caso contrário).
    #    Esta é a mesma lógica de decodificação encontrada na implementação em C++.
    solution = (random_keys > 0.5).int()
    
    # 2. Calcula o peso total e o valor total da solução.
    #    O PyTorch faz essas operações de forma vetorizada, o que é muito eficiente na GPU.
    total_weight = torch.sum(solution * weights)
    total_value = torch.sum(solution * values)
    
    # 3. Aplica uma penalidade para soluções infactíveis
    #    (quando o peso total excede a capacidade).
    infeasible_penalty = torch.zeros_like(total_value)
    if total_weight > capacity:
        infeasible_penalty = 100000.0 * (total_weight - capacity) # A mesma penalidade da versão C++.
    
    # 4. Calcula a função objetivo final.
    #    A penalidade é subtraída do valor total.
    objective_function_value = total_value - infeasible_penalty
    
    # 5. Converte o problema de maximização para minimização, negando o valor final.
    #    Esta é uma etapa padrão para que se encaixe no framework do RKO, que busca minimizar.
    cost = -objective_function_value
    
    return cost
    
# Exemplo de uso
if __name__ == '__main__':
    # Define os dados do problema em tensores do PyTorch
    # (Exemplo baseado nos dados de kp10.txt)
    # 'cuda' para executar na GPU, 'cpu' para a CPU
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    values = torch.tensor([7, 4, 4, 6, 2, 3, 7, 3, 2, 8], dtype=torch.float32, device=device)
    weights = torch.tensor([2, 3, 4, 5, 1, 5, 4, 2, 3, 7], dtype=torch.float32, device=device)
    capacity = 20.0

    # Cria um lote de 4 vetores de chaves aleatórias para testar a decodificação
    # Na sua busca local, você geraria vários desses e os passaria de uma vez
    random_keys_batch = torch.rand(40000, 10, device=device)
    
    print(f"Executando na: {device.type}")
    print("\n--- Exemplo de Decodificação em Lote ---")
    start = time.time()
    for i, keys in enumerate(random_keys_batch):
        cost = knapsack_decoder(keys, values, weights, capacity)
        
        # Converte a solução binária para ser exibida
        solution = (keys > 0.5).int()
        
        # Calcula o peso e valor para exibir (agora na CPU para o print)
        total_weight = torch.sum(solution * weights).item()
        total_value = torch.sum(solution * values).item()
        
        # print(f"\nSolução {i+1}:")
        # print(f"  Chaves Aleatórias: {keys.tolist()}")
        # print(f"  Solução Binária: {solution.tolist()}")
        # print(f"  Peso Total: {total_weight}")
        # print(f"  Valor Total: {total_value}")
        # print(f"  Custo (Função Objetivo): {cost.item():.2f}")
    print(f"\nTempo total de execução: {time.time() - start:.4f} segundos")