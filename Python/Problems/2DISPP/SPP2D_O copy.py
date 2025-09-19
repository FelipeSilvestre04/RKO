import sys
import os
import ast
from collections import Counter
import time
import numpy as np
from PIL import Image, ImageDraw
import matplotlib.pyplot as plt
import turtle
import math
from botao import Botao
from nfp_teste import combinar_poligonos, triangulate_shapely,NoFitPolygon, interpolar_pontos_poligono
from shapely import intersection_all
import shapely
from shapely import Polygon, MultiPolygon, unary_union, LineString, MultiLineString, MultiPoint, LinearRing, GeometryCollection, Point
from shapely.prepared import prep
import itertools
from scipy.spatial import ConvexHull
import numpy as np
import cv2
import copy
import pyautogui
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon as MPolygon, Rectangle
import random
import math
from typing import List, Tuple, Union
import sys
from shapely import affinity
import matplotlib.pyplot as plt
from shapely.geometry import Polygon, MultiPolygon, Point, LineString
from shapely.ops import unary_union
from shapely.affinity import translate
sys.path.append(os.path.abspath("C:\\Users\\felip\\Documents\\GitHub\\RKO\\Python"))

from RKO_v3 import RKO


def tratar_lista(lista_poligonos, Escala):
    def remove_vertices_repetidos(polygon):
        seen = set()
        unique_polygon = []
        for vertex in polygon:
            if vertex not in seen:
                unique_polygon.append(vertex)
                seen.add(vertex)
        return unique_polygon
    
    nova_lista = []
    nova_lista_completa = []
    
    for pol in lista_poligonos:
        novo_pol = []
        
        # Escalando os vértices
        for cor in pol:
            novo_pol.append((int(cor[0] * Escala), int(cor[1] * Escala)))
        
        # Removendo qualquer vértice repetido
        novo_pol = remove_vertices_repetidos(novo_pol)
        
        
        # Gerando pontos intermediários

        nova_lista.append(novo_pol)
    
    return nova_lista
def draw_cutting_area(pieces, area_width, area_height, legenda=None, filename=None):
    """
    Desenha a área de corte e as peças na escala correta, sem flip nem espelhamento.

    pieces: lista de peças (cada peça é lista de tuplas (x, y)), já posicionadas no espaço de corte
    area_width, area_height: dimensões do retângulo da área de corte
    legenda: texto da legenda (string ou lista de strings)
    filename: caminho para salvar a figura (string), opcional
    """
    # cria figura e eixo
    fig, ax = plt.subplots()

    # desenha retângulo da área de corte (contorno apenas)
    ax.add_patch(Rectangle((0, 0), area_width, area_height,
                           edgecolor='black', facecolor='none', linewidth=1.5))

    # desenha cada peça usando suas coordenadas absolutas
    for verts in pieces:
        poly = MPolygon(verts, closed=True,
                        facecolor=(173/255, 216/255, 230/255), edgecolor='black')
        ax.add_patch(poly)

    # configura limites do eixo para mostrar toda a área de corte
    ax.set_xlim(0, area_width)
    ax.set_ylim(0, area_height)
    ax.set_aspect('equal')

    # configura legenda ou título
    if legenda:
        if isinstance(legenda, (list, tuple)):
            ax.legend(legenda)
        else:
            ax.set_title(legenda)

    plt.tight_layout()

   
    directory = os.path.dirname(filename)

    if legenda == '[]':
        plt.show()

    else:
        
        if directory:
            os.makedirs(directory, exist_ok=True)
        if filename:
            plt.savefig(filename, dpi=150)
    
def offset_polygon(vertices, offset):
    """
    Cria um novo polígono com offset a partir de um polígono original usando Shapely
    
    Args:
        vertices: Lista de tuplas (x, y) representando os vértices do polígono
        offset: Valor do offset (positivo para expandir, negativo para contrair)
    
    Returns:
        Lista de tuplas (x, y) representando os vértices do novo polígono
    """
    if offset > 0:
        # Cria um polígono Shapely
        poly = Polygon(vertices)
        
        # Verifica se o polígono é válido
        if not poly.is_valid:
            return vertices
        
        # Aplica o buffer (offset)
        # join_style=1 (round) para suavizar cantos
        # mitre_limit controla quanto os cantos podem se estender
        buffered = poly.buffer(offset, join_style=1, mitre_limit=2.0)
        
        # Se o resultado for vazio ou inválido, retorna o original
        if buffered.is_empty or not buffered.is_valid:
            return vertices
        
        # Extrai os vértices do polígono resultante
        if buffered.geom_type == 'Polygon':
            # Pega apenas o exterior do polígono
            new_vertices = list(buffered.exterior.coords)[:-1]  # Remove o último ponto (duplicado)
        else:
            # Se o resultado for um MultiPolygon, pega o maior polígono
            largest = max(buffered.geoms, key=lambda x: x.area)
            new_vertices = list(largest.exterior.coords)[:-1]
        
        return new_vertices
    
    else:
        return vertices


def multiplicar_tudo(d, multiplicador):
    novo_dicionario = {}

    for chave, valor in d.items():
        # Multiplicar a chave
        nova_chave = tuple(
            multiplicar_elemento(e, multiplicador) for e in chave
        )

        # Multiplicar os valores (listas de tuplas)
        novo_valor = [
            tuple(x * multiplicador for x in ponto) for ponto in valor
        ]

        novo_dicionario[nova_chave] = novo_valor

    return novo_dicionario

def ler_poligonos(arquivo, escala=1):
    with open( 'C:\\Users\\felip\\Documents\\GitHub\\RKO\\Python\\Problems\\2DISPP\\' + arquivo + '.dat', 'r') as f:
        conteudo = f.read().strip()

    # Divide o conteúdo em linhas
    linhas = conteudo.split('\n')

    # Lê o número total de polígonos
    num_poligonos = int(linhas[0].strip())
    #print(f"Número total de polígonos: {num_poligonos}")

    poligonos = []
    i = 1  # Começa a leitura a partir da segunda linha

    while i < len(linhas):
        # Verifica se a linha não está vazia
        if linhas[i].strip():
            try:
                # Lê o número de vértices
                num_vertices = int(linhas[i].strip())
                #print(f"Lendo polígono com {num_vertices} vértices")  # Depuração
                i += 1

                # Lê os vértices
                vertices = []
                for _ in range(num_vertices):
                    # Verifica se a linha não está vazia
                    while i < len(linhas) and not linhas[i].strip():
                        i += 1
                    if i < len(linhas):
                        coords = linhas[i].strip().split()
                        if len(coords) != 2:
                            raise ValueError(f"Esperado 2 valores por linha, mas obteve {len(coords)}: '{linhas[i].strip()}'")
                        x, y = map(float, coords)
                        vertices.append((x * escala, y * escala))
                        i += 1
                    else:
                        raise ValueError(f"Esperado {num_vertices} vértices, mas o arquivo terminou prematuramente.")

                poligonos.append(vertices)
            except ValueError as ve:
                print(f"Erro ao processar a linha {i}: {linhas[i].strip()} - {ve}")
                i += 1
        else:
            i += 1
    if num_poligonos == len(poligonos):
        pass
        print(f'Todos os {num_poligonos} poligonos foram lidos com sucesso!')
    return poligonos




def multiplicar_elemento(e, multiplicador):
    if isinstance(e, (int, float)):
        return e * multiplicador
    elif isinstance(e, tuple):
        return tuple(multiplicar_elemento(x, multiplicador) for x in e)
    else:
        return e  # se aparecer algo que não seja número/tupla, mantém igual

def projetar_vertices_em_poligono(poligono_principal, lista_poligonos):
    """
    Projeta os vértices de uma lista de polígonos em um polígono principal.
    
    Para cada vértice dos polígonos na lista, cria duas retas:
    - Uma paralela ao eixo X
    - Uma paralela ao eixo Y
    E adiciona ao polígono principal os pontos de interseção dessas retas com o polígono.
    
    Args:
        poligono_principal: Lista de tuplas (x, y) representando os vértices do polígono principal.
        lista_poligonos: Lista de polígonos, onde cada polígono é uma lista de tuplas (x, y).
        
    Returns:
        Lista de tuplas (x, y) representando o polígono principal com as projeções adicionadas.
    """
    import math
    from functools import cmp_to_key
    
    # Verificação de entrada
    if not poligono_principal or len(poligono_principal) < 3:
        return poligono_principal.copy() if poligono_principal else []
    
    # Cria uma cópia do polígono principal para não modificar o original
    poligono_resultado = poligono_principal.copy()
    
    # Coletar todos os vértices dos polígonos da lista
    todos_vertices = []
    for poligono in lista_poligonos:
        if poligono:  # Verificar se o polígono não está vazio
            todos_vertices.extend(poligono)
    
    # Para cada vértice, encontrar as interseções das retas paralelas aos eixos com o polígono principal
    for vertice in todos_vertices:
        x_vertice, y_vertice = vertice
        
        # Encontrar interseções da reta horizontal (paralela ao eixo X) com o polígono principal
        for i in range(len(poligono_principal)):
            p1 = poligono_principal[i]
            p2 = poligono_principal[(i + 1) % len(poligono_principal)]
            
            # Verificar se o segmento cruza a linha horizontal y = y_vertice
            if not ((p1[1] <= y_vertice <= p2[1]) or (p2[1] <= y_vertice <= p1[1])):
                continue
                
            # Se os pontos têm a mesma coordenada y (segmento horizontal)
            if abs(p1[1] - p2[1]) < 1e-10:  # Usar uma pequena tolerância para comparação
                # Se o y do vértice coincide com o y do segmento horizontal
                if abs(p1[1] - y_vertice) < 1e-10:
                    # Adicionar os pontos do segmento horizontal que estão entre xmin e xmax
                    x_min = min(p1[0], p2[0])
                    x_max = max(p1[0], p2[0])
                    if x_min <= x_vertice <= x_max:
                        ponto_intersecao = (x_vertice, y_vertice)
                        if ponto_intersecao not in poligono_resultado:
                            poligono_resultado.append(ponto_intersecao)
            else:
                # Segmento não horizontal - calcular interseção
                t = (y_vertice - p1[1]) / (p2[1] - p1[1])
                if 0 <= t <= 1:  # Verificar se a interseção está dentro do segmento
                    x_intersecao = p1[0] + t * (p2[0] - p1[0])
                    ponto_intersecao = (x_intersecao, y_vertice)
                    if ponto_intersecao not in poligono_resultado:
                        poligono_resultado.append(ponto_intersecao)
        
        # Encontrar interseções da reta vertical (paralela ao eixo Y) com o polígono principal
        for i in range(len(poligono_principal)):
            p1 = poligono_principal[i]
            p2 = poligono_principal[(i + 1) % len(poligono_principal)]
            
            # Verificar se o segmento cruza a linha vertical x = x_vertice
            if not ((p1[0] <= x_vertice <= p2[0]) or (p2[0] <= x_vertice <= p1[0])):
                continue
                
            # Se os pontos têm a mesma coordenada x (segmento vertical)
            if abs(p1[0] - p2[0]) < 1e-10:  # Usar uma pequena tolerância para comparação
                # Se o x do vértice coincide com o x do segmento vertical
                if abs(p1[0] - x_vertice) < 1e-10:
                    # Adicionar os pontos do segmento vertical que estão entre ymin e ymax
                    y_min = min(p1[1], p2[1])
                    y_max = max(p1[1], p2[1])
                    if y_min <= y_vertice <= y_max:
                        ponto_intersecao = (x_vertice, y_vertice)
                        if ponto_intersecao not in poligono_resultado:
                            poligono_resultado.append(ponto_intersecao)
            else:
                # Segmento não vertical - calcular interseção
                t = (x_vertice - p1[0]) / (p2[0] - p1[0])
                if 0 <= t <= 1:  # Verificar se a interseção está dentro do segmento
                    y_intersecao = p1[1] + t * (p2[1] - p1[1])
                    ponto_intersecao = (x_vertice, y_intersecao)
                    if ponto_intersecao not in poligono_resultado:
                        poligono_resultado.append(ponto_intersecao)
    
    # Se temos menos de 3 pontos, não podemos formar um polígono
    if len(poligono_resultado) < 3:
        return poligono_resultado
    
    # Reordenar os pontos no sentido anti-horário
    # Calcular o centroide
    cx = sum(x for x, _ in poligono_resultado) / len(poligono_resultado)
    cy = sum(y for _, y in poligono_resultado) / len(poligono_resultado)
    
    # Função para comparar pontos baseada no ângulo com respeito ao centroide
    def comparar_pontos(p1, p2):
        angulo1 = math.atan2(p1[1] - cy, p1[0] - cx)
        angulo2 = math.atan2(p2[1] - cy, p2[0] - cx)
        return -1 if angulo1 < angulo2 else (1 if angulo1 > angulo2 else 0)
    
    # Reordenar os pontos
    poligono_resultado.sort(key=cmp_to_key(comparar_pontos))
    
    # Remover pontos duplicados ou muito próximos
    i = 0
    while i < len(poligono_resultado):
        j = (i + 1) % len(poligono_resultado)
        p1 = poligono_resultado[i]
        p2 = poligono_resultado[j]
        
        # Verificar se os pontos são muito próximos
        distancia = math.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)
        if distancia < 1e-6:  # Tolerância para pontos próximos
            # Remover o ponto j
            poligono_resultado.pop(j if j < i else i)
        else:
            i += 1
    
    return poligono_resultado

def dimensions(dataset: str): 

    specs = {
        'fu':          (34.0,     38.0,     1, [0,1,2,3]),
        'jackobs1':    (13.0,     40.0,     1, [0,1,2,3]),
        'jackobs2':    (28.2,     70.0,     1, [0,1,2,3]),
        'shapes0':     (65.0,     40.0,     1, [0]),
        'shapes1':     (65.0,     40.0,     1, [0,2]),
        'shapes2':     (27.3,     15.0,     1, [0,2]),
        'dighe1':      (138.14,   100.0,    1, [0]),
        'dighe2':      (134.05,   100.0,    1, [0]),
        'albano':      (10122.63, 4900.0,   1, [0,2]),
        'dagli':       (65.6,     60.0,     1, [0,2]),
        'mao':         (2058.6,   2550.0,   1, [0,1,2,3]),
        'marques':     (83.6,     104.0,    1, [0,1,2,3]),
        'shirts':      (63.13,    40.0,     1, [0,2]),
        'swim':        (6568.0,   5752.0,   1, [0,2]),
        'trousers':    (245.75,   79.0,     1, [0,2]),
    }

    return specs.get(dataset, (None, None, None, None))

def pre_processar_NFP(rotacoes, lista_pecas, offset, env):
    tabela_nfps = {}
    lista_unica = []
    for peca in lista_pecas:
        if peca not in lista_unica:
            lista_unica.append(peca)
    
    total = len(lista_unica) * len(rotacoes) * len(lista_unica) * len(rotacoes)
    atual = 0
    
    for pecaA in lista_unica:
        for grauA in rotacoes:
            for pecaB in lista_unica:
                for grauB in rotacoes:
                    atual += 1
                    porcentagem = (atual / total) * 100
                    print(f"\rPré-processando NFPs: {porcentagem:.1f}% concluído", end="")
                    
                    chave = (tuple(pecaA), grauA, tuple(pecaB), grauB)
                    
                    # nfp agora é um objeto Polygon
                    nfp, intersec = NFP(pecaA, grauA, pecaB, grauB, env)
            
                    # --- ALTERAÇÃO PRINCIPAL AQUI ---
                    # Armazenamos o objeto Polygon diretamente.
                    # Como offset_polygon não faz nada, podemos simplesmente atribuir.
                    tabela_nfps[chave] = [list(nfp.exterior.coords),intersec]

    return tabela_nfps
import matplotlib.pyplot as plt
from matplotlib.patches import PathPatch
from matplotlib.path import Path
# (mantenha suas outras importações como Polygon, unary_union, etc.)

# --- FUNÇÃO AUXILIAR PARA DESENHAR POLÍGONOS DO SHAPELY ---
def plot_shapely_geometry(ax, geom, facecolor='lightblue', edgecolor='black', alpha=0.5, linewidth=1.0):
    geom = Polygon(geom) if isinstance(geom, list) else geom
    """
    Desenha uma geometria do Shapely (Polygon, MultiPolygon, etc.) em um eixo Matplotlib.
    Esta função sabe como lidar com buracos.
    """
    if geom is None or geom.is_empty:
        return

    # Trata coleções de geometrias (como MultiPolygon)
    geoms_to_plot = getattr(geom, 'geoms', [geom])
    
    for g in geoms_to_plot:
        if not isinstance(g, Polygon):
            continue

        path_verts = list(g.exterior.coords)
        path_codes = [Path.MOVETO] + [Path.LINETO] * (len(path_verts) - 2) + [Path.CLOSEPOLY]
        
        for interior in g.interiors:
            interior_verts = list(interior.coords)
            path_verts.extend(interior_verts)
            path_codes.extend([Path.MOVETO] + [Path.LINETO] * (len(interior_verts) - 2) + [Path.CLOSEPOLY])

        path = Path(path_verts, path_codes)
        patch = PathPatch(path, facecolor=facecolor, edgecolor=edgecolor, alpha=alpha, linewidth=linewidth)
        ax.add_patch(patch)
        
      
def NFP(PecaA, grauA, PecaB, grauB, env):
    """
    Executa a lógica da sua função NFP e plota os resultados intermediários,
    incluindo ambas as peças para referência.
    """
    # --- ETAPA 1: Cálculos (idêntico à sua função NFP) ---
    pontos_pol_A = env.rot_pol(env.lista.index(PecaA), grauA)
    pontos_pol_B = env.rot_pol(env.lista.index(PecaB), grauB)
    


    if Polygon(pontos_pol_B).equals(Polygon(pontos_pol_B).convex_hull):
        convex_partsB = [pontos_pol_B] 
    else:
        convex_partsB = triangulate_shapely(pontos_pol_B)
    
    if Polygon(pontos_pol_A).equals(Polygon(pontos_pol_A).convex_hull):
        convex_partsA = [pontos_pol_A]
    else:
        convex_partsA = triangulate_shapely(pontos_pol_A)

    nfps_convx = []
    intersec_parts = []
    for cb_poly in convex_partsB:
        intersec_B = []
        for ca_poly in convex_partsA:
            nfp_part = NoFitPolygon(ca_poly, cb_poly)
            if nfp_part and not nfp_part.is_empty:
                nfps_convx.append(nfp_part)
                intersec_B.append(nfp_part)
        if intersec_B:
            intersec_parts.append(intersec_B)

    if not nfps_convx:
        print("Nenhum NFP parcial foi gerado.")
        while True:
            print(len(convex_partsA), len(convex_partsB))
            print(PecaA)
            print(PecaB)
        return Polygon(), None

    nfp_unido = unary_union(nfps_convx)
    pontos_candidatos = set()
    for subgrupo in intersec_parts:
        for ponto in pontos_pol_A:
            pontos_candidatos.add(Point(ponto))
            
        for ponto in pontos_pol_B:
            pontos_candidatos.add(Point(ponto))
        # 1. Adiciona TODOS os vértices originais de cada NFP parcial no subgrupo
        for nfp in subgrupo:
            for ponto in nfp.exterior.coords:
                pontos_candidatos.add(ponto)
        
        # 2. Adiciona os NOVOS vértices criados na interseção par a par
        if len(subgrupo) > 1:
            for p1, p2 in itertools.combinations(subgrupo, 2):
                intersec = p1.boundary.intersection(p2.boundary)
                if not intersec.is_empty:
                    # Usa sua função extrair_vertices para pegar os pontos da interseção
                    for ponto in extrair_vertices(intersec):
                        pontos_candidatos.add(ponto)
    if len(nfps_convx) > 1:
        intersec_total = intersection_all([nfp.boundary for nfp in nfps_convx])
        if not intersec_total.is_empty:
            for ponto in extrair_vertices(intersec_total):
                pontos_candidatos.add(ponto)
        for p1, p2 in itertools.combinations(nfps_convx, 2):
                intersec = p1.boundary.intersection(p2.boundary)
                if not intersec.is_empty:
                    # Usa sua função extrair_vertices para pegar os pontos da interseção
                    for ponto in extrair_vertices(intersec):
                        pontos_candidatos.add(ponto)
                
    # # Une todas as geometrias de interseção encontradas
    # min_x, min_y, max_x, max_y = nfp_unido.bounds
    # padding = 1.0 

    # # Itere sobre os vértices da Peça A
    # for ponto in pontos_pol_A:
    #     x_vertice, y_vertice = ponto

    #     # --- Raio Vertical para Cima ---
    #     linha_cima = LineString([ponto, (x_vertice, max_y + padding)])
    #     intersecao_cima = nfp_unido.boundary.intersection(linha_cima)
    #     pontos_candidatos.update(extrair_vertices(intersecao_cima))

    #     # --- Raio Vertical para Baixo ---
    #     linha_baixo = LineString([ponto, (x_vertice, min_y - padding)])
    #     intersecao_baixo = nfp_unido.boundary.intersection(linha_baixo)
    #     pontos_candidatos.update(extrair_vertices(intersecao_baixo))

    #     # --- Raio Horizontal para Direita ---
    #     linha_direita = LineString([ponto, (max_x + padding, y_vertice)])
    #     intersecao_direita = nfp_unido.boundary.intersection(linha_direita)
    #     pontos_candidatos.update(extrair_vertices(intersecao_direita))

    #     # --- Raio Horizontal para Esquerda ---
    #     linha_esquerda = LineString([ponto, (min_x - padding, y_vertice)])
    #     intersecao_esquerda = nfp_unido.boundary.intersection(linha_esquerda)
    #     pontos_candidatos.update(extrair_vertices(intersecao_esquerda))
        
    # for ponto in pontos_pol_B:
    #     pontos_candidatos.add(Point(ponto))
    

  
    intersec = MultiPoint(list(pontos_candidatos))
    
    nfp_f = []
    inter = []
    for ponto in extrair_vertices(intersec):
        if nfp_unido.touches(Point(ponto)):
            nfp_f.append(ponto)
        else:
            inter.append(ponto)
            
    intersec = MultiPoint(inter)
    nfp_unido = unary_union([nfp_unido, MultiPoint(nfp_f)])
        
    # # print(intersec)
    polyA = Polygon(pontos_pol_A)
    polyB = Polygon(pontos_pol_B)
    
    pontos_de_encontro_validos = []
    for ponto in extrair_vertices(intersec):
    
        # Move a Peça B original (rotacionada) para a posição do ponto candidato
        polyB_na_posicao = affinity.translate(polyB, xoff=ponto[0], yoff=ponto[1])
        
        # O método .overlaps verifica se os interiores se cruzam.
        # .touches seria verdadeiro, .overlaps deve ser falso.
        if not polyA.overlaps(polyB_na_posicao) and polyA.touches(polyB_na_posicao):
            # # Adicionalmente, verificamos se não está muito longe (evita pontos inválidos)
            # if polyA.distance(polyB_na_posicao) < 1e-6:
                pontos_de_encontro_validos.append(ponto)
    intersec = MultiPoint(pontos_de_encontro_validos)

    
    # # --- ETAPA 2: Visualização ---

    # # --- ETAPA 2: Visualização em Layout 2x2 ---
    # fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 16))
    # fig.suptitle(f"Análise do NFP entre Peça A (rot {grauA}°) e Peça B (rot {grauB}°)", fontsize=16)

    # # Gráfico 1: Decomposição da Peça A
    # ax1.set_title("1. Decomposição Convexa da Peça A (Fixa)")
    # plot_shapely_geometry(ax1, polyA, facecolor='gray', alpha=0.3)
    # for i, part in enumerate(convex_partsA):
    #     plot_shapely_geometry(ax1, part, facecolor=f'C{i}', alpha=0.5)
    
    # # Gráfico 2: Decomposição da Peça B
    # ax2.set_title("2. Decomposição Convexa da Peça B (Rotacional)")
    # plot_shapely_geometry(ax2, polyB, facecolor='gray', alpha=0.3)
    # for i, part in enumerate(convex_partsB):
    #     plot_shapely_geometry(ax2, part, facecolor=f'C{i+len(convex_partsA)}', alpha=0.5)

    # # Gráfico 3: NFPs Parciais
    # ax3.set_title("3. Peça A + NFPs Parciais")
    # plot_shapely_geometry(ax3, polyA, facecolor='gray', alpha=0.9)
    # for nfp_part in nfps_convx:
    #     plot_shapely_geometry(ax3, nfp_part, facecolor='cyan', alpha=0.2, edgecolor='blue')

    # # Gráfico 4: NFP Final Unido
    # ax4.set_title("4. Peça A + NFP Final (União)")
    # plot_shapely_geometry(ax4, polyA, facecolor='gray', alpha=0.9)
    # plot_shapely_geometry(ax4, nfp_unido, facecolor='red', alpha=0.4, edgecolor='red')
    
    # # --- NOVO BLOCO DE CÓDIGO AQUI ---
    # # Plota a geometria de interseção nos gráficos 3 e 4 para destaque
    # if intersec and not intersec.is_empty:
    #     geoms_to_plot = getattr(intersec, 'geoms', [intersec])
    #     for geom in geoms_to_plot:
    #         if isinstance(geom, (Point, MultiPoint)):
    #             # Se for um ponto, plota como um círculo preto grande
    #             ax3.plot(geom.x, geom.y, 'ko', markersize=10, label='Ponto de Interseção Total')
    #             ax4.plot(geom.x, geom.y, 'ko', markersize=10)
    #         elif isinstance(geom, (LineString, MultiLineString)):
    #              # Se for uma linha, plota como uma linha preta grossa
    #             ax3.plot(*geom.xy, color='black', linewidth=3, label='Linha de Interseção Total')
    #             ax4.plot(*geom.xy, color='black', linewidth=3)
    
    # ax3.legend(loc='upper right') # Adiciona a legenda ao gráfico 3
    # # --- FIM DO NOVO BLOCO ---
    
    # # Ajusta os limites de todos os eixos
    # try:
    #     all_geoms = [polyA, polyB] + nfps_convx + [nfp_unido]
    #     min_x, min_y, max_x, max_y = unary_union(all_geoms).bounds
    #     padding = 5
    #     for ax in [ax1, ax2, ax3, ax4]:
    #         ax.set_aspect('equal', adjustable='box')
    #         ax.grid(True, linestyle='--', alpha=0.6)
    #         ax.set_xlim(min_x - padding, max_x + padding)
    #         ax.set_ylim(min_y - padding, max_y + padding)
    # except Exception as e:
    #     print(f"Erro ao ajustar eixos do gráfico: {e}")

    # plt.tight_layout(rect=[0, 0, 1, 0.96])
    # plt.show()
    
   
    return nfp_unido, extrair_vertices(intersec)

def extrair_vertices(encaixes):
    if encaixes is None or encaixes.is_empty:
        return []
    vertices = []
    if isinstance(encaixes, MultiPolygon):
        for poly in encaixes.geoms:
            vertices.extend(list(poly.exterior.coords))
            for hole in poly.interiors:
                vertices.extend(list(hole.coords))
    elif isinstance(encaixes, Polygon):
        vertices.extend(list(encaixes.exterior.coords))
        for hole in encaixes.interiors:
            vertices.extend(list(hole.coords))
    elif isinstance(encaixes, MultiLineString):
        for line in encaixes.geoms:
            vertices.extend(list(line.coords))
    elif isinstance(encaixes, LineString):
        vertices.extend(list(encaixes.coords))
    elif isinstance(encaixes, Point):
        vertices.append((encaixes.x, encaixes.y))
    elif isinstance(encaixes, MultiPoint):
        for pt in encaixes.geoms:
            vertices.append((pt.x, pt.y))

    elif isinstance(encaixes, LinearRing):
        vertices.extend(list(encaixes.coords))

    elif isinstance(encaixes, GeometryCollection):
        encaixe = encaixes
        for encaixes in encaixe.geoms:
            if isinstance(encaixes, MultiPolygon):
                for poly in encaixes.geoms:
                    vertices.extend(list(poly.exterior.coords))
                    for hole in poly.interiors:
                        vertices.extend(list(hole.coords))
            elif isinstance(encaixes, Polygon):
                vertices.extend(list(encaixes.exterior.coords))
                for hole in encaixes.interiors:
                    vertices.extend(list(hole.coords))
            elif isinstance(encaixes, MultiLineString):
                for line in encaixes.geoms:
                    vertices.extend(list(line.coords))
            elif isinstance(encaixes, LineString):
                vertices.extend(list(encaixes.coords))

            elif isinstance(encaixes, Point):
                vertices.append((encaixes.x, encaixes.y))
            elif isinstance(encaixes, MultiPoint):
                for pt in encaixes.geoms:
                    vertices.append((pt.x, pt.y))

            elif isinstance(encaixes, LinearRing):
                vertices.extend(list(encaixes.coords))

    return vertices
def rotate_point(x: float, y: float, angle_deg: float) -> Tuple[float, float]:
    """
    Rotaciona o ponto (x, y) em torno da origem por angle_deg graus (sentido anti-horário).
    Retorna a tupla (x_rot, y_rot).
    """
    rad = math.radians(angle_deg % 360)
    cos_a = math.cos(rad)
    sin_a = math.sin(rad)
    x_rot = x * cos_a - y * sin_a
    y_rot = x * sin_a + y * cos_a
    return x_rot, y_rot
def calcular_shrink_factor(x):

    if x < 0.1:
        return 0.9 - x
    else:
        # Para x=0.1, log10(0.1) = -1, resultando em 1 + 0.1*(-1) = 0.9.
        # Para x=1.0, log10(1.0) = 0, resultando em 1 + 0.1*(0) = 1.0.
        return 1 + 0.1 * math.log10(x)

class SPP2D():
    def __init__(self,dataset='fu',Base=None,Altura=None,Escala=None, Graus = None, tabela = None, margem = 0, tempo=200, decoder = 'D1', pairwise = False):
        self.decoder_type = decoder
        
        self.save_q_learning_report = True
        self.counter = 0
        self.BRKGA_parameters = {
            'p': [100, 50],          # Tamanho da população
            'pe': [0.20, 0.15],        # Fração da população que compõe a elite
            'pm': [0.05],        # Fração da população de mutantes
            'rhoe': [0.70]       # Probabilidade de herança do alelo do pai de elite
        }
        # Exemplo Online: self.BRKGA_parameters = {'p': [300, 400, 500], 'pe': [0.10, 0.15, 0.20], ...}

        # Parâmetros para Simulated Annealing (SA)
        self.SA_parameters = {
            'SAmax': [10, 5],     # Número de iterações por temperatura
            'alphaSA': [0.5, 0.7],   # Fator de resfriamento
            'betaMin': [0.01, 0.03],   # Intensidade mínima da perturbação (shaking)
            'betaMax': [0.05, 0.1],   # Intensidade máxima da perturbação (shaking)
            'T0': [10]      # Temperatura inicial
        }

        # Parâmetros para Iterated Local Search (ILS)
        self.ILS_parameters = {
            'betaMin': [0.10,0.5],   # Intensidade mínima da perturbação (shaking)
            'betaMax': [0.20,0.15]    # Intensidade máxima da perturbação (shaking)
        }
        # Exemplo Online: self.ILS_parameters = {'betaMin': [0.05, 0.10, 0.15], 'betaMax': [0.15, 0.20, 0.25]}

        # Parâmetros para Variable Neighborhood Search (VNS)
        self.VNS_parameters = {
            'kMax': [5,3],         # Número máximo de estruturas de vizinhança
            'betaMin': [0.05, 0.1]    # Fator base para a intensidade da perturbação
        }

        # Parâmetros para Particle Swarm Optimization (PSO)
        self.PSO_parameters = {
            'PSize': [100,50],      # Tamanho do enxame (número de partículas)
            'c1': [2.05],        # Coeficiente cognitivo
            'c2': [2.05],        # Coeficiente social
            'w': [0.73]          # Fator de inércia
        }

        # Parâmetros para Genetic Algorithm (GA)
        self.GA_parameters = {
            'sizePop': [100,50],    # Tamanho da população
            'probCros': [0.98],  # Probabilidade de crossover
            'probMut': [0.005, 0.01]   # Probabilidade de mutação
        }

        # Parâmetros para Large Neighborhood Search (LNS)
        self.LNS_parameters = {
            'betaMin': [0.10],   # Intensidade mínima da destruição
            'betaMax': [0.30],   # Intensidade máxima da destruição
            'TO': [100],       # Temperatura inicial
            'alphaLNS': [0.95,0.9]   # Fator de resfriamento
        }
        self.max_time = tempo
        self.start_time = time.time()
        self.dataset = dataset
        self.instance_name = dataset
        if Base == None and Altura == None and Escala == None:
            self.base, self.altura, self.escala, self.graus = dimensions(dataset)
        else:
            self.base = Base
            self.altura = Altura
            self.escala = Escala
            self.graus = Graus
            
        # self.base*= 2
        self.area = self.base * self.altura
        self.base_inicial = self.base
        self.inicial = False
        
        
        lista = ler_poligonos(self.dataset)
        
        lista.sort(
                key=lambda coords: Polygon(coords).area,
                reverse=True
            )
  
        # print(lista)
        self.lista_original = lista
        
        
        self.lista = copy.deepcopy(self.lista_original)
        

        porcentagens_por_dataset = {
            'albano': 0.0,
            'dagli': 0.20,
            'dighe1': 0.0,
            'dighe2': 0.0,
            'fu': 0.0,
            'jackobs1': 0.20,
            'jackobs2': 0.40,
            'mao': 0.0,
            'marques': 0.10,
            'shapes0': 0.0,
            'shapes1': 0.0,
            'shapes2': 0.25,
            'shirts': 0.0,
            'swim': 0.15,
            'trousers': 0.0
        }
        
        porcentagem = porcentagens_por_dataset.get(self.dataset.lower(), 0.0)



            

            
            
      
            
        self.cordenadas_area = ( [0,0] , [self.base,0] , [self.base,self.altura] , [0,self.altura] )
        
        self.pecas_posicionadas = []
        self.indices_pecas_posicionadas = []
        self.dict_nfps = {}
        # self.regras = {
        #     0: self.UL,
        #     1: self.BL,
        # }

        

        self.dict_sol = {}

        #     self.regras = {
        #     0: self.BL,
        #     1: self.UL,

        # } 
            
        self.LS_type = 'Best'
        self.greedy = []
        self.dict_best = {
                    "fu": -92.41,
                    "jackobs1": -89.10,
                    "jackobs2": -87.73,
                    "shapes0": -68.79,
                    "shapes1": -76.73,
                    "shapes2": -84.84,
                    "dighe1": -100.00,
                    "dighe2": -100.00,
                    "albano": -89.58,
                    "dagli": -89.51,
                    "mao": -85.44,
                    "marques": -90.59,
                    "shirts": -88.96,
                    "swim": -75.94,
                    "trousers": -91.00
                }
        
    
        self.dict_feasible = {}
        self.lista_anterior = []
        self.best_fit = 100000

        if tabela is not None:
            self.tabela_nfps = tabela
        
        elif os.path.exists(f"nfp_{self.dataset}.txt") and  (not pairwise or porcentagem ==0.0):
            with open(f"nfp_{self.dataset}.txt", "r") as f:
                conteudo = f.read()
            self.tabela_nfps = ast.literal_eval(conteudo)
            
        elif os.path.exists(f"nfp_{self.dataset}_pairwise.txt") and pairwise:

            with open(f"nfp_{self.dataset}.txt", "r") as f:
                conteudo = f.read()
            self.tabela_nfps = ast.literal_eval(conteudo)

            porcentagem = porcentagens_por_dataset.get(self.dataset.lower(), 0.0)
            pares_selecionados = self.pairwise(porcentagem_cluster=porcentagem)
            self.lista = self.criar_lista_clusterizada(pares_selecionados)
            self.lista_original = copy.deepcopy(self.lista)
        
            with open(f"nfp_{self.dataset}_pairwise.txt", "r") as f:
                conteudo = f.read()
            self.tabela_nfps = ast.literal_eval(conteudo)

        elif not pairwise:
            self.tabela_nfps = pre_processar_NFP(self.graus, self.lista, margem, self)
            with open(f"nfp_{self.dataset}.txt", "w") as f:
                f.write(repr(self.tabela_nfps))
        elif pairwise:
            with open(f"nfp_{self.dataset}.txt", "r") as f:
                conteudo = f.read()
            self.tabela_nfps = ast.literal_eval(conteudo)
        
            porcentagem = porcentagens_por_dataset.get(self.dataset.lower(), 0.0)
            pares_selecionados = self.pairwise(porcentagem_cluster=porcentagem)
            self.lista = self.criar_lista_clusterizada(pares_selecionados)
            self.lista_original = copy.deepcopy(self.lista)

            self.tabela_nfps = pre_processar_NFP(self.graus, self.lista, margem, self)
            with open(f"nfp_{self.dataset}_pairwise.txt", "w") as f:
                f.write(repr(self.tabela_nfps))

        self.max_pecas = len(self.lista_original)
        if self.decoder_type == 'D1_A' or self.decoder_type == 'D2_A':
            self.tam_solution = 2 * self.max_pecas + 1
            self.regras = {
            0: self.BL,
            1: self.LB,
            2: self.BR,
            3: self.RB,
            4: self.UL,
            5: self.LU,
            6: self.UR,
            7: self.RU,
        }        
        elif self.decoder_type == 'D1_B' or self.decoder_type == 'D2_B':
            self.tam_solution = 3 * self.max_pecas + 1
            self.regras = {
            0: self.BL,
            1: self.LB,
            2: self.BR,
            3: self.RB,
            4: self.UL,
            5: self.LU,
            6: self.UR,
            7: self.RU,
        }
            
        elif self.decoder_type == 'D0_A' or self.decoder_type == 'D0':
            self.tam_solution = 2 * self.max_pecas 
            self.regras = {
            0: self.BL,
            1: self.LB,
            2: self.UL,
            3: self.LU,

        }   
            self.regras = {
            0: self.BL,
            1: self.UL,

        }   
        elif self.decoder_type == 'D0_B':
            self.tam_solution = 3 * self.max_pecas
            self.regras = {
            0: self.BL,
            1: self.LB,
            2: self.UL,
            3: self.LU,

        }   

        # 3. Obtenha a porcentagem correta para o dataset atual (ou 0% como padrão)
        # Em seu script main ou onde você está executando a lógica

        # As duas primeiras linhas estão corretas

        # print(nova_lista)



      
    def plot_pairwise_geometries(self, peca1_poly, peca2_poly, nfp_poly, titulo="", filepath="."):
        """
        Salva uma imagem com as geometrias envolvidas na análise de um par de peças.
        """
        fig, ax = plt.subplots(figsize=(10, 8))
        
        # Plota a Peça 1 (fixa, na origem)
        x, y = peca1_poly.exterior.xy
        ax.fill(x, y, alpha=0.6, fc='blue', ec='black', label='Peça 1 (Fixa)')

        # Plota a Peça 2 (posicionada em um vértice do NFP)
        x, y = peca2_poly.exterior.xy
        ax.fill(x, y, alpha=0.6, fc='green', ec='black', label='Peça 2 (Móvel)')

        # Plota o NFP
        if nfp_poly and not nfp_poly.is_empty:
            x, y = nfp_poly.exterior.xy
            ax.plot(x, y, color='red', linestyle='--', linewidth=2, label='NFP')

        ax.set_aspect('equal', adjustable='box')
        ax.legend()
        # Usa apenas a primeira linha do título para o nome do arquivo, removendo caracteres inválidos
        safe_filename = titulo.split('\n')[0].replace(' | ', '_').replace(':', '').replace('.', 'p') + ".png"
        ax.set_title(titulo)
        ax.grid(True, which='both', linestyle='--', linewidth=0.5)
        
        # Garante que o diretório para salvar a imagem exista
        os.makedirs(filepath, exist_ok=True)
        full_path = os.path.join(filepath, safe_filename)
        
        plt.savefig(full_path, bbox_inches='tight')
        plt.close(fig) # Fecha a figura para liberar memória

    def find_dominant_point(self, nfp_poly):
        """
        Encontra o "Dominant Point" de um No-Fit Polygon côncavo.
        O Dp é o vértice do NFP mais distante da borda do seu fecho convexo.

        Args:
            nfp_poly (Polygon): O polígono NFP a ser analisado.

        Returns:
            tuple: As coordenadas (x, y) do ponto dominante, ou None se não for encontrado.
        """
        if nfp_poly is None or nfp_poly.is_empty or not hasattr(nfp_poly, 'exterior'):
            return None

        # O conceito de "Dominant Point" só se aplica a polígonos côncavos.
        ch = nfp_poly.convex_hull
        # Usamos uma pequena tolerância para problemas de ponto flutuante
        if nfp_poly.area >= ch.area - 1e-9:
            return None # Polígono é convexo, não há Dp.

        # As "vacancies" são as áreas de concavidade (diferença entre o fecho e o polígono)
        vacancies = ch.difference(nfp_poly)
        if vacancies.is_empty:
            return None

        # Garante que as vacancies sejam sempre uma lista de polígonos
        if isinstance(vacancies, Polygon):
            vacancies = [vacancies]
        else:
            vacancies = list(vacancies.geoms)

        max_dist = -1
        dominant_point = None

        # Para cada área de concavidade (vacancy)...
        for vacancy in vacancies:
            # A "opponent side" é a borda que a concavidade compartilha com o fecho convexo
            opponent_side = vacancy.boundary.intersection(ch.boundary)
            
            if opponent_side.is_empty or not isinstance(opponent_side, (LineString, Point)):
                continue
                
            # Os pontos a serem testados são os vértices do NFP que formam essa concavidade
            points_to_check_geom = nfp_poly.boundary.intersection(vacancy)
            
            all_coords = []
            if hasattr(points_to_check_geom, 'geoms'):
                for geom in points_to_check_geom.geoms:
                    all_coords.extend(list(geom.coords))
            elif hasattr(points_to_check_geom, 'coords'):
                all_coords = list(points_to_check_geom.coords)
            
            # Encontra o ponto (vértice) com a maior distância até a borda do fecho convexo
            for p_coords in all_coords:
                p = Point(p_coords)
                dist = p.distance(opponent_side)
                if dist > max_dist:
                    max_dist = dist
                    dominant_point = p_coords

        return dominant_point
    def pairwise(self, porcentagem_cluster=0.2):
        """
        Processa todas as combinações de pares, utilizando cache para evitar cálculos
        repetidos de formas. Calcula o ClusterValue para os pontos mais promissores
        e, ao final, salva imagens das N melhores configurações encontradas.
        """

        if porcentagem_cluster == 0:
            return
        avaliacoes = []
        pairwise_cache = {} # Dicionário para cache dos cálculos geométricos

        pecas_a_analisar = range(len(self.lista_original))
        rotacoes_a_analisar = self.graus

        # Calculando o total de combinações de TIPOS de peça para o progresso
        tipos_unicos = []
        for peca in self.lista_original:
            if peca not in tipos_unicos:
                tipos_unicos.append(peca)
        num_tipos = len(tipos_unicos)
        num_pares_tipos = (num_tipos * (num_tipos - 1)) / 2 + num_tipos # Pares (A,B) + Pares (A,A)
        total_steps = int(num_pares_tipos * (len(rotacoes_a_analisar)**2))
        step = 0
        print(f"Iniciando análise de pairwise... Total de combinações de TIPOS de peça/rotações: ~{total_steps}")

        for i in pecas_a_analisar:
            for j in range(i + 1, len(pecas_a_analisar)):
                for grau1_idx in rotacoes_a_analisar:
                    for grau2_idx in rotacoes_a_analisar:
                        
                        # --- Cria uma chave canônica para o cache baseada nas FORMAS ---
                        shape1_coords = tuple(map(tuple, self.lista_original[i]))
                        shape2_coords = tuple(map(tuple, self.lista_original[j]))
                        key_shapes = tuple(sorted((shape1_coords, shape2_coords)))
                        cache_key = (key_shapes[0], grau1_idx, key_shapes[1], grau2_idx)

                        # --- Verifica o cache ---
                        if cache_key in pairwise_cache:
                            cached_result = pairwise_cache[cache_key]
                            if cached_result:
                                avaliacoes.append({**cached_result, 'i': i, 'j': j, 'grau1': grau1_idx, 'grau2': grau2_idx})
                            continue # Pula para a próxima iteração, evitando recalcular

                        # --- Se não está no cache, faz o cálculo completo ---
                        step += 1
                        print(f"Progresso (cálculo novo): {step}/{total_steps}", end='\r')
                        
                        self.reset()
                        self.acao(i, 0, 0, grau1_idx)
                        nfp_result, intersec_result = self.nfp(self.lista.index(self.lista_original[j]), grau2_idx)
                        self.remover_ultima_acao()

                        if not nfp_result or nfp_result.is_empty:
                            pairwise_cache[cache_key] = None
                            continue

                        pontos_candidatos = []
                        dominant_point = self.find_dominant_point(nfp_result)
                        if dominant_point: pontos_candidatos.append(dominant_point)
                        if intersec_result and not intersec_result.is_empty:
                            pontos_candidatos.extend(extrair_vertices(intersec_result))
                        
                        if not pontos_candidatos:
                            pairwise_cache[cache_key] = None
                            continue
                        
                        pontos_candidatos = list(dict.fromkeys(pontos_candidatos))
                        
                        best_value_for_key = -1
                        best_config_for_key = None

                        for ponto_encaixe in pontos_candidatos:
                            poly1 = Polygon(self.rot_pol(i, grau1_idx))
                            poly2_coords = self.rot_pol(j, grau2_idx)
                            poly2 = Polygon([(p[0] + ponto_encaixe[0], p[1] + ponto_encaixe[1]) for p in poly2_coords])
                            
                            ch_poly1 = poly1.convex_hull
                            ch_poly2 = poly2.convex_hull
                            ch_uniao = unary_union([poly1, poly2]).convex_hull
                            
                            # --- VERSÃO 1: Cr1 Granular (Correta segundo a Figura 7 do artigo) ---
                            # chv1 = ch_poly1.difference(poly1)
                            # chv2 = ch_poly2.difference(poly2)
                            # score_p2_em_p1_list = []
                            # if not chv1.is_empty:
                            #     chv1_polys = list(chv1.geoms) if hasattr(chv1, 'geoms') else [chv1]
                            #     for vacancy in chv1_polys:
                            #         if vacancy.area > 1e-9:
                            #             score_p2_em_p1_list.append(poly2.intersection(vacancy).area / vacancy.area)
                            # score_p2_em_p1 = max(score_p2_em_p1_list) if score_p2_em_p1_list else 0
                            # score_p1_em_p2_list = []
                            # if not chv2.is_empty:
                            #     chv2_polys = list(chv2.geoms) if hasattr(chv2, 'geoms') else [chv2]
                            #     for vacancy in chv2_polys:
                            #         if vacancy.area > 1e-9:
                            #             score_p1_em_p2_list.append(poly1.intersection(vacancy).area / vacancy.area)
                            # score_p1_em_p2 = max(score_p1_em_p2_list)  if score_p1_em_p2_list else 0
                            # cr1 = max(score_p1_em_p2, score_p2_em_p1)

                            # --- VERSÃO 2: Cr1 Alternativa (Sua proposta para testes) ---
                            ratio1_alt = ch_poly1.intersection(poly2).area / ch_poly1.area if ch_poly1.area > 1e-9 else 0
                            ratio2_alt = ch_poly2.intersection(poly1).area / ch_poly2.area if ch_poly2.area > 1e-9 else 0
                            cr1 = max(ratio1_alt, ratio2_alt) # Descomente esta linha para usar a versão 2
                            
                            cr2 = (poly1.area + poly2.area) / ch_uniao.area if ch_uniao.area > 1e-9 else 0
                            cluster_value = cr1 * cr2
                            
                            if cluster_value > best_value_for_key:
                                best_value_for_key = cluster_value
                                best_config_for_key = {
                                    'value': cluster_value, 'ponto_encaixe': ponto_encaixe,
                                    'cr1': cr1, 'cr2': cr2
                                }
                        
                        pairwise_cache[cache_key] = best_config_for_key
                        if best_config_for_key:
                            avaliacoes.append({**best_config_for_key, 'i': i, 'j': j, 'grau1': grau1_idx, 'grau2': grau2_idx})

        print("\nAnálise concluída. Selecionando e salvando imagens dos melhores resultados...")

    # print("\nAnálise concluída. Selecionando pares únicos para plotagem...")

        # --- Ordena e Salva as Imagens dos Melhores Resultados SEM REPETIR PEÇAS ---
        if not avaliacoes:
            print("Nenhuma configuração válida foi encontrada.")
            return

        avaliacoes.sort(key=lambda x: x['value'], reverse=True)

        # MUDANÇA 2: Calculamos o número de pares desejado com base na porcentagem
        num_total_pecas = len(self.lista_original)
        num_pares_desejado = int((num_total_pecas * porcentagem_cluster) / 2)

        pares_finais_para_plotar = []
        pecas_ja_agrupadas = set()

        for aval in avaliacoes:
            # MUDANÇA 3: O critério de parada agora usa o número de pares calculado
            if len(pares_finais_para_plotar) >= num_pares_desejado:
                break

            peca1_idx = aval['i']
            peca2_idx = aval['j']

            if peca1_idx not in pecas_ja_agrupadas and peca2_idx not in pecas_ja_agrupadas:
                pares_finais_para_plotar.append(aval)
                pecas_ja_agrupadas.add(peca1_idx)
                pecas_ja_agrupadas.add(peca2_idx)

        # MUDANÇA 4: O print final reflete a nova lógica
        print(f"\n--- Salvando imagens para {len(pares_finais_para_plotar)}/{num_pares_desejado} pares encontrados ({porcentagem_cluster:.0%}) ---")
        

        # print(f"\n--- Salvando as imagens dos Top {len(pares_finais)} Melhores Encaixes ---")
        save_path = f"C:\\Users\\felip\\OneDrive\\Documentos\\GitHub\\RKO\\Python\\pairwise_results_20\\{self.dataset}"
        for rank, aval in enumerate(pares_finais_para_plotar):
            self.reset()
            poly1 = Polygon(self.rot_pol(aval['i'], aval['grau1']))
            ponto = aval['ponto_encaixe']
            poly2_coords = self.rot_pol(aval['j'], aval['grau2'])
            poly2 = Polygon([(p[0] + ponto[0], p[1] + ponto[1]) for p in poly2_coords])
            
            self.acao(aval['i'], 0, 0, aval['grau1'])
            nfp_plot, _ = self.nfp(self.lista.index(self.lista_original[aval['j']]), aval['grau2'])
            self.remover_ultima_acao()

            titulo = (f"Rank #{rank + 1} | CV {aval['value']:.3f}\n"
                    f"P ({aval['i']},{aval['j']}) G ({aval['grau1']},{aval['grau2']})\n"
                    f"Cr1 {aval['cr1']:.3f}, Cr2 {aval['cr2']:.3f}")
            
            self.plot_pairwise_geometries(poly1, poly2, nfp_plot, titulo, filepath=save_path) 
   
        return pares_finais_para_plotar


    def criar_lista_clusterizada(self, pares_selecionados):
        """
        Recebe os pares selecionados pela função 'pairwise' e retorna uma nova
        lista de polígonos, com as peças dos pares substituídas por suas
        respectivas "meta-peças".

        Retorna:
            list: A nova lista de coordenadas de polígonos.
        """
        if not pares_selecionados:
            print("INFO: Nenhum par selecionado. Retornando a lista original.")
            return self.lista_original

        print(f"INFO: Criando nova lista de peças com {len(pares_selecionados)} clusters...")

        pecas_agrupadas_indices = set()
        nova_lista_de_pecas = []

        # --- Passo 1: Criar e adicionar as meta-peças à nova lista ---
        for par in pares_selecionados:
            i, j = par['i'], par['j']
            g1, g2 = par['grau1'], par['grau2']
            ponto_encaixe = par['ponto_encaixe']
            
            pecas_agrupadas_indices.add(i)
            pecas_agrupadas_indices.add(j)

            poly1 = Polygon(self.rot_pol(i, g1))
            poly2_coords = self.rot_pol(j, g2)
            poly2_translated = Polygon([(p[0] + ponto_encaixe[0], p[1] + ponto_encaixe[1]) for p in poly2_coords])
            
            # --- Escolha como a meta-peça será criada ---

            # Opção A (Recomendado e Robusto): Usa unary_union e garante um polígono simples
            uniao = unary_union([poly1, poly2_translated]).buffer(0)
            if hasattr(uniao, 'geoms'):  # Se for um MultiPolygon
                uniao = max(uniao.geoms, key=lambda p: p.area) # Pega apenas o maior componente
            # Pega apenas o contorno externo, ignorando buracos, o que simplifica o polígono
            meta_peca = [(round(x,2), round(y,2)) for x,y in uniao.exterior.coords]
            # print(meta_peca)
            meta_peca_poly = []
            for cor in meta_peca:
                if cor not in meta_peca_poly:
                    meta_peca_poly.append(cor)

            ring = LinearRing(meta_peca_poly)
            if ring.is_ccw:
                coords_ccw = list(ring.coords)
            else:
                # Se a ordem for horária (CW), simplesmente invertemos a lista
                coords_ccw = list(ring.coords)[::-1]

            # 4. Remove o último vértice se ele for uma repetição do primeiro
            #    A biblioteca de decomposição geralmente espera uma lista sem o ponto final repetido.
            if coords_ccw and coords_ccw[0] == coords_ccw[-1]:
                coords_finais_para_lib = coords_ccw[:-1]
            else:
                coords_finais_para_lib = coords_ccw

            meta_peca_poly = coords_finais_para_lib
            # meta_peca_poly 
            print(meta_peca_poly)

            # # Opção B (Experimental): "Costurando" os vértices a partir do ponto de encaixe
            # # AVISO: Risco de gerar polígonos inválidos (auto-interseção). Use com cautela.
            # p1_coords = list(poly1.exterior.coords)
            # p2_coords_translated = list(poly2_translated.exterior.coords)
            # ponto_em_p1 = min(p1_coords, key=lambda p: Point(p).distance(Point(ponto_encaixe)))
            # idx_insercao = p1_coords.index(ponto_em_p1)
            # nova_lista_coords = p1_coords[:idx_insercao + 1] + p2_coords_translated[:-1] + p1_coords[idx_insercao + 1:]
            # meta_peca_poly = Polygon(nova_lista_coords)
            
            nova_lista_de_pecas.append(meta_peca_poly)

        # --- Passo 2: Adicionar as peças que não foram agrupadas ---
        for i, peca_coords in enumerate(self.lista_original):
            if i not in pecas_agrupadas_indices:
                nova_lista_de_pecas.append(peca_coords)
                
        print(f"INFO: Nova lista de peças criada com {len(nova_lista_de_pecas)} itens.")
        
        return nova_lista_de_pecas  
    def acao(self,peca,x,y,grau_idx):
        peca_posicionar = self.rot_pol(peca, grau_idx)

  
        pontos_posicionar = [(x + cor[0], y + cor[1]) for cor in peca_posicionar]
        
        self.pecas_posicionadas.append(pontos_posicionar)
        
        self.indices_pecas_posicionadas.append([x,y,grau_idx,self.lista_original.index(self.lista[peca])])
        
        self.lista_anterior.append(copy.deepcopy(self.lista))
        self.lista.pop(peca)
        

        
    def reset(self):
        self.lista = copy.deepcopy(self.lista_original)
        self.pecas_posicionadas = []
        self.indices_pecas_posicionadas = []
        
    def remover_ultima_acao(self):
        if self.pecas_posicionadas:
            self.lista = copy.deepcopy(self.lista_anterior[-1])
            self.lista_anterior.pop()
            self.pecas_posicionadas.pop()
            self.indices_pecas_posicionadas.pop()
    
    def rot_pol(self,pol, grau_indice):
        

        pontos = self.lista[pol]
        # print(pontos)


        px, py = pontos[0]  # pivô
        resultado = []

        for x, y in pontos:
            dx, dy = x - px, y - py
            if grau_indice == 0:      # 0° → sem mudança
                nx, ny = dx, dy
            elif grau_indice == 1:    # +90° → (x, y) → (-y, x)
                nx, ny = -dy, dx
            elif grau_indice == 2:    # 180° → (x, y) → (-x, -y)
                nx, ny = -dx, -dy
            elif grau_indice == 3:    # 270° → (x, y) → (y, -x)
                nx, ny = dy, -dx

            resultado.append([px + nx, py + ny])
        min_x = min(p[0] for p in resultado)
        min_y = min(p[1] for p in resultado)

        if min_x < 0 or min_y < 0:
            resultado = [(x - min_x if min_x < 0 else x,
                        y - min_y if min_y < 0 else y) for x, y in resultado]
        # print(resultado)
        
        return resultado
 
    
    def ifp(self, peca_idx,grau_indice):
        peca = self.rot_pol(peca_idx,grau_indice)
        # print(peca)
        maxx = max([x for x,y in peca])
        maxy = max([y for x,y in peca])

        minx = min([x for x,y in peca])
        miny = min([y for x,y in peca])
    
        if (maxx - minx) > (self.base) or (maxy - miny) > (self.altura):
            return []
        
        cords = self.cordenadas_area
        # print(cords)

        v0 = (0 - minx, 0 - miny)

        # Vértice inferior direito (base, 0)
        v1 = (self.base - maxx, 0 - miny)

        # Vértice superior direito (base, altura)
        v2 = (self.base - maxx, self.altura - maxy)

        # Vértice superior esquerdo (0, altura)
        v3 = (0 - minx, self.altura - maxy)
        
        ifp = [v0,v1,v2,v3]
 
        # print('\n')
        return ifp
    
    def nfp(self, peca, grau_indice):
        nfps = []
        todos_pontos_de_encontro = []

        # cria chave de prefixo (estado atual das peças posicionadas + peça nova)
        chaves = []
        for x2, y2, grau1, pol_idx in self.indices_pecas_posicionadas:
            chave = (
                tuple(self.lista_original[pol_idx]), grau1,
                tuple(self.lista[peca]), grau_indice
            )
            chaves.append((chave, x2, y2))
        prefixo_t = tuple(chaves)

        # se já calculou esse prefixo antes, retorna direto do cache
        if prefixo_t in self.dict_nfps:
            return self.dict_nfps[prefixo_t]

        # caso contrário, calcula normalmente
        for (chave, x2, y2) in chaves:
            nfp_salvo = self.tabela_nfps.get(chave)
            if not nfp_salvo:
                continue

            coords_base_nfp = nfp_salvo[0]
            pontos_intersec_base = nfp_salvo[1]

            base_nfp = Polygon(coords_base_nfp)
            if base_nfp.is_empty:
                continue

            # translada o NFP
            p = affinity.translate(base_nfp, xoff=x2, yoff=y2)
            nfps.append(p)

            # translada pontos de interseção
            if pontos_intersec_base:
                pontos_transladados = [(pt[0] + x2, pt[1] + y2) for pt in pontos_intersec_base]
                todos_pontos_de_encontro.append((pontos_transladados, p))

        if not nfps:
            return None, None

        # ocupado = unary_union(nfps)
        ocupado = unary_union([nfp.buffer(-0.000001) for nfp in nfps])
        # print(ocupado)
        # self.prepared_nfps = [prep(nfp) for nfp in nfps]
        # self.boundaries = [nfp.boundary for nfp in nfps]
        

        # pontos_validos = []
        # print((pontos_validos))

        pontos_validos = []
        if todos_pontos_de_encontro:
            for pontos, nfp_origem in todos_pontos_de_encontro:
                for ponto in pontos:
                    valido = True
                    pt = Point(ponto)
                    for nfp in nfps:
                        if nfp == nfp_origem:
                            continue
                        if nfp.contains(pt):
                            valido = False
                            break
                    if valido:
                        pontos_validos.append(ponto)




        intersec_final = MultiPoint(pontos_validos ) if pontos_validos else None

        # salva no cache antes de retornar
        self.dict_nfps[prefixo_t] = (ocupado, intersec_final)

        return ocupado, intersec_final    
    
    def feasible(self, peca, grau_indice, area=False):
        # chave = tuple([peca, grau_indice, tuple(map(tuple, self.pecas_posicionadas)), self.base, self.altura])

        # # 1. Verifica cache do feasible
        # if chave in self.dict_feasible:
        #     cached_result = self.dict_feasible[chave]
        #     if area:
        #         return cached_result['vertices'], cached_result['area']
        #     return cached_result['vertices']

        # 2. Calcula o Inner-Fit Polygon (área permitida dentro do bin)
        ifp_coords = self.ifp(peca, grau_indice)
        if not ifp_coords:
           
            return ([], 0) if area else []

        ifp_polygon = Polygon(ifp_coords)

        # 3. Caso não haja peças já posicionadas → retorno simples
        if not self.pecas_posicionadas:
            vertices = list(ifp_polygon.exterior.coords)
            # self.dict_feasible[chave] = {'vertices': vertices, 'area': ifp_polygon.area}
            if area:
                return vertices, ifp_polygon.area
            
            return vertices

        # 4. Calcula o No-Fit Polygon combinado (já com cache agressivo no nfp)
        nfp_polygon, nfp_intersec = self.nfp(peca, grau_indice)

        # 5. Interseção entre bordas IFP × NFP
        intersec = ifp_polygon.boundary.intersection(nfp_polygon.boundary) if nfp_polygon else None
        pts = []
        if intersec and not intersec.is_empty:
            if intersec.geom_type == 'Point':
                pts = [(intersec.x, intersec.y)]
            else:
                for part in getattr(intersec, 'geoms', [intersec]):
                    if hasattr(part, 'coords'):
                        pts.extend(list(part.coords))

        # 6. Diferença (área factível)
        if nfp_polygon and not nfp_polygon.is_empty:
            encaixes = ifp_polygon.difference(nfp_polygon)
        else:
            encaixes = ifp_polygon

        # 7. Extrai vértices da área
        vertices = extrair_vertices(encaixes)
        vertices.extend(pts)

        # 8. Considera interseções extras vindas do NFP
        # if nfp_intersec:
            
        #     for ponto in extrair_vertices(nfp_intersec):
        #         vertices.append(ponto)
                
        if nfp_intersec:
            intersecao = extrair_vertices(nfp_intersec.intersection(ifp_polygon))
            for ponto in intersecao:
                if ponto not in vertices:
                    vertices.append(ponto)

        # 9. Salva no cache
        encaixes_area = encaixes.area if encaixes else 0
        # self.dict_feasible[chave] = {'vertices': vertices, 'area': encaixes_area}

        # 10. Retorna
        if area:
            return vertices, encaixes_area
        else:
            
            return vertices
 
    def BL(self, peca, grau_indice):
        positions = self.feasible(peca,grau_indice)
        if not positions:
            return []      
        positions_bl = sorted(positions, key=lambda ponto: (ponto[0], ponto[1]))        
        bl = positions_bl[0]        
        return bl
    
    def NBL(self, peca, grau_indice):
        positions = self.feasible(peca,grau_indice)
        if not positions:
            return []      
        positions_bl = sorted(positions, key=lambda ponto: (ponto[0]**2 + ponto[1]**2))        
        nbl = positions_bl[0] 
              
        return nbl
    
    def NUL(self, peca, grau_indice):
        positions = self.feasible(peca,grau_indice)
        if not positions:
            return []      
        positions_bl = sorted(positions, key=lambda ponto: (ponto[0]**2 + (self.base - ponto[1])**2))        
        nul = positions_bl[0]        
        return nul
        
    def LB(self, peca, grau_indice):
        positions = self.feasible(peca,grau_indice)
        if not positions:
            return []        
        positions_lb = sorted(positions, key=lambda ponto: (ponto[1], ponto[0]))       
        lb = positions_lb[0]        
        return lb
    
        
    def BR(self, peca, grau_indice):
        positions = self.feasible(peca, grau_indice)
        if not positions:
            return [] 
        positions_br = sorted(positions, key=lambda ponto: (-ponto[0], ponto[1]))
        br = positions_br[0]
        return br

   
    def RB(self, peca, grau_indice):
        positions = self.feasible(peca, grau_indice)
        if not positions:
            return [] 
        positions_rb = sorted(positions, key=lambda ponto: (ponto[1], -ponto[0]))
        rb = positions_rb[0]
        return rb

   
    def UL(self, peca, grau_indice):
        positions = self.feasible(peca, grau_indice)
        if not positions:
            return [] 
        positions_ul = sorted(positions, key=lambda ponto: (ponto[0], -ponto[1]))
        ul = positions_ul[0]
        return ul

   
    def LU(self, peca, grau_indice):
        positions = self.feasible(peca, grau_indice)
        if not positions:
            return []         
        positions_lu = sorted(positions, key=lambda ponto: (-ponto[1], ponto[0]))
        lu = positions_lu[0]
        return lu

    
    def UR(self, peca, grau_indice):
        positions = self.feasible(peca, grau_indice)
        if not positions:
            return []         
        positions_ur = sorted(positions, key=lambda ponto: (-ponto[0], -ponto[1]))
        ur = positions_ur[0]
        return ur

   
    def RU(self, peca, grau_indice):
        positions = self.feasible(peca, grau_indice)
        if not positions:
            return []         
        positions_ru = sorted(positions, key=lambda ponto: (-ponto[1], -ponto[0]))
        ru = positions_ru[0]
        return ru
    
    def avaliar_posicoes_onepass(self, pos, peca_idx, grau_indice):
        medias = []
        # print(peca_idx, len(self.lista))
        self.acao(peca_idx, pos[0], pos[1], grau_indice)
        for peca in self.lista:
            for grau in self.graus:
                positions = self.feasible(self.lista.index(peca), grau)
                media = sum([x for x, y in positions]) / len(positions) 
                # print(medias)
                medias.append(media)
                # self.remover_ultima_acao()
        
        if medias == []:
            fit = pos[0]
        else:
            fit = pos[0] * (sum(medias) / len(medias)) 
        self.remover_ultima_acao()
        return fit
                

    def OnePass(self, peca, grau_indice):
        positions = self.feasible(peca, grau_indice)
        best_pos = None
        best_fit = float('inf')
        for pos in positions:
            fit = self.avaliar_posicoes_onepass(pos, peca, grau_indice)
            if fit < best_fit:
                best_fit = fit
                best_pos = pos
            
        return best_pos

    def key_nfp(self, key, peca, grau_indice):
        positions = self.feasible(peca, grau_indice)
        
        if not positions:
            return []
        
        return positions[int(key * len(positions))]
        
    def pack(self, peca, grau_indice, regra_idx, regra = True):
        if regra:
            
            pos = self.regras[regra_idx](peca, grau_indice)
            
            
            if pos:
                # print(pos)
                self.acao(peca, pos[0], pos[1], grau_indice)
                # base = self.base
                # self.base
                # if time.time() - self.start_time > 20:
                #     self.plot(legenda='[]')
                return True
            return False
        
        else:
            pos = self.key_nfp(regra_idx, peca, grau_indice)
            if pos:
                # print(pos)
                self.acao(peca, pos[0], pos[1], grau_indice)
                # self.plot(legenda='[]')
                return True
            return False
        
    def decoder(self, keys):
        
        if self.decoder_type == 'D0':
            # print('1')
            rot = keys[:self.max_pecas]
            pieces = keys[self.max_pecas:]
            
            rot_idx = []
            tipos_rot = len(self.graus)
            for key in rot:
                rot_idx.append(self.graus[int(key * tipos_rot)])
                
            pieces_idx = np.argsort(pieces)
            
            return list(pieces_idx) + rot_idx
            
        elif self.decoder_type == 'D0_A':
            rot = keys[:self.max_pecas]
            regras = keys[self.max_pecas:]
            
            
            rot_idx = []
            tipos_rot = len(self.graus)
            for key in rot:
                rot_idx.append(self.graus[int(key * tipos_rot)])
                
            regras_idx = []
            tipos_regras = 2
            for key in regras:
                regras_idx.append(int(key * tipos_regras))
                
            # print(rot_idx + regras_idx)
            return rot_idx + regras_idx 
        elif self.decoder_type == 'D0_B':
            pieces = keys[:self.max_pecas]            
            rot = keys[self.max_pecas:2*self.max_pecas]
            regras = keys[2*self.max_pecas:]
            
            pieces_idx = np.argsort(pieces)
            
            rot_idx = []
            tipos_rot = len(self.graus)
            for key in rot:
                rot_idx.append(self.graus[int(key * tipos_rot)])
                
            regras_idx = []
            tipos_regras = 4
            for key in regras:
                regras_idx.append(int(key * tipos_regras))
                
            # print(rot_idx + regras_idx)
            return list(pieces_idx) + rot_idx + regras_idx 
            
                
        elif self.decoder_type == 'D1_A':
            rot = keys[:self.max_pecas]
            regras = keys[self.max_pecas:-1]
            shrink_key = keys[-1]
            
            rot_idx = []
            tipos_rot = len(self.graus)
            for key in rot:
                rot_idx.append(self.graus[int(key * tipos_rot)])
                
            regras_idx = []
            tipos_regras = 8
            for key in regras:
                regras_idx.append(int(key * tipos_regras))
                
            # print(rot_idx + regras_idx)
            return rot_idx + regras_idx + [shrink_key]
          
        elif self.decoder_type == 'D1_B':
            pieces = keys[:self.max_pecas]            
            rot = keys[self.max_pecas:2*self.max_pecas]
            nfp_key = keys[2*self.max_pecas:]
            
            pieces_idx = np.argsort(pieces)

            rot_idx = []
            tipos_rot = len(self.graus)
            for key in rot:
                rot_idx.append(self.graus[int(key * tipos_rot)])
                
     
                
            return list(pieces_idx) + rot_idx + list(nfp_key)
       
        elif self.decoder_type == 'D2_A':
            rot = keys[:self.max_pecas]
            regras = keys[self.max_pecas:-1]
            shrink_key = (keys[-1]/5)+0.8
            
            rot_idx = []
            tipos_rot = len(self.graus)
            for key in rot:
                rot_idx.append(self.graus[int(key * tipos_rot)])
                
            regras_idx = []
            tipos_regras = 8
            for key in regras:
                regras_idx.append(int(key * tipos_regras))
                
            # print(rot_idx + regras_idx)
            return rot_idx + regras_idx + [shrink_key]
        elif self.decoder_type == 'D2_B':
                       
            pieces = keys[:self.max_pecas]            
            rot = keys[self.max_pecas:2*self.max_pecas]
            regras = keys[2*self.max_pecas:]
            
            pieces_idx = np.argsort(pieces)

            rot_idx = []
            tipos_rot = len(self.graus)
            for key in rot:
                rot_idx.append(self.graus[int(key * tipos_rot)])
                
            regras_idx = []
            tipos_regras = 8
            for key in regras:
                regras_idx.append(int(key * tipos_regras))
                
            return pieces_idx + rot_idx + regras_idx

            
    
    def cost(self, sol, tag = 0, save  =True):
        if self.decoder_type == 'D0':
            if tuple(sol) in self.dict_sol:
                return self.dict_sol[tuple(sol)]
            else:
                pieces = sol[:self.max_pecas]
                rot = sol[self.max_pecas:]                                
                
                
              
                i = 0
                for idx in pieces:
                    # print(i)
                    self.pack(self.lista.index(self.lista_original[idx]),rot[i], 0)
                    i+=1
                    # self.plot()
                
                
                # if self.lista == []:
                #     self.plot()
                pecas = len(self.pecas_posicionadas) 
                  
                
              
                fit = -1 * self.area_usada()
                # print(self.counter, fit)
                self.dict_sol[tuple(sol)] = fit
                if fit < self.best_fit:
                    self.best_fit = fit
                    # if fit == self.dict_best[self.instance_name]:
                    
                    self.plot(f"{round(self.start_time - time.time(),2)} | {fit} | {len(self.pecas_posicionadas)}/{self.max_pecas}")
                    
                self.reset()
                # self.base = self.base / shrink_factor
                # print(self.base) 
                return fit
        
        
        
        elif self.decoder_type == 'D0_A':
            if tuple(sol) in self.dict_sol:
                return self.dict_sol[tuple(sol)]
            else:
                rot = sol[:self.max_pecas]
                regras = sol[self.max_pecas:]                                
                
                
              
                i = 0
                for peca in self.lista_original:
                    # print(i)
                    self.pack(self.lista.index(peca),rot[i], regras[i])
                    i+=1
                    # self.plot()
                
                
                # if self.lista == []:
                #     self.plot()
                pecas = len(self.pecas_posicionadas) 
                  
                
              
                fit = -1 * self.area_usada()
                # print(self.counter, fit)
                self.dict_sol[tuple(sol)] = fit
                if fit < self.best_fit:
                    self.best_fit = fit
                    # if fit == self.dict_best[self.instance_name]:
                    
                    self.plot(f"{round(self.start_time - time.time(),2)} | {fit} | {len(self.pecas_posicionadas)}/{self.max_pecas}")
                    
                self.reset()
                # self.base = self.base / shrink_factor
                # print(self.base) 
                return fit
            
        elif self.decoder_type == 'D0_B':
            if tuple(sol) in self.dict_sol:
                return self.dict_sol[tuple(sol)]
            else:
                pieces = sol[:self.max_pecas]
                rot = sol[self.max_pecas:2*self.max_pecas]
                regras = sol[self.max_pecas*2:]                              
                
                
              
                i = 0
                for idx in pieces:
                    # print(i)
                    self.pack(self.lista.index(self.lista_original[idx]),rot[i], regras[i])
                    i+=1
                    # self.plot()
                
                
                # if self.lista == []:
                #     self.plot()
                pecas = len(self.pecas_posicionadas) 
                  
                
              
                fit = -1 * self.area_usada()
                # print(self.counter, fit)
                self.dict_sol[tuple(sol)] = fit
                if fit < self.best_fit:
                    self.best_fit = fit
                    # if fit == self.dict_best[self.instance_name]:
                    
                    self.plot(f"{round(self.start_time - time.time(),2)} | {fit} | {len(self.pecas_posicionadas)}/{self.max_pecas}")
                    
                self.reset()
                # self.base = self.base / shrink_factor
                # print(self.base) 
                return fit
      
        
        elif self.decoder_type == 'D1_A':
            if tuple(sol) in self.dict_sol:
                return self.dict_sol[tuple(sol)]
            else:
                rot = sol[:self.max_pecas]
                regras = sol[self.max_pecas:-1]
                shrink_factor = sol[-1]
                
                base_antigo = self.base
                if self.inicial == False and self.base == self.base_inicial:
                    pass
                else:
                    self.base = ((self.base - 0.99*self.base) * shrink_factor) + 0.99*self.base
                # print(base_antigo, self.base, shrink_factor)
                
                i = 0
                nao_posicionadas = []
                for peca in self.lista_original:
                    # print(i)
                    packed = self.pack(self.lista.index(peca),rot[i], regras[i])
                    if not packed:
                        nao_posicionadas.append((peca,i))
                        
                    i+=1
                    # self.plot()
                
                
                # if self.lista == []:
                #     self.plot()
                pecas = len(self.pecas_posicionadas)
                
                
                if len(self.pecas_posicionadas) == self.max_pecas: 
                    self.inicial = True   
                    fit = -1 * self.area_usada()
                    # print(self.counter, fit)
                    self.dict_sol[tuple(sol)] = fit
                    # if save:
                    #     if fit == self.dict_best[self.instance_name]:
                    #         self.plot(f"{round(self.start_time - time.time(),2)} | {fit} | {len(self.pecas_posicionadas)}/{self.max_pecas}")
                    # if round(fit,1) < round(self.best_fit,1) and fit < 0.95 * self.dict_best[self.instance_name]:
                    #     self.best_fit = fit
                    #     # if fit == self.dict_best[self.instance_name]:
                        
                    #     self.plot(f"{round(self.start_time - time.time(),2)} | {fit} | {len(self.pecas_posicionadas)}/{self.max_pecas}")
                    self.reset()
                    
                    return fit
                else:
                    d = random.randint(0,10000)
                   
                    # print(d,self.base, base_antigo)
                    self.base = base_antigo 
                    # print(d,self.base, base_antigo)
                    
                    
                    for peca,idx in nao_posicionadas:
     
                        self.pack(self.lista.index(peca),rot[idx], regras[idx])
              
                        
                    fit = -1 * self.area_usada()                    
                    
                   
                    # print(d,self.base, base_antigo)
                        
                    

                        
                    if len(self.pecas_posicionadas) != self.max_pecas:
                        # print(len(self.pecas_posicionadas), self.max_pecas, fit)
                        fit = sum([Polygon(pol).area for pol in self.lista]) * 100 / (self.base * self.altura)
                        self.reset()
                        self.dict_sol[tuple(sol)] = fit
                        return fit
                    
                    # if round(fit,1) < round(self.best_fit,1) and fit < 0.95 * self.dict_best[self.instance_name]:
                    #     self.best_fit = fit
                    #     # if fit == self.dict_best[self.instance_name]:
                        
                    #     self.plot(f"{round(self.start_time - time.time(),2)} | {fit} | {len(self.pecas_posicionadas)}/{self.max_pecas}")
                    
                    if len(self.pecas_posicionadas) != self.max_pecas:
                        print("EROOROROROROR")
                    self.inicial = True
                    self.reset()
                    self.dict_sol[tuple(sol)] = fit
                    return fit
        elif self.decoder_type == 'D1_B':
            if tuple(sol) in self.dict_sol:
                return self.dict_sol[tuple(sol)]
            else:
                pieces = sol[:self.max_pecas]
                rot = sol[self.max_pecas:2*self.max_pecas]
                nfp_key = sol[self.max_pecas*2:]
              
                
                i = 0
            for idx in pieces:
                # print(i)
                self.pack(self.lista.index(self.lista_original[idx]),rot[i], nfp_key[i], regra=False)
                i+=1
                    # self.plot()
                
            fit = -1 * self.area_usada()
            # if self.lista == []:
            #     self.plot()
            pecas = len(self.pecas_posicionadas)    
                
            # print(self.counter, fit)
            self.dict_sol[tuple(sol)] = fit
            if save:
                if fit == self.dict_best[self.instance_name]:
                    self.plot(f"{round(self.start_time - time.time(),2)} | {fit} | {len(self.pecas_posicionadas)}/{self.max_pecas}")
                    
                    
            self.reset()
            return fit
        
        elif self.decoder_type == 'D2_A':
            if tuple(sol) in self.dict_sol:
                return self.dict_sol[tuple(sol)]
            else:
                rot = sol[:self.max_pecas]
                regras = sol[self.max_pecas:-1]                                
                shrink_factor = sol[-1]
                
                self.base = self.base * shrink_factor
                i = 0
                for peca in self.lista_original:
                    # print(i)
                    self.pack(self.lista.index(peca),rot[i], regras[i])
                    i+=1
                    # self.plot()
                
                
                # if self.lista == []:
                #     self.plot()
                pecas = len(self.pecas_posicionadas) 
                  
                # print(self.base, shrink_factor) 
                if len(self.pecas_posicionadas) == self.max_pecas:    
                    fit = -1 * self.area_usada()
                    # print(self.counter, fit)
                    self.dict_sol[tuple(sol)] = fit
                    if save:
                        if fit == self.dict_best[self.instance_name]:
                            self.plot(f"{round(self.start_time - time.time(),2)} | {fit} | {len(self.pecas_posicionadas)}/{self.max_pecas}")
                    self.reset()
                    self.base = self.base / shrink_factor
                    
                    return fit
                else:
                    
                    # fit = sum([Polygon(pol).area for pol in self.lista]) * 100 / (self.base * self.altura)
                    fit = -1 * self.area_usada()
                    self.reset()
                    self.base = self.base / shrink_factor
                    return fit
        
        elif self.decoder_type == 'D2_B':
            if tuple(sol) in self.dict_sol:
                return self.dict_sol[tuple(sol)]
            else:
                pieces = sol[:self.max_pecas]
                rot = sol[self.max_pecas:2*self.max_pecas]
                regras = sol[self.max_pecas*2:]
                
                i = 0
            for idx in pieces:
                # print(i)
                self.pack(self.lista.index(self.lista_original[idx]),rot[i], regras[i])
                i+=1
                    # self.plot()
                
            fit = -1 * self.area_usada()
            # if self.lista == []:
            #     self.plot()
            pecas = len(self.pecas_posicionadas)    
                
            # print(self.counter, fit)
            self.dict_sol[tuple(sol)] = fit
            if save:
                if fit == self.dict_best[self.instance_name]:
                    self.plot(f"{round(self.start_time - time.time(),2)} | {fit} | {len(self.pecas_posicionadas)}/{self.max_pecas}")
            self.reset()
            return fit

                    
            
                
        
            
    def plot(self, legenda):
        draw_cutting_area(self.pecas_posicionadas, self.base, self.altura ,legenda=legenda, filename=f'C:\\Users\\felip\\Documents\\GitHub\\RKO\\Python\\Images\\SPP\\{self.instance_name}\\{self.instance_name}_{time.time()}.png')
    
    def get_used_width(self):
        original_counter = Counter(tuple(map(tuple, pol)) for pol in self.lista_original)
        nao_usado_counter = Counter(tuple(map(tuple, pol)) for pol in self.lista)

        # Subtrai pra obter os usados
        usados_counter = original_counter - nao_usado_counter

        # Reconstrói a lista de polígonos usados
        usados = []
        for pol, count in usados_counter.items():
            usados.extend([list(pol) for _ in range(count)])

        # Soma as áreas
        area_total = sum(Polygon(pol).area for pol in usados)

        # Área total do bin
        coords = []
        for pol in self.pecas_posicionadas:
            for x,y in pol:
                coords.append(x)
        if coords == []:
            return 0
        larg = max(coords) - min(coords)
        area_bin = (larg / self.escala) * (self.altura / self.escala)
        # area_bin = (self.base / self.escala) * (self.altura / self.escala)

        return larg 
    
    def get_efficiency(self):
        original_counter = Counter(tuple(map(tuple, pol)) for pol in self.lista_original)
        nao_usado_counter = Counter(tuple(map(tuple, pol)) for pol in self.lista)

        # Subtrai pra obter os usados
        usados_counter = original_counter - nao_usado_counter

        # Reconstrói a lista de polígonos usados
        usados = []
        for pol, count in usados_counter.items():
            usados.extend([list(pol) for _ in range(count)])

        # Soma as áreas
        area_total = sum(Polygon(pol).area for pol in usados)

        # Área total do bin
        coords = []
        for pol in self.pecas_posicionadas:
            for x,y in pol:
                coords.append(x)
        if coords == []:
            return 0
        larg = max(coords) - min(coords)
        area_bin = (larg / self.escala) * (self.altura / self.escala)
        # area_bin = (self.base / self.escala) * (self.altura / self.escala)

        return round((area_total / area_bin) * 100, 2) 
        
    def area_usada(self):
        original_counter = Counter(tuple(map(tuple, pol)) for pol in self.lista_original)
        nao_usado_counter = Counter(tuple(map(tuple, pol)) for pol in self.lista)

        # Subtrai pra obter os usados
        usados_counter = original_counter - nao_usado_counter

        # Reconstrói a lista de polígonos usados
        usados = []
        for pol, count in usados_counter.items():
            usados.extend([list(pol) for _ in range(count)])

        # Soma as áreas
        area_total = sum(Polygon(pol).area for pol in usados)

        # Área total do bin
        coords = []
        for pol in self.pecas_posicionadas:
            for x,y in pol:
                coords.append(x)
        
        larg = max(coords) - min(coords)
        area_bin = (larg / self.escala) * (self.altura / self.escala)
        # area_bin = (self.base / self.escala) * (self.altura / self.escala)

        return round((area_total / area_bin) * 100, 2) 

# while True:
#     env = Knapsack2D(dataset='shapes1')
    
#     keys = np.random.random(2 * env.max_pecas)
#     sol = env.decoder(keys)
#     env.cost(sol)
#     # print(env.pecas_posicionadas)
#     print(env.area_usada())   
        
    
    
#     env.plot()

if __name__ == '__main__':
    instancias = ["shapes2","swim","jackobs2"]    
    # instancias = [ "shapes2","swim","fu","jackobs1","trousers", "jackobs2","shapes0","shapes1","shapes2","albano","shirts","dighe1","dighe2","dagli","mao","marques","swim"] 
    # decoders = ['D0','D0_A','D2_A','D0_B','D1_A','D1_B',  'D2_B']
    decoders = ['D1_A']
    # for ins in instancias:
    # env = SPP2D(dataset=instancias[0], tempo=10, decoder='D0')
    for fd in range(10):
        for tempo in [2400]:    
            for restart in [1/6]:                
                for ins in instancias:
                    for decoder in decoders:
                        list_time = []
                        list_cost = []
                        
                        env = SPP2D(dataset=ins, tempo=tempo * restart, decoder=decoder, pairwise=True)
                        # i = 0
                        # start = time.time()
                        # while time.time() - start < 10:
                        #     keys = np.random.random(env.tam_solution)
                        #     sol = env.decoder(keys)
                        #     print(env.cost(sol, save=False))
                            
                        #     i += 1
                            
                        # print(i)

                        print(len(env.lista), sum(Polygon(pol).area for pol in env.lista)/env.area)
                        solver = RKO(env, print_best=True, save_directory=f'c:\\Users\\felip\\Documents\\GitHub\\RKO\\Python\\testes_SPP\\{decoder}_SPP_{tempo}_{restart}\\testes_RKO.csv')
                        cost,sol, temp = solver.solve(tempo,brkga=1,ms=0,sa=3,vns=1,ils=0, lns=0, pso=1, ga=0, restart= restart,  runs=1)

