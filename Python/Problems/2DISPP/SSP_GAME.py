from SPP2D import SPP2D

import pygame
import sys
import random

# --- Classe do Visualizador ---


import pygame
import sys
import copy

import pygame
import sys
import copy

import pygame
import sys
import copy
from collections import OrderedDict

import math

# --- Classe auxiliar para Botões (sem alterações) ---
class Button:
    def __init__(self, rect, text, bg_color=(200,200,200), text_color=(0,0,0)):
        self.rect, self.text, self.bg_color, self.text_color = pygame.Rect(rect), text, bg_color, text_color
        self.font = pygame.font.SysFont('Arial', 18, bold=True)
    def draw(self, screen):
        pygame.draw.rect(screen, self.bg_color, self.rect, border_radius=5); pygame.draw.rect(screen, (50,50,50), self.rect, width=2, border_radius=5)
        text_surf = self.font.render(self.text, True, self.text_color); screen.blit(text_surf, text_surf.get_rect(center=self.rect.center))
    def is_clicked(self, pos): return self.rect.collidepoint(pos)

# --- Classe Principal do Visualizador ---
class Visualizer:
    def __init__(self, spp_env, screen_width=1280, screen_height=800):
        self.spp_env = spp_env
        self.width, self.height = screen_width, screen_height
        self.padding, self.sidebar_width, self.panel_height = 50, 300, 140
        pygame.init()
        self.screen = pygame.display.set_mode((self.width, self.height))
        pygame.display.set_caption("GEOMETRY-ITOR: Visualizador SPP 2D")
        self.font = pygame.font.SysFont('Arial', 20); self.font_metrics = pygame.font.SysFont('Arial', 22, bold=True); self.font_count = pygame.font.SysFont('Arial', 18, bold=True)
        self.clock = pygame.time.Clock()
        self.BG_COLOR = (240,240,240); self.BIN_COLOR = (20,20,40); self.SIDEBAR_COLOR = (210,210,210)
        self.PIECE_COLORS = [(227,88,88),(98,194,91),(91,157,194),(230,155,79),(170,102,204),(240,228,66)]
        
        # --- ALTERAÇÃO AQUI: Variáveis de estado para o scroll ---
        self.selected_piece_idx_in_list = None
        self.selected_rotation = 0
        self.unique_pieces_view = []
        self.feasible_vertices = []
        self.current_vertex_idx = 0
        self.sidebar_scroll_y = 0
        self.sidebar_content_height = 0

        self._setup_layout_and_buttons()
        self.scale_factor = self._calculate_scale()

    def run(self):
        running = True
        while running:
            mouse_pos = pygame.mouse.get_pos()
            for event in pygame.event.get():
                if event.type == pygame.QUIT: running = False
                if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                    self._handle_clicks(mouse_pos)
                
                # --- ALTERAÇÃO AQUI: Tratamento do evento de Roda do Mouse ---
                if event.type == pygame.MOUSEWHEEL:
                    # Verifica se o mouse está sobre a sidebar
                    if mouse_pos[0] < self.sidebar_width:
                        self.sidebar_scroll_y -= event.y * 20 # Multiplicador de velocidade
                        # Limita o scroll para não sair do conteúdo
                        max_scroll = self.sidebar_content_height - self.height
                        if max_scroll < 0: max_scroll = 0
                        self.sidebar_scroll_y = max(0, min(self.sidebar_scroll_y, max_scroll))

            self._draw()
            pygame.display.flip()
            self.clock.tick(60)
        pygame.quit(); sys.exit()

    def _draw_sidebar(self):
        self._prepare_unique_pieces_view()
        sidebar_area = pygame.Rect(0, 0, self.sidebar_width, self.height)
        pygame.draw.rect(self.screen, self.SIDEBAR_COLOR, sidebar_area)
        
        title_surf = self.font.render("Peças Disponíveis", True, (0,0,0))
        self.screen.blit(title_surf, (10, 10))

        thumb_w, thumb_h, thumb_pad = 120, 90, 10
        start_x, start_y = thumb_pad, 50
        current_x, current_y = start_x, start_y
        
        # --- LÓGICA DE SCROLL ---
        # 1. Calcula a posição de todas as peças para saber a altura total
        content_rects = []
        for i, piece_info in enumerate(self.unique_pieces_view):
            if current_y + thumb_h > self.height and current_x == start_x: # Se precisa de mais de 1 coluna
                current_y = start_y
                current_x += thumb_w + thumb_pad
            
            # Posição relativa ao conteúdo, não à tela
            content_rects.append(pygame.Rect(current_x, current_y, thumb_w, thumb_h))
            current_y += thumb_h + thumb_pad
        
        self.sidebar_content_height = current_y # Altura total do conteúdo
        
        # 2. Desenha apenas as peças visíveis, aplicando o offset do scroll
        for i, piece_info in enumerate(self.unique_pieces_view):
            content_rect = content_rects[i]
            # Posição na tela = posição no conteúdo - offset do scroll
            screen_rect = content_rect.move(0, -self.sidebar_scroll_y)
            
            # Otimização: Só desenha se estiver visível na tela
            if sidebar_area.colliderect(screen_rect):
                peca = piece_info['original_peca']
                min_x,min_y = min(p[0] for p in peca),min(p[1] for p in peca)
                norm_peca = [(p[0]-min_x, p[1]-min_y) for p in peca]
                max_x,max_y = max(p[0] for p in norm_peca),max(p[1] for p in norm_peca)
                scale = min((thumb_w-thumb_pad)/max(1,max_x), (thumb_h-thumb_pad)/max(1,max_y))
                
                thumb_coords = [(screen_rect.left+thumb_pad/2+p[0]*scale, screen_rect.top+thumb_pad/2+p[1]*scale) for p in norm_peca]
                piece_info['rect'] = screen_rect # Atualiza o rect clicável para a posição atual na tela
                
                pygame.draw.rect(self.screen, (225,225,225), screen_rect, border_radius=3)
                pygame.draw.polygon(self.screen, self.PIECE_COLORS[i % len(self.PIECE_COLORS)], thumb_coords)
                
                count_surf = self.font_count.render(f"x{piece_info['count']}", True, (0,0,0))
                self.screen.blit(count_surf, (screen_rect.right - 30, screen_rect.top + 5))
                
                is_selected = piece_info['first_idx'] == self.selected_piece_idx_in_list
                pygame.draw.polygon(self.screen, (255,0,0) if is_selected else (50,50,50), thumb_coords, 3 if is_selected else 1)
        
        # 3. Desenha a barra de rolagem se necessário
        visible_height = self.height
        if self.sidebar_content_height > visible_height:
            scrollbar_track = pygame.Rect(self.sidebar_width - 12, 0, 12, visible_height)
            pygame.draw.rect(self.screen, (200, 200, 200), scrollbar_track)
            
            handle_height = visible_height * (visible_height / self.sidebar_content_height)
            handle_height = max(20, handle_height) # Altura mínima
            
            scroll_ratio = self.sidebar_scroll_y / (self.sidebar_content_height - visible_height)
            handle_y = (visible_height - handle_height) * scroll_ratio
            
            scrollbar_handle = pygame.Rect(self.sidebar_width - 12, handle_y, 12, handle_height)
            pygame.draw.rect(self.screen, (130, 130, 130), scrollbar_handle, border_radius=6)

        pygame.draw.line(self.screen, (150,150,150), (self.sidebar_width, 0), (self.sidebar_width, self.height), 2)
    
    # --- Demais métodos (sem alterações) ---
    def _setup_layout_and_buttons(self):
        p_y,base_x=self.height - self.panel_height + 20,self.sidebar_width
        GRAY_BLUE,GRAY_GREEN,GRAY_RED,GRAY_ORANGE=(140,160,180),(140,180,160),(180,140,140),(180,160,140)
        self.buttons={'BL':Button((base_x+20,p_y+20,80,40),"BL",bg_color=GRAY_BLUE),'LB':Button((base_x+110,p_y+20,80,40),"LB",bg_color=GRAY_GREEN),'UL':Button((base_x+20,p_y+70,80,40),"UL",bg_color=GRAY_RED),'LU':Button((base_x+110,p_y+70,80,40),"LU",bg_color=GRAY_ORANGE),'PLACE_V':Button((base_x+275,p_y-5,180,40),"Posicionar Vértice",bg_color=(170,102,204)),'PREV_V':Button((base_x+295,p_y+70,50,40),"<"),'NEXT_V':Button((base_x+355,p_y+70,50,40),">"),'ROT':Button((self.width-210,p_y+15,190,40),"Rotacionar"),'UNDO':Button((self.width-210,p_y+65,90,40),"Undo",bg_color=(240,170,0)),'RESET':Button((self.width-110,p_y+65,90,40),"Reset",bg_color=(220,50,50))}
    def _calculate_scale(self):
        env_w,env_h=self.spp_env.base,self.spp_env.altura;scale_x=(self.width-self.sidebar_width-2*self.padding)/env_w;scale_y=(self.height-self.panel_height-2*self.padding)/env_h;return min(scale_x,scale_y)*0.98
    def _transform_coords(self,poly_pts,offset=(0,0)):return[(int(self.padding+self.sidebar_width+(x+offset[0])*self.scale_factor),int(self.height-self.padding-self.panel_height-((y+offset[1])*self.scale_factor)))for x,y in poly_pts]
    def _update_feasible_region(self):
        if self.selected_piece_idx_in_list is not None:self.feasible_vertices=self.spp_env.feasible(self.selected_piece_idx_in_list,self.selected_rotation);self.current_vertex_idx=0
        else:self.feasible_vertices=[]
    def _handle_clicks(self,mouse_pos):
        if self.buttons['RESET'].is_clicked(mouse_pos):self.spp_env.reset();self._update_feasible_region()
        if self.buttons['UNDO'].is_clicked(mouse_pos):self.spp_env.remover_ultima_acao();self._update_feasible_region()
        if self.buttons['ROT'].is_clicked(mouse_pos)and self.selected_piece_idx_in_list is not None:self.selected_rotation=(self.selected_rotation+1)%4;self._update_feasible_region()
        num_vertices=len(self.feasible_vertices)
        if num_vertices>0:
            if self.buttons['NEXT_V'].is_clicked(mouse_pos):self.current_vertex_idx=(self.current_vertex_idx+1)%num_vertices
            if self.buttons['PREV_V'].is_clicked(mouse_pos):self.current_vertex_idx=(self.current_vertex_idx-1+num_vertices)%num_vertices
            if self.buttons['PLACE_V'].is_clicked(mouse_pos):pos=self.feasible_vertices[self.current_vertex_idx];self.spp_env.acao(self.selected_piece_idx_in_list,pos[0],pos[1],self.selected_rotation);self.selected_piece_idx_in_list=None;self.feasible_vertices=[];return
        for name in['BL','LB','UL','LU']:
            if self.buttons[name].is_clicked(mouse_pos)and self.selected_piece_idx_in_list is not None:
                pos=getattr(self.spp_env,name)(self.selected_piece_idx_in_list,self.selected_rotation)
                if pos:self.spp_env.acao(self.selected_piece_idx_in_list,pos[0],pos[1],self.selected_rotation);self.selected_piece_idx_in_list=None;self.feasible_vertices=[];return
        for piece_info in self.unique_pieces_view:
            if piece_info.get('rect') and piece_info['rect'].collidepoint(mouse_pos):self.selected_piece_idx_in_list=piece_info['first_idx'];self.selected_rotation=0;self._update_feasible_region();return
    def _prepare_unique_pieces_view(self):
        """
        Analisa self.spp_env.lista, agrupa peças idênticas e prepara uma
        estrutura de dados para a renderização na sidebar.
        """
        counts = OrderedDict()
        for i, peca in enumerate(self.spp_env.lista):
            # Adicionado para robustez: ignora listas de peças vazias que podem causar erros
            if not peca:
                continue

            # Esta é a lógica que estava comprimida e causando o erro
            min_x = min(p[0] for p in peca)
            min_y = min(p[1] for p in peca)
            
            # Normaliza a peça para que peças transladadas sejam consideradas iguais
            # A ordenação (sorted) garante que peças com vértices em ordens diferentes
            # mas com a mesma geometria sejam agrupadas.
            norm_peca = tuple(sorted((p[0] - min_x, p[1] - min_y) for p in peca))
            
            if norm_peca not in counts:
                counts[norm_peca] = {'count': 0, 'first_idx': i, 'original_peca': peca}
            
            counts[norm_peca]['count'] += 1
        
        self.unique_pieces_view = list(counts.values())

    def _draw_panel_and_buttons(self):
            panel_rect = pygame.Rect(self.sidebar_width, self.height - self.panel_height, self.width - self.sidebar_width, self.panel_height)
            pygame.draw.rect(self.screen, (220, 220, 220), panel_rect)
            pygame.draw.line(self.screen, (150, 150, 150), (self.sidebar_width, self.height - self.panel_height), (self.width, self.height - self.panel_height), 2)

            # --- CORREÇÃO APLICADA AQUI ---
            # A posição Y do título foi ajustada para um local fixo no painel,
            # resolvendo a sobreposição.
            p_y_base = self.height - self.panel_height + 20 # Y de referência do painel
            
            # Títulos dos painéis
            self.screen.blit(self.font.render("Regras Heurísticas", True, (0, 0, 0)), (self.buttons['BL'].rect.left, p_y_base - 15))
            self.screen.blit(self.font.render("Controle de Vértices", True, (0, 0, 0)), (self.buttons['PLACE_V'].rect.left, self.buttons['PLACE_V'].rect.bottom + 5))

            for button in self.buttons.values():
                button.draw(self.screen)

            # Contador de vértices
            num_v = len(self.feasible_vertices)
            v_text = f"{self.current_vertex_idx + 1}/{num_v}" if num_v > 0 else "N/A"
            v_surf = self.font.render(v_text, True, (0, 0, 0))
            text_rect = v_surf.get_rect(center=(self.buttons['NEXT_V'].rect.right + 50, self.buttons['NEXT_V'].rect.centery))
            self.screen.blit(v_surf, text_rect)

    def _draw_heuristic_previews(self):
        """Desenha as pré-visualizações para as 4 regras heurísticas."""
        if self.selected_piece_idx_in_list is None:
            return
        
        for name in ['BL', 'LB', 'UL', 'LU']:
            pos = getattr(self.spp_env, name)(self.selected_piece_idx_in_list, self.selected_rotation)
            if pos:
                peca_rot = self.spp_env.rot_pol(self.selected_piece_idx_in_list, self.selected_rotation)
                preview_surface = pygame.Surface((self.width, self.height), pygame.SRCALPHA)
                peca_screen_coords = self._transform_coords(peca_rot, offset=pos)
                # Usa a cor do botão correspondente com transparência
                preview_color = list(self.buttons[name].bg_color) + [128]
                pygame.draw.polygon(preview_surface, preview_color, peca_screen_coords)
                self.screen.blit(preview_surface, (0, 0))

    def _draw_vertex_preview(self):
        """Desenha a pré-visualização para o vértice atualmente selecionado."""
        if self.selected_piece_idx_in_list is not None and self.feasible_vertices:
            pos = self.feasible_vertices[self.current_vertex_idx]
            peca_rot = self.spp_env.rot_pol(self.selected_piece_idx_in_list, self.selected_rotation)
            
            preview_surface = pygame.Surface((self.width, self.height), pygame.SRCALPHA)
            peca_screen_coords = self._transform_coords(peca_rot, offset=pos)
            pygame.draw.polygon(preview_surface, (170, 102, 204, 180), peca_screen_coords) # Cor roxa
            self.screen.blit(preview_surface, (0, 0))

            pos_x, pos_y = self._transform_coords([pos])[0]
            pygame.draw.circle(self.screen, (200, 0, 0), (pos_x, pos_y), 5) # Círculo vermelho no vértice

    def _draw(self):
        self.screen.fill(self.BG_COLOR)
        
        # Desenha o Bin e as peças já posicionadas
        bin_rect = [(0, 0), (self.spp_env.base, 0), (self.spp_env.base, self.spp_env.altura), (0, self.spp_env.altura)]
        pygame.draw.polygon(self.screen, self.BIN_COLOR, self._transform_coords(bin_rect), 2)
        for i, peca in enumerate(self.spp_env.pecas_posicionadas):
            pygame.draw.polygon(self.screen, self.PIECE_COLORS[i % len(self.PIECE_COLORS)], self._transform_coords(peca))
            pygame.draw.polygon(self.screen, (50, 50, 50), self._transform_coords(peca), 1)

        # --- CORREÇÃO APLICADA AQUI ---
        # A lógica "if/else" foi removida. Agora ambas as pré-visualizações são chamadas.
        self._draw_heuristic_previews()
        self._draw_vertex_preview()

        # UI (sidebar, painel, métricas) desenhada por cima de tudo
        self._draw_sidebar()
        self._draw_panel_and_buttons()
        
        used_width, eff = self.spp_env.get_used_width(), self.spp_env.get_efficiency()
        self.screen.blit(self.font_metrics.render(f"Largura: {used_width:.2f}", True, (0, 0, 0)), (self.sidebar_width + self.padding, 10))
        self.screen.blit(self.font_metrics.render(f"Eficiência: {eff:.2f}%", True, (0, 0, 0)), (self.sidebar_width + self.padding, 40))

if __name__ == '__main__':
    

    meu_ambiente_spp = SPP2D(dataset='fu')



    # 4. CRIE A INSTÂNCIA DO VISUALIZADOR, PASSANDO SEU AMBIENTE
    visualizador = Visualizer(spp_env=meu_ambiente_spp, screen_width=1024, screen_height=768)
    
    # 5. EXECUTE O VISUALIZADOR
    visualizador.run()

