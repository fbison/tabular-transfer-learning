import numpy as np
from collections import deque
import math

# --- Funções Auxiliares ---

def exponential_moving_average(sequence, span):
    """
    Calcula a Média Móvel Exponencial (EMA) para uma sequência de valores.
    A EMA suaviza a série temporal, reduzindo o ruído e destacando a tendência subjacente.
    """
    # Fator de suavização (alpha = 2 / (N + 1))
    alpha = 2 / (span + 1.0)
    
    smoothed_series = []
    # 'e' armazena o valor EMA anterior (ou o primeiro valor da sequência)
    ema_value = None
    
    for value in sequence:
        if ema_value is None:
            # Inicializa com o primeiro valor
            ema_value = value
        else:
            # Fórmula EMA: EMA_t = alpha * Valor_t + (1 - alpha) * EMA_{t-1}
            ema_value = alpha * value + (1 - alpha) * ema_value
            
        smoothed_series.append(ema_value)
        
    return np.array(smoothed_series)

def compute_slope_and_stderr(y_values):
    """
    Calcula a inclinação (slope) e o Erro Padrão do Slope (stderr) usando regressão linear.
    Isso é usado para avaliar a significância estatística da tendência na série.
    """
    # x representa os índices de tempo (0, 1, 2, ..., N-1)
    x_values = np.arange(len(y_values))
    
    # Médias para cálculo
    x_mean = x_values.mean()
    y_mean = y_values.mean()
    
    # Soma dos Quadrados de X (Sxx)
    Sxx = ((x_values - x_mean)**2).sum()
    
    if Sxx == 0:
        # Evita divisão por zero (ocorre com uma janela de tamanho 1)
        return 0.0, float('inf')
        
    # Cálculo da Inclinação (Slope, m) da Regressão Linear Simples (y = mx + b)
    # m = Sxy / Sxx
    slope = ((x_values - x_mean) * (y_values - y_mean)).sum() / Sxx
    
    # Cálculo dos resíduos (erros)
    # Previsão da linha de regressão: y_hat = slope * x + (y_mean - slope * x_mean)
    residuals = y_values - (slope * x_values + (y_mean - slope * x_mean))
    
    # Graus de Liberdade (DOF): N - 2, onde N é o tamanho da amostra e 2 são os parâmetros (slope e intercepto)
    degrees_of_freedom = max(1, len(y_values) - 2)
    
    # Erro Quadrático Médio (MSE - Mean Squared Error) do modelo de regressão
    mean_squared_error = (residuals**2).sum() / degrees_of_freedom
    
    # Erro Padrão do Slope (Standard Error - stderr)
    # Mede a incerteza da estimativa do slope
    stderr = math.sqrt(mean_squared_error / Sxx)
    
    return slope, stderr

# --- Detector Principal ---

class PlateauDetector:
    """
    Detecta quando a perda (loss) de um modelo entra em um platô (estagnação) 
    baseado em melhoria relativa e significância estatística da tendência (slope).
    """
    def __init__(self, *,
                 window_size=30,             # Tamanho da janela para calcular o slope e a EMA
                 ema_span=None,              # Período de suavização da EMA
                 patience_steps=7,           # Número de passos de estagnação consecutivos antes de sinalizar o platô
                 min_rel_improvement=1e-3,   # 0.1%. Melhoria relativa mínima exigida para evitar o platô
                 slope_significance_threshold=1.0, # Limiar do "Z-score" (abs(slope) / stderr) para significância
                 require_negative_slope_to_stop=False): # Exigir slope positivo/nulo para contar o passo de paciência
        
        self.window_size = window_size
        # Define o span da EMA, se não for fornecido (default: metade do tamanho da janela, min 3)
        self.ema_span = ema_span if ema_span is not None else max(3, window_size // 2)
        
        self.patience_steps = patience_steps
        self.min_rel_improvement = min_rel_improvement
        self.slope_significance_threshold = slope_significance_threshold
        self.require_negative_slope_to_stop = require_negative_slope_to_stop

        # Estado interno
        self.all_losses = []             # Armazena todas as perdas observadas
        self.stagnation_counter = 0      # Contador de passos consecutivos de estagnação (patience counter)
        self.best_smoothed_loss = float('inf') # A menor perda suavizada já registrada
        self.best_loss = float('inf')    # A menor perda já registrada

    def step(self, current_loss):
        """
        Processa uma nova perda e verifica se o platô foi atingido.
        Retorna True se o platô for detectado, False caso contrário.
        """
        self.all_losses.append(float(current_loss))
        
        # O detector só começa a operar após preencher a janela mínima
        if len(self.all_losses) < self.window_size:
            return False  # Ainda não há dados suficientes para análise
        
        # 1. Suavização da série de perdas
        smoothed_series = exponential_moving_average(self.all_losses, self.ema_span)
        # Pega apenas a janela de interesse da série suavizada
        recent_smoothed_window = smoothed_series[-self.window_size:]
        current_smoothed_loss = recent_smoothed_window[-1]

        # 2. Cálculo da Melhoria Relativa
        # Quanto a perda atual está 'atrás' da melhor já vista (quanto maior o valor, pior/mais distante)
        # Nota: (best - current) é negativo se a perda atual for maior que a best, mas o max(1e-12, ...) garante a segurança da divisão.
        relative_improvement = 0.00
        if self.best_smoothed_loss < float('inf'):    
            relative_improvement = (self.best_smoothed_loss - current_smoothed_loss) / max(1e-12, self.best_smoothed_loss)
        
        # Atualiza a melhor perda suavizada observada
        self.best_smoothed_loss = min(self.best_smoothed_loss, current_smoothed_loss)
        
        # Se atingiu um novo melhor valor histórico, nunca estamos em plateau,
        # porém só valida aqui, para manter o best_smoothed_loss consistente com a lógica
        #if current_loss < self.best_loss:
        #    self.best_loss = current_loss
        #    self.stagnation_counter = 0
        #    return False

        # 3. Análise Estatística da Tendência (Slope)
        # Calcula a inclinação da linha de regressão e a incerteza dessa estimativa
        trend_slope, slope_stderr = compute_slope_and_stderr(recent_smoothed_window)
        
        # Calcula o quão “confiável” é esse slope
        # Se o valor for alto, o slope é estatisticamente diferente de zero
        slope_z_score = abs(trend_slope) / (slope_stderr + 1e-12)

        # --- Regras para Detecção de Platô (Stagnation) ---
        
        # Condição 1: Houve melhoria relativa suficiente desde o melhor ponto anterior?
        # Se falso, a melhora foi muito pequena ou inexistente
        cond_sufficient_improvement = relative_improvement >= self.min_rel_improvement

        # Condição 2: Há uma tendência estatisticamente clara (positiva ou negativa)?
        # Se falso, a tendência é incerta ou quase zero
        cond_slope_significant = slope_z_score >= self.slope_significance_threshold

        # Slope negativo significa que o loss ainda está DESCENDO
        slope_is_negative = trend_slope < 0
        
        # Se existe uma tendência negativa clara, NÃO contar como estagnação
        # mesmo que a melhora seja lenta. Ainda está havendo progresso real.
        cond_clear_improving_trend = slope_is_negative and cond_slope_significant

        # Regra de Estagnação:
        # Contar estagnação somente quando:
        # 1. Não houve melhoria relativa suficiente E
        # 2. Não há uma tendência clara (ou a tendência é de piora)
        if (not cond_sufficient_improvement) and (not cond_clear_improving_trend):
            # Aqui consideramos como um passo de platô
            self.stagnation_counter += 1
        else:
            # Qualquer evidência de progresso limpa o contador
            self.stagnation_counter = 0

        # Interrompe se o contador ultrapassar a paciência configurada
        return self.stagnation_counter >= self.patience_steps
