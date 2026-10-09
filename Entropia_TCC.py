#Bibliotecas
import os
import pandas as pd
import re
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from scipy.interpolate import UnivariateSpline
import smplotlib

"""
Em comparação com a Literatura, 
a adição de Mn para uma estrutura hexagonal\cite{JAYARAMAN20111411}
é prevista que diminua a temperatura de curie sem alterar significativamente a 
magnitude da variação de entropia magnética, a qual a nossa amostra caso sofreu um aumento.
"""

def encontrar_massa_amostra(caminho_arquivo):
    """
    Encontra SAMPLE_MASS no cabeçalho do arquivo VSM/PPMS.
    Trata, por exemplo:
        INFO,28.53,SAMPLE_MASS
    e também:
        INFO,28,53,SAMPLE_MASS
    Assume que SAMPLE_MASS está em mg e retorna a massa em gramas.
    """
    with open(caminho_arquivo, "r", encoding="latin1") as arquivo:
        for linha in arquivo:
            linha = linha.strip()
            if not linha:
                continue

            partes = [parte.strip() for parte in linha.split(",")]

            if (
                partes[0].upper() == "INFO"
                and partes[-1].upper() == "SAMPLE_MASS"
            ):

                valores = partes[1:-1]

                # Caso normal:
                # INFO,28.53,SAMPLE_MASS
                if len(valores) == 1:
                    massa_texto = valores[0].replace(",", ".")

                # Caso com vírgula decimal:
                # INFO,28,53,SAMPLE_MASS
                elif len(valores) == 2:
                    massa_texto = f"{valores[0]}.{valores[1]}"

                else:
                    raise ValueError(
                        f"Formato de SAMPLE_MASS não reconhecido: {linha}"
                    )

                try:
                    massa_mg = float(massa_texto)

                except ValueError:
                    raise ValueError(
                        f"Não foi possível converter SAMPLE_MASS: {linha}"
                    )

                if massa_mg <= 0:
                    raise ValueError(
                        f"Massa inválida encontrada: {massa_mg} mg"
                    )

                # mg -> g
                massa_g = massa_mg * 1e-3

                print(
                    f"Massa da amostra encontrada: "
                    f"{massa_mg:.6g} mg"
                )

                print(
                    f"Massa usada na normalização: "
                    f"{massa_g:.8f} g"
                )

                return massa_g

            # SAMPLE_MASS fica no cabeçalho.
            # Não precisamos procurar depois de [Data].
            if linha == "[Data]":
                break

    raise ValueError(
        "SAMPLE_MASS não encontrado no cabeçalho do arquivo."
    )

def integral_trapezio(x, y):
    """Calcula a integral filtrando NaNs."""
    # Filtrar NaNs
    mask = ~np.isnan(y)
    x_clean = x[mask]
    y_clean = y[mask]

    if len(x_clean) < 2:
        return 0.0

    # Calcular integral
    n = len(x_clean)
    integral = 0.0

    for i in range(n - 1):
        h = x_clean[i + 1] - x_clean[i]
        area = (y_clean[i] + y_clean[i + 1]) * h / 2.0
        integral += area

    return integral

def arredondamento_inteiro(num,multiplo=500,torr=20):
    resto = num % multiplo

    if min(resto, multiplo - resto) <= torr:
        return round(num / multiplo) * multiplo
    return num

def arredondamento_array(arr, multiplo=500, torr=20):
    """Aplica arredondamento_inteiro em cada elemento do array"""
    return np.array([arredondamento_inteiro(x, multiplo, torr) for x in arr])

def encontrar_linhas_pular(caminho_arquivo):
    with open(caminho_arquivo, "r", encoding="latin1") as arquivo:
        for numero_linha, linha in enumerate(arquivo, start=1):
            if linha.strip() == "[Data]":
                return numero_linha

    raise ValueError(f"Marcador [Data] não encontrado em {caminho_arquivo}")

def ler_dados_ppms(caminho_arquivo, linhas_pular=None):
    if linhas_pular is None:
        linhas_pular = encontrar_linhas_pular(caminho_arquivo)

    dados = pd.read_csv(caminho_arquivo, skiprows=linhas_pular)

    colunas_necessarias = [
        "Moment (emu)",
        "Temperature (K)",
        "Magnetic Field (Oe)",
    ]
    colunas_faltando = [coluna for coluna in colunas_necessarias if coluna not in dados.columns]
    if colunas_faltando:
        raise ValueError(
            "Colunas obrigatórias ausentes no arquivo "
            f"{os.path.basename(caminho_arquivo)}: {colunas_faltando}"
        )

    for coluna in colunas_necessarias:
        dados[coluna] = pd.to_numeric(dados[coluna], errors="coerce")

    linhas_antes = len(dados)
    dados = dados.dropna(subset=colunas_necessarias).reset_index(drop=True)
    linhas_removidas = linhas_antes - len(dados)

    if linhas_removidas:
        print(f"Linhas ignoradas por dados incompletos: {linhas_removidas}")

    if dados.empty:
        raise ValueError(f"Nenhuma linha válida foi lida de {caminho_arquivo}")

    return dados

def separar_por_campo(arquivo,saida,nome_da_amostra,linhas_pular,campos_calculo=None):
    controle = 0
    dados_por_campo = {}
    diferença = []
    caminho_arquivo = os.path.join(diretorio, arquivo)
    dados = ler_dados_ppms(caminho_arquivo, linhas_pular)
    massa = encontrar_massa_amostra(caminho_arquivo)
    print(f"Massa da amostra (g): {massa}")
    mag_norm = np.array(dados["Moment (emu)"])/massa #emu/g
    Temp = np.array(dados["Temperature (K)"])
    Campo = np.array(dados["Magnetic Field (Oe)"])
    Campo_round = arredondamento_array(np.round(Campo))
    campos_unicos = np.unique(Campo_round)

    if campos_calculo is not None:
        campos_calculo = np.array(campos_calculo)
        campos_faltando = np.setdiff1d(campos_calculo, campos_unicos)

        if len(campos_faltando) > 0:
            raise ValueError(f"Campos solicitados ausentes no arquivo: {campos_faltando}")

        campos_unicos = campos_calculo

    print("Campos usados no calculo (Oe): ",campos_unicos)

    for i, H in enumerate(campos_unicos):
        if i < len(campos_unicos) - 1:
            diferença.append(campos_unicos[i+1] - campos_unicos[i])

        mask = Campo_round == H
        Temp_H = Temp[mask]
        mag_H = mag_norm[mask]
        # apenas ruido/estabilizacao do PPMS, entao nao devem cortar a curva.
        dT = np.diff(Temp_H)
        idx = np.where(dT < -1.0)[0]

        if len(idx) > 0:
            corte = idx[0] + 1
            Temp_H = Temp_H[:corte]
            mag_H = mag_H[:corte]

        dados_por_campo[H] = (Temp_H, mag_H)

        if controle == 1:
            nome_arquivo_saida = f"{nome_da_amostra}_{int(H)}Oe.txt"
            caminho_saida = os.path.join(saida, nome_arquivo_saida)

            df_saida = pd.DataFrame({
                "Temperature (K)": Temp_H,
                "Magnetization (emu/g)": mag_H
            })
            df_saida.to_csv(caminho_saida, sep='\t', index=False)
            print(f"Arquivo salvo: {caminho_saida}")

    campos_ordenados = np.sort(list(dados_por_campo.keys()))

    return diferença, dados_por_campo, campos_ordenados

def plot_MxT(dados_por_campo,nome_da_amostra,saida):
    controle = 0

    #figura completa
    fig, ax = plt.subplots()
    for H, (Temp_H, mag_H) in dados_por_campo.items():
        ax.plot(Temp_H, mag_H, "o", label=f'{int(H)} Oe')

    ax.text(
        0.02,
        0.98,
        "Aquecimento da curva MxT",
        transform=ax.transAxes,
        ha="left",
        va="top",
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.8),
    )

    ax.set_xlabel('Temperatura (K)')
    ax.set_ylabel('Magnetização (emu/g)')
    ax.legend()
    ax.grid()

    if controle == 1:
        caminho_figura = os.path.join(saida, f"{nome_da_amostra}_MxT.png")
        fig.savefig(caminho_figura)
        plt.close()
        print(f"Figura salva: {caminho_figura}")

    plt.show()

def plot_deltaS(temp_comum,delta_S_total):
    controle = 0
    plt.figure()

    for chave, delta_S in delta_S_total.items():
        plt.plot(temp_comum,delta_S, "o")

    plt.xlabel('Índice de Temperatura')
    plt.ylabel('ΔS (J/kg·K)')
    #plt.legend()
    plt.grid()
    if controle == 1:
        caminho_figura = os.path.join(saida, f"{nome_da_amostra}_DeltaS.png")
        plt.savefig(caminho_figura)
        plt.close()
        print(f"Figura salva: {caminho_figura}")

    plt.show()

def operação(derivadas, diferença):

    deltaS = 0.0

    for i in range(len(derivadas) - 1):
        deltaS += (
            0.5
            * (derivadas[i] + derivadas[i + 1])
            * diferença[i]
            * 1e-4
        )

    return deltaS

def calculo_da_entropia(dados_por_campo, campos_ordenados, diferença):
    """Calcula Delta S interpolando todos os campos na mesma grade de temperatura."""
    temp_min = max(np.min(dados_por_campo[H][0]) for H in campos_ordenados)
    temp_max = min(np.max(dados_por_campo[H][0]) for H in campos_ordenados)
    passos = []

    for H in campos_ordenados:
        Temp_H = np.sort(dados_por_campo[H][0])
        dT = np.diff(Temp_H)
        passos.extend(dT[dT > 0])

    if len(passos) == 0:
        raise ValueError("Nao foi possivel montar uma grade comum de temperatura.")

    passo_temp = np.median(passos)
    temp_comum = np.arange(temp_min, temp_max, passo_temp)

    if len(temp_comum) == 0 or temp_comum[-1] < temp_max:
        temp_comum = np.append(temp_comum, temp_max)
    dados_alinhados = {}
    for H in campos_ordenados:
        Temp_H = dados_por_campo[H][0]
        Mag_H = dados_por_campo[H][1]
        ordem = np.argsort(Temp_H)
        dados_alinhados[H] = np.interp(temp_comum, Temp_H[ordem], Mag_H[ordem])

    # Calcular deltaS
    deltaS = []
    for i in range(len(temp_comum) - 1):
        derivadas = []
        diff_x = temp_comum[i + 1] - temp_comum[i]

        # Evitar divisão por zero
        if abs(diff_x) < 1e-10:
            continue

        for H in campos_ordenados:
            mag_H = dados_alinhados[H]
            diff_y = mag_H[i + 1] - mag_H[i]
            derivadas.append(diff_y / diff_x)

        # Só calcular se coletou derivadas de todos os campos
        if len(derivadas) == len(campos_ordenados):
            deltaS.append(operação(derivadas, diferença))

    if not deltaS:  # Se deltaS estiver vazio
        delta_S_total = 0

    else:
        delta_S_total = np.sum(deltaS)

    return temp_comum, delta_S_total, deltaS

#inputs
diretorio = "Exemplos_de_entrada"
#nome_da_amostra = r"Fe$_{5}Sn_{3}$"
nome_da_amostra = r"Fe$_{2.5}$Mn$_{1}$Co$_{1.5}$Si$_{3}$"
linhas_pular = None  # detecta automaticamente a linha depois de [Data]
campos_calculo = [5000,10000,15000,20000,25000,30000,35000,40000,45000,50000]
saida = os.path.join(diretorio,"separados_por_campo")
saida_entropia = os.path.join(diretorio, "entropia_txt")
arquivos = [f for f in os.listdir(diretorio) if f.endswith(".dat") or f.endswith(".DAT")]
print(arquivos)
os.makedirs(saida, exist_ok=True)
os.makedirs(saida_entropia, exist_ok=True)

for arquivo in arquivos:
    print(f"Processando arquivo: {arquivo}")

    diferença, dados_por_campo, campos_ordenados = separar_por_campo(
        arquivo,
        saida,
        nome_da_amostra,
        linhas_pular,
        campos_calculo,
    )

    #print(f"ΔH entre campos (Oe): {diferença}")
    print(f"Campos ordenados (Oe): {campos_ordenados}")
    plot_MxT(dados_por_campo,nome_da_amostra,saida)

    temp_comum,delta_S_total,dS = calculo_da_entropia(dados_por_campo,campos_ordenados,diferença)

    dS = np.array(dS)  # Converter para array para facilitar o plot

    df_entropia = pd.DataFrame({
        "Temperature (K)": temp_comum[:-1],
        "DeltaS (J/kg.K)": dS,
    })

    nome_base = os.path.splitext(arquivo)[0]
    sufixo_campos = f"{int(campos_ordenados[0] / 1000)}k-{int(campos_ordenados[-1] / 1000)}kOe"
    caminho_entropia = os.path.join(saida_entropia, f"{nome_base}_DeltaS_MxT_{sufixo_campos}.txt")
    df_entropia.to_csv(caminho_entropia, sep="\t", index=False)
    print(f"Arquivo de entropia salvo: {caminho_entropia}")

    # Na parte do plot, faça:
    plt.plot(temp_comum[:-1], dS, "o-", label = f"{nome_da_amostra}")  # Exclui o último ponto de temp_comum
    plt.xlabel("Temperature (K)")
    plt.ylabel("ΔS (J/kg·K)")
    plt.legend()
    plt.show()