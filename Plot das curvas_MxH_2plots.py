import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
import smplotlib

def separacao_por_temperatura(diretorio, arquivos):
    resultado = {}

    for nome_arquivo in arquivos:
        caminho_completo = os.path.join(diretorio, nome_arquivo)

        dados = pd.read_csv(
            caminho_completo,
            sep=",",
            header=0,
            skiprows=34
        )

        magnetizacao = dados["Moment (emu)"]
        temperatura = np.round(dados["Temperature (K)"]).astype(int)
        print(f"Temperaturas lidas no arquivo {nome_arquivo} (K):", np.unique(temperatura))
        campo = dados["Magnetic Field (Oe)"]

        massa = float(dados["Mass (grams)"].iloc[0])

        for temp in np.unique(temperatura):
            mask = temperatura == temp

            campo_temp = campo[mask].values
            magnetizacao_temp = magnetizacao[mask].values / massa

            resultado[(nome_arquivo, temp)] = (campo_temp, magnetizacao_temp)

    return resultado


diretorio = "/home/gabriel/Documents/IC/Medidas/unmsm med magn/MXH /"

arquivos = sorted([
    f for f in os.listdir(diretorio)
    if f.lower().endswith(".dat")
])

print("Arquivos encontrados:", arquivos)

nomes_personalizados = {}

print("\nDigite um nome para cada amostra.")
print("Pressione Enter para manter o nome original do arquivo.\n")

for arquivo in arquivos:
    nome = input(f"Nome para '{arquivo}': ").strip()

    if nome == "":
        nome = os.path.splitext(arquivo)[0]

    nomes_personalizados[arquivo] = nome

dados = separacao_por_temperatura(diretorio, arquivos)

# pega todas as temperaturas existentes em todos os arquivos
temperaturas_unicas = sorted(set(temp for arquivo, temp in dados.keys()))

# Lista de arquivos realmente presentes no dicionário
arquivos_plot = sorted(set(arquivo for arquivo, temp in dados.keys()))

# Cria um gráfico para cada arquivo, lado a lado
fig, eixos = plt.subplots(
    nrows=1,
    ncols=len(arquivos_plot),
    figsize=(4 * len(arquivos_plot), 6),
    sharey=True
)

# Evita erro caso exista somente um arquivo
eixos = np.atleast_1d(eixos)

mapa_cores = plt.get_cmap("coolwarm")

for eixo, arquivo_plot in zip(eixos, arquivos_plot):

    curvas_do_arquivo = [
        (temp, campo, magnetizacao)
        for (arquivo, temp), (campo, magnetizacao) in dados.items()
        if arquivo == arquivo_plot
    ]

    # Ordena da menor para a maior temperatura
    curvas_do_arquivo.sort(key=lambda curva: curva[0])

    # Temperaturas existentes somente neste arquivo
    temperaturas_arquivo = [
        temp for temp, campo, magnetizacao in curvas_do_arquivo
    ]

    normalizacao = plt.Normalize(
        vmin=min(temperaturas_arquivo),
        vmax=max(temperaturas_arquivo)
    )

    for temp, campo, magnetizacao in curvas_do_arquivo:
        cor = mapa_cores(normalizacao(temp))

        eixo.plot(
            campo,
            magnetizacao,
            marker="o",
            linestyle=" ",
            markersize=2,
            linewidth=1,
            color=cor,
        )
    eixo.set_xlabel("Campo (kOe)")
    eixo.grid()
    eixo.text(
        0.05,
        0.95,
        r"${}$".format(nomes_personalizados[arquivo_plot]),
        transform=eixo.transAxes,
        fontsize=12,
        fontweight="bold",
        verticalalignment="top",
        horizontalalignment="left"
    )

    # Barra de temperatura específica deste gráfico
    mapeamento = plt.cm.ScalarMappable(
        norm=normalizacao,
        cmap=mapa_cores
    )

    mapeamento.set_array([])

    barra_cores = fig.colorbar(
        mapeamento,
        ax=eixo,
        pad=0.02
    )

    barra_cores.set_label("Temperatura (K)")

# Como o eixo Y é compartilhado, basta identificar no primeiro gráfico
eixos[0].set_ylabel("Magnetização (emu/g)")
eixos[1].set_ylabel("Magnetização (emu/g)")

plt.tight_layout()
plt.show()