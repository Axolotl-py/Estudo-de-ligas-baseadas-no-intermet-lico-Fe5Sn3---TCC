import pandas as pd
import os
import matplotlib.pyplot as plt
import numpy as np
import smplotlib


def encontrar_linhas_pular(caminho_arquivo):
    with open(caminho_arquivo, "r", encoding="latin1") as arquivo:
        for numero_linha, linha in enumerate(arquivo, start=1):
            if linha.strip() == "[Data]":
                return numero_linha

    raise ValueError(
        f"Marcador [Data] não encontrado em {caminho_arquivo}"
    )


def ler_dados_ppms(caminho_arquivo):
    linhas_pular = encontrar_linhas_pular(caminho_arquivo)

    dados = pd.read_csv(
        caminho_arquivo,
        skiprows=linhas_pular
    )

    colunas_necessarias = [
        "Mass (grams)",
        "Moment (emu)",
        "Temperature (K)",
        "Magnetic Field (Oe)"
    ]

    for coluna in colunas_necessarias:
        if coluna not in dados.columns:
            raise ValueError(
                f"Coluna '{coluna}' não encontrada em {caminho_arquivo}"
            )

        dados[coluna] = pd.to_numeric(
            dados[coluna],
            errors="coerce"
        )

    dados = dados.dropna(
        subset=colunas_necessarias
    ).reset_index(drop=True)

    if dados.empty:
        raise ValueError(
            f"Nenhum dado válido encontrado em {caminho_arquivo}"
        )

    return dados


# ============================================================
# GRÁFICO PRINCIPAL
# ============================================================

arquivo = (
    "/home/gabriel/Documents/IC/Medidas/unmsm med magn/1FMCS currie temp DC principal/"
    "1FMCS-01-MT1.DAT"
)

dados = ler_dados_ppms(arquivo)

Temp_H = dados["Temperature (K)"].to_numpy(dtype=float)
mag_H = dados["Moment (emu)"].to_numpy(dtype=float)

massa = np.mean(dados["Mass (grams)"]) * 1e-3

mag_H = mag_H / massa


# ============================================================
# DADOS ZFC E FC
# ============================================================

arquivo2 = (
    "/home/gabriel/Documents/IC/Medidas/unmsm med magn/1FMCS MT 2 currie 50/"
    "1FMCS-01-MT2-MEIO-KMIN_00001.dat"
)

dados2 = ler_dados_ppms(arquivo2)

temp = dados2["Temperature (K)"].to_numpy(dtype=float)
mag = dados2["Moment (emu)"].to_numpy(dtype=float)

massa = np.mean(dados2["Mass (grams)"]) * 1e-3

mag = mag / massa


# Separação das duas regiões da medida
'''
parte1_temp = temp[:339]
parte1_mag = mag[:339]

parte2_temp = temp[339:]
parte2_mag = mag[339:]
'''
parte1_temp = temp[:178]
parte1_mag = mag[:178]

parte2_temp = temp[178:]
parte2_mag = mag[178:]
# ============================================================
# PLOT
# ============================================================

fig, ax = plt.subplots(figsize=(10, 7))


# Gráfico principal
ax.plot(
    Temp_H,
    mag_H,
    "o-",
    markersize=3,
    linewidth=1,
    label=r"$\mathrm{Fe_{2.5}Mn_{1.0}Co_{1.5}Sn_{3}}$"
)

ax.set_xlabel("Temperatura (K)")
ax.set_ylabel("Magnetização (emu/g)")

ax.grid()
ax.legend(loc="upper right")


# ============================================================
# INSET: ZFC / FC
# ============================================================

ax_interno = ax.inset_axes(
    [0.10, 0.18, 0.48, 0.48]
)

ax_interno.plot(
    parte1_temp,
    parte1_mag,
    "o-",
    markersize=2,
    linewidth=0.8,
    label="Mzfc"
)

ax_interno.plot(
    parte2_temp,
    parte2_mag,
    "o-",
    markersize=2,
    linewidth=0.8,
    label="Mfc"
)

ax_interno.set_xlabel(
    "Temperatura (K)",
    fontsize=9
)

ax_interno.set_ylabel(
    "Magnetização (emu/g)",
    fontsize=9
)

ax_interno.tick_params(
    axis="both",
    labelsize=8
)

ax_interno.grid()

ax_interno.legend(
    fontsize=9,
    loc="best"
)


# ============================================================
# TEMPERATURA DE CURIE
# ============================================================

Tc = 325
mag_Tc = np.interp(Tc, Temp_H, mag_H)

ax.plot(Tc, mag_Tc, "o", color="red", markersize=8, zorder=5)

ax.annotate(
    r"$T_c = 325\ \mathrm{K}$",
    xy=(Tc, mag_Tc),          # ponto marcado
    xytext=(325, 1.1),        # posição da box
    textcoords="data",        # garante que xytext está em coordenadas do gráfico
    fontsize=12,
    color="black",
    bbox=dict(
        boxstyle="round,pad=0.35",
        facecolor="white",
        edgecolor="red",
        linewidth=1.2
    ),
    arrowprops=dict(
        arrowstyle="->",
        color="red",
        linewidth=1.5
    ),
    ha="left",
    va="center",
    annotation_clip=False
)


plt.tight_layout()
plt.show()