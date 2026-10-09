from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from scipy.signal import savgol_filter
from scipy import sparse
from scipy.sparse.linalg import spsolve


# ============================================================
# CONFIGURAÇÕES
# ============================================================

# Arquivo experimental de DRX
arquivo = "/home/gabriel/Documents/TCC DRX/DRX/Fe5Sn3_M5_TT(1100K)_Annealing.xy" #Fe_{5}Sn_{3}
arquivo2 = "/home/gabriel/Documents/TCC DRX/DRX/1FMCS-03B_1g.xy" #Fe_{2.5}Mn_{1}Co_{1.5}Sn_{3}
# Arquivo cristalográfico usado para calcular o padrão teórico
ARQUIVO_CIF = (
    "/home/gabriel/Documents/TCC DRX/"
    "Fe5Sn3 modificado.cif"
)

# Intervalo apresentado no gráfico final
INTERVALO_2THETA = (10.0, 90.0)

# Identificação usada na legenda do gráfico final
NOME_AMOSTRA = r"Fe$_{5}$Sn$_{3}$"
NOME_AMOSTRA2 = r"Fe$_{2.5}$Mn$_{1}$Co$_{1.5}$Sn$_{3}$"

# Salvar gráfico final
SALVAR_GRAFICO_FINAL = True
NOME_GRAFICO_FINAL = "drx_com_cif.png"

# Comprimento de onda da radiação Cu Kα
# O pymatgen/XRDCalculator utiliza o comprimento de onda em angstrom.
COMPRIMENTO_ONDA = 0.15406  # nm

# Suavização Savitzky-Golay
JANELA_SAVGOL = 11
ORDEM_SAVGOL = 3


# ============================================================
# PADRÃO TEÓRICO CALCULADO DO ARQUIVO CIF
# ============================================================

def calcular_padrao_cif(caminho_cif, intervalo_2theta):
    """
    Calcula as posições e intensidades relativas das reflexões
    de DRX descritas no arquivo CIF.
    """

    try:
        from pymatgen.core import Structure
        from pymatgen.analysis.diffraction.xrd import XRDCalculator
    except ModuleNotFoundError as erro:
        raise ModuleNotFoundError(
            "O cálculo do padrão do CIF requer o pacote pymatgen. "
            "Instale-o no mesmo ambiente Python usado para executar "
            "este programa."
        ) from erro

    caminho_cif = Path(caminho_cif).expanduser()

    if not caminho_cif.is_file():
        raise FileNotFoundError(
            f"Arquivo CIF não encontrado: {caminho_cif}"
        )

    estrutura = Structure.from_file(str(caminho_cif))

    calculador = XRDCalculator(
        wavelength=COMPRIMENTO_ONDA * 10.0
    )

    padrao = calculador.get_pattern(
        estrutura,
        two_theta_range=intervalo_2theta
    )

    return (
        np.asarray(padrao.x, dtype=float),
        np.asarray(padrao.y, dtype=float)
    )


# ============================================================
# BACKGROUND - ASYMMETRIC LEAST SQUARES
# ============================================================

def baseline_als(y, lam=1e6, p=0.01, niter=15):
    """
    Estima o background usando Asymmetric Least Squares.

    lam:
        controla a suavidade do background.

    p:
        controla a assimetria.
    """

    y = np.asarray(y, dtype=float)
    tamanho = len(y)

    D = sparse.diags(
        diagonals=[1.0, -2.0, 1.0],
        offsets=[0, 1, 2],
        shape=(tamanho - 2, tamanho),
        format="csc"
    )

    pesos = np.ones(tamanho)

    for _ in range(niter):

        W = sparse.spdiags(
            pesos,
            0,
            tamanho,
            tamanho
        ).tocsc()

        Z = (W + lam * D.T @ D).tocsc()

        background = spsolve(
            Z,
            pesos * y
        )

        pesos = (
            p * (y > background)
            + (1 - p) * (y < background)
        )

    return background


# ============================================================
# 1. CARREGAR ARQUIVO EXPERIMENTAL
# ============================================================

dados = pd.read_csv(
    arquivo,
    header=None,
    sep=r"\s+",
    names=["2theta", "intensidade"],
    dtype=float
)
dados2 = pd.read_csv(
    arquivo2,
    header=None,
    sep=r"\s+",
    names=["2theta", "intensidade"],
    dtype=float
)

print("\nVERIFICAÇÃO DO ARQUIVO CARREGADO")
print("Caminho:", Path(arquivo).resolve())
print("Número de pontos:", len(dados))
print("2θ inicial:", dados["2theta"].min())
print("2θ final:", dados["2theta"].max())
print(dados.head())

two_theta = dados["2theta"].to_numpy()
intensidade = dados["intensidade"].to_numpy()

two_theta2 = dados2["2theta"].to_numpy()
intensidade2 = dados2["intensidade"].to_numpy()

# ============================================================
# 2. NORMALIZAR
# ============================================================

intensidade_normalizada = (
    intensidade
    / np.max(intensidade)
)

intensidade_normalizada2 = (
    intensidade2
    / np.max(intensidade2)
)
# ============================================================
# 3. CALCULAR BACKGROUND
# ============================================================

background = baseline_als(
    intensidade_normalizada,
    lam=1e6,
    p=0.01,
    niter=15
)

background2 = baseline_als(
    intensidade_normalizada2,
    lam=1e6,
    p=0.01,
    niter=15
)
# ============================================================
# 4. REMOVER BACKGROUND
# ============================================================

intensidade_corrigida = (
    intensidade_normalizada
    - background
)
intensidade_corrigida2 = (
    intensidade_normalizada2
    - background2
)
# Pequenos valores negativos gerados pelo ajuste
# são colocados em zero
intensidade_corrigida[
    intensidade_corrigida < 0
] = 0

intensidade_corrigida2[
    intensidade_corrigida2 < 0
] = 0
# ============================================================
# 5. SUAVIZAÇÃO LEVE
# ============================================================

intensidade_suave = savgol_filter(
    intensidade_corrigida,
    window_length=JANELA_SAVGOL,
    polyorder=ORDEM_SAVGOL
)


intensidade_suave2 = savgol_filter(
    intensidade_corrigida2,
    window_length=JANELA_SAVGOL,
    polyorder=ORDEM_SAVGOL
)
# ============================================================
# 7. CALCULAR O PADRÃO DO CIF
# ============================================================

two_theta_cif, intensidade_cif = calcular_padrao_cif(
    ARQUIVO_CIF,
    INTERVALO_2THETA
)


# ============================================================
# 8. GRÁFICO FINAL: EXPERIMENTAL + CIF
# ============================================================

fig, (ax_exp, ax_cif) = plt.subplots(
    2,
    1,
    figsize=(14, 9),
    sharex=True,
    layout="constrained",
    gridspec_kw={
        "height_ratios": [4.0, 1.25]
    }
)

# Difratograma experimental após remoção do background e suavização
ax_exp.plot(
    two_theta,
    intensidade_suave,
    color="black",
    linewidth=1.4,
    label=NOME_AMOSTRA
)
ax_exp.plot(
    two_theta2,
    intensidade_suave2,
    color="red",
    linewidth=1.4,
    label=NOME_AMOSTRA2
)

# Ajuste dos limites verticais do painel experimental
y_minimo = min(
    0.0,
    float(np.nanmin(intensidade_corrigida)),
    float(np.nanmin(intensidade_suave))
)

y_maximo = max(
    float(np.nanmax(intensidade_corrigida)),
    float(np.nanmax(intensidade_suave))
)

amplitude_y = max(
    y_maximo - y_minimo,
    1e-12
)

ax_exp.set_ylim(
    y_minimo - 0.03 * amplitude_y,
    y_maximo + 0.08 * amplitude_y
)

# Reflexões teóricas calculadas a partir do arquivo CIF
ax_cif.vlines(
    two_theta_cif,
    0,
    intensidade_cif,
    color="#1f77b4",
    linewidth=1.4,
    label="Picos calculados do CIF"
)

ax_exp.set_ylabel("Intensidade")
ax_exp.set_yticks([])
ax_exp.grid(alpha=0.18)
ax_exp.legend(loc="upper right")

ax_cif.set_xlabel(r"$2\theta$ (°)")
ax_cif.set_ylabel("CIF\nInt. rel. (%)")
ax_cif.set_ylim(0, 105)
ax_cif.grid(axis="x", alpha=0.18)
ax_cif.legend(loc="upper right")

# O mesmo intervalo de 2θ é aplicado aos dois painéis
ax_exp.set_xlim(20,90)
ax_cif.set_xlim(20,90)


# ============================================================
# 9. SALVAR / EXIBIR
# ============================================================

if SALVAR_GRAFICO_FINAL:
    fig.savefig(
        NOME_GRAFICO_FINAL,
        dpi=300,
        bbox_inches="tight"
    )

plt.show()

if SALVAR_GRAFICO_FINAL:
    print(
        f"\nGráfico final salvo em: {NOME_GRAFICO_FINAL}"
    )
