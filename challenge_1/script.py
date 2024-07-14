"""
Script para gerar um dataset de exemplo para o projeto de Machine Learning.
O dataset consiste em um conjunto de dados fictícios de planos de saúde.
Os valores dos encargos são calculados com base em algumas variáveis como idade, gênero, IMC, fumante, região e quantidade de filhos.

O dataset gerado é salvo em um arquivo CSV chamado dataset.csv.
"""

import argparse
import numpy as np
import pandas as pd
import random

REGIOES = (
    "Norte",
    "Nordeste",
    "Centro-Oeste",
    "Sul",
    "Sudeste",
)

GENEROS = (
    "Masculino",
    "Feminino",
)

FUMANTES = (
    "Não",
    "Sim",
)


def get_base_value_for_age(idade):
    """
    Retorna o valor base dos encargos de acordo com a idade.
    """
    if 0 <= idade <= 18:
        return 295
    elif 19 <= idade <= 30:
        return 725
    elif 31 <= idade <= 40:
        return 1857
    elif 41 <= idade <= 50:
        return 2726
    elif 51 <= idade <= 60:
        return 3421
    elif 61 <= idade <= 70:
        return 4318
    elif 71 <= idade <= 80:
        return 5938
    else:
        return 6930


def get_imc_coef(imc):
    """
    Retorna o coeficiente de ajuste dos encargos de acordo com o IMC.
    """
    if 0 <= imc <= 18.5:  #  abaixo do peso
        return 1.57
    if 25 <= imc <= 29.9:  #  sobrepeso
        return 1.41
    if 30 <= imc <= 34.9:  #  obesidade g1
        return 2.08
    if 35 <= imc <= 39.9:  # obesidade g2
        return 0.53
    else:
        return 2.48


def get_region_coef(regiao):
    """
    Retorna o coeficiente de ajuste dos encargos de acordo com a região.
    """
    if regiao in ("Norte", "Nordeste"):
        return 1.15
    if regiao == "Centro-Oeste":
        return 1.10
    else:
        return 1.05


def gerador(size, seed):
    random.seed(seed)
    np.random.seed(seed)

    idade = np.random.randint(18, 75, size)
    imc = np.random.randint(15, 45, size)
    filhos = np.random.randint(0, 5, size)
    genero = random.choices(GENEROS, k=size)
    fumante = random.choices(FUMANTES, k=size)
    regiao = random.choices(REGIOES, k=size)
    encargos = np.zeros(size)

    for i in range(size):
        # Idade
        valor_faixa = get_base_value_for_age(idade[i])
        encargos[i] = np.random.randint(valor_faixa * 0.9, valor_faixa * 1.05)

        # Genero
        if genero[i] == "Feminino":
            encargos[i] = encargos[i] * 1.1085

        # IMC
        encargos[i] *= get_imc_coef(imc[i])

        # Fumante
        if fumante[i] == "Sim":
            encargos[i] *= 1.460

        # Região
        encargos[i] = encargos[i] * get_region_coef(regiao[i])

    # quantidade de filhos
    encargos = encargos * (1 + filhos / 100)

    df = pd.DataFrame(
        {
            "idade": idade,
            "gênero": genero,
            "imc": imc,
            "filhos": filhos,
            "fumante": fumante,
            "região": regiao,
            "encargos": encargos,
        }
    )

    return df


def main():
    parser = argparse.ArgumentParser(
        description="Gerador de dados aleatórios para fins de teste."
    )
    parser.add_argument("size", type=int, help="Tamanho dos dados a serem gerados.")
    parser.add_argument(
        "seed", type=int, help="Semente para geração de números aleatórios."
    )

    args = parser.parse_args()

    # Chamar a função geradora com os parâmetros fornecidos
    df = gerador(args.size, args.seed)

    # Salvar os dados gerados em um arquivo CSV
    df.to_csv("dataset.csv", index=False)
    print(f'Dados gerados salvos em "dataset.csv".')


if __name__ == "__main__":
    main()
