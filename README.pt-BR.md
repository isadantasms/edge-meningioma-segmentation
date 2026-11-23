<div align="center">

# Segmentação Semântica de Meningiomas via Destilação de Conhecimento e IA Explicável para Ambientes com Restrição Computacional

[![Language: English](https://img.shields.io/badge/Language-English-blue.svg)](README.md)
[![Versão em Português](https://img.shields.io/badge/Vers%C3%A3o-Portugu%C3%AAs-green.svg)](#)
[![Instituição: UFU](https://img.shields.io/badge/UFU-FACOM-003366.svg)](https://facom.ufu.br/)
[![Programa: PET Sistemas de Informação](https://img.shields.io/badge/PET-Sistemas%20de%20Informa%C3%A7%C3%A3o-orange.svg)](https://www.petfacom.ufu.br/)
[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-Deep%20Learning-EE4C2C.svg)](https://pytorch.org/)
[![Status](https://img.shields.io/badge/Status-Em%20Andamento-brightgreen.svg)]()
[![Documentação](https://img.shields.io/badge/Documentos-Oficiais%20Anexados-success.svg)](#-documentações-e-anexos-do-projeto)
[![Notion Workspace](https://img.shields.io/badge/Notion-Notas%20de%20Pesquisa-000000.svg?logo=notion&logoColor=white)](https://app.notion.com/p/Segmenta-o-Sem-ntica-de-Meningiomas-via-Destila-o-de-Conhecimento-e-Intelig-ncia-Artificial-Explic-4070fdd8eda580a9812afa1ed45d73f6)

**[🇺🇸 English](README.md) | [🇧🇷 Português](README.pt-BR.md)**

<p align="justify">
Projeto de pesquisa científica individual desenvolvido no âmbito do <b>Programa de Educação Tutorial (PET - Sistemas de Informação)</b> da <b>Faculdade de Computação (FACOM)</b> da <b>Universidade Federal de Uberlândia (UFU)</b>.
</p>

</div>

---

## 📌 Identificação do Projeto

- **Pesquisadora (Discente):** Isabela Dantas de Moraes ([isabeladantasms@ufu.br](mailto:isabeladantasms@ufu.br) / Matrícula: `12521BSI221`)
- **Professor Orientador:** Prof. Dr. Marcelo Zanchetta do Nascimento ([marcelo.nascimento@ufu.br](mailto:marcelo.nascimento@ufu.br) / SIAPE: `1546624`)
- **Tutor do Grupo PET:** Prof. Dr. Thiago Pirola Ribeiro
- **Unidade Acadêmica:** Faculdade de Computação (FACOM / UFU)
- **Duração do Projeto:** 12 meses (Novembro/2026 – Outubro/2027)
- **Diário de Bordo / Notas Interativas:** [Espaço de Trabalho no Notion](https://app.notion.com/p/Segmenta-o-Sem-ntica-de-Meningiomas-via-Destila-o-de-Conhecimento-e-Intelig-ncia-Artificial-Explic-4070fdd8eda580a9812afa1ed45d73f6)

---

## 📖 Resumo e Visão Geral

Os **meningiomas** são os tumores primários mais frequentes do Sistema Nervoso Central (SNC), respondendo por mais de 40% dos casos diagnosticados. A delimitação tumoral precisa em exames de **Ressonância Magnética (RM)** é determinante para o planejamento cirúrgico e acompanhamento clínico, mas a segmentação manual exige esforço contínuo de especialistas altamente qualificados.

Embora métodos modernos de aprendizado profundo (*Deep Learning*) apresentem alta acurácia na segmentação automática, suas arquiteturas convencionais demandam elevado poder de processamento (GPUs de alta performance) e grande consumo de memória — recursos frequentemente escassos em hospitais regionais e unidades básicas do Sistema Único de Saúde (SUS).

Este projeto investiga o desenvolvimento de uma abordagem de **segmentação semântica bidimensional de meningiomas** baseada em **Destilação de Conhecimento (*Knowledge Distillation* - KD)**, na qual um modelo robusto de alta capacidade (*rede professora*) orienta o treinamento de uma arquitetura leve e compacta (*rede aluna*). O objetivo é viabilizar inferência rápida em **CPUs convencionais** sem perda substancial de acurácia, complementada por técnicas de **Inteligência Artificial Explicável (XAI)** como o LIME (*Local Interpretable Model-agnostic Explanations*) e mapas de ativação para apoiar a auditabilidade e interpretabilidade dos limites preditos.

---

## 🎯 Objetivos

### Objetivo Geral
Desenvolver e avaliar uma abordagem de segmentação semântica de meningiomas em imagens de ressonância magnética por destilação de conhecimento, investigando o equilíbrio entre **desempenho de segmentação**, **eficiência computacional em CPU** e **explicabilidade das predições**.

### Objetivos Específicos
1. **Caracterização e Curadoria de Dados:** Caracterizar a base de dados pública **BRISC**, selecionando as imagens de meningiomas nos planos axial, sagital e coronal, e padronizar o pré-processamento de intensidade e geometria.
2. **Particionamento Sem Vazamento:** Definir protocolos rigorosos de separação entre conjuntos de treino, validação e teste com base em pacientes, evitando vazamento de dados (*data leakage*).
3. **Modelagem de Referência:** Implementar e treinar uma **Rede Professora** (alta capacidade) e uma **Rede Aluna Não Destilada** (treinada do zero) como linhas de base (*baselines*).
4. **Destilação de Conhecimento:** Treinar a rede aluna sob destilação combinando:
   - Supervisão direta pelas máscaras anotadas ($\text{Cross-Entropy} + \text{Dice Loss}$);
   - Alinhamento probabilístico das saídas com suavização por temperatura (*soft labels*);
   - Alinhamento de representações intermediárias (*feature-aligned distillation*).
5. **Avaliação Quantitativa e de Eficiência:** Comparar os modelos quanto à acurácia de segmentação (*Dice, IoU, HD95*) e métricas de execução em CPU (latência por imagem, consumo de RAM, contagem de parâmetros e tamanho em disco).
6. **Interpretabilidade e Auditoria (XAI):** Aplicar LIME adaptado para saídas densas e inspecionar mapas de ativação intermediários, confrontando-os com as regiões tumorais reais.
7. **Disponibilização e Reprodutibilidade:** Versionar o código, parâmetros de configuração, sementes e documentação técnica para reprodutibilidade científica completa.

---

## 🔬 Metodologia

```
+-------------------------------------------------------------+
|                     Base de Dados: BRISC                    |
|      (RM T1 ponderada c/ contraste - Axial, Coronal, Sagital)|
+-------------------------------------------------------------+
                               |
                               v
+-------------------------------------------------------------+
|                     Pré-processamento                       |
|   (Padronização, Normalização, Augmentation no Treinamento)  |
+-------------------------------------------------------------+
                               |
            +------------------+------------------+
            |                                     |
            v                                     v
+------------------------+             +----------------------+
|    Rede Professora     |             |      Rede Aluna      |
|    (Alta Capacidade)   |             | (Compacta / Edge-CPU)|
+------------------------+             +----------------------+
            |                                     |
            +------- Destilação de Conhecimento --+
            |   - Soft logits c/ temperatura      |
            |   - Alinhamento intermediário       |
            |   - Loss combinada (CE + Dice + KD) |
            |                                     |
            +-----------------> v <---------------+
                                |
             +------------------+------------------+
             |                                     |
             v                                     v
+-------------------------+           +-------------------------+
| Métricas de Segmentação |           | Eficiência Computacional|
|  - Dice Similarity (DSC)|           |  - Latência em CPU (ms) |
|  - IoU (Jaccard Index)  |           |  - Memória RAM alocada  |
|  - HD95 (Hausdorff 95%) |           |  - Qtd. de Parâmetros   |
+-------------------------+           +-------------------------+
             \                                     /
              \                                   /
               +----------------+----------------+
                                |
                                v
               +---------------------------------+
               | Explicabilidade & Auditoria XAI |
               |  - LIME em regiões ativadas     |
               |  - Mapas de ativação por camada |
               +---------------------------------+
```

### Formulação da Função de Perda (Loss) de Destilação
$$\mathcal{L}_{\text{total}} = \alpha \mathcal{L}_{\text{seg}}(y, \hat{y}_{\text{aluna}}) + (1 - \alpha) \mathcal{L}_{\text{KD}}(\hat{y}_{\text{aluna}}, \hat{y}_{\text{professora}}; T) + \beta \mathcal{L}_{\text{feat}}(F_{\text{aluna}}, F_{\text{professora}})$$

Onde:
- $\mathcal{L}_{\text{seg}}$ é a perda supervisionada ($\text{Dice} + \text{Cross-Entropy}$) contra a anotação manual $y$.
- $\mathcal{L}_{\text{KD}}$ é a divergência Kullback-Leibler calculada sobre os logits suavizados pela temperatura $T > 1$.
- $\mathcal{L}_{\text{feat}}$ minimiza a distância das características intermediárias entre camadas selecionadas.
- $\alpha, \beta$ são hiperparâmetros de ponderação ajustados no conjunto de validação.

---

## 📓 Experimentos Preliminares e Notebooks

As primeiras atividades exploratórias e pipelines de base desenvolvidos na fase inicial do projeto estão organizados no diretório [`notebooks/`](./notebooks/):

| Notebook | Foco | Descrição | Link |
| :--- | :---: | :--- | :---: |
| 🧪 **01 — Classificação (ConvNeXt-Atto)** | Classificação Leve (Edge) | Pipeline inicial de classificação multiclasse (*glioma, meningioma, tumor hipofisário e sem tumor*) utilizando a arquitetura leve ConvNeXt-Atto com partições reprodutíveis. | [Acessar Notebook](./notebooks/01_classification_convnext_atto.ipynb) |
| 🎯 **02 — Segmentação (SegFormer Baseline)** | Segmentação Semântica | Pipeline inicial de segmentação semântica binária validando extração de patches, ciclo de treinamento e métricas Dice/IoU para tumores cerebrais. | [Acessar Notebook](./notebooks/02_segmentation_segformer.ipynb) |

---

## 📅 Cronograma de Atividades (12 Meses)

| Período | Atividade Prevista |
| :---: | :--- |
| **11/2026** | Revisão bibliográfica; caracterização da base BRISC e verificação das condições de uso; organização do ambiente computacional; implementação inicial do pré-processamento. |
| **01/2027** | Definição das partições de dados; seleção das arquiteturas; treinamento inicial da professora e da aluna sem destilação. |
| **03/2027** | Implementação da destilação das saídas; ajuste de hiperparâmetros no conjunto de validação; registro dos primeiros experimentos. |
| **05/2027** | Investigação da transferência de características intermediárias; integração inicial das explicações (LIME); comparação das configurações e repetição dos experimentos prioritários. |
| **07/2027** | Avaliação final no conjunto de teste; medição de tempo e memória em CPU; análise de erros e experimentos complementares de explicabilidade. |
| **10/2027** | Consolidação e discussão dos resultados; materiais de reprodutibilidade; redação do relatório final e preparação de produção científica. |

---

## 📂 Documentações e Anexos do Projeto

Os documentos oficiais homologados e apresentados para este projeto encontram-se disponíveis no repositório na pasta [`docs/`](./docs/) com nomenclatura padronizada em inglês:

| Documento | Descrição | Link |
| :--- | :--- | :---: |
| 📄 **Plano de Trabalho PET** | Plano de trabalho detalhado de Pesquisa Individual PET aprovado pela UFU | [Visualizar PDF (`docs/research_work_plan.pdf`)](./docs/research_work_plan.pdf) |
| 📝 **Registro de Pesquisa DIREN** | Formulário oficial de Registro de Pesquisa Individual PET com assinaturas digitais (DIREN/PROGRAD/DIFDI) | [Visualizar PDF (`docs/research_registration.pdf`)](./docs/research_registration.pdf) |
| 📊 **Pôster Científico (WTDCC 2026)** | Pôster acadêmico apresentado no Workshop de Teses e Dissertações em Ciência da Computação | [Visualizar PDF (`docs/scientific_poster_wtdcc_2026.pdf`)](./docs/scientific_poster_wtdcc_2026.pdf) |

---

## 🚀 Guia de Início Rápido & Configuração do Ambiente

### 1. Clonar o Repositório
```bash
git clone https://github.com/isadantasms/edge-meningioma-segmentation.git
cd edge-meningioma-segmentation
```

### 2. Configurar o Ambiente Virtual
Crie um ambiente virtual com Python 3.10+ e instale os pacotes necessários:
```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### 3. Notebooks de Experimentação
Os experimentos e etapas do pipeline encontram-se documentados no diretório [`notebooks/`](./notebooks/):
* `01_classification_convnext_atto.ipynb`: Classificação preliminar com ConvNeXt-Atto ($224 \times 224$).
* `02_segmentation_segformer.ipynb`: Linha de base para segmentação semântica com SegFormer ($256 \times 256$).

---

## 📁 Estrutura do Repositório

```text
edge-meningioma-segmentation/
│
├── docs/                               # Documentações oficiais e acadêmicas (em inglês)
│   ├── research_work_plan.pdf         # Plano de trabalho detalhado PET/UFU
│   ├── research_registration.pdf      # Termo de registro de pesquisa homologado
│   └── scientific_poster_wtdcc_2026.pdf # Pôster acadêmico (WTDCC 2026)
│
├── notebooks/                          # Notebooks de experimentação e exploração
│   ├── 01_classification_convnext_atto.ipynb # Tarefa de classificação preliminar
│   └── 02_segmentation_segformer.ipynb        # Linha de base de segmentação semântica
│
├── data/                               # Instruções e scripts de preparação da base BRISC
├── src/                                # Código-fonte do projeto
│   ├── data/                           # Dataloaders e transformações de pré-processamento
│   ├── models/                         # Arquiteturas (Professora, Aluna e Adaptadores)
│   ├── distillation/                   # Módulos de perda de destilação e alinhamento
│   ├── explainability/                 # Scripts de XAI (LIME e mapas de ativação)
│   └── utils/                          # Métricas (Dice, IoU, HD95) e benchmark de CPU
│
├── .gitignore                          # Regras de exclusão do Git
├── requirements.txt                    # Dependências do projeto Python
├── README.md                           # Documentação principal (Inglês)
└── README.pt-BR.md                     # Documentação completa em Português
```

---

## 📚 Referências Bibliográficas

1. **COLLI, B. O. et al.** Epidemiological features of meningiomas: a single Brazilian center’s experience with 993 cases. *Arquivos de Neuro-Psiquiatria*, v. 79, n. 8, 2021.
2. **FATEH, A. et al.** BRISC: annotated dataset for brain tumor segmentation and classification with Swin-HAFNet. *arXiv preprint*, arXiv:2506.14318, 2025.
3. **FREIRE, T.** SUS realiza seis em cada dez exames de imagem no Brasil, diz estudo. *Agência Brasil*, Reportagem sobre o Atlas da Radiologia no Brasil 2025 (Colégio Brasileiro de Radiologia e Diagnóstico por Imagem), 2025.
4. **GOLDBRUNNER, R. et al.** EANO guideline on the diagnosis and management of meningiomas. *Neuro-Oncology*, v. 23, n. 11, p. 1821–1834, 2021.
5. **HINTON, G.; VINYALS, O.; DEAN, J.** Distilling the knowledge in a neural network. *arXiv preprint*, arXiv:1503.02531, 2015.
6. **ISLAM, M. S. et al.** Optimizing brain tumor classification for edge devices with knowledge distillation. *IEEE ICCIT*, 2025.
7. **LABELLA, D. et al.** A multi-institutional meningioma MRI dataset for automated multi-sequence image segmentation. *Scientific Data*, v. 11, artigo 496, 2024.
8. **PINTO, L. M. et al.** Comparative analysis of mortality trends from meningioma across regions of Brazil from 2012 to 2022. *Brazilian Journal of Oncology*, v. 21, 2025.
9. **PRICE, M. et al.** CBTRUS statistical report: primary brain and other central nervous system tumors diagnosed in the United States in 2018–2022. *Neuro-Oncology*, v. 27, suplemento 4, 2025.
10. **RAHMAN, M. A. et al.** FAKD-XAI: feature-aligned knowledge distillation with explainable AI for efficient brain tumor classification. In: *ICDSAIA*, Springer, 2025.
11. **RIBEIRO, M. T.; SINGH, S.; GUESTRIN, C.** “Why should I trust you?”: explaining the predictions of any classifier. *ACM SIGKDD*, 2016.

---

## 🏛️ Apoio e Agradecimentos

Este trabalho é apoiado por:
- **PET Sistemas de Informação** — Universidade Federal de Uberlândia (UFU)
- **Faculdade de Computação (FACOM / UFU)**
- **Pró-Reitoria de Graduação (PROGRAD / UFU)**
- **Diretoria de Ensino (DIREN / UFU)**
- **Divisão de Formação Discente (DIFDI / UFU)**
