# Segmentação Semântica de Meningiomas via Destilação de Conhecimento e IA Explicável para Ambientes com Restrição Computacional

[![Instituição: UFU](https://img.shields.io/badge/UFU-FACOM-003366.svg)](https://facom.ufu.br/)
[![Programa: PET Sistemas de Informação](https://img.shields.io/badge/PET-Sistemas%20de%20Informa%C3%A7%C3%A3o-orange.svg)](https://www.petfacom.ufu.br/)
[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-Deep%20Learning-EE4C2C.svg)](https://pytorch.org/)
[![Status](https://img.shields.io/badge/Status-Em%20Andamento-brightgreen.svg)]()
[![Documentação](https://img.shields.io/badge/Documentos-Oficiais%20Anexados-success.svg)](#-documentações-e-anexos-do-projeto)

Projeto de pesquisa científica individual desenvolvido no âmbito do **Programa de Educação Tutorial (PET - Sistemas de Informação)** da **Faculdade de Computação (FACOM)** da **Universidade Federal de Uberlândia (UFU)**.

---

## 📌 Identificação do Projeto

- **Pesquisadora (Discente):** Isabela Dantas de Moraes ([isabeladantasms@ufu.br](mailto:isabeladantasms@ufu.br) / Matrícula: 12521BSI221)
- **Professor Orientador:** Prof. Dr. Marcelo Zanchetta do Nascimento ([marcelo.nascimento@ufu.br](mailto:marcelo.nascimento@ufu.br) / SIAPE: 1546624)
- **Tutor do Grupo PET:** Prof. Dr. Thiago Pirola Ribeiro
- **Unidade Acadêmica:** Faculdade de Computação (FACOM / UFU)
- **Duração do Projeto:** 12 meses (Novembro/2026 – Outubro/2027)

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
1. **Caracterização da Base de Dados:** Estruturar e analisar a base pública BRISC (imagens de meningioma, planos axial, sagital e coronal, distribuição de lesões e metadados).
2. **Pipeline de Pré-processamento:** Padronizar dimensões, normalizar intensidades e organizar partições de treino, validação e teste sem vazamento de dados (*data leakage* por paciente).
3. **Modelos de Linha de Base (*Baselines*):** Treinar uma rede professora de alta capacidade e uma rede aluna compacta sem destilação para servir de referência comparativa.
4. **Treinamento por Destilação de Conhecimento:** Implementar o treinamento da rede aluna destilada combinando:
   - Supervisão pelas máscaras anotadas (Cross-Entropy + Dice Loss);
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

### 1. Base de Dados
- **Dataset BRISC** (*Brain Tumor Segmentation and Classification dataset* - Fateh et al., 2025): 6.000 imagens de RM ponderadas em T1 com contraste, com anotações manuais validadas por especialistas médicos e radiologistas.
- **Recorte Temático:** Amostras focadas em **meningiomas**, abrangendo planos axial, coronal e sagital.

### 2. Destilação de Conhecimento (KD)
- Treinamento da **Rede Professora** para atingir desempenho de referência.
- Treinamento da **Rede Aluna sem Destilação** (controle).
- Treinamento da **Rede Aluna Destilada**:
  $$\mathcal{L}_{\text{total}} = \alpha \mathcal{L}_{\text{seg}}(y, \hat{y}_{\text{aluna}}) + (1 - \alpha) \mathcal{L}_{\text{KD}}(\hat{y}_{\text{aluna}}, \hat{y}_{\text{professora}}; T) + \beta \mathcal{L}_{\text{feat}}(F_{\text{aluna}}, F_{\text{professora}})$$
  onde $T$ é a temperatura de suavização e $\mathcal{L}_{\text{feat}}$ alinha representações intermediárias.

### 3. Avaliação de Eficiência em CPU
- Medição de tempo de inferência por imagem sob condições controladas (aquecimento prévio, registro de threads e isolamento de etapas de pré/pós-processamento).
- Análise de consumo de RAM e pegada de memória do modelo compilado/serializado.

### 4. Inteligência Artificial Explicável (XAI)
- Avaliação exploratória com **LIME** e mapas de ativação para verificar a coerência das decisões espaciais e identificar casos de falha/borda sem depender exclusivamente da métrica Dice.

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

Os documentos oficiais homologados e apresentados para este projeto encontram-se disponíveis no repositório na pasta [`docs/`](./docs/):

| Documento | Descrição | Link |
| :--- | :--- | :---: |
| 📄 **Plano de Trabalho PET** | Plano de trabalho detalhado de Pesquisa Individual PET aprovado pela UFU | [Visualizar PDF](./docs/plano_de_trabalho.pdf) |
| 📝 **Registro de Pesquisa DIREN** | Formulário oficial de Registro de Pesquisa Individual PET com assinaturas digitais (DIREN/PROGRAD/DIFDI) | [Visualizar PDF](./docs/registro_de_pesquisa.pdf) |
| 📊 **Pôster Científico (WTDCC 2026)** | Pôster acadêmico apresentado no Workshop de Teses e Dissertações em Ciência da Computação | [Visualizar PDF](./docs/poster_wtdcc_2026.pdf) |

---

## 📁 Estrutura do Repositório

```text
edge-meningioma-segmentation/
│
├── docs/                               # Documentações oficiais e acadêmicas
│   ├── plano_de_trabalho.pdf          # Plano de trabalho detalhado PET/UFU
│   ├── registro_de_pesquisa.pdf       # Termo de registro de pesquisa homologado
│   └── poster_wtdcc_2026.pdf          # Pôster acadêmico (WTDCC 2026)
│
├── data/                               # Instruções e scripts de preparação da base BRISC
├── notebooks/                          # Notebooks de experimentação e exploração
├── src/                                # Código-fonte do projeto
│   ├── data/                           # Dataloaders e transformações de pré-processamento
│   ├── models/                         # Arquiteturas (Professora, Aluna e Adaptadores)
│   ├── distillation/                   # Módulos de perda de destilação e alinhamento
│   ├── explainability/                 # Scripts de XAI (LIME e mapas de ativação)
│   └── utils/                          # Métricas (Dice, IoU, HD95) e benchmark de CPU
│
├── .gitignore                          # Regras de exclusão do Git
└── README.md                           # Apresentação do projeto e documentação
```

---

## 📚 Referências Bibliográficas

- **COLLI, B. O. et al.** Epidemiological features of meningiomas: a single Brazilian center’s experience with 993 cases. *Arquivos de Neuro-Psiquiatria*, v. 79, n. 8, 2021.
- **FATEH, A. et al.** BRISC: annotated dataset for brain tumor segmentation and classification with Swin-HAFNet. *arXiv preprint*, arXiv:2506.14318, 2025.
- **FREIRE, T.** SUS realiza seis em cada dez exames de imagem no Brasil, diz estudo. *Agência Brasil*, Reportagem sobre o Atlas da Radiologia no Brasil 2025 (Colégio Brasileiro de Radiologia e Diagnóstico por Imagem), 2025.
- **GOLDBRUNNER, R. et al.** EANO guideline on the diagnosis and management of meningiomas. *Neuro-Oncology*, v. 23, n. 11, p. 1821–1834, 2021.
- **HINTON, G.; VINYALS, O.; DEAN, J.** Distilling the knowledge in a neural network. *arXiv preprint*, arXiv:1503.02531, 2015.
- **ISLAM, M. S. et al.** Optimizing brain tumor classification for edge devices with knowledge distillation. *IEEE ICCIT*, 2025.
- **PRICE, M. et al.** CBTRUS statistical report: primary brain and other central nervous system tumors diagnosed in the United States in 2018–2022. *Neuro-Oncology*, v. 27, suppl. 4, 2025.
- **RAHMAN, M. A. et al.** FAKD-XAI: feature-aligned knowledge distillation with explainable AI for efficient brain tumor classification. In: *ICDSAIA*, Springer, 2025.
- **RIBEIRO, M. T.; SINGH, S.; GUESTRIN, C.** “Why should I trust you?”: explaining the predictions of any classifier. *ACM SIGKDD*, 2016.

---

## 🏛️ Apoio e Agradecimentos

Este trabalho é apoiado por:
- **PET Sistemas de Informação** — Universidade Federal de Uberlândia (UFU)
- **Faculdade de Computação (FACOM / UFU)**
- **Pró-Reitoria de Graduação (PROGRAD / UFU)**
- **Diretoria de Ensino (DIREN / UFU)**
- **Divisão de Formação Discente (DIFDI / UFU)**
