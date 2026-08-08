# Relatório preliminar de reprodução e auditoria

> Documento preservado como registro da etapa anterior à obtenção dos dados. A análise concluída está em `RELATORIO_FINAL.md`.

## 1. Materiais recebidos

- artigo científico publicado em 14 de maio de 2026;
- gravação de 9min20s com o professor Leonardo e os integrantes Renan, João e André;
- nenhum código, dado derivado, modelo treinado ou semente experimental.

O próprio artigo informa que os dados tratados serão disponibilizados “mediante solicitação razoável”. Isso impede uma reprodução bit a bit e torna necessário reconstruir decisões que não foram especificadas.

## 2. Método e resultados declarados

O estudo usa precipitação TRMM 3B42 V7 de 1998 a 2015 em 169 pontos (13x13), calcula SPI nas escalas 1, 3, 6, 9, 12, 18, 24 e 48 meses e compara 17 MLPs. ANN#n usa n anos anteriores como entrada; ANN#15, com 2000-2014, é escolhida para prever 2015. Todas as redes têm uma camada intermediária de 11 neurônios, sigmoide na camada oculta, saída linear e treinamento Levenberg-Marquardt.

O artigo declara uma divisão sequencial 70%/15%/15%, mas não define inequivocamente a unidade amostral. As Tabelas 2 e 3 apresentam R e kappa para todos os 169 pontos em cada mês de 2015. Os valores publicados foram preservados em `reference/article_table2_r.csv` e `reference/article_table3_kappa.csv`.

## 3. O que a reunião esclareceu

O professor orientou a equipe a:

1. reproduzir primeiro o procedimento descrito, mesmo com suas ambiguidades;
2. medir quanto do alto desempenho de SPI-24/SPI-48 decorre de meses já conhecidos;
3. esclarecer a divisão 70/15/15 e manter 2015 fora de treino/validação;
4. escolher arquitetura por validação, não por MSE de treino;
5. comparar outros regressores e técnicas espaciais antes de concluir que a MLP é superior.

A síntese revisada está em `outputs/transcricao/sintese_reuniao.md`.

## 4. Auditoria metodológica antes da execução real

| Ponto | Evidência atual | Classificação |
|---|---|---|
| Sobreposição das janelas SPI-k | SPI-48 de jan/2015 contém 47 meses anteriores à origem; em dez/2015 contém 36 | Propriedade matemática confirmada; não é vazamento por si só, mas favorece persistência |
| Treino/validação/teste 70/15/15 | Figuras 3-4 indicam anos como entradas e pares mês-ponto como amostras; o rótulo é 2015 | Forte suspeita de uso de rótulos de 2015 no treino; confirmação depende da reprodução |
| Escolha de ANN#15 | O artigo escolhe entre 17 modelos pelo desempenho das previsões de 2015 | Reutilização do teste para seleção de modelo confirmada pelo texto |
| Ajuste do SPI | Não há intervalo de calibração separado; os dados declarados vão até 2015 | Se a distribuição foi ajustada com 2015, há vazamento de transformação; não documentado |
| Escolha de 11 neurônios | 1-20 neurônios testados até o MSE de treino deixar de melhorar | Seleção inadequada/overfitting; deveria usar validação |
| Métricas nos 169 pontos | Tabelas mensais usam todos os pontos, embora 30% devessem ser validação/teste | Possível avaliação sobre amostras de treino |
| Dependência espacial | Pontos vizinhos de 0,25° são altamente relacionados; não há bloqueio espacial | Erro preditivo pode estar subestimado |
| Comprimento do registro | Apenas 18 anos para SPI até 48 meses | Limitação grave confirmada pelas recomendações da WMO |
| Reprodutibilidade | Sem código, semente, parâmetros do SPI ou dados tratados | Reprodução exata não garantida |

### Nota importante sobre SPI-48

O guia da World Meteorological Organization recomenda idealmente pelo menos 20-30 anos de dados mensais, preferindo 50-60 anos. Para escalas acima de 24 meses, alerta que mesmo 50-60 anos podem ser insuficientes para estimar as caudas e que 80-100 anos seriam necessários para maior confiança. O artigo usa 18 anos e ainda tenta empregar SPI-48 a partir de 2000, embora os 48 primeiros meses completos só existam no fim de 2001. Isso exige dado anterior não declarado, janela parcial ou imputação não documentada.

## 5. Protocolos implementados

### 5.1 Reconstrução paper-like

Para cada escala e número de anos de entrada:

- cada mês-ponto é uma amostra;
- o mesmo mês/ponto nos n anos anteriores forma os atributos;
- o valor correspondente em 2015 é o rótulo;
- as amostras são divididas sequencialmente em 70/15/15;
- uma MLP sigmoide/linear com 11 neurônios é treinada;
- R, R², kappa, MAE e RMSE são calculados no treino, validação, teste e conjunto completo.

Essa interpretação é a mais compatível com a dimensionalidade mostrada nas Figuras 3 e 4. Se ela reproduzir as tabelas apenas quando as métricas incluem treino, o vazamento fica demonstrado.

### 5.2 Protocolo causal

- ajuste da distribuição do SPI: somente 1998-2013;
- treino: apenas origens cujos 12 alvos terminam até dezembro de 2013;
- validação: origem dezembro de 2013, alvos de 2014;
- teste: origem dezembro de 2014, alvos de 2015;
- teste usado uma única vez;
- comparação com persistência do último mês e persistência sazonal;
- seleção de Ridge, MLP ou Extra Trees pela RMSE de 2014.

## 6. Técnicas de PDI e deep learning implementadas

Os campos 13x13 são preservados como imagens, evitando o achatamento prematuro que elimina vizinhança. A CNN espacial:

- recebe os 12 mapas anteriores como 12 canais;
- usa convoluções 3x3 para explorar dependência local;
- aprende um resíduo em relação ao último mapa, em vez de reconstruir todo o campo do zero;
- combina MSE com perda de gradiente espacial para preservar transições e extremos;
- escolhe a época por validação em 2014;
- prevê 2015 recursivamente sem observar meses de 2015.

O modelo só será considerado melhoria se superar persistências e modelos tabulares em RMSE/MAE sem degradar kappa e desempenho de extremos.

## 7. Critério para concluir se houve vazamento

O diagnóstico não será baseado apenas em uma queda de desempenho. Será considerado **confirmado** se ocorrer pelo menos uma das condições abaixo:

1. os valores publicados só forem reproduzidos quando rótulos de 2015 entram no ajuste 70/15/15;
2. as métricas publicadas coincidirem com o conjunto completo, mas não com a partição de teste;
3. ANN#15 tiver sido selecionada pelo próprio 2015 e perder a vantagem quando a seleção for feita em 2014;
4. o SPI global (ajustado até 2015) for materialmente melhor que o SPI calibrado apenas no passado.

Se nenhuma condição ocorrer, a sobreposição de janelas será tratada como persistência inerente ao alvo, e o foco passará às demais limitações: registro curto, ausência de baselines, dependência espacial, hiperparâmetros escolhidos incorretamente e falta de incerteza.

## 8. Estado de validação

Os testes automatizados cobrem:

- 47/48 meses conhecidos no SPI-48 de janeiro de 2015;
- separação das origens anuais de treino, validação e teste;
- estabilidade do SPI de treino quando apenas 2014-2015 são alterados;
- dimensões da reconstrução paper-like;
- preservação da forma 13x13 pela CNN.

Resultado atual: **6 testes aprovados**. A execução com dados reais aguarda somente a exportação autenticada do TRMM.

## 9. Fontes metodológicas adicionais

- World Meteorological Organization. *Standardized Precipitation Index User Guide*, WMO-No. 1090, 2012.
- Roberts et al. (2017). *Cross-validation strategies for data with temporal, spatial, hierarchical, or phylogenetic structure*. Ecography 40:913-929. DOI 10.1111/ecog.02881.
- Kapoor e Narayanan (2023). *Leakage and the reproducibility crisis in machine-learning-based science*. Patterns 4(9):100804.
