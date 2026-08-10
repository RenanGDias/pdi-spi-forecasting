# Reprodução e auditoria de previsão espacial do SPI

Este repositório reproduz e audita o artigo **“Artificial neural network-based forecasting of SPI for extreme event analysis in the Upper São Francisco sub-basin”** (Santos et al., 2026). O objetivo é separar três perguntas:

1. É possível obter números próximos aos publicados seguindo a descrição do artigo?
2. Quais resultados sobrevivem quando 2015 é mantido totalmente fora de treino, transformação, validação e seleção de modelo?
3. Técnicas espaciais de PDI e modelos mais adequados melhoram o desempenho contra baselines simples em uma comparação causal?

## Resultado concluído

O diagnóstico final encontrou vazamento no protocolo descrito pelo artigo. A seleção da ANN#15 entre 17 modelos não se reproduz: a métrica usada pelo artigo cresce monotonicamente com o número de anos de entrada, porque é calculada sobre os 169 pontos da grade e 70% deles estão no treino. Sob protocolo causal, o R médio cai de 0,82 para 0,26–0,37 e o kappa colapsa de 0,70 para cerca de 0,07. A CNN espacial causal melhorou o RMSE frente ao regressor selecionado em 6 das 8 escalas, mas persistência permaneceu competitiva. Consulte `RELATORIO_FINAL.md` para a conclusão, evidências, figuras e limitações.

## Conteúdo principal

- `outputs/transcricao/`: transcrição automática e síntese revisada da conversa com o professor Leonardo.
- `reference/`: valores de R e kappa publicados nas Tabelas 2 e 3.
- `scripts/gee_export_trmm.js`: exportação da grade mensal TRMM/3B42 V7 (13x13, 1998-2015).
- `scripts/run_audit.py`: reprodução paper-like, auditoria causal, baselines e modelos melhorados.
- `scripts/analyze_paper_orderings.py`: testa a divisão 70/15/15 ordenada por ponto espacial.
- `scripts/run_paper_claims.py`: coloca à prova as afirmações do artigo — a varredura ANN#1–#17, o efeito do early stopping declarado e quanto da correlação a sobreposição de janelas explica.
- `scripts/run_deep_only.py`: executa a CNN espacial causal sem repetir as ANNs do artigo.
- `scripts/build_final_plots.py`: gera as figuras finais da auditoria.
- `src/spi_audit/`: cálculo do SPI, desenhos amostrais, métricas, modelos e CNN espacial.
- `configs/article.yaml`: todas as decisões declaradas pelo artigo e pelo protocolo corrigido.
- `tests/`: testes automatizados para datas de corte, janelas de SPI e dimensões dos dados/modelos.

## Preparação

No PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m pip install --no-build-isolation -e .
.\.venv\Scripts\python.exe -m pytest -q
```

## Obtenção do TRMM

O artigo usa o produto de três horas `TRMM/3B42`, versão 7. No Earth Engine Code Editor, execute `scripts/gee_export_trmm.js`. O script:

- soma `precipitation × 3 h` em cada mês;
- amostra as 169 coordenadas de 0,25° declaradas pelo artigo;
- produz 36.504 linhas (216 meses × 169 pontos);
- imprime um link direto e também prepara a exportação de `trmm_3b42_monthly_1998_2015_upper_sao_francisco.csv`.

Coloque o CSV em `data/raw/`.

O CSV já obtido foi validado com 36.504 linhas, 216 meses, 169 pontos, zero ausências e zero duplicações.

## Execução

Auditoria principal, primeiro com ANN#15:

```powershell
.\.venv\Scripts\python.exe scripts\run_audit.py `
  data\raw\trmm_3b42_monthly_1998_2015_upper_sao_francisco.csv
```

Varredura dos 17 comprimentos de entrada e CNN espacial:

```powershell
.\.venv\Scripts\python.exe scripts\run_audit.py `
  data\raw\trmm_3b42_monthly_1998_2015_upper_sao_francisco.csv `
  --all-paper-models --deep
```

## Protocolos comparados

### Paper-like

- SPI calibrado em 1998-2015, incluindo o ano-alvo.
- Janelas iniciais parciais são aceitas para testar a única forma de usar SPI-48 já em 2000 com dados iniciados em 1998.
- Anos anteriores são atributos e combinações mês-ponto são amostras, interpretação compatível com as Figuras 3 e 4.
- Divisão sequencial 70/15/15 e MLP sigmoide com 11 neurônios.
- Métricas são emitidas separadamente para treino, validação, teste e conjunto completo.

Este cenário não é recomendado; ele existe para descobrir se reproduz as tabelas.

### Causal corrigido

- Distribuição do SPI ajustada somente até 2013.
- Treino contém apenas previsões cujo alvo termina até dezembro de 2013.
- O ano de 2014 é usado para validação e seleção.
- A origem final é dezembro de 2014 e os 12 meses de 2015 são usados uma única vez como teste.
- Pré-processadores são ajustados apenas no treino.
- Modelos são comparados contra persistência simples e persistência sazonal.

## Interpretação do vazamento

A sobreposição de meses em um índice acumulado não é, isoladamente, prova de vazamento: meses anteriores à origem são informação causalmente disponível. Ela torna escalas longas previsíveis por persistência e exige que o horizonte seja declarado com precisão. Há vazamento quando dados/rótulos de 2015 influenciam o ajuste do SPI, o treinamento, a escolha de ANN#15/11 neurônios ou qualquer pré-processamento.

Duas sobreposições distintas aparecem no projeto e não devem ser confundidas. `known_window_fraction` mede quanto da janela do alvo já é conhecido em uma data de origem, e vale para o protocolo causal. `annual_lag_overlap` mede quanto a janela do alvo compartilha com a do mesmo mês em um ano anterior, `max(0, escala − 12) / escala`, e é a que vale para o desenho do artigo, cujos atributos são o mesmo mês em anos anteriores. Pela segunda medida, escalas de até 12 meses não têm sobreposição alguma.

O diagnóstico final é feito pela diferença entre:

- métricas paper-like no conjunto completo e no teste real de 15%;
- desempenho do modelo selecionado em 2015 versus validação em 2014;
- modelos aprendidos versus persistência;
- calibração global do SPI versus calibração encerrada em 2013.

## Reprodutibilidade

- Todas as datas de corte estão em `configs/article.yaml`.
- Sementes aleatórias são fixadas.
- As tabelas publicadas foram transcritas para CSV e não são usadas no treinamento.
- A execução salva SPI global e causal separadamente, além de métricas por mês, escala, modelo e subconjunto.
