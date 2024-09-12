# Desafio - Fase 3

Executar o fine-tuning de um foundation model (Llama, BERT, MINSTREL etc.), utilizando o dataset "The AmazonTitles-1.3MM".

- Receber perguntas com um contexto obtido por meio do arquivo json “trn.json” que está contido dentro do dataset.
- A partir do prompt formado pela pergunta do usuário sobre o título do produto, o modelo deverá gerar uma resposta baseada na pergunta do usuário trazendo como resultado do aprendizado do fine-tuning os dados da sua descrição.


## Como Rodar

1. Utilize o notebook `download_dataset.ipynb` para baixar o dataset original. O arquivo deve ficar salvo na pasta `/datasets`.
2. Execute as células de pré-processamento em `pre_processing.ipynb`. Um arquivo `.csv` será criado com dados limpos e prontos para serem usados.