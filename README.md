# Yolo Detector Java

Implementação do algoritmo YOLO de detecção de objetos para imagens, em Java.

Compilado para Java 9 (major version: 53)

## Manual de Uso

O diretório 'out/artifacts/YOLO_Java_jar' contém o arquivo .jar correspondente ao servidor com o algoritmo de detecção YOLO, acompanhando de arquivos .bat para iniciá-lo.

Após a inicialização do servidor, esse é associado à porta 8080 para o recebimento dos seguintes tipos de requisições:


### HTTP - GET

Tal método funciona para uso local do servidor, pois envolve passar o diretório da imagem via uma requisição HTTP (URLs não funcionam!!); não é necessário adicionar autenticação ao cabeçalho. A URL para a requisção tem o formato:

`localhost:8080/Detect?path=[CAMINHO DA IMAGEM]&thresh=[THRESHOLD]`

* CAMINHO DA IMAGEM - diretório absoluto do local da imagem a ser analisada; pode-se usar tanto barras normais quanto barras invertidas;
* THRESHOLD - valor entre opcional entre 0 e 1 (padrão = 0.5) para determinar qual o nível mínimo de confiança para aceitar uma detecção de objeto

### HTTP - POST

