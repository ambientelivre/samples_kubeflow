```python
!pip3 install kfp --upgrade
```

```python
import kfp
import kfp.components as comp
import kfp.dsl as dsl
```


```python
def treina(feat:str, label:str,file:str):
    import pandas
    from sklearn.linear_model import LinearRegression
    df = pandas.read_csv(file)
    reglin = LinearRegression()
    reglin.fit(df[[feat]], df[label])
    return (reglin.coef_[0], reglin.intercept_)
```


```python
comp_treina = comp.create_component_from_func(treina,output_component_file='treina_component.yaml',base_image='tensorflow/tensorflow:1.11.0-py3')
```


```python
def valida (rl: (float,float),feat:str, label:str, file:str) -> float :
    import pandas
    from sklearn.linear_model import LinearRegression
    df = pandas.read_csv(file)
    reglin = LinearRegression()
    reglin.fit(df[[feat]], df[label])
    r2 = reglin.score(df[[feat]], df[label])
    return r2
    
```


```python
comp_valida = comp.create_component_from_func(valida,output_component_file='valida_component.yaml',base_image='tensorflow/tensorflow:1.11.0-py3')
```


```python
def previsao(r2:float, feat:str, label:str, valor:float, file:str) -> float:
    import pandas
    from sklearn.linear_model import LinearRegression
    df = pandas.read_csv(file)
    reglin = LinearRegression()
    reglin.fit(df[[feat]], df[label])
    future = pandas.DataFrame({feat:[valor]})
    return reglin.predict(future)[0]
    
```


```python
comp_previsao = comp.create_component_from_func(previsao,output_component_file='previsao_component.yaml',base_image='tensorflow/tensorflow:1.11.0-py3')
```


```python
@dsl.pipeline(name='Pipeline Data',description='Dados Bolsa X Dolar')
def pipe(feat, label, valor, file):
    task_treina   = comp_treina(feat,label,file)
    task_valida   = comp_valida(task_treina.output,feat,label,file)
    task_previsao = comp_previsao(task_valida.output,feat,label,valor,file)
```


```python
client = kfp.Client(host='http://localhost:8080/pipeline')
FILE = 'https://raw.githubusercontent.com/ambientelivre/samples_kubeflow/main/datasets/bolsa.csv'
args = {'feat':'bolsa','label':'usd','file':FILE,'valor':'100'}
##print(client.list_experiments())
client.create_run_from_pipeline_func(pipe,arguments=args)
```
