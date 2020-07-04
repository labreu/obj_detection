import pandas as pd
d = pd.read_csv('annotations/train_labels.csv')
x = pd.Series(d['class'].value_counts().index).reset_index().rename(columns={'index':'id', 0:'name'}).to_json(orient='records')
print(x.replace(',', '\n').replace('"', '').replace("{", "item {\n").replace("}", "\n}").replace("id", "    id").replace("name","    name").replace(':', ': '))


