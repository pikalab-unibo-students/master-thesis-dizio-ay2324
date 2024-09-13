import fairlib as fl

df = fl.DataFrame({
    'name': ['Alice', 'Bob', 'Carla', 'Davide', 'Elena'],
    'age': [25, 32, 45, 29, 34],
    'sex': ['F', 'M', 'F', 'M', 'F'],
    'income': ['<40000', '40000..49999', '50000..59999', '60000..69999', '>=70000']
})

print(df)
df.targets = 'income'
print(df.targets) # {'income'}

df.sensitive = {'age', 'sex'}
print(df.sensitive) # {'age', 'sex'}

try:
    df.sensitive = {'age', 'sex', 'missing'}
except Exception as e:
    print(e) # Column missing not found

df2 = df.drop(['name'], axis=1) # random operation creating another DataFrame. are attributes preserved? yes
print(df2.targets) # {'income'}
print(df2.sensitive) # {'age', 'sex'}


df3 = df.drop(['sex'], axis=1) # random operation creating another DataFrame. what if the operation changes the columns?
print(df3.targets) # {'income'}
print(df3.sensitive) # {'age'}
