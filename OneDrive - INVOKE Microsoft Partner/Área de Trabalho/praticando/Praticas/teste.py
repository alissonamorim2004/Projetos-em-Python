#Nesse programa a pessoa que vai fazer a verificação precisa informar as temperaturas em °C e o código vai converter e fazer a média

# Criei a função para converter Celsius para FAH
def celsius_para_fahrenheit(celsius):
    return (celsius * 9/5) + 32

# Essa lista vai armazenar as temperaturas em Celsius das verificaçõe do inspetor
temperaturas_celsius = []

# Aqui vai rodar um loop onde o inspetor vai inserir as temperaturas em Celsius e serão adicionados na lista de temperaturas
for i in range(4):
    temp_celsius = float(input(f"Digite a temperatura medida (em °C) após {i*6} horas: "))
    temperaturas_celsius.append(temp_celsius)

#  Aqui vai rodar um loop que salva as temperaturas de Celsius dentro da variavel FAh e usa a função que faz a conversão, então já vai salvar em FAh
temperaturas_fahrenheit = [celsius_para_fahrenheit(temp) for temp in temperaturas_celsius]

# Calcular a média das temperaturas em Fahrenheit
media_fahrenheit = sum(temperaturas_fahrenheit) / len(temperaturas_fahrenheit)

# Exibir os resultados
print("\nTemperaturas em Fahrenheit:")
for i, temp in enumerate(temperaturas_fahrenheit):
    print(f"{i*6} horas: {temp:.2f} °F")

print(f"\nMédia das temperaturas: {media_fahrenheit:.2f} °F")
