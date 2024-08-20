#Módulos

# import math #importing bibliotec of math

# print(math.pow(2, 3)) # expoente

# numero = 7.6

# # resultado = math.ceil(numero)# round the number


# resultado = math.trunc(numero)# cut fractional number

# print(resultado)

# ----------------------------------------
# import os

# if os.path.exists("herois.txt"):
#     os.remove("herois.txt") #remove files
# else:
#     print("O arquivo n existe")
    
#for remove one folder, use rmdir


# ----------------
# use dir() to know all the things that have modules

# import math

# print(dir(math))
# =----------------------------

# import calculadora #creating your module

# print(calculadora.soma(10,30))
# print(calculadora.subtracao(30,10))


numeros = range(1,11)

for nun in numeros:
   if nun % 2 == 0:
       print(f"É par {nun}")
    