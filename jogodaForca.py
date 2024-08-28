import random

palavras = ["banana", "abacaxi", "morango", "laranja", "manga"]

palavra = random.choice(palavras)

tabuleiro = ["_"] * len(palavra)

tentativas = 6
letras_erradas = []

while tentativas > 0 and "_" in tabuleiro:
    print("\n" + " ".join(tabuleiro))
    print(f"Tentativas restantes: {tentativas}")
    print("Letras erradas: " + ", ".join(letras_erradas))
    
    letra = input("Adivinhe uma letra: ").lower()
    
    if letra in palavra:
        for i, l in enumerate(palavra):
            if l == letra:
                tabuleiro[i] = letra
    else:
        tentativas -= 1
        letras_erradas.append(letra)

if "_" not in tabuleiro:
    print("\nParabéns! Você adivinhou a palavra:", palavra)
else:
    print("\nVocê perdeu! A palavra era:", palavra)
