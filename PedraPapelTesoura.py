import random

while True:
    opcoes = ['Pedra', 'Papel', 'Tesoura']

    escolha_usuario = input('Escolha pedra, Papel ou Tesoura: ')

    escolha_computador = random.choice(opcoes)

    if escolha_usuario == escolha_computador:
        print(f'Empate! Os dois escolheram {escolha_usuario}')

    elif (escolha_usuario == 'Pedra' and        escolha_computador == 'Tesoura') or \
        (escolha_usuario == "Papel" and escolha_computador == "Pedra") or \
        (escolha_usuario == "Tesoura" and escolha_computador == "Papel"):
        print(f"Voce perdeu! {escolha_usuario} vence computador {escolha_computador}.")
    else:
        print(f"Computador perdeu! Escolheu: {escolha_computador}, usuário  vence, escolhou: {escolha_usuario}")
        break