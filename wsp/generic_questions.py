"""Jeu de RÉFÉRENCE de questions Oui/Non générales (hors tâche).

Sert uniquement à calculer m_g, l'état final moyen sur des questions Oui/Non quelconques,
pour que l'édition de lecture n'ait pas d'effet moyen hors de la tâche. Ces questions sont
distinctes des 30 questions factuelles d'évaluation (wsp/side_effects.py). Équilibré Oui/Non,
plusieurs langues et thèmes ; les réponses ne sont pas utilisées.
"""
REF = [
    "Rome est-elle en Italie ?", "Le lion est-il un félin ?", "Un hexagone a-t-il six côtés ?",
    "Le café contient-il de la caféine ?", "La Seine traverse-t-elle Paris ?", "Mars est-elle une planète ?",
    "Is London the capital of the United Kingdom?", "Do bees make honey?", "Is ice made of water?",
    "¿Es el Amazonas un río?", "¿Tiene un año doce meses?", "¿Es el español una lengua?",
    "Le vélo a-t-il deux roues ?", "L'or est-il un métal ?", "Mozart était-il compositeur ?",
    "La pluie est-elle de l'eau ?", "Les oiseaux ont-ils des plumes ?", "Le Canada est-il en Amérique ?",
    "Faut-il de l'air pour respirer ?", "Le chocolat est-il fait à partir de cacao ?",
    "Lyon est-elle la capitale de la Belgique ?", "Les vaches pondent-elles des œufs ?",
    "Un cercle a-t-il des angles ?", "Le plomb flotte-t-il sur l'eau ?", "Le désert du Sahara est-il en Asie ?",
    "Mercure est-elle la planète la plus éloignée du Soleil ?", "Is the Pacific a desert?",
    "Do fish have fur?", "Is 10 smaller than 3?", "¿Es Lisboa la capital de Francia?",
    "¿Vuelan los elefantes?", "¿Es la nieve caliente?", "Les tomates poussent-elles sous la mer ?",
    "Napoléon était-il japonais ?", "Une heure compte-t-elle 100 minutes ?", "Le citron est-il sucré ?",
    "Les serpents ont-ils des jambes ?", "Le Mont Blanc est-il en Australie ?",
    "Un piano est-il un animal ?", "La glace brûle-t-elle ?",
]


def ref_prompt(tok, q):
    return tok.apply_chat_template([{"role": "user", "content": q + " Réponds Oui ou Non."}], tokenize=False,
                                   add_generation_prompt=True, enable_thinking=False)
