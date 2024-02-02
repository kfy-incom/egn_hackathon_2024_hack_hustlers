from src.dronning_chat import dronning_chat

questions = [
    "I hvilket år holdt Dronning Margrethe en nytårstale, der markerede hendes 40-års regeringsjubilæum?",
    "Hvem er Danmarks Dronning i februar 2024?",
    "Hvornår blev Kongens første datter født?",
    "Hvilke lande nævner Dronningen altid i sine nytårstaler?",
    "Udover dansk, hvilket andet sprog benytter Dronning Margrethe undertiden i sine nytårstaler?",
    "I hvilken nytårstale reflekterede Dronning Margrethe over tabet af Prins Henrik?",
    "Hvem kalder Dronningen for Søens Folk?",
]
for question in questions:
    print(question)
    answer = dronning_chat(question, [])
    print(answer)
