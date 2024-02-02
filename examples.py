import src.app_interface as app_interface

if __name__ == "__main__":

    questions = [
        "I hvilket år holdt Dronning Margrethe en nytårstale, der markerede hendes 40-års regeringsjubilæum?",
        "Hvem er Danmarks Dronning i februar 2024?",
        "Hvornår blev Kongens første datter født?",
        "Hvilke lande nævner Dronningen altid i sine nytårstaler?",
        "Udover dansk, hvilket andet sprog benytter Dronning Margrethe undertiden i sine nytårstaler?",
        "I hvilken nytårstale reflekterede Dronning Margrethe over tabet af Prins Henrik?",
        "Hvem kalder Dronningen for Søens Folk?"
    ]
    
    # Test question answering
    #question = "Hvad var den største overraskelse i talen fra 2023?"
    #question = "Hvornår blev Kongens første datter født?"
    for q in questions:
        answer = app_interface.combine_results(q, None)
        print(answer)