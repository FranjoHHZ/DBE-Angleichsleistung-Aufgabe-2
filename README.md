# DBE-Angleichungsmodul-Aufgabe2

# Ziele der Aufgabe:
- Implementierung von ML basierten SW Systemen
- Betrieb und Ablauf von ML basierter Software transparent gestalten
- Automatisches Testen von ML basierten SW Systemen

# Ausführen MyBinder
1.Öffnen Sie den Link zu Binder, den Sie im Binder-Badge in der README-Datei finden. Dadurch wird eine Binder-Umgebung gestartet.

2.Starten Sie ein neues Terminal innerhalb der Binder-Umgebung.

3.Geben Sie den folgenden Befehl ein:

      pip install -r requirements.txt
   
4.Geben Sie den folgenden Befehl ein, um das Hauptskript auszuführen:

      python main_runtime.py
   
5.Geben Sie den folgenden Befehl ein, um die Unit-Tests auszuführen:

      python -m unittest test_runtime.py
      

[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/FranjoHHZ/DBE-Angleichsleistung-Aufgabe-2/HEAD)



# Ausführen Colabs

Klicken Sie auf den folgenden Badge, um das Projekt direkt auf Google Colab zu öffnen. Bitte führen Sie dort durch das Drücken "Shift"+ "Enter" die Befehle aus.


[![Open in Google Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/FranjoHHZ/DBE-Angleichsleistung-Aufgabe-2/blob/main/run_script.ipynb)

### Manuell:

1.

      !git clone https://github.com/FranjoHHZ/DBE-Angleichsleistung-Aufgabe-2.git
      %cd DBE-Angleichsleistung-Aufgabe-2

2.

      !pip install -r requirements.txt

3.

       !python main_runtime.py

4.

  
      !python -m unittest test_runtime.py
  




# Erwartete Ergebnis:
Ergebnis nach der Ausführung "main_runtime.py":

![image](https://github.com/user-attachments/assets/fa4bddec-aa21-463b-aeb2-784177d82353)


Ergebnis nach der Ausführung der "test_runtime.py":

![image](https://github.com/user-attachments/assets/a0b46db7-e888-438d-b736-d705d1768552)




Die Ergebnisse, wie die Confusion-Matrix sowie die Accuracies, sind nochmals im Ordner "Data" zu finden.

 




