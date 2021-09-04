# Road Signs Recognition on Raspberry Pi

**W celu uruchomienia detekcji konieczne jest rozpakowanie archiwum "wagi" i umieszczenie znajdującego się w nim pliku z rozszerzeniem .weights w głównym katalogu**

---

Rozpoznawanie znaków drogowych w czasie rzeczywistym na Rasbperry Pi z obrazu kamery internetowej

<p>
    <img src="/resources/device.jpg" width="50%" height=auto />
</p>

Przy projekcie wykorzystano:

- Raspberry Pi 4 Model B (4GB RAM)
- Kamera internetowa Logitech C920
- Tensorflow 2 + Keras API
- Google Colaboratory (do trenowania modeli)
- [Darknet](https://github.com/AlexeyAB/darknet) (DNN framework w C oraz CUDA)

## Dotychczasowe rezultaty:

#### Film w serwisie YouTube (wymaga kliknięcia w miniaturę):

[![Film na YouTube](https://img.youtube.com/vi/3R9dNx7FXng/0.jpg)](https://www.youtube.com/watch?v=3R9dNx7FXng)

#### Zrzut ekranu przedstawiający detekcję:

<p>
    <img src="/resources/screenshot.png" width="50%" height=auto />
</p>

---

## Struktura kodu:

- **CNN.ipynb** - notatnik zawierający trenowanie klasyfikatora CNN
- **YOLOv4.ipynb** - notatnik zawierający trenowanie sieci YOLOv4
- **Detect.ipynb** - notatnik służący do detekcji znaków drogowych
- **webcam.py** - skrypt umożliwiający detekcję oraz informujący głosowo o wykrytych znakach
- **labels.csv** - plik zawierający ID oraz odpowiadające im nazwy klas
- **Folder "Test Dataset"** - zawiera zdjęcia wykorzystane do oceny skuteczności
- **Folder "Test Dataset after detection"** - zawiera zdjęcia po uruchomieniu na nich detekcji

Pliki CNN.ipynb oraz YOLOv4.ipynb przeznaczone są do wgrania do platformy Google Colaboratory.

Pliki **Detect.ipynb** oraz **webcam.py** przeznaczone są do lokalnego uruchomienia na Raspberry Pi, jednak notatnik **Detect.ipynb** może być bezproblemowo uruchomiony również w środowisku Google Colaboratory. W tym celu oprócz wgrania go do Google Colaboratory, wymagane jest dodanie i uruchomienie na początku notatnika następujących dwóch instrukcji:

```
!git clone https://github.com/nexif/Road-Signs-Recognition.git darknet_for_colab
%cd darknet_for_colab
```
