# 1. Generierung antiker Münzbilder mittels Few-Shot GANs (FSGANs)

In dieser Arbeit wurde der Few-shot-GAN genutzt: [Few-shot-GAN GitHub Repository](https://github.com/e-271/few-shot-gan/tree/master). Dieser erfordert TensorFlow Version 1.14 oder 1.15. Diese älteren TensorFlow-Versionen können möglicherweise nicht kompatibel mit neueren Nvidia-Grafikkarten sein. Daher wird Docker empfohlen: [Docker Dokumentation](https://docs.docker.com).

Das folgende Docker Image kann für die Ausführung genutzt werden: [Nvidia TensorFlow Release Notes 20-11](https://docs.nvidia.com/deeplearning/frameworks/tensorflow-release-notes/rel_20-11.html#rel_20-11). Das Docker Image kann über den Dockerfile erstellt werden. Für GPU-Unterstützung muss eventuell das Nvidia Container Toolkit installiert werden: [Nvidia Container Toolkit Installationsanleitung](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html).

Es ist wichtig zu beachten, dass dies möglicherweise nicht mit anderen Grafikkarten kompatibel ist. Diese Arbeit wurde mit einer Nvidia RTX 3090 auf einem Ubuntu-Rechner durchgeführt.

Der Nutzer muss sich für die folgenden Schritte im Pfad des fsgan befinden.

## Vorbereitung der Daten

Die Trainingsdatensätze müssen in TFDS Records umgewandelt werden. Dies kann mit dem folgenden Befehl von der Konsole aus gemacht werden:

    python dataset_tool.py create_from_images /path/to/target/tfds /path/to/source/folder --resolution 256

Alternativ kann das Bash-Skript `processData.sh` genutzt werden, welches die TFRecords für alle Trainingsordner erstellt, die in einem Überordner gespeichert sind:

    bash processData.sh /pathTo/FolderWithAllTrainingset /pathTo/tfds/FolderWhereTheTfdsSetsAreSaved


## Training eines FSGAN

Das Training wird mit folgendem Befehl von der Konsole aus gestartet:

    python run_training.py --config=config-ada-sv-flat --result-dir=results/whereTheResultsAreSaved/ --data-dir=tfds/ --dataset-train=train/TrainingSet/ --dataset-eval=train/EvaluationSet/ --resume-pkl="stylegan2-coin-config-f-2800kimg.pkl" --total-kimg=100 --metrics=None --img-ticks=10 --net-ticks=20


Für den FSGAN muss `config-ada-sv-flat` genutzt werden. Wenn kein Evaluationsset vorhanden ist, wird `dataset-eval` standardmäßig auf das Trainingsset gesetzt. `stylegan2-ffhq-config-f.pkl` ist ein trainiertes StyleGAN2-Modell auf dem gesamten Münzdatensatz. `total-kimg` gibt die Trainingslänge an. `img-ticks` gibt an, wie oft Fake-Bilder während des Trainings generiert werden sollen. `net-ticks` gibt an, wie oft ein Zwischenergebnis gespeichert werden soll.

Alternativ kann das Skript `startTrainingFSGAN.sh` genutzt werden. Dies führt mehrere Trainingsvorgänge hintereinander aus, für alle Unterordner im angegebenen Pfad. Befehl:
    
    bash startTrainingFSGAN.sh pathTo/tfds/trainset pathTo/resultDirectory 80


`trainset` enthält alle Trainingsdatensätze, für die das Training durchgeführt wird. `resultDirectory` gibt an, wo die Ergebnisse gespeichert werden sollen. Für jedes Trainingsset wird automatisch ein Ordner in `resultDirectory` erstellt.

## Generieren von Bildern

Mit folgendem Befehl können Bilder generiert werden:

    python run_generator.py generate-images --network=pathToNetwork --seeds=0-100 --result-dir=pathToSaveFolder


# 2. Training der Base Learner (CNNs) für das Ensemble

Für das Training der CNNs wurde folgendes Docker Image benutzt: [TensorFlow Docker Image](https://hub.docker.com/layers/tensorflow/tensorflow/2.10.0-gpu/images/sha256-3aeb6a5489ad8221d79ab50ec09e0b09afc483dfdb4b868ea38cfb9335269049?context=explore).

Für das Training wird die Datei `cnnTraining.py` benutzt. Das Training wird mit vortrainierten Gewichten ausgeführt. Die Datei `evaluationData.csv` muss existieren, da hier die Ergebnisse gespeichert werden. Die Gewichte stammen von TensorFlow. Ein Training kann von der Konsole wie folgt gestartet werden:

    python3 cnnTraining.py side='SeiteDerMuenze' path/to/trainset path/to/testset path/to/evalset batch_size model='ResNet50V2' 'SaveNameForCNN' 'FreezeUntilLayer' path/to/secondEvalset --augmentation


`path/to/secondEvalset` und `--augmentation` sind optionale Argumente. Wenn die Flag `--augmentation` gesetzt ist, wird Data Augmentation auf die Trainingsdaten angewendet. `FreezeUntilLayer` gibt an, bis zu welchem Layer die Gewichte nicht neu trainiert werden sollen (Default: `Conv4` für die ResNets). `model='ResNet50V2'` gibt an, welches Modell trainiert werden soll. ResNet50V2, ResNet101V2, ResNet152V2 und VGG16 sind unterstützt.

Alternativ können mehrere Trainings mit Hilfe des Skripts `startTrainingCNN.sh` durchgeführt werden. Dafür müssen die gewünschten Trainingsparameter im Abschnitt "Configurations" des Skriptes angegeben werden. Ausführung:

    bash startTrainingCNN.sh NumOfRepeats

`NumOfRepeats` gibt an, wie oft das Training für eine Konfiguration wiederholt werden soll.


# 3. Vorhersagen für die einzelnen CNNs erstellen

Für die Erstellung der Predictions für das Ensemble Learning wird ModelEvaluator.py verwendet:

    python3 ModelEvaluator.py path/to/model path/to/data 64

Durch Setzen der --eval-Flag können die Confusion Matrix und die falschen Vorhersagen erstellt werden.

Alternativ kann das Skript getPredictions.sh genutzt werden, um mehrere Predictions gleichzeitig zu erstellen. Das Skript erstellt für jedes CNN in path/to/folderWithAllCnns/ alle Predictions für die Datensätze in path/to/folderWithAllEvalsets/:

    bash getPredictions.sh path/to/folderWithAllCnns/ path/to/folderWithAllEvalsets/

# 5. Ensemble Learning zur finalen Klassifizierung von Münztypen 
Zum Ausführen des Ensemble Learnings wird die
Ensemble.py Datei benötigt.
Der Nutzer kann es im Terminal starten, wobei er im Pfad der Ensemble.py Datei sein muss.

Die Eingabe ist allgemein wie folgt:
##
        python3 Ensemble.py Path/To/JSON_File.json Path/To/Dictionary_Pickle.pkl voting_or_stacking_function Top-X-value --eval
Das setzen von `--eval` wird für die Evaluation benötigt. Mit diesem Flag werden die Ergebnisse ausgegeben und die Konfusionsmatrizen und CSV-Dateien gespeichert.

Mit den JSON-Dateien kann der Nutzer der Ensemble-Klasse entscheiden, welche Netzwerke (CNNs) ein Ensemble bilden sollen.
In diesen Dateien sind für jedes Netzwerk die Pfade zu den Vorhersagen zu den Test- und Validierungsdaten enthalten.
Die Datei hat im allgemeinen folgende Struktur:
```json
{
"CNN_1":"Path_to_Predictions_on_Test_Data/Test_Prediction_File.npy,Path_to_Predictions_on_Validation_Data/Validation_Prediction_File.npy",
"CNN_2":"Path_to_Predictions_on_Test_Data/Test_Prediction_File.npy,Path_to_Predictions_on_Validation_Data/Validation_Prediction_File.npy",
"CNN_3":"Path_to_Predictions_on_Test_Data/Test_Prediction_File.npy,Path_to_Predictions_on_Validation_Data/Validation_Prediction_File.npy"
}
```
Anmerkung: In diesem Fall würde das Ensemble aus drei Netwerken bestehen.

