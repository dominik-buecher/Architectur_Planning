
Nachdem unser Modell das Grid komplett abgefahren ist, jedoch nicht gewartet hat wenn ein Schaf vor dem Mäher war, haben wir nochmal einen Versuch gestaret dieses Problem zu lösen.
Dafür haben wir dem Mäher einen Reward von 10 gegeben wenn er die action "Stay" (action = 4) nutz, also wartet wenn sich ein Schaf auf einem benachbartem Feld befindet. 
Außerdem haben wir unseren state Tensor erweitert so dass der Tensor nun 15x15x4 Dimensionen hat. Somit haben wir eine Schicht in dem Tensor die die Position des Mähers angibt, 
eine Schicht die die Position der Schafe angibt, eine Schicht die die Position des Ziels angibt und eine Schicht welche angibt ob ein feld schon besucht wurde oder nicht.
Mit diesen änderungen war es uns möglich ein Modell zu trainieren welches die Aufgabe nun richtig löst. Das heißt der Mäher wartet wenn sich ein Schaf vor ihm befindet,
und fährt nicht mehr gegen das Schaf. Erst wenn das Schaf wieder weg ist, fährt der Mäher weiter. 
Bis auf einzelne Situationen in denen der mäher ein paar Felder ausläst, fährt er genau wie das alte Modell alle Felder ab!

Die verschiedenen Modell sind in den jeweiligen Ordner zu finden, wobei das neuste Modell welches wartet in dem Ordner "Model_waiting_for_sheep" zu finden ist.


Informationen bezüglich den Modellen:

-Das Modell "dqn_model_nocow.pth" kann das gesamte Grid abfahren, jedoch ohne Schafe.

-Das Modell "dqn_model_5cow_overrun_sheep.pth" kann das gesamte grid abfahren auch wenn sich Schafe in dem Grid befinden. Jedoch fährt der Roboter gegen Kühe anstatt zu warten.

-Das Modell "dqn_model_5cow_5_waiting.pth" kann das gesamte grid abfahren auch wenn sich Schafe in dem Grid befinden. Dabei nutzt der Rasenmäher die action "Stay" falls, 
 ein Schaf im Weg ist. Das heißt der Mäher wartet bis das Schaf weg ist bevor er weiter fährt. Ab und zu kommt es vor das er ein paar Felder ausläst,
 wenn er auf ein Schaf getroffen ist, im normal Fall fährt er jedoch jedes fällt ab. 
 Um zu erkennen wann das Modell Wartet schauen Sie sich die ausgegebenen actions und rewards an, die action = 4 und der Reward = 10 falls der mäher wartet.









English version:

After our model completed traversing the grid but failed to wait when a sheep was in front of the mower, we made another attempt to solve this problem. 
For this purpose, we assigned a reward of 10 to the mower when it used the action "Stay" (action = 4), meaning it waits if a sheep is on an adjacent field. 
Additionally, we expanded our state tensor so that the tensor now has dimensions of 15x15x4. Thus, we have a layer in the tensor indicating the position of the mower, 
a layer indicating the position of the sheep, a layer indicating the position of the target, and a layer indicating whether a field has been visited or not.

With these changes, we were able to train a model that now solves the task correctly. This means the mower waits when a sheep is in front of it and no longer runs into the sheep. 
It only continues mowing once the sheep moves away. Apart from isolated situations where the mower skips a few fields, it mows all fields exactly like the old model!

The different models can be found in their respective folders, with the latest model that waits located in the "Model_waiting_for_sheep" folder.



Information regarding the models:

- The model "dqn_model_nocow.pth" can traverse the entire grid but without sheep.
- The model "dqn_model_5cow_overrun_sheep.pth" can traverse the entire grid even when there are sheep in the grid. However, the mower runs into cows instead of waiting.
- The model "dqn_model_5cow_5_waiting.pth" can traverse the entire grid even when there are sheep in the grid. In this model, the mower uses the action "Stay" if a sheep 
  is in the way. This means the mower waits until the sheep moves away before continuing. Occasionally, it skips a few fields when it encounters a sheep, but typically, 
  it mows every field. To identify when the model is waiting, look at the outputted actions and rewards, where action = 4 and the reward = 10 if the mower is waiting.