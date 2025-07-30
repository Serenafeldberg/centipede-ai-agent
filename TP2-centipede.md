# Trabajo Práctico 2: Agente de Centipede

## Introducción

[Centipede](https://gymnasium.farama.org/environments/atari/centipede/) es un juego de video desarrollado por Atari en el que interpretas a un elfo en un bosque encantado lleno de hongos. Tu objetivo es proteger estos hongos de las amenazas de arañas, pulgas y ciempiés utilizando tus varitas mágicas. Si alguno de estos enemigos te muerde, quedarás temporalmente paralizado y perderás una varita. El juego termina una vez que se han perdido todas las varitas, aunque puedes obtener varitas adicionales acumulando suficientes puntos.

![CentipedeGame](/assets/centipede.gif)

## Objetivo

El propósito de este trabajo práctico es desarrollar un agente inteligente capaz de jugar al Centipede. Se evaluará la eficacia del agente en términos de su rendimiento durante las partidas de prueba y su habilidad para implementar estrategias de juego avanzadas.

## Forma de trabajo

La forma de trabajo para este trabajo práctico será la siguiente:

1. Hacer un fork del repositorio del trabajo práctico en GitHub. Esto creará una copia del repositorio en tu cuenta de GitHub.
1. Clonar el repositorio forked en tu entorno local.
1. Desarrollar el código del agente en tu entorno local.
1. Realiza commits y pushs frecuentes a tu repositorio forked para mantener un registro de tus cambios.
1. Cuando hayas terminado, realiza un pull request desde tu repositorio forked hacia el repositorio original. Esto enviará tu código para su revisión y evaluación.

**Nota:** La rama que se intente _mergear_ mediante el pull request debe contener exclusivamente el código fuente del agente y ningún otro archivo más. Cualquier _merge conflict_ o archivo extra se calificará de forma negativa.

## Entregables

El entregable para cada fase será el código del agente desarrollado en Python 3.11. Asegúrate de que tu código esté bien comentado y sea fácil de entender. No es necesario un informe, pero se espera que cualquier decisión de diseño o implementación importante esté claramente explicada en los docstrings y comentarios del código.

### Checkpoint 0: Extracción del estado del juego

Para el checkpoint, es necesario demostrar que puedes extraer el estado del juego a partir de las observaciones. Deberás crear un método que tome como entrada la observación del juego (una imagen) y devuelva una representación compacta del estado del juego. Esta representación debería incluir, al menos, la posición de todos los elementos del juego.

El código debe ser utilizado de la siguiente manera:

```python
import gymnasium as gym
from agents.turing.state_extractor import StateExtractor

extractor = StateExtractor()
env = gym.make('ALE/Centipede-v5', render_mode='human')
env.reset()

action = env.action_space.sample()
obs, reward, terminated, truncated, info = env.step(action)
state = extractor.extract(obs)
```

**Fecha de entrega**: Domingo 22 de Octubre a las 23:59

### Entrega Final: Agente Inteligente

Para la entrega final, se espera un agente que muestre un comportamiento inteligente. Tu agente debería ser capaz de implementar estrategias de tomar decisiones basadas en el estado actual del juego. Debes usar alguna técnica de aprendizaje automático para crear un agente que siempre supere a un agente aleatorio. Puedes utilizar cualquiera de las tecnicas vistas en clase, incluyendo aprendizaje por refuerzo, aprendizaje supervisado o aprendizaje no supervisado.

El código debe ser utilizado de la siguiente manera:

```python
import gymnasium as gym
from agents.turing.agent import TuringAgent
from agents.turing.state_extractor import StateExtractor

agent = TuringAgent()
extractor = StateExtractor()
env = gym.make('ALE/Centipede-v5', render_mode='human')
obs, _ = env.reset()

while True:
    state = extractor.extract(obs)
    action = agent.action(state)
    obs, reward, terminated, truncated, info = env.step(action)
    if terminated or truncated:
        break

print(agent.name())
```
```bash
>> {nombre:'Alan', apellido:'Turing', legajo:123456}
```

**Fecha de entrega**: Domingo 12 de Noviembre a las 23:59

## Librerías
Además de las librerias base de Python 3.11 , se permite el uso de las siguientes librerías extra:
- numpy
- SciPy
- pandas
- scikit-learn 1.3
- pytorch 2.
- tensorflow 2.9

  
Hacer un pull request con anticiapción para asegurarse que las dependecias andan bien, por cualquier otra librería consultar con la cátedra. En caso de usar un modelo debe esntregarse el código que se usó para entrenarlo, no hace falta entregar la base de datos. El agente debe cargar un modelo pre entrenado guardado en disco, no se puede pushear ningun modelo de mas de 25 Mb de espacio en disco

## Cálculo de la Nota Final


El trabajo será evaluado en tres dimensiones:

1. **Correctitud de las implementaciones**: Verificación de que las implementaciones se ajustan a las especificaciones y realizan las tareas requeridas.
2. **Rendimiento del agente**: Haremos que tu agente juegue 5 partidas y calcularemos el puntaje promedio de puntos obtenido.
3. **Calidad del código y documentación**: Se evaluará la legibilidad, modularidad y calidad general del código, así como la documentación adjunta en forma de comentarios.

El primer checkpoint debe estar aprobada para poder entregar el TP.

La nota final será calculada utilizando la siguiente fórmula:

$`
Nota = 2 \times \text{implementaciones} + 2 \times \text{calidad\_y\_documentacion} + 6 \times \frac{(\text{Puntaje\_avg} - \text{Puntaje\_min})}{(\text{Puntaje\_max} - \text{Puntaje\_min})}
`$

Donde Puntaje_avg es el puntaje promedio obtenido por tu agente en 5 partidas de prueba, Puntaje_min es el puntaje obtenido por un agente que juega de manera aleatoria y Puntaje_max es el puntaje más alto optenido por cualquier agente de este curso.

Se proporcionarán más detalles acerca de las expectativas el checkpoint-0 y la entrega final durante las clases. Este trabajo debe ser realizado individualmente. ¡Buena suerte!
